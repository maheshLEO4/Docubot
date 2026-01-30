import os
import streamlit as st
import logging
from typing import List, Dict, Tuple
import hashlib

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from agents.workflow import AgentWorkflow
from vector_store import get_vector_store, get_bm25_retriever

logger = logging.getLogger(__name__)

# ==========================
# HYBRID RETRIEVER WITH SCORES
# ==========================
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float] = None
    final_k: int = 6
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        all_docs_with_scores = []
        
        for idx, retriever in enumerate(self.retrievers):
            try:
                results = retriever.invoke(
                    query,
                    config={"callbacks": run_manager.get_child() if run_manager else None}
                )
                
                # Add scores based on position and retriever weight
                for i, doc in enumerate(results):
                    # Create unique key for deduplication
                    source = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', 0)
                    content_preview = doc.page_content[:100].replace(' ', '_')
                    key = f"{source}_{page}_{hashlib.md5(content_preview.encode()).hexdigest()[:10]}"
                    
                    # Calculate score (higher rank = better)
                    position_score = 1.0 / (i + 1)  # 1st: 1.0, 2nd: 0.5, 3rd: 0.33, etc.
                    
                    # Apply weight if provided
                    if self.weights and idx < len(self.weights):
                        position_score *= self.weights[idx]
                    
                    # Store retriever type for identification
                    retriever_type = "bm25" if idx == 0 else "vector"
                    
                    # Store the original score if available
                    original_score = None
                    if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                        original_score = doc.metadata.get('score')
                    
                    all_docs_with_scores.append({
                        'doc': doc,
                        'key': key,
                        'position_score': position_score,
                        'original_score': original_score,
                        'retriever_type': retriever_type,
                        'retriever_idx': idx,
                        'position': i
                    })
            except Exception as e:
                logger.error(f"Retriever {idx} failed: {e}")
                continue
        
        # Deduplicate and sort by combined score
        seen_keys = set()
        unique_docs = []
        
        # Sort by position score (highest first)
        for item in sorted(all_docs_with_scores, key=lambda x: x['position_score'], reverse=True):
            if item['key'] not in seen_keys:
                seen_keys.add(item['key'])
                
                # Add score information to metadata
                doc = item['doc']
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                
                # Store scoring information
                doc.metadata['retriever_scores'] = {
                    'retriever_type': item['retriever_type'],
                    'position_score': item['position_score'],
                    'original_score': item['original_score'],
                    'position': item['position'],
                    'retriever_idx': item['retriever_idx'],
                    'combined_score': item['position_score']
                }
                
                unique_docs.append(doc)
                
                if len(unique_docs) >= self.final_k:
                    break
        
        return unique_docs
    
    def get_detailed_scores(self, query: str) -> Dict:
        """Get detailed scoring information for all retrievers"""
        scores_info = {
            'query': query,
            'bm25_results': [],
            'vector_results': [],
            'combined_results': []
        }
        
        try:
            # Get BM25 scores if available
            if len(self.retrievers) > 0 and hasattr(self.retrievers[0], 'get_scores'):
                bm25_scores = self.retrievers[0].get_scores(query)
                scores_info['bm25_results'] = bm25_scores
            
            # Get results from each retriever
            all_items = []
            for idx, retriever in enumerate(self.retrievers):
                results = retriever.invoke(query)
                retriever_type = "bm25" if idx == 0 else "vector"
                
                for i, doc in enumerate(results):
                    item = {
                        'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        'source': doc.metadata.get('source', 'unknown'),
                        'retriever_type': retriever_type,
                        'position': i,
                        'position_score': 1.0 / (i + 1)
                    }
                    
                    if retriever_type == "bm25":
                        scores_info['bm25_results'].append(item)
                    else:
                        scores_info['vector_results'].append(item)
                    
                    all_items.append(item)
            
            # Sort combined results
            scores_info['combined_results'] = sorted(
                all_items, 
                key=lambda x: x['position_score'], 
                reverse=True
            )[:self.final_k]
            
        except Exception as e:
            logger.error(f"Error getting detailed scores: {e}")
        
        return scores_info

# ==========================
# QA CHAIN
# ==========================
@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(groq_api_key, user_id):
    try:
        vector_store = get_vector_store(user_id)
        if not vector_store:
            return None

        system_prompt = (
            "Use the provided context to answer the question. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        bm25 = get_bm25_retriever(user_id)

        retriever = HybridRetriever(
            retrievers=[bm25, vector_retriever] if bm25 else [vector_retriever],
            weights=[0.3, 0.7] if bm25 else [1.0],  # Give vector search more weight
            final_k=5
        )

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=groq_api_key
        )

        return {
            "retriever": retriever,
            "llm": llm,
            "prompt": prompt
        }

    except Exception as e:
        logger.exception(f"Error creating QA chain: {e}")
        return None

# ==========================
# SOURCE FORMATTER WITH SCORES
# ==========================
def format_source_documents(docs):
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        if isinstance(page, int):
            page += 1
        
        # Extract scoring information
        scores_info = meta.get('retriever_scores', {})
        retriever_type = scores_info.get('retriever_type', 'unknown')
        position_score = scores_info.get('position_score', 0)
        original_score = scores_info.get('original_score')
        
        # Format score for display
        score_display = ""
        if retriever_type == 'bm25':
            score_display = f"BM25 Score: {position_score:.3f}"
        elif retriever_type == 'vector':
            score_display = f"Vector Score: {position_score:.3f}"
        
        if original_score is not None:
            score_display += f" (Original: {original_score:.3f})"
        
        sources.append({
            "document": os.path.basename(source),
            "page": page,
            "excerpt": doc.page_content[:200] + "...",
            "type": "web" if 'http' in source.lower() else "pdf",
            "retriever_type": retriever_type,
            "score": score_display,
            "raw_score": position_score,
            "original_score": original_score
        })
    
    # Sort by score (highest first)
    sources.sort(key=lambda x: x['raw_score'], reverse=True)
    return sources

# ==========================
# PROCESS QUERY WITH SCORES
# ==========================
def process_query(prompt, groq_api_key, user_id, use_agentic=True):
    qa = get_cached_qa_chain(groq_api_key, user_id)
    if not qa:
        return {"success": False, "error": "Knowledge base not ready"}

    retriever = qa["retriever"]
    llm = qa["llm"]
    chat_prompt = qa["prompt"]

    try:
        if use_agentic:
            # Get detailed scores from hybrid retriever
            scores_info = {}
            if hasattr(retriever, 'get_detailed_scores'):
                scores_info = retriever.get_detailed_scores(prompt)
            
            workflow = AgentWorkflow()
            result = workflow.full_pipeline(prompt, retriever)
            docs = retriever.invoke(prompt)
            
            # Add scores information to result
            result_with_scores = {
                "success": True,
                "answer": result.get("draft_answer"),
                "sources": format_source_documents(docs[:5]),
                "verification_report": result.get("verification_report"),
                "retrieval_scores": scores_info  # Add scores to response
            }
            return result_with_scores

        # Classic mode
        docs = retriever.invoke(prompt)
        context = "\n".join(d.page_content for d in docs)
        message = chat_prompt.format(input=prompt, context=context)
        answer = llm.invoke(message)
        
        # Get scores for classic mode too
        scores_info = {}
        if hasattr(retriever, 'get_detailed_scores'):
            scores_info = retriever.get_detailed_scores(prompt)

        return {
            "success": True,
            "answer": answer.content,
            "sources": format_source_documents(docs[:5]),
            "verification_report": None,
            "retrieval_scores": scores_info
        }

    except Exception as e:
        logger.exception("Query failed")
        return {"success": False, "error": str(e)}
