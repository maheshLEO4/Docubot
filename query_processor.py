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
# IMPROVED HYBRID RETRIEVER
# ==========================
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float] = None
    final_k: int = 5  # Limit final results
    
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
                    # Create unique key for deduplication (FIXED: content + source + page)
                    source = doc.metadata.get('source', 'unknown')
                    page = doc.metadata.get('page', 0)
                    
                    # Clean content for better deduplication
                    content_clean = doc.page_content.lower().strip()
                    content_clean = ' '.join(content_clean.split())  # Remove extra whitespace
                    
                    # Create key from content hash + source + page
                    content_hash = hashlib.md5(content_clean[:500].encode()).hexdigest()[:10]
                    key = f"{source}_{page}_{content_hash}"
                    
                    # Calculate score (higher rank = better)
                    position_score = 1.0 / (i + 1)  # 1st: 1.0, 2nd: 0.5, 3rd: 0.33, etc.
                    
                    # Apply weight if provided (Intelligent weighting)
                    if self.weights and idx < len(self.weights):
                        position_score *= self.weights[idx]
                    
                    # Store retriever type for identification
                    retriever_type = "bm25" if idx == 0 else "vector"
                    
                    all_docs_with_scores.append({
                        'doc': doc,
                        'key': key,
                        'position_score': position_score,
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
        
        # Sort by position score (highest first) - Intelligent ranking
        for item in sorted(all_docs_with_scores, key=lambda x: x['position_score'], reverse=True):
            if item['key'] not in seen_keys:
                seen_keys.add(item['key'])
                unique_docs.append(item['doc'])
                
                if len(unique_docs) >= self.final_k:  # Limit final results
                    break
        
        return unique_docs

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

        # Create improved hybrid retriever with intelligent weighting
        retriever = HybridRetriever(
            retrievers=[bm25, vector_retriever] if bm25 else [vector_retriever],
            weights=[0.3, 0.7] if bm25 else [1.0],  # Vector search gets more weight
            final_k=5  # Limit to 5 best results
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
# SOURCE FORMATTER
# ==========================
def format_source_documents(docs):
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        if isinstance(page, int):
            page += 1

        sources.append({
            "document": os.path.basename(source),
            "page": page,
            "excerpt": doc.page_content[:200] + "...",
            "type": "web" if 'http' in source.lower() else "pdf"
        })
    return sources

# ==========================
# PROCESS QUERY
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
            workflow = AgentWorkflow()
            result = workflow.full_pipeline(prompt, retriever)
            docs = retriever.invoke(prompt)
            return {
                "success": True,
                "answer": result.get("draft_answer"),
                "sources": format_source_documents(docs[:5]),
                "verification_report": result.get("verification_report")
            }

        # Classic mode
        docs = retriever.invoke(prompt)
        context = "\n".join(d.page_content for d in docs)
        message = chat_prompt.format(input=prompt, context=context)
        answer = llm.invoke(message)

        return {
            "success": True,
            "answer": answer.content,
            "sources": format_source_documents(docs[:5]),
            "verification_report": None
        }

    except Exception as e:
        logger.exception("Query failed")
        return {"success": False, "error": str(e)}
