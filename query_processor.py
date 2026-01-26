import os
import streamlit as st
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever

from agents.workflow import AgentWorkflow
from vector_store import get_vector_store, get_bm25_retriever

import logging

logger = logging.getLogger(__name__)

# ==========================
# HYBRID RETRIEVER
# ==========================
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs: List[Document] = []
        seen = set()

        for retriever in self.retrievers:
            results = retriever.invoke(
                query, 
                config={"callbacks": run_manager.get_child() if run_manager else None}
            )
            for doc in results:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(doc)
        return docs

# ==========================
# QA CHAIN (Modernized)
# ==========================
@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(groq_api_key, user_id):
    try:
        db = get_vector_store(user_id)
        if not db:
            return None

        system_prompt = (
            "Use the pieces of information provided in the context to answer user's question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
            "Don't provide anything out of the given context.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # ==========================
        # VECTOR STORE + BM25
        # ==========================
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        vector_store = QdrantVectorStore(
            client=db.qdrant_client,  # assuming get_vector_store exposes qdrant_client
            collection_name=f"user_{user_id}_collection",
            embedding=embeddings,
        )

        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = get_bm25_retriever(user_id)

        if bm25_retriever:
            retriever = HybridRetriever(retrievers=[bm25_retriever, vector_retriever])
        else:
            retriever = vector_retriever

        # ==========================
        # LLM
        # ==========================
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=groq_api_key,
        )

        combine_docs_chain = llm  # modern chains can directly use the LLM
        return {
            "retriever": retriever,
            "llm": llm,
            "prompt": prompt
        }

    except Exception as e:
        logger.exception("Error creating QA chain")
        return None

# ==========================
# SOURCE FORMATTER
# ==========================
def format_source_documents(source_documents):
    formatted_sources = []
    for doc in source_documents:
        try:
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            source_type = "web" if isinstance(source, str) and source.startswith(("http://", "https://")) else "pdf"
            source_name = os.path.basename(str(source)) if source_type == "pdf" else source
            page_num = metadata.get("page", "N/A")
            if isinstance(page_num, int):
                page_num += 1
            excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            formatted_sources.append({
                "document": source_name,
                "page": page_num,
                "excerpt": excerpt,
                "type": source_type
            })
        except Exception as e:
            logger.warning(f"Error formatting source: {e}")
            continue
    return formatted_sources

# ==========================
# PROCESS QUERY
# ==========================
def process_query(prompt, groq_api_key, user_id, use_agentic=True):
    try:
        qa_data = get_cached_qa_chain(groq_api_key, user_id)
        if not qa_data:
            return {"success": False, "error": "Knowledge base not ready. Please add documents first."}

        retriever = qa_data["retriever"]
        llm = qa_data["llm"]
        chat_prompt = qa_data["prompt"]

        if use_agentic:
            workflow = AgentWorkflow()
            agent_result = workflow.full_pipeline(question=prompt, retriever=retriever)
            retrieved_docs = retriever.invoke(prompt)
            return {
                "success": True,
                "answer": agent_result.get("draft_answer", "No answer generated."),
                "sources": format_source_documents(retrieved_docs[:5]),
                "verification_report": agent_result.get("verification_report", "")
            }
        else:
            retrieved_docs = retriever.invoke(prompt)
            # Use LLM + prompt for classical QA
            input_text = chat_prompt.format(input=prompt, context="\n".join([d.page_content for d in retrieved_docs]))
            answer = llm.invoke({"input": input_text}).get("answer", "No answer generated.")
            return {
                "success": True,
                "answer": answer,
                "sources": format_source_documents(retrieved_docs[:5]),
                "verification_report": None
            }

    except Exception as e:
        logger.exception(f"Query processing error: {e}")
        return {"success": False, "error": str(e)}
