import os
import streamlit as st
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from agents.workflow import AgentWorkflow
from vector_store import get_vector_store, get_bm25_retriever

logger = logging.getLogger(__name__)

# ==========================
# HYBRID RETRIEVER (ORIGINAL VERSION - ONLY ADDED RESULT LIMIT)
# ==========================
class HybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = []
        seen = set()

        for retriever in self.retrievers:
            results = retriever.invoke(
                query,
                config={"callbacks": run_manager.get_child() if run_manager else None}
            )
            for doc in results:
                key = hash(doc.page_content)
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
                    
                    # ✅ ONLY ADDITION: Limit to 5 results total
                    if len(docs) >= 5:
                        return docs

        return docs

# ==========================
# QA CHAIN (EXACTLY ORIGINAL)
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

        # ✅ ORIGINAL: No weights, just the retriever list
        retriever = HybridRetriever(retrievers=[bm25, vector_retriever] if bm25 else [vector_retriever])

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

    except Exception:
        logger.exception("Error creating QA chain")
        return None

# ==========================
# SOURCE FORMATTER (EXACTLY ORIGINAL)
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
            "type": "web" if doc.get('type') == 'web' else "pdf"  # ✅ ORIGINAL
        })
    return sources

# ==========================
# PROCESS QUERY (EXACTLY ORIGINAL WITH VERIFICATION)
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
                "verification_report": result.get("verification_report")  # ✅ ORIGINAL VERIFICATION
            }

        docs = retriever.invoke(prompt)
        context = "\n".join(d.page_content for d in docs)
        message = chat_prompt.format(input=prompt, context=context)
        answer = llm.invoke(message)

        return {
            "success": True,
            "answer": answer.content,
            "sources": format_source_documents(docs[:5]),
            "verification_report": None  # ✅ ORIGINAL: None for classic mode
        }

    except Exception as e:
        logger.exception("Query failed")
        return {"success": False, "error": str(e)}
