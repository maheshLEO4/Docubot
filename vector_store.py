import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from data_processing import get_document_chunks
from web_scraper import scrape_urls_to_chunks
from config import get_qdrant_config
from database import MongoDBManager

db_manager = MongoDBManager()

# ==========================
# EMBEDDINGS
# ==========================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

# ==========================
# QDRANT
# ==========================
def get_user_collection_name(user_id):
    return f"docubot_user_{user_id}" if user_id else "docubot_default"

@st.cache_resource
def get_qdrant_client():
    cfg = get_qdrant_config()
    return QdrantClient(
        url=cfg["url"],
        api_key=cfg["api_key"],
        timeout=30,
    )

@st.cache_resource
def get_qdrant_vector_store(user_id):
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    collection_name = get_user_collection_name(user_id)

    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

# ==========================
# BM25
# ==========================
@st.cache_resource(show_spinner=False)
def get_bm25_retriever(user_id):
    chunks, _ = get_document_chunks(user_id)
    if not chunks:
        return None

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return bm25

# ==========================
# DATA MANAGEMENT
# ==========================
def clear_all_data(user_id):
    client = get_qdrant_client()
    client.delete_collection(get_user_collection_name(user_id))
    return "Cleared vector store"

def remove_documents_from_store(user_id, source, doc_type):
    client = get_qdrant_client()
    collection = get_user_collection_name(user_id)

    points, _ = client.scroll(collection_name=collection, limit=10_000)
    delete_ids = []

    for p in points:
        meta = (p.payload or {}).get("metadata", {})
        stored = meta.get("source", "")

        if doc_type == "pdf":
            if stored.endswith(source):
                delete_ids.append(p.id)
        else:
            if stored == source:
                delete_ids.append(p.id)

    if delete_ids:
        client.delete(collection_name=collection, points_selector=delete_ids)

    return True

# ==========================
# BUILDERS
# ==========================
def build_vector_store_from_pdfs(user_id, uploaded_files, append=False):
    if not append:
        clear_all_data(user_id)

    store = get_qdrant_vector_store(user_id)
    chunks, files = get_document_chunks(user_id)

    if chunks:
        store.add_documents(chunks)

    return store, "created"

def build_vector_store_from_urls(user_id, urls, append=False):
    if not append:
        clear_all_data(user_id)

    store = get_qdrant_vector_store(user_id)
    chunks = scrape_urls_to_chunks(urls)

    if chunks:
        store.add_documents(chunks)

    return store, "created"

# ==========================
# ACCESS
# ==========================
def get_vector_store(user_id):
    return get_qdrant_vector_store(user_id)

def vector_store_exists(user_id):
    try:
        client = get_qdrant_client()
        info = client.get_collection(get_user_collection_name(user_id))
        return info.points_count > 0
    except Exception:
        return False
