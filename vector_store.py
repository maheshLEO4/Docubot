import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
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
# EMBEDDINGS MODEL
# ==========================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# ==========================
# QDRANT HELPERS
# ==========================
def get_user_collection_name(user_id):
    return f"docubot_user_{user_id}" if user_id else "docubot_default"

@st.cache_resource
def get_qdrant_client():
    cfg = get_qdrant_config()
    return QdrantClient(
        url=cfg['url'],
        api_key=cfg['api_key'],
        timeout=30
    )

@st.cache_resource
def get_qdrant_vector_store(user_id):
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    collection_name = get_user_collection_name(user_id)

    # create collection if it doesn't exist
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model
    )

# ==========================
# BM25 RETRIEVER
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
    collection_name = get_user_collection_name(user_id)
    try:
        client.delete_collection(collection_name=collection_name)
        return "Cleared all vector data!"
    except Exception as e:
        return f"Error clearing data: {str(e)}"

def remove_documents_from_store(user_id, source, doc_type):
    client = get_qdrant_client()
    collection_name = get_user_collection_name(user_id)
    scroll_result = client.scroll(collection_name=collection_name, limit=10000)
    points_to_delete = []

    for point in scroll_result[0]:
        if point.payload and 'metadata' in point.payload:
            metadata_source = point.payload['metadata'].get('source', '')
            if doc_type == 'pdf':
                if os.path.basename(metadata_source) == source or metadata_source.endswith(source):
                    points_to_delete.append(point.id)
            else:  # web
                if metadata_source == source:
                    points_to_delete.append(point.id)

    if points_to_delete:
        client.delete(collection_name=collection_name, points_selector=points_to_delete)

    return True

# ==========================
# BUILD VECTOR STORE
# ==========================
def build_vector_store_from_pdfs(user_id, uploaded_files, append=False):
    if not user_id:
        raise ValueError("User not authenticated")

    if not append:
        clear_all_data(user_id)

    db = get_qdrant_vector_store(user_id)
    chunks, processed_files = get_document_chunks(user_id)
    if not chunks:
        return None, "no_documents"

    db.add_documents(chunks)

    for filename in processed_files:
        db_manager.log_file_upload(
            user_id=user_id,
            filename=os.path.basename(filename),
            file_size=os.path.getsize(filename) if os.path.exists(filename) else 0,
            pages_processed=len([c for c in chunks if c.metadata.get('source') == filename])
        )

    return db, "created" if not append else "updated"

def build_vector_store_from_urls(user_id, urls, append=False):
    if not user_id:
        raise ValueError("User not authenticated")

    if not append:
        clear_all_data(user_id)

    db = get_qdrant_vector_store(user_id)
    chunks = scrape_urls_to_chunks(urls)
    if not chunks:
        return None, "failed"

    db.add_documents(chunks)
    successful_urls = list(set(chunk.metadata.get('source') for chunk in chunks))
    db_manager.log_web_scrape(user_id=user_id, urls=urls, successful_urls=successful_urls, total_chunks=len(chunks))

    return db, "created" if not append else "updated"

# ==========================
# ACCESSORS
# ==========================
def get_vector_store(user_id):
    return get_qdrant_vector_store(user_id)

def vector_store_exists(user_id):
    client = get_qdrant_client()
    collection_name = get_user_collection_name(user_id)
    try:
        info = client.get_collection(collection_name=collection_name)
        return info.points_count > 0
    except:
        return False
