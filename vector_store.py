import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
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
# BM25 - UPDATED TO USE VECTOR DB TEXT
# ==========================
@st.cache_resource(show_spinner=False)
def get_bm25_retriever(user_id):
    """Get BM25 retriever from ALL content in vector store (PDFs + Websites)"""
    try:
        # Get all documents from Qdrant vector store
        client = get_qdrant_client()
        collection_name = get_user_collection_name(user_id)
        
        # Check if collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist yet
            return None
        
        # Fetch all documents from vector store
        all_points = []
        next_offset = None
        
        # Scroll through all points in collection
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            all_points.extend(points)
            
            if next_offset is None:
                break
        
        if not all_points:
            return None
        
        # Convert points to LangChain Documents
        documents = []
        for point in all_points:
            payload = point.payload or {}
            page_content = payload.get('page_content', '')
            
            if not page_content or len(page_content.strip()) == 0:
                continue
                
            # Extract metadata
            metadata = payload.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Ensure type field exists for consistency
            if 'type' not in metadata:
                if 'scraping_method' in metadata:
                    metadata['type'] = 'web'
                else:
                    metadata['type'] = 'pdf'
            
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        if not documents:
            return None
            
        # Create BM25 retriever
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 5
        
        print(f"✅ BM25 loaded {len(documents)} documents from vector store")
        return bm25
        
    except Exception as e:
        print(f"❌ Error creating BM25 retriever from vector store: {e}")
        return None

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
