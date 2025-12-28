import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from config import get_config
from data_processing import get_document_chunks

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(user_id):
    client = QdrantClient(
        url=get_config('QDRANT_URL'), 
        api_key=get_config('QDRANT_API_KEY')
    )
    return Qdrant(
        client=client, 
        collection_name=f"user_{user_id}", 
        embeddings=get_embeddings()
    )

def build_vector_store_from_pdfs(user_id, file_paths):
    chunks = get_document_chunks(user_id, file_paths)
    vector_store = get_vector_store(user_id)
    vector_store.add_documents(chunks)