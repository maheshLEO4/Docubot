# Fix the imports and connection logic:

import os
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import streamlit as st
from config import config

class VectorStoreManager:
    """Optimized vector store manager with caching"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection_name = f"docubot_{user_id}"
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache in session state"""
        if 'vector_cache' not in st.session_state:
            st.session_state.vector_cache = {}
    
    @st.cache_resource(ttl=3600)
    def _get_embedding_model(_self):
        """Cached embedding model"""
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return None
    
    @st.cache_resource(ttl=300)
    def _get_qdrant_client(_self):
        """Cached Qdrant client"""
        try:
            qdrant_config = config.get_qdrant_config()
            if not qdrant_config['api_key'] or not qdrant_config['url']:
                st.error("Qdrant configuration missing. Check your API keys.")
                return None
            
            client = QdrantClient(
                url=qdrant_config['url'],
                api_key=qdrant_config['api_key'],
                timeout=30,
            )
            
            # Test connection
            client.get_collections()
            return client
            
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            st.error(f"Failed to connect to Qdrant: {e}")
            return None
    
    def get_store(self) -> Optional[Qdrant]:
        """Get vector store for current user"""
        cache_key = f"store_{self.user_id}"
        
        if cache_key in st.session_state.vector_cache:
            return st.session_state.vector_cache[cache_key]
        
        try:
            client = self._get_qdrant_client()
            if not client:
                return None
            
            embeddings = self._get_embedding_model()
            if not embeddings:
                return None
            
            # Create collection if it doesn't exist
            try:
                client.get_collection(self.collection_name)
            except Exception:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            
            store = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=embeddings
            )
            
            # Cache the store
            st.session_state.vector_cache[cache_key] = store
            return store
            
        except Exception as e:
            print(f"Error getting vector store: {e}")
            st.error(f"Failed to initialize vector store: {e}")
            return None
    
    def add_documents(self, documents: List, doc_type: str = "pdf") -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                return False
            
            store = self.get_store()
            if not store:
                st.error("Vector store not available")
                return False
            
            # Add documents
            store.add_documents(documents)
            
            # Clear cache to force refresh
            cache_key = f"store_{self.user_id}"
            if cache_key in st.session_state.vector_cache:
                del st.session_state.vector_cache[cache_key]
            
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            st.error(f"Failed to add documents: {e}")
            return False