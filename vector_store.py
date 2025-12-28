import os
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
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
                print("Qdrant configuration missing. Check your API keys.")
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
            return None
    
    def exists(self) -> bool:
        """Check if vector store exists and has data"""
        try:
            client = self._get_qdrant_client()
            if not client:
                return False
            
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                return False
            
            # Check if collection has data
            collection_info = client.get_collection(self.collection_name)
            return collection_info.points_count > 0
            
        except Exception as e:
            print(f"Error checking vector store existence: {e}")
            return False
    
    def add_documents(self, documents: List, doc_type: str = "pdf") -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                return False
            
            store = self.get_store()
            if not store:
                print("Vector store not available")
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
            return False
    
    def clear(self) -> bool:
        """Clear vector store"""
        try:
            client = self._get_qdrant_client()
            if not client:
                return False
            
            # Delete the collection
            client.delete_collection(self.collection_name)
            
            # Clear cache
            cache_key = f"store_{self.user_id}"
            if cache_key in st.session_state.vector_cache:
                del st.session_state.vector_cache[cache_key]
            
            return True
            
        except Exception as e:
            # If collection doesn't exist, that's fine
            if "not found" in str(e).lower():
                return True
            print(f"Error clearing vector store: {e}")
            return False
    
    def remove_document(self, source: str, doc_type: str) -> bool:
        """Remove specific document from store"""
        try:
            client = self._get_qdrant_client()
            if not client:
                return False
            
            # Try to get collection
            try:
                client.get_collection(self.collection_name)
            except:
                # Collection doesn't exist
                return True
            
            # Delete using filter
            if doc_type == 'pdf':
                # For PDFs, match by filename in source
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.source",
                                match=MatchValue(value=source)
                            )
                        ]
                    )
                )
            else:
                # For web, match by exact URL
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.source",
                                match=MatchValue(value=source)
                            )
                        ]
                    )
                )
            
            # Clear cache
            cache_key = f"store_{self.user_id}"
            if cache_key in st.session_state.vector_cache:
                del st.session_state.vector_cache[cache_key]
            
            return True
            
        except Exception as e:
            print(f"Error removing document: {e}")
            return False