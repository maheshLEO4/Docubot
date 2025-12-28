import os
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
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
            # Use the new import path
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            # Fallback if new package not installed
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except ImportError:
                # Direct use of sentence-transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                class SimpleEmbeddings:
                    def embed_query(self, text: str):
                        return model.encode(text).tolist()
                    def embed_documents(self, texts: List[str]):
                        return [model.encode(text).tolist() for text in texts]
                
                return SimpleEmbeddings()
    
    @st.cache_resource(ttl=300)
    def _get_qdrant_client(_self):
        """Cached Qdrant client"""
        try:
            qdrant_config = config.get_qdrant_config()
            if not qdrant_config['api_key'] or not qdrant_config['url']:
                print("Qdrant configuration missing")
                return None
            
            client = QdrantClient(
                url=qdrant_config['url'],
                api_key=qdrant_config['api_key'],
                timeout=30,
                check_compatibility=False
            )
            
            # Test connection
            try:
                client.get_collections()
                return client
            except Exception as e:
                print(f"Qdrant connection failed: {e}")
                return None
                
        except Exception as e:
            print(f"Error creating Qdrant client: {e}")
            return None
    
    def get_store(self):
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
            
            # Use QdrantVectorStore instead of Qdrant
            try:
                # Try the new class name
                from langchain_qdrant import QdrantVectorStore
                store = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embeddings=embeddings
                )
            except (ImportError, AttributeError):
                # Fallback to old name
                try:
                    from langchain_qdrant import Qdrant
                    store = Qdrant(
                        client=client,
                        collection_name=self.collection_name,
                        embeddings=embeddings
                    )
                except (ImportError, AttributeError):
                    # Fallback to community version
                    from langchain_community.vectorstores import Qdrant
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
            store = self.get_store()
            if not store:
                return False
            
            # Try a simple search to see if there's data
            try:
                results = store.similarity_search("test", k=1)
                return len(results) > 0
            except:
                return False
                
        except Exception as e:
            print(f"Error checking existence: {e}")
            return False
    
    def add_documents(self, documents: List, doc_type: str = "pdf") -> bool:
        """Add documents to vector store"""
        try:
            if not documents:
                return False
            
            store = self.get_store()
            if not store:
                return False
            
            store.add_documents(documents)
            
            # Clear cache
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
            
            try:
                client.delete_collection(self.collection_name)
            except:
                pass  # Collection might not exist
            
            # Clear cache
            cache_key = f"store_{self.user_id}"
            if cache_key in st.session_state.vector_cache:
                del st.session_state.vector_cache[cache_key]
            
            return True
            
        except Exception as e:
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
                return True  # Collection doesn't exist
            
            # Delete using filter
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