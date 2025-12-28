import os
from typing import Optional, List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import streamlit as st
from config import config
from database import MongoDBManager

class VectorStoreManager:
    """Optimized vector store manager with caching"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.collection_name = f"docubot_{user_id}"
        self.db_manager = MongoDBManager()
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache in session state"""
        if 'vector_cache' not in st.session_state:
            st.session_state.vector_cache = {}
    
    @st.cache_resource(ttl=3600)
    def _get_embedding_model(_self):
        """Cached embedding model"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    @st.cache_resource(ttl=300)
    def _get_qdrant_client(_self):
        """Cached Qdrant client"""
        qdrant_config = config.get_qdrant_config()
        return QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key'],
            timeout=30,
            prefer_grpc=True
        )
    
    def get_store(self) -> Optional[Qdrant]:
        """Get vector store for current user"""
        cache_key = f"store_{self.user_id}"
        
        if cache_key in st.session_state.vector_cache:
            return st.session_state.vector_cache[cache_key]
        
        try:
            client = self._get_qdrant_client()
            embeddings = self._get_embedding_model()
            
            # Create collection if it doesn't exist
            try:
                client.get_collection(self.collection_name)
            except:
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
            collection_info = client.get_collection(self.collection_name)
            return collection_info.points_count > 0
        except:
            return False
    
    def add_documents(self, documents: List, doc_type: str = "pdf") -> bool:
        """Add documents to vector store"""
        try:
            store = self.get_store()
            if not store:
                return False
            
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
            client.delete_collection(self.collection_name)
            
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
            
            # Scroll through points and delete matching ones
            points_to_delete = []
            offset = None
            
            while True:
                scroll_result = client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset
                )
                
                points = scroll_result[0]
                if not points:
                    break
                
                for point in points:
                    if point.payload and 'metadata' in point.payload:
                        metadata = point.payload['metadata']
                        doc_source = metadata.get('source', '')
                        
                        if doc_type == 'pdf':
                            if os.path.basename(doc_source) == source:
                                points_to_delete.append(point.id)
                        else:  # web
                            if doc_source == source:
                                points_to_delete.append(point.id)
                
                offset = scroll_result[1]
                if offset is None:
                    break
            
            # Delete points
            if points_to_delete:
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=points_to_delete
                )
                print(f"Removed {len(points_to_delete)} points for {source}")
                
                # Clear cache
                cache_key = f"store_{self.user_id}"
                if cache_key in st.session_state.vector_cache:
                    del st.session_state.vector_cache[cache_key]
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error removing document: {e}")
            return False