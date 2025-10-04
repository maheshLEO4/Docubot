import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from data_processing import get_document_chunks
from web_scraper import scrape_urls_to_chunks
from auth import AuthManager
from config import get_qdrant_config
from database import MongoDBManager

# Initialize managers
auth_manager = AuthManager()
db_manager = MongoDBManager()

def get_user_collection_name():
    """Get user-specific Qdrant collection name"""
    return auth_manager.get_user_collection_name()

def get_qdrant_vector_store(collection_name=None):
    """Get Qdrant vector store for current user"""
    if collection_name is None:
        collection_name = get_user_collection_name()
    
    qdrant_config = get_qdrant_config()
    if not qdrant_config['api_key'] or not qdrant_config['url']:
        raise ValueError("Qdrant Cloud not configured")
    
    try:
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key']
        )
        
        embedding_model = get_embedding_model()
        
        # Create collection if it doesn't exist
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model
        )
        
        return vector_store
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        raise

def get_embedding_model():
    """Cached embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def clear_all_data():
    """Clear user's Qdrant collection and MongoDB records"""
    user_id = auth_manager.get_user_id()
    if not user_id:
        return "No user logged in"
    
    try:
        # Clear Qdrant collection
        qdrant_config = get_qdrant_config()
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key']
        )
        collection_name = get_user_collection_name()
        client.delete_collection(collection_name=collection_name)
        
        # Note: We don't delete MongoDB records for audit purposes
        # You can add soft delete if needed
        
        return "🧹 Cleared all vector data!"
    except Exception as e:
        return f"Error clearing data: {str(e)}"

def build_vector_store_from_pdfs(uploaded_files, append=False):
    """Build vector store from PDF documents"""
    user_id = auth_manager.get_user_id()
    if not user_id:
        raise ValueError("User not authenticated")
    
    try:
        # Get or create vector store
        if append:
            db = get_qdrant_vector_store()
        else:
            # For replace mode, clear existing collection
            clear_all_data()
            db = get_qdrant_vector_store()
        
        # Process PDFs and get chunks
        chunks, processed_files = get_document_chunks()
        
        if not chunks:
            return None, "no_documents"
        
        # Add documents to vector store
        db.add_documents(chunks)
        
        # Log file uploads in MongoDB
        for filename in processed_files:
            db_manager.log_file_upload(
                user_id=user_id,
                filename=filename,
                file_size=os.path.getsize(filename) if os.path.exists(filename) else 0,
                pages_processed=len([chunk for chunk in chunks if chunk.metadata.get('source') == filename])
            )
        
        return db, "created" if not append else "updated"
        
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None, "failed"

def build_vector_store_from_urls(urls, append=False):
    """Build vector store from webpage URLs"""
    user_id = auth_manager.get_user_id()
    if not user_id:
        raise ValueError("User not authenticated")
    
    try:
        # Get or create vector store
        if append:
            db = get_qdrant_vector_store()
        else:
            # For replace mode, clear existing collection
            clear_all_data()
            db = get_qdrant_vector_store()
        
        # Scrape URLs and get chunks
        chunks = scrape_urls_to_chunks(urls)
        
        if not chunks:
            return None, "failed"
        
        # Add documents to vector store
        db.add_documents(chunks)
        
        # Log web scrape in MongoDB
        successful_urls = list(set([chunk.metadata.get('source') for chunk in chunks]))
        db_manager.log_web_scrape(
            user_id=user_id,
            urls=urls,
            successful_urls=successful_urls,
            total_chunks=len(chunks)
        )
        
        return db, "created" if not append else "updated"
        
    except Exception as e:
        print(f"Error building vector store from URLs: {e}")
        return None, "failed"

def get_vector_store():
    """Get vector store for current user"""
    return get_qdrant_vector_store()

def vector_store_exists():
    """Check if vector store exists for current user"""
    try:
        qdrant_config = get_qdrant_config()
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key']
        )
        collection_name = get_user_collection_name()
        collection_info = client.get_collection(collection_name=collection_name)
        return collection_info.points_count > 0
    except:
        return False