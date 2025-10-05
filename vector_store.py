import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from data_processing import get_document_chunks
from web_scraper import scrape_urls_to_chunks
from config import get_qdrant_config
from database import MongoDBManager

# Initialize database manager
db_manager = MongoDBManager()

def get_user_collection_name(user_id):
    """Get user-specific Qdrant collection name"""
    return f"docubot_user_{user_id}" if user_id else "docubot_default"

def get_qdrant_vector_store(user_id, collection_name=None):
    """Get Qdrant vector store for current user"""
    if collection_name is None:
        collection_name = get_user_collection_name(user_id)
    
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

def clear_all_data(user_id):
    """Clear user's Qdrant collection"""
    if not user_id:
        return "No user logged in"
    
    try:
        # Clear Qdrant collection
        qdrant_config = get_qdrant_config()
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key']
        )
        collection_name = get_user_collection_name(user_id)
        client.delete_collection(collection_name=collection_name)
        
        return "🧹 Cleared all vector data!"
    except Exception as e:
        return f"Error clearing data: {str(e)}"

def remove_documents_from_store(user_id, source, doc_type):
    """Remove specific documents from vector store by source"""
    if not user_id:
        return False
    
    try:
        qdrant_config = get_qdrant_config()
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key']
        )
        collection_name = get_user_collection_name(user_id)
        
        # Delete points with matching source in metadata
        # For PDFs, match basename; for URLs, match full URL
        if doc_type == 'pdf':
            # Search for documents with source containing this filename
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=10000
            )
            
            points_to_delete = []
            for point in scroll_result[0]:
                if point.payload and 'metadata' in point.payload:
                    metadata_source = point.payload['metadata'].get('source', '')
                    if os.path.basename(metadata_source) == source or metadata_source.endswith(source):
                        points_to_delete.append(point.id)
            
            if points_to_delete:
                client.delete(
                    collection_name=collection_name,
                    points_selector=points_to_delete
                )
        else:  # web
            # For web URLs, match exact URL
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=10000
            )
            
            points_to_delete = []
            for point in scroll_result[0]:
                if point.payload and 'metadata' in point.payload:
                    metadata_source = point.payload['metadata'].get('source', '')
                    if metadata_source == source:
                        points_to_delete.append(point.id)
            
            if points_to_delete:
                client.delete(
                    collection_name=collection_name,
                    points_selector=points_to_delete
                )
        
        print(f"✅ Removed {len(points_to_delete)} points for source: {source}")
        return True
        
    except Exception as e:
        print(f"Error removing documents: {e}")
        return False

def build_vector_store_from_pdfs(user_id, uploaded_files, append=False):
    """Build vector store from PDF documents"""
    if not user_id:
        raise ValueError("User not authenticated")
    
    try:
        # Get or create vector store
        if append:
            db = get_qdrant_vector_store(user_id)
        else:
            # For replace mode, clear existing collection
            clear_all_data(user_id)
            db = get_qdrant_vector_store(user_id)
        
        # Process PDFs and get chunks
        chunks, processed_files = get_document_chunks(user_id)
        
        if not chunks:
            return None, "no_documents"
        
        # Add documents to vector store
        db.add_documents(chunks)
        
        # Log file uploads in MongoDB
        for filename in processed_files:
            db_manager.log_file_upload(
                user_id=user_id,
                filename=os.path.basename(filename),
                file_size=os.path.getsize(filename) if os.path.exists(filename) else 0,
                pages_processed=len([chunk for chunk in chunks if chunk.metadata.get('source') == filename])
            )
        
        return db, "created" if not append else "updated"
        
    except Exception as e:
        print(f"Error building vector store: {e}")
        return None, "failed"

def build_vector_store_from_urls(user_id, urls, append=False):
    """Build vector store from webpage URLs"""
    if not user_id:
        raise ValueError("User not authenticated")
    
    try:
        # Get or create vector store
        if append:
            db = get_qdrant_vector_store(user_id)
        else:
            # For replace mode, clear existing collection
            clear_all_data(user_id)
            db = get_qdrant_vector_store(user_id)
        
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

def get_vector_store(user_id):
    """Get vector store for current user"""
    return get_qdrant_vector_store(user_id)

def vector_store_exists(user_id):
    """Check if vector store exists for current user"""
    try:
        qdrant_config = get_qdrant_config()
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key']
        )
        collection_name = get_user_collection_name(user_id)
        collection_info = client.get_collection(collection_name=collection_name)
        return collection_info.points_count > 0
    except:
        return False