import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from data_processing import get_document_chunks, save_uploaded_files, load_pdf_files, split_documents_into_chunks
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
    """Clear all data from Qdrant for a user"""
    try:
        client = get_qdrant_client()
        collection_name = get_user_collection_name(user_id)
        client.delete_collection(collection_name)
        print(f"🗑️ Cleared Qdrant collection: {collection_name}")
        return "Cleared vector store"
    except Exception as e:
        print(f"⚠️ Error clearing Qdrant data: {e}")
        return "Cleared vector store"

def remove_documents_from_store(user_id, source, doc_type, db_manager=None):
    """Remove documents from Qdrant and optionally clean temp files"""
    client = get_qdrant_client()
    collection = get_user_collection_name(user_id)
    
    try:
        # Find ALL points to delete from Qdrant
        all_delete_ids = []
        next_offset = None
        deleted_count = 0
        
        while True:
            # Scroll through all points
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            # Check each point for matching source
            for p in points:
                meta = (p.payload or {}).get("metadata", {})
                stored = str(meta.get("source", ""))
                
                should_delete = False
                if doc_type == "pdf":
                    # For PDFs, check if filename matches (end of path)
                    if stored.endswith(source) or os.path.basename(stored) == source:
                        should_delete = True
                        deleted_count += 1
                else:  # web
                    # For web, check if URL matches
                    if stored == source or source in stored:
                        should_delete = True
                        deleted_count += 1
                
                if should_delete:
                    all_delete_ids.append(p.id)
            
            if next_offset is None:
                break
        
        # Delete from Qdrant
        if all_delete_ids:
            print(f"🗑️ Deleting {len(all_delete_ids)} chunks from Qdrant for {source}")
            client.delete(collection_name=collection, points_selector=all_delete_ids)
        
        # If db_manager provided and it's a PDF, also clean temp files
        if db_manager and doc_type == "pdf":
            try:
                # Clean temp files
                from data_processing import get_user_data_path
                data_path = get_user_data_path(user_id)
                if data_path and os.path.exists(data_path):
                    file_path = os.path.join(data_path, source)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"🗑️ Deleted temp file: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")
        
        return deleted_count > 0  # Return True if anything was deleted
        
    except Exception as e:
        print(f"❌ Error deleting documents: {e}")
        return False

# ==========================
# BUILDERS
# ==========================
def build_vector_store_from_pdfs(user_id, uploaded_files, append=False):
    """Build vector store from uploaded PDF files and log to MongoDB"""
    if not append:
        clear_all_data(user_id)
        db_manager.clear_user_data(user_id)  # Clear MongoDB too

    store = get_qdrant_vector_store(user_id)
    
    # Save uploaded files to temp storage
    file_paths = save_uploaded_files(uploaded_files, user_id)
    
    # Get chunks from these files
    all_chunks = []
    file_stats = []  # To track file info for MongoDB
    
    for file_path in file_paths:
        try:
            # Load documents from this PDF
            documents = load_pdf_files([file_path])
            if documents:
                # Split into chunks
                chunks = split_documents_into_chunks(documents)
                all_chunks.extend(chunks)
                
                # Count pages for this file
                pages = len(documents)
                file_stats.append({
                    'filename': os.path.basename(file_path),
                    'pages': pages,
                    'chunks': len(chunks)
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if all_chunks:
        # Add to Qdrant
        store.add_documents(all_chunks)
        
        # Log to MongoDB for each file
        for file, stats in zip(uploaded_files, file_stats):
            db_manager.log_file_upload(
                user_id=user_id,
                filename=stats['filename'],
                file_size=file.size,
                pages_processed=stats['pages']
            )
        
        print(f"✅ Added {len(all_chunks)} chunks from {len(file_stats)} files to Qdrant")
        print(f"✅ Logged {len(file_stats)} files to MongoDB")
        return store, "added"
    
    return None, "no_documents"

def build_vector_store_from_urls(user_id, urls, append=False):
    """Build vector store from URLs and log to MongoDB"""
    if not append:
        clear_all_data(user_id)
        db_manager.clear_user_data(user_id)

    store = get_qdrant_vector_store(user_id)
    chunks = scrape_urls_to_chunks(urls)

    if chunks:
        store.add_documents(chunks)
        
        # Log to MongoDB
        successful_urls = []
        for chunk in chunks:
            url = chunk.metadata.get('source')
            if url and url not in successful_urls:
                successful_urls.append(url)
        
        db_manager.log_web_scrape(
            user_id=user_id,
            urls=urls,
            successful_urls=successful_urls,
            total_chunks=len(chunks)
        )
        
        print(f"✅ Added {len(chunks)} chunks from {len(successful_urls)} URLs to Qdrant")
        print(f"✅ Logged {len(successful_urls)} URLs to MongoDB")
        return store, "added"
    
    return None, "no_new_urls"

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
