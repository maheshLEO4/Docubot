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

    # FIXED: Better error handling for collection creation
    try:
        # Try to get the collection
        collection_info = client.get_collection(collection_name)
        print(f"✅ Found existing Qdrant collection: {collection_name}")
        print(f"   Points count: {collection_info.points_count}")
    except Exception as e:
        # Collection doesn't exist, create it
        print(f"⚠️ Collection '{collection_name}' not found, creating it...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # This MUST match the embedding model dimension
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created new Qdrant collection: {collection_name}")
        except Exception as create_error:
            print(f"❌ FAILED to create collection '{collection_name}': {create_error}")
            # Check if it's a permission issue
            if "permission" in str(create_error).lower() or "forbidden" in str(create_error).lower():
                print("💡 Possible issue: Your Qdrant API key might not have create collection permissions")
            raise create_error

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
            print(f"⚠️ BM25: Collection '{collection_name}' doesn't exist yet")
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
            print(f"⚠️ BM25: No points found in collection '{collection_name}'")
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
            print(f"⚠️ BM25: No valid documents found in points")
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
        
        # Check if collection exists before trying to delete
        try:
            client.get_collection(collection_name)
            client.delete_collection(collection_name)
            print(f"🗑️ Cleared Qdrant collection: {collection_name}")
            return "Cleared vector store"
        except Exception as e:
            # Collection doesn't exist, that's fine
            print(f"⚠️ Collection '{collection_name}' doesn't exist, nothing to clear")
            return "Collection didn't exist"
            
    except Exception as e:
        print(f"⚠️ Error in clear_all_data: {e}")
        return f"Error: {e}"

def remove_documents_from_store(user_id, source, doc_type, db_manager=None):
    """Remove documents from Qdrant and optionally clean temp files"""
    client = get_qdrant_client()
    collection = get_user_collection_name(user_id)
    
    try:
        # First check if collection exists
        try:
            client.get_collection(collection)
        except Exception:
            print(f"⚠️ Collection '{collection}' doesn't exist, nothing to delete")
            return False
        
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
            return True
        
        print(f"⚠️ No documents found to delete for {source}")
        return False
        
    except Exception as e:
        print(f"❌ Error deleting documents: {e}")
        return False

# ==========================
# BUILDERS
# ==========================
def build_vector_store_from_pdfs(user_id, uploaded_files, append=False):
    """Build vector store from uploaded PDF files and log to MongoDB"""
    print(f"📥 Starting PDF processing for user {user_id}")
    print(f"   Mode: {'Append' if append else 'Replace'}")
    print(f"   Files: {[f.name for f in uploaded_files]}")
    
    if not append:
        print("🔄 Clearing existing data...")
        clear_all_data(user_id)
        db_manager.clear_user_data(user_id)  # Clear MongoDB too
    else:
        print("➕ Adding to existing data...")
    
    # FIXED: Get vector store FIRST (this creates collection if needed)
    try:
        store = get_qdrant_vector_store(user_id)
    except Exception as e:
        print(f"❌ FAILED to get/create vector store: {e}")
        return None, "failed"
    
    # Save uploaded files to temp storage
    try:
        file_paths = save_uploaded_files(uploaded_files, user_id)
        print(f"✅ Saved {len(file_paths)} files to temp storage")
    except Exception as e:
        print(f"❌ Failed to save files: {e}")
        return None, "failed"
    
    # Get chunks from these files
    all_chunks = []
    file_stats = []  # To track file info for MongoDB
    
    for file_path in file_paths:
        try:
            print(f"📄 Processing: {os.path.basename(file_path)}")
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
                print(f"   Created {len(chunks)} chunks from {pages} pages")
            else:
                print(f"⚠️ No documents loaded from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    if all_chunks:
        # Add to Qdrant
        print(f"📤 Adding {len(all_chunks)} total chunks to Qdrant...")
        try:
            store.add_documents(all_chunks)
            print(f"✅ Added {len(all_chunks)} chunks to Qdrant")
        except Exception as e:
            print(f"❌ Failed to add documents to Qdrant: {e}")
            return None, "failed"
        
        # Log to MongoDB for each file
        print("📝 Logging files to MongoDB...")
        for file, stats in zip(uploaded_files, file_stats):
            try:
                db_manager.log_file_upload(
                    user_id=user_id,
                    filename=stats['filename'],
                    file_size=file.size,
                    pages_processed=stats['pages']
                )
                print(f"   Logged: {stats['filename']} ({stats['pages']} pages)")
            except Exception as e:
                print(f"⚠️ Failed to log {stats['filename']} to MongoDB: {e}")
        
        print(f"✅ Successfully processed {len(file_stats)} files")
        return store, "added"
    
    print("❌ No chunks were created from the uploaded files")
    return None, "no_documents"

def build_vector_store_from_urls(user_id, urls, append=False):
    """Build vector store from URLs and log to MongoDB"""
    print(f"🌐 Starting URL processing for user {user_id}")
    print(f"   URLs: {urls}")
    
    if not append:
        print("🔄 Clearing existing data...")
        clear_all_data(user_id)
        db_manager.clear_user_data(user_id)
    else:
        print("➕ Adding to existing data...")
    
    # FIXED: Get vector store FIRST (this creates collection if needed)
    try:
        store = get_qdrant_vector_store(user_id)
    except Exception as e:
        print(f"❌ FAILED to get/create vector store: {e}")
        return None, "failed"
    
    chunks = scrape_urls_to_chunks(urls)

    if chunks:
        print(f"📤 Adding {len(chunks)} chunks to Qdrant...")
        try:
            store.add_documents(chunks)
            print(f"✅ Added {len(chunks)} chunks to Qdrant")
        except Exception as e:
            print(f"❌ Failed to add documents to Qdrant: {e}")
            return None, "failed"
        
        # Log to MongoDB
        print("📝 Logging URLs to MongoDB...")
        successful_urls = []
        for chunk in chunks:
            url = chunk.metadata.get('source')
            if url and url not in successful_urls:
                successful_urls.append(url)
        
        try:
            db_manager.log_web_scrape(
                user_id=user_id,
                urls=urls,
                successful_urls=successful_urls,
                total_chunks=len(chunks)
            )
            print(f"✅ Logged {len(successful_urls)} URLs to MongoDB")
        except Exception as e:
            print(f"⚠️ Failed to log URLs to MongoDB: {e}")
        
        return store, "added"
    
    print("❌ No chunks were created from the URLs")
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
        exists = info.points_count > 0
        print(f"🔍 Vector store exists check: {exists} (points: {info.points_count})")
        return exists
    except Exception as e:
        print(f"🔍 Vector store doesn't exist: {e}")
        return False
