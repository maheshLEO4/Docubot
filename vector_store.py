import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from data_processing import get_document_chunks, get_existing_pdf_files
from web_scraper import scrape_urls_to_chunks
from auth import AuthManager
from config import get_qdrant_config, validate_qdrant_config

# Initialize auth manager
auth_manager = AuthManager()

def get_user_data_path():
    """Get user-specific data path"""
    return auth_manager.get_user_data_path()

def get_user_db_faiss_path():
    """Get user-specific vectorstore path"""
    return auth_manager.get_user_vectorstore_path()

def get_user_collection_name():
    """Get user-specific Qdrant collection name"""
    user_id = auth_manager.get_user_id()
    return f"docubot_user_{user_id}" if user_id else "docubot_default"

def get_vector_store_type():
    """Determine which vector store to use (Qdrant Cloud or local FAISS)"""
    qdrant_config = validate_qdrant_config()
    return "qdrant" if qdrant_config else "faiss"

def clear_all_data():
    """Clears ALL data and vector store for current user."""
    vector_store_type = get_vector_store_type()
    
    if vector_store_type == "faiss":
        user_db_path = get_user_db_faiss_path()
        user_data_path = get_user_data_path()
        
        if os.path.exists(user_db_path):
            shutil.rmtree(user_db_path)
        if os.path.exists(user_data_path):
            shutil.rmtree(user_data_path)
        
        os.makedirs(user_data_path, exist_ok=True)
    else:
        # Clear Qdrant collection
        try:
            qdrant_config = get_qdrant_config()
            client = QdrantClient(
                url=qdrant_config['url'],
                api_key=qdrant_config['api_key']
            )
            collection_name = get_user_collection_name()
            client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Warning: Could not clear Qdrant collection: {e}")
    
    return "🧹 Cleared ALL data and vector store!"

def get_embedding_model():
    """Cached embedding model to avoid reloading."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_qdrant_vector_store(collection_name=None):
    """Get Qdrant vector store for current user"""
    if collection_name is None:
        collection_name = get_user_collection_name()
    
    qdrant_config = get_qdrant_config()
    if not qdrant_config['api_key'] or not qdrant_config['url']:
        return None
    
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
        return None

def build_vector_store(append=False):
    """Builds or updates the vector store from PDF documents for current user."""
    vector_store_type = get_vector_store_type()
    data_path = get_user_data_path()
    
    # Track processed files
    processed_files_path = os.path.join(data_path, "processed_files.txt")
    
    if vector_store_type == "qdrant":
        db = get_qdrant_vector_store()
        if append and db:
            # For Qdrant, we always add documents (deduplication happens at query time)
            existing_files = set()
            if os.path.exists(processed_files_path):
                with open(processed_files_path, 'r') as f:
                    existing_files = set(line.strip() for line in f)
        else:
            db = None
            existing_files = set()
    else:
        # FAISS logic
        db_faiss_path = get_user_db_faiss_path()
        if append and os.path.exists(db_faiss_path):
            print("🔄 Loading existing vector store...")
            embedding_model = get_embedding_model()
            try:
                db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
                if os.path.exists(processed_files_path):
                    with open(processed_files_path, 'r') as f:
                        existing_files = set(line.strip() for line in f)
                else:
                    existing_files = set()
            except Exception as e:
                print(f"Error loading existing vector store: {str(e)}")
                db = None
                existing_files = set()
        else:
            db = None
            existing_files = set()
    
    # Get chunks and file information
    if append and db is not None:
        # For append mode, only process new files
        all_files = get_existing_pdf_files(data_path)
        new_files = [f for f in all_files if f not in existing_files]
        
        if not new_files:
            print("ℹ️ No new PDF files to process. All files are already in the vector store.")
            return db, "no_new_files"
        
        file_paths = [os.path.join(data_path, f) for f in new_files]
        chunks, processed_files = get_document_chunks(data_path, file_paths)
        files_to_mark = new_files
    else:
        # For new vector store, process all files
        chunks, processed_files = get_document_chunks(data_path)
        files_to_mark = [os.path.basename(f) for f in processed_files] if processed_files else []
        existing_files = set()
    
    if not chunks:
        print("❌ No chunks generated from documents.")
        return None, "no_documents"

    print("🔧 Creating/updating embeddings...")
    embedding_model = get_embedding_model()

    print("💾 Building/updating vector database...")
    
    if db is None:
        if vector_store_type == "qdrant":
            db = get_qdrant_vector_store()
            if db:
                db.add_documents(chunks)
                action = "created"
            else:
                # Fallback to FAISS if Qdrant fails
                db = FAISS.from_documents(chunks, embedding_model)
                action = "created_faiss_fallback"
        else:
            db = FAISS.from_documents(chunks, embedding_model)
            action = "created"
        
        # Mark all files as processed
        with open(processed_files_path, 'w') as f:
            for file_name in files_to_mark:
                f.write(file_name + '\n')
    else:
        # Add to existing vector store
        db.add_documents(chunks)
        action = "updated"
        # Append new files to processed list
        with open(processed_files_path, 'a') as f:
            for file_name in files_to_mark:
                f.write(file_name + '\n')
    
    # Save FAISS locally if using FAISS
    if vector_store_type == "faiss" and isinstance(db, FAISS):
        db_faiss_path = get_user_db_faiss_path()
        db.save_local(db_faiss_path)
    
    # Show what was processed
    if files_to_mark:
        print(f"✅ Processed {len(files_to_mark)} file(s): {', '.join(files_to_mark)}")
    else:
        print("✅ Vector store updated successfully!")
    
    return db, action

def build_vector_store_from_urls(urls, append=False):
    """
    Build vector store from webpage URLs with deduplication for current user
    Uses your existing web scraping functionality
    """
    vector_store_type = get_vector_store_type()
    data_path = get_user_data_path()
    
    # Ensure data directory exists
    os.makedirs(data_path, exist_ok=True)
    
    # Track processed URLs
    processed_urls_path = os.path.join(data_path, "processed_urls.txt")
    
    # Convert single URL to list
    if isinstance(urls, str):
        urls = [urls]
    
    # Load existing processed URLs
    existing_urls = set()
    if append and os.path.exists(processed_urls_path):
        with open(processed_urls_path, 'r') as f:
            existing_urls = set(line.strip() for line in f if line.strip())
    
    # Filter out already processed URLs
    new_urls = [url for url in urls if url not in existing_urls]
    
    if not new_urls:
        print("ℹ️ No new URLs to process. All URLs are already in the knowledge base.")
        # Try to load existing vector store
        if vector_store_type == "qdrant":
            db = get_qdrant_vector_store()
            return db, "no_new_urls" if db else None
        else:
            db_faiss_path = get_user_db_faiss_path()
            if os.path.exists(db_faiss_path):
                embedding_model = get_embedding_model()
                db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
                return db, "no_new_urls"
        return None, "no_new_urls"
    
    print(f"🌐 Starting web scraping for {len(new_urls)} new URL(s)...")
    print(f"📋 URLs to scrape: {', '.join(new_urls)}")
    
    # Use your existing web scraping function
    chunks = scrape_urls_to_chunks(new_urls)
    
    if not chunks:
        print("❌ No content could be scraped from the URLs")
        return None, "failed"
    
    print("🔧 Creating/updating embeddings...")
    embedding_model = get_embedding_model()
    
    if vector_store_type == "qdrant":
        if append:
            db = get_qdrant_vector_store()
            if db:
                db.add_documents(chunks)
                action = "updated"
            else:
                db = get_qdrant_vector_store()
                if db:
                    db.add_documents(chunks)
                    action = "created"
                else:
                    # Fallback to FAISS
                    db = FAISS.from_documents(chunks, embedding_model)
                    action = "created_faiss_fallback"
        else:
            db = get_qdrant_vector_store()
            if db:
                db.add_documents(chunks)
                action = "created"
            else:
                db = FAISS.from_documents(chunks, embedding_model)
                action = "created_faiss_fallback"
    else:
        # FAISS logic
        db_faiss_path = get_user_db_faiss_path()
        if append and os.path.exists(db_faiss_path):
            print("🔄 Loading existing vector store...")
            try:
                db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
                db.add_documents(chunks)
                action = "updated"
            except Exception as e:
                print(f"Error loading existing store: {e}")
                db = FAISS.from_documents(chunks, embedding_model)
                action = "created"
        else:
            db = FAISS.from_documents(chunks, embedding_model)
            action = "created"
    
    # Save FAISS locally if using FAISS
    if isinstance(db, FAISS):
        db_faiss_path = get_user_db_faiss_path()
        db.save_local(db_faiss_path)
    
    # Save processed URLs
    if action in ["created", "created_faiss_fallback"]:
        # Write all URLs (overwrite)
        with open(processed_urls_path, 'w') as f:
            for url in new_urls:
                f.write(url + '\n')
    else:
        # Append new URLs
        with open(processed_urls_path, 'a') as f:
            for url in new_urls:
                f.write(url + '\n')
    
    print(f"✅ Vector store {action} from {len(new_urls)} web URL(s)")
    print(f"📝 Processed URLs: {', '.join(new_urls)}")
    
    return db, action

def get_vector_store():
    """Get vector store for current user"""
    vector_store_type = get_vector_store_type()
    
    if vector_store_type == "qdrant":
        return get_qdrant_vector_store()
    else:
        # FAISS fallback
        db_faiss_path = get_user_db_faiss_path()
        try:
            embedding_model = get_embedding_model()
            db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None

def vector_store_exists():
    """Check if vector store exists for current user"""
    vector_store_type = get_vector_store_type()
    
    if vector_store_type == "qdrant":
        db = get_qdrant_vector_store()
        if db:
            # Check if collection has any vectors
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
        return False
    else:
        db_faiss_path = get_user_db_faiss_path()
        return os.path.exists(db_faiss_path)