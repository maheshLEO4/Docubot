import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings  # Fixed import
from langchain_community.vectorstores import FAISS
from data_processing import get_document_chunks, get_existing_pdf_files
from web_scraper import scrape_urls_to_chunks

# --- Configuration ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def clear_all_data():
    """Clears ALL data and vector store (manual option only)."""
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH, exist_ok=True)
    return "🧹 Cleared ALL data and vector store!"

def get_embedding_model():
    """Cached embedding model to avoid reloading."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_vector_store(append=False, data_path=DATA_PATH, db_faiss_path=DB_FAISS_PATH):
    """Builds or updates the FAISS vector store from PDF documents."""
    
    # Track processed files
    processed_files_path = os.path.join(data_path, "processed_files.txt")
    
    if append and os.path.exists(db_faiss_path):
        print("🔄 Loading existing vector store...")
        embedding_model = get_embedding_model()
        try:
            db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
            
            # Load already processed files
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
        existing_files = set()  # Reset for new vector store
    
    if not chunks:
        print("❌ No chunks generated from documents.")
        return None, "no_documents"

    print("🔧 Creating/updating embeddings...")
    embedding_model = get_embedding_model()

    print("💾 Building/updating vector database...")
    
    if db is None:
        # Create new vector store
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
    
    db.save_local(db_faiss_path)
    
    # Show what was processed
    if files_to_mark:
        print(f"✅ Processed {len(files_to_mark)} file(s): {', '.join(files_to_mark)}")
    else:
        print("✅ Vector store updated successfully!")
    
    return db, action

def build_vector_store_from_urls(urls, append=False, data_path=DATA_PATH, db_faiss_path=DB_FAISS_PATH):
    """
    Build vector store from webpage URLs with deduplication
    """
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
        print("ℹ️ No new URLs to process. All URLs are already in the vector store.")
        # Try to load existing vector store
        if os.path.exists(db_faiss_path):
            embedding_model = get_embedding_model()
            db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
            return db, "no_new_urls"
        return None, "no_new_urls"
    
    print(f"🌐 Starting web scraping for {len(new_urls)} new URL(s)...")
    print(f"📋 URLs to scrape: {', '.join(new_urls)}")
    
    chunks = scrape_urls_to_chunks(new_urls)
    
    if not chunks:
        print("❌ No content could be scraped from the URLs")
        return None, "failed"
    
    print("🔧 Creating/updating embeddings...")
    embedding_model = get_embedding_model()
    
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
            existing_urls = set()  # Reset for new vector store
    else:
        db = FAISS.from_documents(chunks, embedding_model)
        action = "created"
        existing_urls = set()  # Reset for new vector store
    
    db.save_local(db_faiss_path)
    
    # Save processed URLs
    if action == "created":
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

def get_vector_store(db_faiss_path=DB_FAISS_PATH):
    """Cached vector store to avoid reloading."""
    try:
        embedding_model = get_embedding_model()
        db = FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None

def vector_store_exists(db_faiss_path=DB_FAISS_PATH):
    """Check if vector store exists."""
    return os.path.exists(db_faiss_path)