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

def build_vector_store_from_urls(urls, append=False, db_faiss_path=DB_FAISS_PATH):
    """
    Build vector store from webpage URLs
    """
    print("🌐 Starting web scraping...")
    
    chunks = scrape_urls_to_chunks(urls)
    
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
    else:
        db = FAISS.from_documents(chunks, embedding_model)
        action = "created"
    
    db.save_local(db_faiss_path)
    print(f"✅ Vector store {action} from web URLs")
    
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