import os
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from auth import AuthManager

# Initialize auth manager
auth_manager = AuthManager()

def get_user_data_path():
    """Get user-specific data path"""
    return auth_manager.get_user_data_path()

def get_existing_pdf_files(data_path=None):
    """Get list of existing PDF files in user's data directory."""
    if data_path is None:
        data_path = get_user_data_path()
    
    if not os.path.exists(data_path):
        return []
    return [f for f in os.listdir(data_path) if f.endswith('.pdf')]

def save_uploaded_files(uploaded_files, data_path=None):
    """Saves uploaded files to the user's data directory without clearing existing files."""
    if data_path is None:
        data_path = get_user_data_path()
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    new_files = []
    for file in uploaded_files:
        file_path = os.path.join(data_path, file.name)
        # Check if file already exists to avoid duplicates
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            new_files.append(file.name)
    
    return new_files

def load_pdf_files(file_paths=None, data_path=None):
    """Load PDF files with progress tracking."""
    if data_path is None:
        data_path = get_user_data_path()
    
    if file_paths is None:
        file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]
    
    def load_single_pdf(file_path):
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Warning: Error loading {os.path.basename(file_path)}: {str(e)}")
            return []
    
    # Use thread pool for parallel loading
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(file_paths))) as executor:
        results = list(executor.map(load_single_pdf, file_paths))
    
    # Flatten results
    documents = []
    for doc_list in results:
        documents.extend(doc_list)
    
    return documents

def split_documents_into_chunks(documents):
    """Split documents into chunks using recursive text splitter."""
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def get_document_chunks(data_path=None, file_paths=None):
    """Main function to load PDFs and return chunks."""
    if data_path is None:
        data_path = get_user_data_path()
    
    print("📂 Scanning for PDF documents...")
    
    # Get all PDF files in data directory
    if file_paths is None:
        all_pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]
    else:
        all_pdf_files = file_paths
    
    if not all_pdf_files:
        print("❌ No PDF files found to process.")
        return None, []
    
    print(f"📄 Found {len(all_pdf_files)} PDF file(s) to process...")
    documents = load_pdf_files(all_pdf_files, data_path)
    
    if not documents:
        print("❌ No documents could be loaded from PDF files.")
        return None, []
    
    chunks = split_documents_into_chunks(documents)
    return chunks, all_pdf_files