import os
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Remove AuthManager import from here - we'll pass user_id as parameter

def get_user_data_path(user_id):
    """Get user-specific temporary data path"""
    if not user_id:
        return None
    temp_path = f"temp_uploads/user_{user_id}"
    os.makedirs(temp_path, exist_ok=True)
    return temp_path

def save_uploaded_files(uploaded_files, user_id):
    """Save uploaded files to temporary storage and return file paths"""
    data_path = get_user_data_path(user_id)
    if not data_path:
        raise ValueError("User not authenticated")
    
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(data_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths

def get_existing_pdf_files(user_id):
    """Get list of existing PDF files in user's data directory."""
    data_path = get_user_data_path(user_id)
    if not data_path or not os.path.exists(data_path):
        return []
    return [f for f in os.listdir(data_path) if f.endswith('.pdf')]

def load_pdf_files(file_paths):
    """Load PDF files"""
    def load_single_pdf(file_path):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # Add file source to metadata
            for doc in documents:
                doc.metadata['source'] = file_path
                doc.metadata['type'] = 'pdf'
            return documents
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
    """Split documents into chunks"""
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

def get_document_chunks(user_id, file_paths=None):
    """Main function to load PDFs and return chunks"""
    if not file_paths:
        data_path = get_user_data_path(user_id)
        if not data_path or not os.path.exists(data_path):
            return None, []
        file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]
    
    if not file_paths:
        print("❌ No PDF files found to process.")
        return None, []
    
    print(f"📄 Found {len(file_paths)} PDF file(s) to process...")
    documents = load_pdf_files(file_paths)
    
    if not documents:
        print("❌ No documents could be loaded from PDF files.")
        return None, []
    
    chunks = split_documents_into_chunks(documents)
    return chunks, file_paths
