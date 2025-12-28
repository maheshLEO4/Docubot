import os
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_user_data_path(user_id):
    path = f"temp_uploads/user_{user_id}"
    os.makedirs(path, exist_ok=True)
    return path

def save_uploaded_files(uploaded_files, user_id):
    data_path = get_user_data_path(user_id)
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(data_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def load_single_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def get_document_chunks(user_id, file_paths):
    all_docs = []
    # Fast Parallel Loading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(load_single_pdf, file_paths))
    
    for doc_list in results:
        all_docs.extend(doc_list)
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(all_docs)