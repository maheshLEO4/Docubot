import os
import tempfile
from typing import List, Tuple
import streamlit as st

class DataProcessor:
    """Optimized data processor for PDFs"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.temp_dir = tempfile.mkdtemp(prefix=f"docubot_{user_id}_")
    
    def process_pdfs(self, uploaded_files) -> Tuple[List, List[str]]:
        """Process uploaded PDF files"""
        if not uploaded_files:
            return [], []
        
        documents = []
        processed_files = []
        
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
            for uploaded_file in uploaded_files:
                try:
                    print(f"📄 Processing: {uploaded_file.name}")
                    
                    # Save to temp file
                    temp_path = os.path.join(self.temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load PDF using PyPDFLoader
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(temp_path)
                    pdf_docs = loader.load()
                    
                    print(f"  Loaded {len(pdf_docs)} pages from {uploaded_file.name}")
                    
                    # Add metadata
                    for doc in pdf_docs:
                        doc.metadata.update({
                            'source': uploaded_file.name,
                            'type': 'pdf',
                            'user_id': self.user_id
                        })
                    
                    documents.extend(pdf_docs)
                    processed_files.append(uploaded_file.name)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    print(f"❌ Error processing {uploaded_file.name}: {e}")
                    continue
        
        if not documents:
            st.warning("No content extracted from PDFs")
            return [], []
        
        # Split into chunks
        print(f"✂️ Splitting {len(documents)} documents into chunks...")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        return chunks, processed_files
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass