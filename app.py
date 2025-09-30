import os
import shutil
import streamlit as st
from data_processing import get_existing_pdf_files, save_uploaded_files
from vector_store import (
    clear_all_data, build_vector_store, build_vector_store_from_urls,
    get_vector_store, vector_store_exists, DATA_PATH, DB_FAISS_PATH
)
from query_processor import process_query
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="DocuBot AI 🤖", page_icon="🤖", layout="wide")

# Get API key from Streamlit secrets or .env file
def get_api_key():
    """Get API key from Streamlit secrets or .env file"""
    # First try Streamlit secrets (for deployment)
    if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
        return st.secrets['GROQ_API_KEY']
    # Then try environment variable (for local development)
    elif 'GROQ_API_KEY' in os.environ:
        return os.environ['GROQ_API_KEY']
    else:
        return None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = vector_store_exists()
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False

# Check if API key is available
api_key = get_api_key()
if not api_key:
    st.error("""
    ❌ GROQ_API_KEY not found. 
    
    Please add your API key to:
    - **Streamlit Cloud**: Go to App Settings → Secrets
    - **Local development**: Create a `.env` file
    """)
    st.stop()

# --- Streamlit UI ---
with st.sidebar:
    st.title("📄 DocuBot Controls")
    st.markdown("Upload PDFs or add websites to build your knowledge base.")
    
    # Tab for different input methods
    input_tab1, input_tab2 = st.tabs(["📁 Upload PDFs", "🌐 Add Websites"])
    
    with input_tab1:
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )
    
    with input_tab2:
        website_urls = st.text_area(
            "Website URLs",
            placeholder="Enter one or more URLs (one per line)\nExample:\nhttps://example.com\nhttps://another-site.com",
            help="Add websites to scrape and include in your knowledge base"
        )
        urls_list = [url.strip() for url in website_urls.split('\n') if url.strip()] if website_urls else []

    # Processing options
    st.markdown("---")
    st.subheader("⚙️ Processing Options")
    
    # Check if vector store already exists
    vector_store_exists_flag = vector_store_exists()
    
    if vector_store_exists_flag:
        st.success("✅ Vector store exists")
        existing_files = get_existing_pdf_files()
        st.write(f"**Current documents:** {len(existing_files)}")
        
        processing_mode = st.radio(
            "Processing Mode:",
            ["Add New Content", "Replace All Content"],
            help="Choose whether to add new content or replace everything"
        )
    else:
        st.info("📝 No vector store found. Will create new one.")
        processing_mode = "Replace All Content"
    
    # File size warnings
    if uploaded_files:
        total_size = sum([file.size for file in uploaded_files]) / (1024 * 1024)
        if total_size > 10:
            st.warning(f"Total file size: {total_size:.1f} MB")
    
    # Process buttons
    col1, col2 = st.columns(2)
    
    with col1:
        process_pdfs = st.button("Process PDFs", use_container_width=True, type="primary", 
                                disabled=not uploaded_files)
        
        process_websites = st.button("Scrape Websites", use_container_width=True, type="primary",
                                   disabled=not urls_list)

    with col2:
        if st.button("Clear All", use_container_width=True, type="secondary"):
            result = clear_all_data()
            st.session_state.is_processed = False
            st.session_state.db_loaded = False
            st.session_state.messages = []
            st.toast(result, icon="✨")
            st.rerun()

    # Processing logic
    if process_pdfs and uploaded_files:
        with st.spinner("Processing PDF documents..."):
            new_files = save_uploaded_files(uploaded_files)
            
            if processing_mode == "Replace All Content":
                if os.path.exists(DB_FAISS_PATH):
                    shutil.rmtree(DB_FAISS_PATH)
                db, action = build_vector_store(append=False)
            else:
                db, action = build_vector_store(append=True)
            
            if db is not None and action != "no_new_files":
                st.session_state.is_processed = True
                st.session_state.db_loaded = True
                st.success(f"✅ PDF documents {action} successfully!")
                if new_files:
                    st.toast(f"Added {len(new_files)} new documents", icon="📄")
            elif action == "no_new_files":
                st.info("ℹ️ No new PDF documents to process.")
            else:
                st.error("❌ Failed to process PDF documents.")
            st.rerun()
    
    if process_websites and urls_list:
        with st.spinner(f"Scraping {len(urls_list)} website(s)..."):
            if processing_mode == "Replace All Content":
                if os.path.exists(DB_FAISS_PATH):
                    shutil.rmtree(DB_FAISS_PATH)
                db, action = build_vector_store_from_urls(urls_list, append=False)
            else:
                db, action = build_vector_store_from_urls(urls_list, append=True)
            
            if db is not None:
                st.session_state.is_processed = True
                st.session_state.db_loaded = True
                st.success(f"✅ Websites {action} successfully!")
                st.toast(f"Scraped {len(urls_list)} website(s)", icon="🌐")
            else:
                st.error("❌ Failed to scrape websites.")
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared!", icon="🧹")
    
    # Display document info
    if st.session_state.get('is_processed', False) and os.path.exists(DATA_PATH):
        st.markdown("---")
        st.subheader("📊 Knowledge Base Info")
        pdf_files = get_existing_pdf_files()
        st.write(f"**Total documents:** {len(pdf_files)}")
        if pdf_files:
            for pdf in pdf_files[-5:]:
                file_path = os.path.join(DATA_PATH, pdf)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    st.caption(f"• {pdf} ({file_size:.1f} MB)")
                else:
                    st.caption(f"• {pdf} (file not found)")
            if len(pdf_files) > 5:
                st.caption(f"... and {len(pdf_files) - 5} more documents")

# --- Main Chat ---
st.title("🤖 DocuBot AI: Chat with Your Documents & Websites")
st.markdown("Ask questions about your uploaded PDFs and scraped websites.")

# Display welcome message
if not st.session_state.messages:
    if st.session_state.is_processed:
        doc_count = len(get_existing_pdf_files())
        st.success(f"✅ Ready! Your knowledge base is loaded and ready for questions.")
    else:
        st.info("📚 Upload PDFs or add websites in the sidebar to build your knowledge base.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle user input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if not st.session_state.is_processed:
        st.warning("Please add some content (PDFs or websites) before asking questions.")
    else:
        try:
            with st.spinner("Thinking..."):
                # Use the query processor with the API key
                result = process_query(prompt, api_key)  # Pass the API key
                
                if result['success']:
                    enhanced_result = result['answer']
                    source_documents = result['sources']

                    with st.chat_message('assistant'):
                        st.markdown(enhanced_result)

                        if source_documents:
                            with st.expander("🔍 Source References"):
                                st.caption("Sources from your knowledge base")
                                
                                for i, doc in enumerate(source_documents, 1):
                                    st.markdown(f"**Source {i}:** `{doc['document']}`")
                                    if doc['page'] != 'N/A':
                                        st.caption(f"Page: {doc['page']}")
                                    st.caption(f'"{doc["excerpt"]}"')
                                    st.markdown("---")

                    st.session_state.messages.append({'role': 'assistant', 'content': enhanced_result})
                else:
                    error_msg = result['error']
                    st.error(error_msg)
                    st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

        except Exception as e:
            error_msg = f"⚠️ An error occurred while processing your question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})