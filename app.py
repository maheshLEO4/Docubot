import streamlit as st
from config import config
from auth import render_auth_ui
from database import MongoDBManager
from vector_store import VectorStoreManager
from query_processor import QueryProcessor
from data_processing import DataProcessor
from web_scraper import WebScraper
import time

# Page configuration
st.set_page_config(
    page_title="DocuBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
    }
    .source-doc {
        padding: 0.5rem;
        background-color: #FFF8E1;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding: 0px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'messages': [],
        'sources': {},
        'vector_store_exists': False,
        'cached_user_files': [],
        'cached_user_scrapes': [],
        'processing_mode': 'add',
        'last_query': '',
        'chat_loaded': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Sidebar components
def render_sidebar(user_id):
    """Render optimized sidebar"""
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Knowledge Base Status
        vector_store = VectorStoreManager(user_id)
        st.session_state.vector_store_exists = vector_store.exists()
        
        if st.session_state.vector_store_exists:
            st.success("✅ Knowledge Base Active")
            
            # Knowledge Base Management
            with st.expander("📚 Knowledge Base", expanded=True):
                db = MongoDBManager()
                
                # Files
                files = st.session_state.cached_user_files or db.get_user_files(user_id)
                if files:
                    st.write("**Uploaded Files:**")
                    for file in files[:5]:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.caption(f"📄 {file['filename']}")
                        with col2:
                            if st.button("🗑️", key=f"del_file_{file['upload_id']}"):
                                if vector_store.remove_document(file['filename'], 'pdf'):
                                    db.delete_file(file['upload_id'])
                                    st.success("File removed")
                                    st.rerun()
                else:
                    st.info("No files uploaded")
                
                # Websites
                scrapes = st.session_state.cached_user_scrapes or db.get_user_scrapes(user_id)
                if scrapes:
                    st.write("**Scraped Websites:**")
                    for scrape in scrapes[:3]:
                        for url in scrape.get('successful_urls', [])[:3]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.caption(f"🌐 {url[:40]}...")
                            with col2:
                                if st.button("🗑️", key=f"del_url_{hash(url)}"):
                                    if vector_store.remove_document(url, 'web'):
                                        st.success("URL removed")
                                        st.rerun()
                else:
                    st.info("No websites scraped")
        
        # Input Section
        st.markdown("---")
        st.subheader("➕ Add Content")
        
        # Tabs for input
        input_tab1, input_tab2 = st.tabs(["📄 PDFs", "🌐 Websites"])
        
        with input_tab1:
            uploaded_files = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload PDF documents"
            )
        
        with input_tab2:
            website_urls = st.text_area(
                "Website URLs",
                placeholder="Enter one URL per line\nhttps://example.com\nhttps://docs.streamlit.io",
                height=100
            )
            urls_list = [url.strip() for url in website_urls.split('\n') if url.strip()] if website_urls else []
        
        # Processing Options
        st.markdown("---")
        st.subheader("⚡ Processing")
        
        processing_mode = st.radio(
            "Mode:",
            ["Add to KB", "Replace KB"],
            horizontal=True,
            disabled=not st.session_state.vector_store_exists
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            process_pdfs = st.button(
                "Process PDFs",
                disabled=not uploaded_files,
                use_container_width=True
            )
        
        with col2:
            process_websites = st.button(
                "Scrape Websites",
                disabled=not urls_list,
                use_container_width=True
            )
        
        # Clear buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.sources = {}
                st.success("Chat cleared")
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            if st.button("Clear All", use_container_width=True, type="secondary"):
                vector_store.clear()
                db = MongoDBManager()
                db.clear_user_data(user_id)
                st.session_state.messages = []
                st.session_state.sources = {}
                st.session_state.vector_store_exists = False
                st.session_state.cached_user_files = []
                st.session_state.cached_user_scrapes = []
                st.success("All data cleared")
                time.sleep(0.5)
                st.rerun()
        
        # Handle processing
        if process_pdfs and uploaded_files:
            process_pdfs_files(uploaded_files, user_id, processing_mode == "Add to KB")
        
        if process_websites and urls_list:
            process_website_urls(urls_list, user_id, processing_mode == "Add to KB")

# Processing functions
def process_pdfs_files(uploaded_files, user_id, append=True):
    """Process uploaded PDF files"""
    with st.spinner("Processing PDFs..."):
        try:
            # Process PDFs
            processor = DataProcessor(user_id)
            chunks, processed_files = processor.process_pdfs(uploaded_files)
            
            if not chunks:
                st.error("No content extracted from PDFs")
                return
            
            # Add to vector store
            vector_store = VectorStoreManager(user_id)
            if not append:
                clear_success = vector_store.clear()
                if not clear_success:
                    st.warning("Could not clear existing data, but will try to add new documents")
            
            success = vector_store.add_documents(chunks, 'pdf')
            
            if success:
                # Log to database
                db = MongoDBManager()
                for filename in processed_files:
                    db.log_file_upload(user_id, filename, len(chunks))
                
                # Update cache
                st.session_state.cached_user_files = db.get_user_files(user_id)
                st.session_state.vector_store_exists = True
                
                st.success(f"Added {len(processed_files)} PDF(s) to knowledge base")
                processor.cleanup()
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to add PDFs to knowledge base")
                
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")

def process_website_urls(urls_list, user_id, append=True):
    """Process website URLs"""
    with st.spinner("Scraping websites..."):
        try:
            # Scrape websites
            scraper = WebScraper()
            documents, successful_urls = scraper.scrape_urls(urls_list)
            
            if not documents:
                st.error("No content scraped from websites")
                return
            
            # Split into chunks
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add to vector store
            vector_store = VectorStoreManager(user_id)
            if not append:
                vector_store.clear()
            
            success = vector_store.add_documents(chunks, 'web')
            
            if success:
                # Log to database
                db = MongoDBManager()
                db.log_web_scrape(user_id, urls_list, successful_urls)
                
                # Update cache
                st.session_state.cached_user_scrapes = db.get_user_scrapes(user_id)
                st.session_state.vector_store_exists = True
                
                st.success(f"Added {len(successful_urls)} website(s) to knowledge base")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to add websites to knowledge base")
                
        except Exception as e:
            st.error(f"Error scraping websites: {str(e)}")

# Main chat interface
def render_chat_interface(user_id):
    """Render main chat interface"""
    # Header
    st.markdown('<h1 class="main-header">DocuBot AI 🤖</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your documents and websites</p>', unsafe_allow_html=True)
    
    # Welcome message
    if not st.session_state.messages:
        if st.session_state.vector_store_exists:
            st.success("✅ Your knowledge base is ready! Ask questions about your documents.")
        else:
            st.info("📚 Add PDFs or websites in the sidebar to build your knowledge base.")
    
    # Chat messages
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            role = message['role']
            content = message['content']
            
            with st.chat_message(role):
                st.markdown(content)
                
                # Show sources for assistant messages
                if role == 'assistant' and idx in st.session_state.sources:
                    sources = st.session_state.sources[idx]
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                icon = "🌐" if source['type'] == 'web' else "📄"
                                st.markdown(f"**{icon} Source {i}:** {source['document']}")
                                if source['page'] != 'N/A':
                                    st.caption(f"Page: {source['page']}")
                                st.caption(f'"{source["excerpt"]}"')
                                st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge base..."):
        # Prevent duplicate processing
        if prompt == st.session_state.get('last_query'):
            return
        st.session_state.last_query = prompt
        
        # Add user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Check if knowledge base exists
        if not st.session_state.vector_store_exists:
            st.warning("Please add documents or websites first.")
            return
        
        # Process query
        with st.spinner("Thinking..."):
            try:
                # Get API key
                api_key = config.get_groq_api_key()
                
                # Process query
                processor = QueryProcessor(user_id)
                result = processor.process(prompt, api_key)
                
                if result['success']:
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.markdown(result['answer'])
                        
                        # Store sources
                        message_idx = len(st.session_state.messages)
                        st.session_state.sources[message_idx] = result['sources']
                        
                        # Show sources in expander
                        if result['sources']:
                            with st.expander("📚 Sources", expanded=False):
                                for i, source in enumerate(result['sources'], 1):
                                    icon = "🌐" if source['type'] == 'web' else "📄"
                                    st.markdown(f"**{icon} Source {i}:** {source['document']}")
                                    if source['page'] != 'N/A':
                                        st.caption(f"Page: {source['page']}")
                                    st.caption(f'"{source["excerpt"]}"')
                                    st.divider()
                    
                    # Store message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer']
                    })
                    
                    # Log query
                    try:
                        db = MongoDBManager()
                        db.log_query(user_id, prompt, result['answer'], result['sources'])
                    except:
                        pass
                    
                else:
                    st.error(result['error'])
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

# Main application
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Validate configuration
    try:
        config_valid, missing_keys = config.validate_required_keys()
        if not config_valid:
            st.error(f"Missing configuration: {', '.join(missing_keys)}")
            st.info("Please add the required keys to Streamlit secrets or .env file")
            return
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
        return
    
    # Authentication
    user_id = render_auth_ui()
    if not user_id:
        return
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        render_chat_interface(user_id)
    
    with col2:
        render_sidebar(user_id)

if __name__ == "__main__":
    main()