import os
import shutil
import streamlit as st
from data_processing import get_existing_pdf_files, save_uploaded_files
from vector_store import (
    clear_all_data, build_vector_store, build_vector_store_from_urls,
    get_vector_store, vector_store_exists, DATA_PATH, DB_FAISS_PATH
)
from query_processor import process_query
from config import validate_api_key

# --- Configuration ---
st.set_page_config(page_title="DocuBot AI 🤖", page_icon="🤖", layout="wide")

# Get API key (this will show error if not found)
try:
    api_key = validate_api_key()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = vector_store_exists()
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'scraping_status' not in st.session_state:
    st.session_state.scraping_status = {}

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
            placeholder="Enter one or more URLs (one per line)\nExample:\nhttps://example.com\nhttps://docs.streamlit.io\nhttps://en.wikipedia.org/wiki/Artificial_intelligence",
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
            st.session_state.processing_status = ""
            st.session_state.scraping_status = {}
            st.toast(result, icon="✨")
            st.rerun()

    # Show current processing status
    if st.session_state.processing_status:
        st.info(f"🔄 {st.session_state.processing_status}")
    
    # Show scraping status if any
    if st.session_state.scraping_status:
        with st.expander("🌐 Scraping Progress", expanded=True):
            for url, status in st.session_state.scraping_status.items():
                if "success" in status.lower():
                    st.success(f"✅ {url}: {status}")
                elif "fail" in status.lower() or "error" in status.lower():
                    st.error(f"❌ {url}: {status}")
                else:
                    st.info(f"🔄 {url}: {status}")

    # Processing logic for PDFs
    if process_pdfs and uploaded_files:
        st.session_state.processing_status = "Starting PDF processing..."
        st.session_state.scraping_status = {}  # Clear previous scraping status
        
        with st.spinner("Processing PDF documents..."):
            try:
                # Create status container for real-time updates
                status_container = st.empty()
                status_container.info("📥 Saving uploaded files...")
                
                new_files = save_uploaded_files(uploaded_files)
                status_container.info("🔧 Building vector store...")
                
                if processing_mode == "Replace All Content":
                    if os.path.exists(DB_FAISS_PATH):
                        shutil.rmtree(DB_FAISS_PATH)
                    db, action = build_vector_store(append=False)
                else:
                    db, action = build_vector_store(append=True)
                
                if db is not None and action != "no_new_files":
                    st.session_state.is_processed = True
                    st.session_state.db_loaded = True
                    st.session_state.processing_status = ""
                    status_container.empty()
                    st.success(f"✅ PDF documents {action} successfully!")
                    if new_files:
                        st.toast(f"Added {len(new_files)} new documents", icon="📄")
                elif action == "no_new_files":
                    st.session_state.processing_status = ""
                    status_container.empty()
                    st.info("ℹ️ No new PDF documents to process.")
                else:
                    st.session_state.processing_status = ""
                    status_container.empty()
                    st.error("❌ Failed to process PDF documents.")
                
            except Exception as e:
                st.session_state.processing_status = ""
                status_container.empty()
                st.error(f"❌ Error processing PDFs: {str(e)}")
                st.toast("PDF processing failed", icon="❌")
            
            st.rerun()

    # Processing logic for websites
    if process_websites and urls_list:
        st.session_state.processing_status = f"Starting web scraping for {len(urls_list)} website(s)..."
        
        # Initialize scraping status for all URLs
        for url in urls_list:
            st.session_state.scraping_status[url] = "Queued for scraping"
        
        with st.spinner(f"Scraping {len(urls_list)} website(s)..."):
            try:
                # Create progress area
                progress_container = st.empty()
                
                if processing_mode == "Replace All Content":
                    if os.path.exists(DB_FAISS_PATH):
                        shutil.rmtree(DB_FAISS_PATH)
                    db, action = build_vector_store_from_urls(urls_list, append=False)
                else:
                    db, action = build_vector_store_from_urls(urls_list, append=True)
                
                # Update final status
                successful_scrapes = sum(1 for status in st.session_state.scraping_status.values() 
                                       if "success" in status.lower())
                failed_scrapes = len(urls_list) - successful_scrapes
                
                if db is not None:
                    st.session_state.is_processed = True
                    st.session_state.db_loaded = True
                    st.session_state.processing_status = ""
                    
                    if successful_scrapes > 0:
                        st.success(f"✅ Successfully processed {successful_scrapes}/{len(urls_list)} websites!")
                        if failed_scrapes > 0:
                            st.warning(f"⚠️ {failed_scrapes} website(s) failed to scrape")
                        st.toast(f"Websites {action} successfully!", icon="🌐")
                    else:
                        st.error("❌ All websites failed to scrape")
                else:
                    st.session_state.processing_status = ""
                    st.error("❌ Failed to create vector store from websites")
                
            except Exception as e:
                st.session_state.processing_status = ""
                st.error(f"❌ Error during web scraping: {str(e)}")
                st.toast("Web scraping failed", icon="❌")
            
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared!", icon="🧹")
    
    # Clear status button
    if st.session_state.processing_status or st.session_state.scraping_status:
        if st.button("🗑️ Clear Status", use_container_width=True):
            st.session_state.processing_status = ""
            st.session_state.scraping_status = {}
            st.rerun()
    
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
            with st.spinner("🔍 Searching knowledge base..."):
                # Show processing indicator
                thinking_container = st.empty()
                thinking_container.info("🤔 Analyzing your question...")
                
                # Use the query processor with the API key
                result = process_query(prompt, api_key)
                
                thinking_container.empty()
                
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

# Footer with status information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.is_processed:
        st.success("✅ Knowledge Base: Ready")
    else:
        st.warning("⚠️ Knowledge Base: Not Ready")

with col2:
    if api_key:
        st.success("✅ API Key: Configured")
    else:
        st.error("❌ API Key: Missing")

with col3:
    if st.session_state.processing_status or st.session_state.scraping_status:
        st.info("🔄 Processing: In Progress")
    else:
        st.success("✅ Processing: Idle")