import os
import shutil
import streamlit as st
from data_processing import get_existing_pdf_files, save_uploaded_files
from vector_store import (
    clear_all_data, build_vector_store_from_pdfs, build_vector_store_from_urls,
    get_vector_store, vector_store_exists, remove_documents_from_store
)
from query_processor import process_query
from config import validate_api_key
from auth import setup_authentication
from database import MongoDBManager

# --- Configuration ---
st.set_page_config(page_title="DocuBot AI", page_icon="🤖", layout="wide")

# Get API key (this will show error if not found)
try:
    api_key = validate_api_key()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Setup authentication and get user_id
user_id = setup_authentication()

# Initialize database
db_manager = MongoDBManager()

# Initialize user in database
if user_id and st.session_state.user:
    db_manager.init_user(st.session_state.user)
    db_manager.update_last_login(user_id)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = vector_store_exists(user_id)
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False
if 'user_files' not in st.session_state:
    st.session_state.user_files = []
if 'user_urls' not in st.session_state:
    st.session_state.user_urls = []
if 'source_docs' not in st.session_state:
    st.session_state.source_docs = {}

# Auto-load user data on login
if user_id and not st.session_state.db_loaded:
    with st.spinner("Loading your knowledge base..."):
        try:
            if vector_store_exists(user_id):
                # Load user's files and URLs from database
                st.session_state.user_files = db_manager.get_user_files(user_id)
                st.session_state.user_urls = db_manager.get_user_scrapes(user_id)
                st.session_state.is_processed = True
                st.session_state.db_loaded = True
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")

# --- Streamlit UI ---
with st.sidebar:
    st.title("DocuBot Controls")
    st.markdown("Upload PDFs or add websites to build your knowledge base.")
    
    # Show storage info
    st.success("Using Qdrant Cloud Storage")
    
    # Display existing files and URLs
    if st.session_state.is_processed:
        st.markdown("---")
        st.subheader("Your Knowledge Base")
        
        # Files section
        with st.expander("Uploaded Files", expanded=True):
            user_files = db_manager.get_user_files(user_id)
            if user_files:
                for file_record in user_files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {file_record['filename']}")
                        st.caption(f"Size: {file_record['file_size'] / 1024:.1f} KB")
                    with col2:
                        if st.button("🗑️", key=f"del_file_{file_record['upload_id']}", help="Remove file"):
                            with st.spinner("Removing file..."):
                                success = remove_documents_from_store(user_id, file_record['filename'], 'pdf')
                                if success:
                                    db_manager.delete_file_upload(file_record['upload_id'])
                                    st.success("File removed!")
                                    st.rerun()
                                else:
                                    st.error("Failed to remove file")
            else:
                st.info("No files uploaded yet")
        
        # URLs section
        with st.expander("Scraped Websites", expanded=True):
            user_scrapes = db_manager.get_user_scrapes(user_id)
            if user_scrapes:
                for scrape_record in user_scrapes:
                    for url in scrape_record['successful_urls']:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            display_url = url[:40] + "..." if len(url) > 40 else url
                            st.text(f"🌐 {display_url}")
                        with col2:
                            if st.button("🗑️", key=f"del_url_{scrape_record['scrape_id']}_{url}", help="Remove URL"):
                                with st.spinner("Removing URL..."):
                                    success = remove_documents_from_store(user_id, url, 'web')
                                    if success:
                                        db_manager.delete_web_scrape(scrape_record['scrape_id'])
                                        st.success("URL removed!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to remove URL")
            else:
                st.info("No websites scraped yet")
    
    st.markdown("---")
    
    # Tab for different input methods
    input_tab1, input_tab2 = st.tabs(["Upload PDFs", "Add Websites"])
    
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
            placeholder="Enter one or more URLs (one per line)\nExample:\nhttps://example.com\nhttps://docs.python.org\nhttps://en.wikipedia.org",
            help="Add websites to scrape and include in your knowledge base"
        )
        urls_list = [url.strip() for url in website_urls.split('\n') if url.strip()] if website_urls else []

    # Processing options
    st.markdown("---")
    st.subheader("Processing Options")
    
    # Check if vector store already exists
    vector_store_exists_flag = vector_store_exists(user_id)
    
    if vector_store_exists_flag:
        st.success("Knowledge base exists")
        
        processing_mode = st.radio(
            "Processing Mode:",
            ["Add New Content", "Replace All Content"],
            help="Choose whether to add new content or replace everything"
        )
    else:
        st.info("No knowledge base found. Will create new one.")
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
            result = clear_all_data(user_id)
            db_manager.clear_user_data(user_id)
            st.session_state.is_processed = False
            st.session_state.db_loaded = False
            st.session_state.messages = []
            st.session_state.user_files = []
            st.session_state.user_urls = []
            st.session_state.source_docs = {}
            st.toast(result, icon="✨")
            st.rerun()

    # Processing logic
    if process_pdfs and uploaded_files:
        with st.spinner("Processing PDF documents..."):
            try:
                new_files = save_uploaded_files(uploaded_files, user_id)
                
                db, action = build_vector_store_from_pdfs(
                    user_id, 
                    uploaded_files, 
                    append=(processing_mode == "Add New Content")
                )
                
                if db is not None and action != "no_documents":
                    st.session_state.is_processed = True
                    st.session_state.db_loaded = True
                    st.success(f"PDF documents {action} successfully!")
                    if new_files:
                        st.toast(f"Added {len(new_files)} new documents", icon="📄")
                elif action == "no_documents":
                    st.info("No PDF documents to process.")
                else:
                    st.error("Failed to process PDF documents.")
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")
            st.rerun()
    
    if process_websites and urls_list:
        with st.spinner(f"Scraping {len(urls_list)} website(s)..."):
            try:
                db, action = build_vector_store_from_urls(
                    user_id,
                    urls_list, 
                    append=(processing_mode == "Add New Content")
                )
                
                if db is not None and action not in ["no_new_urls", "failed"]:
                    st.session_state.is_processed = True
                    st.session_state.db_loaded = True
                    st.success(f"Websites {action} successfully!")
                    st.toast(f"Scraped {len(urls_list)} website(s)", icon="🌐")
                elif action == "failed":
                    st.error("Failed to scrape websites.")
                    st.warning("""
                    **Scraping Limitations:**
                    - Works: News sites, blogs, documentation, Wikipedia, most content sites
                    - Doesn't work on cloud: Client-side React/Vue/Angular apps
                    - For React apps: Run this app locally with Chrome/Selenium installed
                    """)
                else:
                    st.error("Failed to scrape websites.")
            except Exception as e:
                st.error(f"Error scraping websites: {str(e)}")
            st.rerun()
            
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_docs = {}
        st.toast("Chat history cleared!", icon="🧹")
        st.rerun()
    
    # Display document info
    if st.session_state.get('is_processed', False):
        st.markdown("---")
        st.subheader("Knowledge Base Info")
        
        # Show user stats from MongoDB
        stats = db_manager.get_user_stats(user_id)
        st.write(f"**Files Uploaded:** {stats['files_uploaded']}")
        st.write(f"**Websites Scraped:** {stats['websites_scraped']}")

# --- Main Chat ---
st.title("DocuBot AI: Chat with Your Documents & Websites")
st.markdown("Ask questions about your uploaded PDFs and scraped websites.")

# Display welcome message
if not st.session_state.messages:
    if st.session_state.is_processed:
        st.success("Ready! Your knowledge base is loaded and ready for questions.")
        st.info("Using Qdrant Cloud for fast, scalable vector search")
    else:
        st.info("Upload PDFs or add websites in the sidebar to build your knowledge base.")

# Display chat messages with sources
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Show sources for assistant messages
        if message['role'] == 'assistant' and idx in st.session_state.source_docs:
            source_documents = st.session_state.source_docs[idx]
            if source_documents:
                with st.expander("Source References"):
                    st.caption("Sources from your knowledge base")
                    
                    for i, doc in enumerate(source_documents, 1):
                        source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                        source_name = doc['document']
                        
                        if len(source_name) > 50:
                            display_name = source_name[:47] + "..."
                        else:
                            display_name = source_name
                        
                        st.markdown(f"**{source_icon} Source {i}:** `{display_name}`")
                        
                        if doc['page'] != 'N/A':
                            st.caption(f"**Page:** {doc['page']}")
                        
                        st.caption(f'**Excerpt:** "{doc["excerpt"]}"')
                        st.markdown("---")

# Handle user input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if not st.session_state.is_processed:
        st.warning("Please add some content (PDFs or websites) before asking questions.")
    else:
        try:
            with st.spinner("Thinking..."):
                import time
                
                # Timing diagnostics
                total_start = time.time()
                
                # Use the query processor with the API key
                result = process_query(prompt, api_key, user_id)
                
                total_time = time.time() - total_start
                
                if result['success']:
                    answer = result['answer']
                    source_documents = result['sources']

                    with st.chat_message('assistant'):
                        st.markdown(answer)
                        
                        # Show timing info (remove this line after testing speed)
                        st.caption(f"⏱️ Response time: {total_time:.2f}s")

                        if source_documents:
                            with st.expander("Source References"):
                                st.caption("Sources from your knowledge base")
                                
                                for i, doc in enumerate(source_documents, 1):
                                    source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                                    source_name = doc['document']
                                    
                                    if len(source_name) > 50:
                                        display_name = source_name[:47] + "..."
                                    else:
                                        display_name = source_name
                                    
                                    st.markdown(f"**{source_icon} Source {i}:** `{display_name}`")
                                    
                                    if doc['page'] != 'N/A':
                                        st.caption(f"**Page:** {doc['page']}")
                                    
                                    st.caption(f'**Excerpt:** "{doc["excerpt"]}"')
                                    st.markdown("---")

                    # Store message and sources
                    message_index = len(st.session_state.messages)
                    st.session_state.messages.append({'role': 'assistant', 'content': answer})
                    st.session_state.source_docs[message_index] = source_documents
                    
                    # Log query in MongoDB
                    db_manager.log_query(
                        user_id=user_id,
                        query=prompt,
                        response=answer,
                        sources_used=source_documents,
                        processing_time=total_time
                    )
                else:
                    error_msg = result['error']
                    st.error(error_msg)
                    st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

        except Exception as e:
            error_msg = f"An error occurred while processing your question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})