import os
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

# Get API key
try:
    api_key = validate_api_key()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Setup authentication
user_id = setup_authentication()

# Initialize database
db_manager = MongoDBManager()

# Initialize session state with caching
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'source_docs' not in st.session_state:
    st.session_state.source_docs = {}
if 'user_data_loaded' not in st.session_state:
    st.session_state.user_data_loaded = False
if 'cached_user_files' not in st.session_state:
    st.session_state.cached_user_files = []
if 'cached_user_scrapes' not in st.session_state:
    st.session_state.cached_user_scrapes = []
if 'vector_store_exists' not in st.session_state:
    st.session_state.vector_store_exists = vector_store_exists(user_id)

# Load user data once
if user_id and not st.session_state.user_data_loaded:
    with st.spinner("Loading your knowledge base..."):
        try:
            if st.session_state.vector_store_exists:
                st.session_state.cached_user_files = db_manager.get_user_files(user_id)
                st.session_state.cached_user_scrapes = db_manager.get_user_scrapes(user_id)
            st.session_state.user_data_loaded = True
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")

# --- Optimized Sidebar ---
with st.sidebar:
    st.title("DocuBot Controls")
    
    # Lightweight storage info
    st.success("Using Qdrant Cloud Storage")
    
    # Knowledge Base Section - Lazy loaded
    if st.session_state.vector_store_exists:
        st.markdown("---")
        st.subheader("Your Knowledge Base")
        
        # Use checkboxes to control expansion
        show_files = st.checkbox("Show Uploaded Files", True)
        if show_files:
            with st.container():
                if st.session_state.cached_user_files:
                    for file_record in st.session_state.cached_user_files[:5]:  # Limit display
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"📄 {file_record['filename']}")
                        with col2:
                            if st.button("🗑️", key=f"del_file_{file_record['upload_id']}"):
                                # Handle deletion without rerun
                                with st.spinner("Removing file..."):
                                    success = remove_documents_from_store(user_id, file_record['filename'], 'pdf')
                                    if success:
                                        db_manager.delete_file_upload(file_record['upload_id'])
                                        # Update cache instead of rerun
                                        st.session_state.cached_user_files = db_manager.get_user_files(user_id)
                                        st.success("File removed!")
                else:
                    st.info("No files uploaded yet")
        
        show_urls = st.checkbox("Show Scraped Websites", True)
        if show_urls:
            with st.container():
                if st.session_state.cached_user_scrapes:
                    # Similar optimized URL display
                    pass
    
    # Input sections
    st.markdown("---")
    input_tab1, input_tab2 = st.tabs(["Upload PDFs", "Add Websites"])
    
    with input_tab1:
        uploaded_files = st.file_uploader(
            "Upload PDF Documents", type=["pdf"], accept_multiple_files=True
        )
    
    with input_tab2:
        website_urls = st.text_area("Website URLs", placeholder="Enter one URL per line")
        urls_list = [url.strip() for url in website_urls.split('\n') if url.strip()] if website_urls else []

    # Processing - optimized
    st.markdown("---")
    st.subheader("Processing Options")
    
    processing_mode = st.radio(
        "Processing Mode:",
        ["Add New Content", "Replace All Content"],
        disabled=not st.session_state.vector_store_exists
    )
    
    # Process buttons with better state management
    process_pdfs_clicked = st.button("Process PDFs", use_container_width=True, 
                                   disabled=not uploaded_files)
    process_websites_clicked = st.button("Scrape Websites", use_container_width=True,
                                       disabled=not urls_list)

    # Handle processing without reruns
    if process_pdfs_clicked:
        with st.spinner("Processing PDF documents..."):
            try:
                new_files = save_uploaded_files(uploaded_files, user_id)
                db, action = build_vector_store_from_pdfs(
                    user_id, uploaded_files, append=(processing_mode == "Add New Content")
                )
                
                if db is not None and action != "no_documents":
                    st.session_state.vector_store_exists = True
                    st.session_state.cached_user_files = db_manager.get_user_files(user_id)
                    st.success(f"PDF documents {action} successfully!")
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

    if process_websites_clicked:
        with st.spinner(f"Scraping {len(urls_list)} website(s)..."):
            try:
                db, action = build_vector_store_from_urls(
                    user_id, urls_list, append=(processing_mode == "Add New Content")
                )
                
                if db is not None and action not in ["no_new_urls", "failed"]:
                    st.session_state.vector_store_exists = True
                    st.session_state.cached_user_scrapes = db_manager.get_user_scrapes(user_id)
                    st.success(f"Websites {action} successfully!")
            except Exception as e:
                st.error(f"Error scraping websites: {str(e)}")

    # Clear buttons
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_docs = {}
        st.toast("Chat history cleared!", icon="🧹")

# --- Optimized Main Chat ---
st.title("DocuBot AI: Chat with Your Documents & Websites")

# Welcome message
if not st.session_state.messages:
    if st.session_state.vector_store_exists:
        st.success("Ready! Your knowledge base is loaded.")
    else:
        st.info("Upload PDFs or add websites to build your knowledge base.")

# Optimized message display
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Only show sources when requested
        if (message['role'] == 'assistant' and 
            idx in st.session_state.source_docs and
            st.session_state.source_docs[idx]):
            
            source_count = len(st.session_state.source_docs[idx])
            if st.button(f"📚 Show Sources ({source_count})", key=f"show_sources_{idx}"):
                with st.expander("Source References", expanded=True):
                    for i, doc in enumerate(st.session_state.source_docs[idx], 1):
                        source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                        source_name = doc['document']
                        display_name = source_name[:47] + "..." if len(source_name) > 50 else source_name
                        
                        st.markdown(f"**{source_icon} Source {i}:** `{display_name}`")
                        if doc['page'] != 'N/A':
                            st.caption(f"**Page:** {doc['page']}")
                        st.caption(f'**Excerpt:** "{doc["excerpt"]}"')
                        st.markdown("---")

# Chat input with optimized processing
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if not st.session_state.vector_store_exists:
        st.warning("Please add content before asking questions.")
    else:
        try:
            with st.spinner("Thinking..."):
                result = process_query(prompt, api_key, user_id)
                
                if result['success']:
                    answer = result['answer']
                    source_documents = result['sources']

                    with st.chat_message('assistant'):
                        st.markdown(answer)

                    # Store message and sources
                    message_index = len(st.session_state.messages)
                    st.session_state.messages.append({'role': 'assistant', 'content': answer})
                    st.session_state.source_docs[message_index] = source_documents
                    
                    # Log query (non-blocking)
                    db_manager.log_query(
                        user_id=user_id,
                        query=prompt,
                        response=answer,
                        sources_used=source_documents,
                        processing_time=0  # You can calculate this if needed
                    )
                else:
                    st.error(result['error'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")