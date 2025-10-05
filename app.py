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
if 'expanded_sections' not in st.session_state:
    st.session_state.expanded_sections = {
        'uploaded_files': True,
        'scraped_websites': True,
        'sources': {}  # Track which message sources are expanded
    }

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
    
    # Knowledge Base Section - With dropdowns
    if st.session_state.vector_store_exists:
        st.markdown("---")
        st.subheader("Your Knowledge Base")
        
        # Uploaded Files Dropdown
        with st.expander("📁 Uploaded Files", expanded=st.session_state.expanded_sections['uploaded_files']):
            if st.session_state.cached_user_files:
                for file_record in st.session_state.cached_user_files[:8]:  # Increased limit
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        # Truncate long filenames for display
                        display_name = file_record['filename']
                        if len(display_name) > 30:
                            display_name = display_name[:27] + "..."
                        st.text(display_name)
                        st.caption(f"{file_record['file_size'] / 1024:.1f} KB")
                    
                    with col2:
                        if st.button("🗑️", key=f"del_file_{file_record['upload_id']}", help="Remove file"):
                            with st.spinner("Removing file..."):
                                success = remove_documents_from_store(user_id, file_record['filename'], 'pdf')
                                if success:
                                    db_manager.delete_file_upload(file_record['upload_id'])
                                    # Update cache
                                    st.session_state.cached_user_files = db_manager.get_user_files(user_id)
                                    st.success(f"Removed {file_record['filename']}")
            else:
                st.info("No files uploaded yet")
        
        # Scraped Websites Dropdown
        with st.expander("🌐 Scraped Websites", expanded=st.session_state.expanded_sections['scraped_websites']):
            if st.session_state.cached_user_scrapes:
                for scrape_record in st.session_state.cached_user_scrapes:
                    for url in scrape_record.get('successful_urls', [])[:6]:  # Limit display
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            # Truncate long URLs for display
                            display_url = url
                            if len(display_url) > 35:
                                display_url = display_url[:32] + "..."
                            st.text(display_url)
                            st.caption(f"Scraped: {scrape_record.get('created_at', 'Unknown')[:10]}")
                        
                        with col2:
                            if st.button("🗑️", key=f"del_url_{scrape_record['scrape_id']}_{hash(url)}", help="Remove URL"):
                                with st.spinner("Removing URL..."):
                                    success = remove_documents_from_store(user_id, url, 'web')
                                    if success:
                                        db_manager.delete_web_scrape(scrape_record['scrape_id'])
                                        # Update cache
                                        st.session_state.cached_user_scrapes = db_manager.get_user_scrapes(user_id)
                                        st.success(f"Removed {display_url}")
            else:
                st.info("No websites scraped yet")
    
    # Input sections
    st.markdown("---")
    input_tab1, input_tab2 = st.tabs(["Upload PDFs", "Add Websites"])
    
    with input_tab1:
        uploaded_files = st.file_uploader(
            "Upload PDF Documents", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Select one or more PDF files to add to your knowledge base"
        )
    
    with input_tab2:
        website_urls = st.text_area(
            "Website URLs", 
            placeholder="Enter one URL per line\nExample:\nhttps://example.com\nhttps://docs.streamlit.io",
            help="Add websites to scrape and include in your knowledge base"
        )
        urls_list = [url.strip() for url in website_urls.split('\n') if url.strip()] if website_urls else []

    # Processing - optimized
    st.markdown("---")
    st.subheader("Processing Options")
    
    processing_mode = st.radio(
        "Processing Mode:",
        ["Add New Content", "Replace All Content"],
        disabled=not st.session_state.vector_store_exists,
        help="Add to existing knowledge base or replace everything"
    )
    
    # Process buttons
    col1, col2 = st.columns(2)
    
    with col1:
        process_pdfs_clicked = st.button(
            "Process PDFs", 
            use_container_width=True, 
            type="primary",
            disabled=not uploaded_files
        )
        
        process_websites_clicked = st.button(
            "Scrape Websites", 
            use_container_width=True,
            type="primary",
            disabled=not urls_list
        )

    with col2:
        clear_all_clicked = st.button(
            "Clear All", 
            use_container_width=True, 
            type="secondary",
            help="Clear entire knowledge base and chat history"
        )

    # Handle processing without reruns
    if process_pdfs_clicked and uploaded_files:
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
                    if new_files:
                        st.toast(f"Added {len(new_files)} new documents", icon="📄")
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

    if process_websites_clicked and urls_list:
        with st.spinner(f"Scraping {len(urls_list)} website(s)..."):
            try:
                db, action = build_vector_store_from_urls(
                    user_id, urls_list, append=(processing_mode == "Add New Content")
                )
                
                if db is not None and action not in ["no_new_urls", "failed"]:
                    st.session_state.vector_store_exists = True
                    st.session_state.cached_user_scrapes = db_manager.get_user_scrapes(user_id)
                    st.success(f"Websites {action} successfully!")
                    st.toast(f"Scraped {len(urls_list)} website(s)", icon="🌐")
            except Exception as e:
                st.error(f"Error scraping websites: {str(e)}")

    # Clear buttons
    if clear_all_clicked:
        with st.spinner("Clearing all data..."):
            result = clear_all_data(user_id)
            db_manager.clear_user_data(user_id)
            st.session_state.vector_store_exists = False
            st.session_state.cached_user_files = []
            st.session_state.cached_user_scrapes = []
            st.session_state.messages = []
            st.session_state.source_docs = {}
            st.toast(result, icon="✨")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_docs = {}
        st.session_state.expanded_sections['sources'] = {}
        st.toast("Chat history cleared!", icon="🧹")
    
    # Display stats with error handling
    if st.session_state.vector_store_exists:
        st.markdown("---")
        st.subheader("Knowledge Base Info")
        
        try:
            stats = db_manager.get_user_stats(user_id)
            
            # Safe stat display with fallbacks
            files_count = stats.get('files_uploaded', len(st.session_state.cached_user_files))
            websites_count = stats.get('websites_scraped', len(st.session_state.cached_user_scrapes))
            queries_count = stats.get('queries_processed', 0)
            
            st.write(f"**Files:** {files_count}")
            st.write(f"**Websites:** {websites_count}")
            st.write(f"**Queries:** {queries_count}")
            
        except Exception as e:
            # Fallback if stats aren't available
            st.write(f"**Files:** {len(st.session_state.cached_user_files)}")
            st.write(f"**Websites:** {len(st.session_state.cached_user_scrapes)}")
            st.write("**Queries:** 0")

# --- Optimized Main Chat ---
st.title("DocuBot AI: Chat with Your Documents & Websites")
st.markdown("Ask questions about your uploaded PDFs and scraped websites.")

# Welcome message
if not st.session_state.messages:
    if st.session_state.vector_store_exists:
        st.success("✅ Ready! Your knowledge base is loaded and ready for questions.")
        st.info("💡 Using Qdrant Cloud for fast, scalable vector search")
    else:
        st.info("📚 Upload PDFs or add websites in the sidebar to build your knowledge base.")

# Optimized message display with auto-expand sources for latest message
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Show sources for assistant messages with auto-expand for latest
        if (message['role'] == 'assistant' and 
            idx in st.session_state.source_docs and
            st.session_state.source_docs[idx]):
            
            source_documents = st.session_state.source_docs[idx]
            source_count = len(source_documents)
            
            # Auto-expand sources for the latest message, others collapsed
            is_latest = idx == len(st.session_state.messages) - 1
            
            with st.expander(f"📚 Source References ({source_count})", expanded=is_latest):
                st.caption("Sources from your knowledge base")
                
                for i, doc in enumerate(source_documents, 1):
                    source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                    source_name = doc['document']
                    
                    # Format display name
                    if len(source_name) > 50:
                        display_name = source_name[:47] + "..."
                    else:
                        display_name = source_name
                    
                    st.markdown(f"**{source_icon} Source {i}:** `{display_name}`")
                    
                    if doc['page'] != 'N/A':
                        st.caption(f"**Page:** {doc['page']}")
                    
                    excerpt = doc["excerpt"]
                    st.caption(f'**Excerpt:** "{excerpt}"')
                    st.markdown("---")

# Chat input with optimized processing
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if not st.session_state.vector_store_exists:
        st.warning("⚠️ Please add some content (PDFs or websites) before asking questions.")
    else:
        try:
            with st.spinner("🔍 Searching knowledge base..."):
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
                    if source_documents:
                        try:
                            db_manager.log_query(
                                user_id=user_id,
                                query=prompt,
                                response=answer,
                                sources_used=source_documents,
                                processing_time=0
                            )
                        except Exception as e:
                            # Silently fail query logging - it's not critical
                            print(f"Query logging failed: {e}")
                else:
                    st.error(f"❌ {result['error']}")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")