import os
import shutil
import streamlit as st
from data_processing import get_existing_pdf_files, save_uploaded_files
from vector_store import (
    clear_all_data, build_vector_store_from_pdfs, build_vector_store_from_urls,
    get_vector_store, vector_store_exists
)
from query_processor import process_query
from config import validate_api_key
from auth import setup_authentication
from database import MongoDBManager

# --- Configuration ---
st.set_page_config(page_title="DocuBot AI 🤖", page_icon="🤖", layout="wide")

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

# --- Streamlit UI ---
with st.sidebar:
    st.title("📄 DocuBot Controls")
    st.markdown("Upload PDFs or add websites to build your knowledge base.")
    
    # Show storage info
    st.success("🔗 Using Qdrant Cloud Storage")
    
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
            placeholder="Enter one or more URLs (one per line)\nExample:\nhttps://example.com\nhttps://docs.python.org\nhttps://en.wikipedia.org",
            help="Add websites to scrape and include in your knowledge base"
        )
        urls_list = [url.strip() for url in website_urls.split('\n') if url.strip()] if website_urls else []

    # Processing options
    st.markdown("---")
    st.subheader("⚙️ Processing Options")
    
    # Check if vector store already exists
    vector_store_exists_flag = vector_store_exists(user_id)
    
    if vector_store_exists_flag:
        st.success("✅ Knowledge base exists")
        existing_files = get_existing_pdf_files(user_id)
        st.write(f"**Current documents:** {len(existing_files)}")
        
        processing_mode = st.radio(
            "Processing Mode:",
            ["Add New Content", "Replace All Content"],
            help="Choose whether to add new content or replace everything"
        )
    else:
        st.info("📝 No knowledge base found. Will create new one.")
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
            st.session_state.is_processed = False
            st.session_state.db_loaded = False
            st.session_state.messages = []
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
                    st.success(f"✅ PDF documents {action} successfully!")
                    if new_files:
                        st.toast(f"Added {len(new_files)} new documents", icon="📄")
                elif action == "no_documents":
                    st.info("ℹ️ No PDF documents to process.")
                else:
                    st.error("❌ Failed to process PDF documents.")
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
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
                    st.success(f"✅ Websites {action} successfully!")
                    st.toast(f"Scraped {len(urls_list)} website(s)", icon="🌐")
                elif action == "failed":
                    st.error("❌ Failed to scrape websites.")
                    st.warning("""
                    **⚠️ Scraping Limitations:**
                    - ✅ Works: News sites, blogs, documentation, Wikipedia, most content sites
                    - ❌ Doesn't work on cloud: Client-side React/Vue/Angular apps
                    - 💡 For React apps: Run this app locally with Chrome/Selenium installed
                    """)
                else:
                    st.error("❌ Failed to scrape websites.")
            except Exception as e:
                st.error(f"❌ Error scraping websites: {str(e)}")
            st.rerun()
            
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared!", icon="🧹")
    
    # Display document info
    if st.session_state.get('is_processed', False):
        st.markdown("---")
        st.subheader("📊 Knowledge Base Info")
        
        # Show user stats from MongoDB
        stats = db_manager.get_user_stats(user_id)
        st.write(f"**Files Uploaded:** {stats['files_uploaded']}")
        st.write(f"**Websites Scraped:** {stats['websites_scraped']}")
        st.write(f"**Queries Made:** {stats['queries_made']}")

# --- Main Chat ---
st.title("🤖 DocuBot AI: Chat with Your Documents & Websites")
st.markdown("Ask questions about your uploaded PDFs and scraped websites.")

# Display welcome message
if not st.session_state.messages:
    if st.session_state.is_processed:
        st.success(f"✅ Ready! Your knowledge base is loaded and ready for questions.")
        st.info("💡 Using Qdrant Cloud for fast, scalable vector search")
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
                import time
                start_time = time.time()
                
                # Use the query processor with the API key
                result = process_query(prompt, api_key)
                processing_time = time.time() - start_time
                
                if result['success']:
                    enhanced_result = result['answer']
                    source_documents = result['sources']

                    with st.chat_message('assistant'):
                        st.markdown(enhanced_result)

                        if source_documents:
                            with st.expander("🔍 Source References"):
                                st.caption("Sources from your knowledge base")
                                
                                for i, doc in enumerate(source_documents, 1):
                                    # Show appropriate icon based on source type
                                    source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                                    source_name = doc['document']
                                    
                                    # Truncate long URLs for display
                                    if len(source_name) > 50:
                                        display_name = source_name[:47] + "..."
                                    else:
                                        display_name = source_name
                                    
                                    st.markdown(f"**{source_icon} Source {i}:** `{display_name}`")
                                    
                                    if doc['page'] != 'N/A':
                                        st.caption(f"**Page:** {doc['page']}")
                                    
                                    st.caption(f'**Excerpt:** "{doc["excerpt"]}"')
                                    st.markdown("---")

                    st.session_state.messages.append({'role': 'assistant', 'content': enhanced_result})
                    
                    # Log query in MongoDB
                    db_manager.log_query(
                        user_id=user_id,
                        query=prompt,
                        response=enhanced_result,
                        sources_used=source_documents,
                        processing_time=processing_time
                    )
                else:
                    error_msg = result['error']
                    st.error(error_msg)
                    st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

        except Exception as e:
            error_msg = f"⚠️ An error occurred while processing your question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})