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
import shutil

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

# Initialize session state with caching and agentic mode
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'source_docs' not in st.session_state:
    st.session_state.source_docs = {}
if 'verification_reports' not in st.session_state:  # ✅ NEW: Store verification reports
    st.session_state.verification_reports = {}
if 'user_data_loaded' not in st.session_state:
    st.session_state.user_data_loaded = False
if 'cached_user_files' not in st.session_state:
    st.session_state.cached_user_files = []
if 'cached_user_scrapes' not in st.session_state:
    st.session_state.cached_user_scrapes = []
if 'vector_store_exists' not in st.session_state:
    st.session_state.vector_store_exists = vector_store_exists(user_id)
if 'last_processed_query' not in st.session_state:
    st.session_state.last_processed_query = ""
if 'use_agentic_mode' not in st.session_state:  # ✅ NEW: Agentic mode toggle
    st.session_state.use_agentic_mode = True

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

# Cache expensive sidebar operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_user_stats_cached(_db_manager, user_id):
    """Cached user stats to prevent repeated DB calls"""
    try:
        return _db_manager.get_user_stats(user_id)
    except Exception:
        return {'files_uploaded': 0, 'websites_scraped': 0}

# Function to parse verification report
def parse_verification_report(report_text):
    """Parse verification report into structured format"""
    if not report_text:
        return {}
    
    parsed = {
        "summary": "",
        "supported": "Unknown",
        "relevant": "Unknown",
        "confidence": "Unknown",
        "notes": []
    }
    
    try:
        lines = report_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for key indicators
            if "Supported:" in line:
                parsed["supported"] = "YES" if "YES" in line.upper() else "NO"
            elif "Relevant:" in line:
                parsed["relevant"] = "YES" if "YES" in line.upper() else "NO"
            elif "Confidence:" in line:
                parsed["confidence"] = line.replace("Confidence:", "").strip()
            elif "Summary:" in line:
                parsed["summary"] = line.replace("Summary:", "").strip()
            elif line.startswith("- "):
                parsed["notes"].append(line[2:])
            elif len(line) > 50 and not parsed["summary"]:  # Use first long line as summary
                parsed["summary"] = line
    except Exception:
        parsed["summary"] = report_text[:200] + "..." if len(report_text) > 200 else report_text
    
    return parsed

# --- Optimized Sidebar ---
with st.sidebar:
    st.title("DocuBot Controls")
    st.markdown("AI-powered assistant for your documents and websites.")
    
    st.markdown("---")
    
    # ✅ NEW: Agentic Mode Toggle
    agentic_mode = st.checkbox(
        "🤖 **Agentic Mode**", 
        value=st.session_state.use_agentic_mode,
        help="Enable advanced agent workflow with relevance checking, research, and verification"
    )
    st.session_state.use_agentic_mode = agentic_mode
    
    if agentic_mode:
        st.caption("🔍 **Features active:** Relevance checking → Research → Verification loop")
        st.success("Using Qdrant Cloud + Agentic Workflow")
    else:
        st.caption("⚡ **Classic mode:** Direct retrieval and answer")
        st.success("Using Qdrant Cloud Storage")
    
    # Knowledge Base Section
    if st.session_state.vector_store_exists:
        st.markdown("---")
        st.subheader("Your Knowledge Base")
        
        # Uploaded Files
        files_container = st.container()
        with files_container:
            with st.expander("Uploaded Files", expanded=True):
                if st.session_state.cached_user_files:
                    for file_record in st.session_state.cached_user_files[:5]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"📄 {file_record['filename']}")
                        with col2:
                            if st.button("🗑️", key=f"del_file_{file_record['upload_id']}"):
                                with st.spinner(f"Removing {file_record['filename']}..."):
                                    # Delete from Qdrant
                                    success = remove_documents_from_store(
                                        user_id, 
                                        file_record['filename'], 
                                        'pdf',
                                        db_manager
                                    )
                                    
                                    # Delete from MongoDB
                                    if success:
                                        db_manager.delete_file_upload(file_record['upload_id'])
                                        
                                        # Update session state
                                        st.session_state.cached_user_files = [
                                            f for f in st.session_state.cached_user_files 
                                            if f['upload_id'] != file_record['upload_id']
                                        ]
                                        
                                        st.success(f"File '{file_record['filename']}' removed from knowledge base!")
                                        st.rerun()
                                    else:
                                        st.warning("File not found in vector store, but removed from records")
                else:
                    st.info("No files uploaded yet")
        
        # Scraped Websites
        websites_container = st.container()
        with websites_container:
            with st.expander("Scraped Websites", expanded=True):
                if st.session_state.cached_user_scrapes:
                    for scrape_record in st.session_state.cached_user_scrapes:
                        for url in scrape_record.get('successful_urls', [])[:5]:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text(f"🌐 {url}")
                            with col2:
                                if st.button("🗑️", key=f"del_url_{scrape_record['scrape_id']}_{hash(url)}"):
                                    with st.spinner(f"Removing {url}..."):
                                        # Delete from Qdrant
                                        success = remove_documents_from_store(user_id, url, 'web')
                                        
                                        # Delete from MongoDB (update the scrape record)
                                        if success:
                                            # Update the scrape record to remove this URL
                                            from pymongo import UpdateOne
                                            db_manager.web_scrapes.update_one(
                                                {'scrape_id': scrape_record['scrape_id']},
                                                {'$pull': {'successful_urls': url}}
                                            )
                                            
                                            # Update session state
                                            st.session_state.cached_user_scrapes = db_manager.get_user_scrapes(user_id)
                                            
                                            st.success(f"URL '{url}' removed from knowledge base!")
                                            st.rerun()
                                        else:
                                            st.warning("URL not found in vector store, but removed from records")
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

    # Processing Options
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
        process_pdfs = st.button(
            "Process PDFs", 
            use_container_width=True, 
            type="primary",
            disabled=not uploaded_files
        )
        
        process_websites = st.button(
            "Scrape Websites", 
            use_container_width=True,
            type="primary",
            disabled=not urls_list
        )

    with col2:
        clear_all = st.button(
            "Clear All", 
            use_container_width=True, 
            type="secondary",
            help="Clear entire knowledge base and chat history"
        )

    # Handle processing
    if process_pdfs and uploaded_files:
        with st.spinner("Processing PDF documents..."):
            try:
                db, action = build_vector_store_from_pdfs(
                    user_id, uploaded_files, append=(processing_mode == "Add New Content")
                )
                
                if db is not None and action != "no_documents":
                    st.session_state.vector_store_exists = True
                    st.session_state.cached_user_files = db_manager.get_user_files(user_id)
                    st.success(f"PDF documents {action} successfully!")
                    if uploaded_files:
                        st.toast(f"Added {len(uploaded_files)} new documents", icon="📄")
                        st.rerun()
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

    if process_websites and urls_list:
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
                    st.rerun()
            except Exception as e:
                st.error(f"Error scraping websites: {str(e)}")

    if clear_all:
        with st.spinner("Clearing all data..."):
            # Clear Qdrant
            result = clear_all_data(user_id)
            
            # Clear MongoDB
            db_manager.clear_user_data(user_id)
            
            # Clear temp files
            try:
                from data_processing import get_user_data_path
                data_path = get_user_data_path(user_id)
                if data_path and os.path.exists(data_path):
                    shutil.rmtree(data_path)
                    print(f"🗑️ Cleared temp directory: {data_path}")
            except Exception as e:
                print(f"⚠️ Could not clear temp files: {e}")
            
            # Clear all session states
            st.session_state.vector_store_exists = False
            st.session_state.cached_user_files = []
            st.session_state.cached_user_scrapes = []
            st.session_state.messages = []
            st.session_state.source_docs = {}
            st.session_state.verification_reports = {}
            
            st.toast("All data cleared from Qdrant, MongoDB, and temp files!", icon="✨")
            st.rerun()

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_docs = {}
        st.session_state.verification_reports = {}
        st.toast("Chat history cleared!", icon="🧹")
    
    # Display stats with caching
    if st.session_state.vector_store_exists:
        st.markdown("---")
        st.subheader("Knowledge Base Info")
        
        stats = get_user_stats_cached(db_manager, user_id)
        files_count = stats.get('files_uploaded', len(st.session_state.cached_user_files))
        websites_count = stats.get('websites_scraped', len(st.session_state.cached_user_scrapes))
        
        st.write(f"**Files:** {files_count}")
        st.write(f"**Websites:** {websites_count}")
        
        # ✅ NEW: Show agent mode indicator
        if st.session_state.use_agentic_mode:
            st.caption("🤖 **Agentic mode:** Active")

# --- Optimized Main Chat ---
st.title("DocuBot AI: Chat with Your Documents & Websites")
st.markdown("Ask questions about your uploaded PDFs and scraped websites.")

# Mode indicator
if st.session_state.use_agentic_mode:
    st.info("🤖 **Agentic Mode Active** - Using advanced workflow with verification")
else:
    st.info("⚡ **Classic Mode** - Fast direct retrieval")

# Welcome message - only show if no messages
if not st.session_state.messages:
    if st.session_state.vector_store_exists:
        st.success("Ready! Your knowledge base is loaded and ready for questions.")
    else:
        st.info("Upload PDFs or add websites in the sidebar to build your knowledge base.")

# Display chat messages - optimized rendering
chat_container = st.container()
with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            # Show verification report for assistant messages in agentic mode
            if (message['role'] == 'assistant' and 
                idx in st.session_state.verification_reports):
                
                verification_data = st.session_state.verification_reports[idx]
                if verification_data and st.session_state.use_agentic_mode:
                    with st.expander("🔍 **Verification Report**", expanded=False):
                        # Parse and display report
                        parsed_report = parse_verification_report(verification_data)
                        
                        # Status badges
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if parsed_report.get("supported") == "YES":
                                st.success("✅ Supported")
                            elif parsed_report.get("supported") == "NO":
                                st.error("❌ Unsupported")
                            else:
                                st.info("🔍 Support Check")
                        
                        with col2:
                            if parsed_report.get("relevant") == "YES":
                                st.success("✅ Relevant")
                            elif parsed_report.get("relevant") == "NO":
                                st.error("❌ Irrelevant")
                            else:
                                st.info("🔍 Relevance")
                        
                        with col3:
                            if parsed_report.get("confidence"):
                                st.info(f"📊 {parsed_report['confidence']}")
                        
                        # Summary
                        if parsed_report.get("summary"):
                            st.markdown("**Summary:**")
                            st.write(parsed_report["summary"])
                        
                        # Notes
                        if parsed_report.get("notes"):
                            st.markdown("**Notes:**")
                            for note in parsed_report["notes"]:
                                st.markdown(f"- {note}")
                        
                        # Raw report (collapsible)
                        with st.expander("📋 View Raw Report"):
                            st.code(verification_data)
            
            # Show resources for assistant messages that have sources
            if (message['role'] == 'assistant' and 
                idx in st.session_state.source_docs):
                
                source_documents = st.session_state.source_docs[idx]
                
                if source_documents and len(source_documents) > 0:
                    with st.expander("📚 **Resources**", expanded=False):
                        st.caption("Resources from your knowledge base")
                        
                        for i, doc in enumerate(source_documents, 1):
                            source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                            source_name = doc['document']
                            
                            # Display source name
                            if len(source_name) > 50:
                                display_name = source_name[:47] + "..."
                            else:
                                display_name = source_name
                            
                            st.markdown(f"**{source_icon} Resource {i}:** `{display_name}`")
                            
                            if doc['page'] != 'N/A':
                                st.caption(f"**Page:** {doc['page']}")
                            
                            excerpt = doc["excerpt"]
                            st.caption(f'**Excerpt:** "{excerpt}"')
                            st.markdown("---")

# Handle user input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    # Prevent processing the same query multiple times
    if prompt != st.session_state.last_processed_query:
        st.session_state.last_processed_query = prompt
        
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        if not st.session_state.vector_store_exists:
            st.warning("Please add some content (PDFs or websites) before asking questions.")
        else:
            try:
                with st.spinner("Thinking..." if not st.session_state.use_agentic_mode else "🤖 Agentic workflow running..."):
                    result = process_query(
                        prompt, 
                        api_key, 
                        user_id,
                        use_agentic=st.session_state.use_agentic_mode
                    )
                    
                    if result['success']:
                        answer = result['answer']
                        source_documents = result['sources']
                        verification_report = result.get('verification_report')

                        with st.chat_message('assistant'):
                            st.markdown(answer)
                            
                            # Display verification report if available
                            if verification_report and st.session_state.use_agentic_mode:
                                with st.expander("🔍 **Verification Report**", expanded=False):
                                    parsed_report = parse_verification_report(verification_report)
                                    
                                    # Status badges
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if parsed_report.get("supported") == "YES":
                                            st.success("✅ Supported")
                                        elif parsed_report.get("supported") == "NO":
                                            st.error("❌ Unsupported")
                                        else:
                                            st.info("🔍 Support Check")
                                    
                                    with col2:
                                        if parsed_report.get("relevant") == "YES":
                                            st.success("✅ Relevant")
                                        elif parsed_report.get("relevant") == "NO":
                                            st.error("❌ Irrelevant")
                                        else:
                                            st.info("🔍 Relevance")
                                    
                                    with col3:
                                        if parsed_report.get("confidence"):
                                            st.info(f"📊 {parsed_report['confidence']}")
                                    
                                    # Summary
                                    if parsed_report.get("summary"):
                                        st.markdown("**Summary:**")
                                        st.write(parsed_report["summary"])
                                    
                                    # Notes
                                    if parsed_report.get("notes"):
                                        st.markdown("**Notes:**")
                                        for note in parsed_report["notes"]:
                                            st.markdown(f"- {note}")
                                    
                                    # Raw report
                                    with st.expander("📋 View Raw Report"):
                                        st.code(verification_report)
                            
                            # Display resources
                            if source_documents:
                                with st.expander("📚 **Resources**", expanded=False):
                                    st.caption("Resources from your knowledge base")
                                    
                                    for i, doc in enumerate(source_documents, 1):
                                        source_icon = "🌐" if doc.get('type') == 'web' else "📄"
                                        source_name = doc['document']
                                        
                                        if len(source_name) > 50:
                                            display_name = source_name[:47] + "..."
                                        else:
                                            display_name = source_name
                                        
                                        st.markdown(f"**{source_icon} Resource {i}:** `{display_name}`")
                                        
                                        if doc['page'] != 'N/A':
                                            st.caption(f"**Page:** {doc['page']}")
                                        
                                        excerpt = doc["excerpt"]
                                        st.caption(f'**Excerpt:** "{excerpt}"')
                                        st.markdown("---")
                        
                        # Store message and associated data
                        message_index = len(st.session_state.messages)
                        st.session_state.messages.append({'role': 'assistant', 'content': answer})
                        st.session_state.source_docs[message_index] = source_documents
                        
                        # ✅ Store verification report for this message
                        if verification_report:
                            st.session_state.verification_reports[message_index] = verification_report
                        
                        # Log query
                        if source_documents:
                            try:
                                db_manager.log_query(
                                    user_id=user_id,
                                    query=prompt,
                                    response=answer,
                                    sources_used=source_documents,
                                    processing_time=0,
                                    agentic_mode=st.session_state.use_agentic_mode,
                                    verification_result=parsed_report.get("supported") if verification_report else None
                                )
                            except Exception:
                                pass  # Silently fail query logging
                    else:
                        st.error(result['error'])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
