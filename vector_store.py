import os
import streamlit as st
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from data_processing import get_document_chunks, save_uploaded_files, load_pdf_files, split_documents_into_chunks
from web_scraper import scrape_urls_to_chunks
from config import get_qdrant_config
from database import MongoDBManager

db_manager = MongoDBManager()

# ==========================
# EMBEDDINGS
# ==========================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

# ==========================
# QDRANT
# ==========================
def get_user_collection_name(user_id):
    return f"docubot_user_{user_id}" if user_id else "docubot_default"

@st.cache_resource
def get_qdrant_client():
    cfg = get_qdrant_config()
    return QdrantClient(
        url=cfg["url"],
        api_key=cfg["api_key"],
        timeout=30,
    )

@st.cache_resource
def get_qdrant_vector_store(user_id):
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    collection_name = get_user_collection_name(user_id)

    # FIXED: Better error handling for collection creation
    try:
        # Try to get the collection
        collection_info = client.get_collection(collection_name)
        print(f"✅ Found existing Qdrant collection: {collection_name}")
        print(f"   Points count: {collection_info.points_count}")
    except Exception as e:
        # Collection doesn't exist, create it
        print(f"⚠️ Collection '{collection_name}' not found, creating it...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # This MUST match the embedding model dimension
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created new Qdrant collection: {collection_name}")
        except Exception as create_error:
            print(f"❌ FAILED to create collection '{collection_name}': {create_error}")
            # Check if it's a permission issue
            if "permission" in str(create_error).lower() or "forbidden" in str(create_error).lower():
                print("💡 Possible issue: Your Qdrant API key might not have create collection permissions")
            raise create_error

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

# ==========================
# DOCUMENT METADATA EXTRACTION
# ==========================
def extract_document_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata and structure from any document file"""
    metadata = {
        "filename": os.path.basename(file_path),
        "title": "",
        "author": "",
        "pages": 0,
        "sections": [],
        "keywords": [],
        "file_type": "pdf",
        "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "processed_at": datetime.now().isoformat(),
        "topics": []
    }
    
    try:
        if file_path.endswith('.pdf'):
            # Try to use PyMuPDF if available
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                metadata["pages"] = len(doc)
                
                # Extract basic metadata
                if doc.metadata:
                    metadata["title"] = doc.metadata.get("title", "")
                    metadata["author"] = doc.metadata.get("author", "")
                    metadata["subject"] = doc.metadata.get("subject", "")
                    keywords = doc.metadata.get("keywords", "")
                    if keywords:
                        metadata["keywords"] = [k.strip() for k in keywords.split(',')]
                
                # Extract structure from first few pages
                max_pages_to_scan = min(15, len(doc))
                for page_num in range(max_pages_to_scan):
                    page = doc[page_num]
                    text = page.get_text()
                    
                    # Look for headings/sections
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if 10 < len(line) < 200:  # Reasonable heading length
                            # Check if it looks like a heading
                            if (line.isupper() or 
                                re.match(r'^(Chapter|Section|Part|Unit|Module|Topic)\s+\d+', line, re.IGNORECASE) or
                                re.match(r'^\d+\.\s+', line) or
                                re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:$', line) or
                                line.endswith(':')):
                                metadata["sections"].append({
                                    "text": line,
                                    "page": page_num + 1
                                })
                
                # Extract topics from content
                content_sample = ""
                for page_num in range(min(5, len(doc))):
                    content_sample += doc[page_num].get_text() + "\n"
                
                metadata["topics"] = extract_topics_from_text(content_sample)
                
                doc.close()
                
            except ImportError:
                # Fallback to simple metadata extraction
                print(f"⚠️ PyMuPDF not available, using basic metadata extraction for {file_path}")
                metadata["title"] = os.path.splitext(os.path.basename(file_path))[0]
        
        # Limit sections for efficiency
        metadata["sections"] = metadata["sections"][:25]
        
    except Exception as e:
        print(f"⚠️ Could not extract metadata from {file_path}: {e}")
        # Basic fallback
        metadata["title"] = os.path.splitext(os.path.basename(file_path))[0]
    
    return metadata

def extract_topics_from_text(text: str, max_topics: int = 10) -> List[str]:
    """Extract potential topics from text"""
    text_lower = text.lower()
    topics = set()
    
    # Common topic patterns
    patterns = [
        # Programming/tech
        r'\b(python|java|javascript|c\+\+|c#|php|ruby|go|rust|swift|kotlin)\b',
        r'\b(html|css|react|angular|vue|node\.js|django|flask|spring)\b',
        r'\b(api|sdk|framework|library|database|sql|nosql|mongodb|postgresql)\b',
        
        # Business
        r'\b(business|marketing|sales|finance|strategy|management|leadership)\b',
        r'\b(startup|enterprise|ecommerce|saas|b2b|b2c|customer|revenue|profit)\b',
        
        # Academic
        r'\b(research|study|analysis|methodology|experiment|theory|hypothesis)\b',
        r'\b(science|engineering|mathematics|physics|chemistry|biology|psychology)\b',
        
        # General
        r'\b(guide|tutorial|manual|handbook|reference|documentation)\b',
        r'\b(introduction|overview|background|conclusion|summary|appendix)\b'
    ]
    
    # Find matches
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            topics.add(match)
    
    # Also look for capitalized phrases (potential proper nouns/titles)
    lines = text.split('\n')
    for line in lines[:20]:  # First 20 lines
        words = line.split()
        for i, word in enumerate(words):
            if (len(word) > 3 and word[0].isupper() and 
                (i == 0 or words[i-1][-1] in ['.', ':', '-']) and
                word.lower() not in ['the', 'and', 'for', 'with', 'from']):
                topics.add(word)
    
    return list(topics)[:max_topics]

def create_document_overview_chunk(file_path: str, metadata: Dict) -> Document:
    """Create a document overview chunk for better metadata queries"""
    filename = os.path.basename(file_path)
    
    overview_text = f"""DOCUMENT OVERVIEW: {filename}

Title: {metadata.get('title', 'Untitled')}
Author: {metadata.get('author', 'Unknown')}
Pages: {metadata.get('pages', 'Unknown')}
File Size: {metadata.get('file_size', 0)} bytes
Processed: {metadata.get('processed_at', 'Unknown')}

"""
    
    # Add sections if available
    if metadata.get('sections'):
        overview_text += "Main Sections:\n"
        for section in metadata['sections'][:15]:  # Show first 15 sections
            overview_text += f"  • Page {section.get('page', '?')}: {section.get('text', '')}\n"
    
    # Add topics if available
    if metadata.get('topics'):
        overview_text += f"\nKey Topics: {', '.join(metadata['topics'][:10])}\n"
    
    # Add keywords if available
    if metadata.get('keywords'):
        overview_text += f"Keywords: {', '.join(metadata['keywords'][:10])}\n"
    
    overview_text += "\n---\nUse this document overview to understand the content structure."
    
    return Document(
        page_content=overview_text,
        metadata={
            "source": file_path,
            "type": "pdf",
            "page": 0,
            "is_overview": True,
            "document_metadata": json.dumps(metadata),
            "filename": filename
        }
    )

# ==========================
# BM25 - UPDATED TO USE VECTOR DB TEXT
# ==========================
@st.cache_resource(show_spinner=False)
def get_bm25_retriever(user_id):
    """Get BM25 retriever from ALL content in vector store (PDFs + Websites)"""
    try:
        # Get all documents from Qdrant vector store
        client = get_qdrant_client()
        collection_name = get_user_collection_name(user_id)
        
        # Check if collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist yet
            print(f"⚠️ BM25: Collection '{collection_name}' doesn't exist yet")
            return None
        
        # Fetch all documents from vector store
        all_points = []
        next_offset = None
        
        # Scroll through all points in collection
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            all_points.extend(points)
            
            if next_offset is None:
                break
        
        if not all_points:
            print(f"⚠️ BM25: No points found in collection '{collection_name}'")
            return None
        
        # Convert points to LangChain Documents
        documents = []
        for point in all_points:
            payload = point.payload or {}
            page_content = payload.get('page_content', '')
            
            if not page_content or len(page_content.strip()) == 0:
                continue
                
            # Extract metadata
            metadata = payload.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Ensure type field exists for consistency
            if 'type' not in metadata:
                if 'scraping_method' in metadata:
                    metadata['type'] = 'web'
                else:
                    metadata['type'] = 'pdf'
            
            # Add is_overview flag if present
            if 'is_overview' in metadata:
                metadata['is_overview'] = True
            
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        if not documents:
            print(f"⚠️ BM25: No valid documents found in points")
            return None
            
        # Create BM25 retriever
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 5
        
        print(f"✅ BM25 loaded {len(documents)} documents from vector store")
        return bm25
        
    except Exception as e:
        print(f"❌ Error creating BM25 retriever from vector store: {e}")
        return None

# ==========================
# DATA MANAGEMENT
# ==========================
def clear_all_data(user_id):
    """Clear all data from Qdrant for a user"""
    try:
        client = get_qdrant_client()
        collection_name = get_user_collection_name(user_id)
        
        # Check if collection exists before trying to delete
        try:
            client.get_collection(collection_name)
            client.delete_collection(collection_name)
            print(f"🗑️ Cleared Qdrant collection: {collection_name}")
            return "Cleared vector store"
        except Exception as e:
            # Collection doesn't exist, that's fine
            print(f"⚠️ Collection '{collection_name}' doesn't exist, nothing to clear")
            return "Collection didn't exist"
            
    except Exception as e:
        print(f"⚠️ Error in clear_all_data: {e}")
        return f"Error: {e}"

def remove_documents_from_store(user_id, source, doc_type, db_manager=None):
    """Remove documents from Qdrant and optionally clean temp files"""
    client = get_qdrant_client()
    collection = get_user_collection_name(user_id)
    
    try:
        # First check if collection exists
        try:
            client.get_collection(collection)
        except Exception:
            print(f"⚠️ Collection '{collection}' doesn't exist, nothing to delete")
            return False
        
        # Find ALL points to delete from Qdrant
        all_delete_ids = []
        next_offset = None
        deleted_count = 0
        
        while True:
            # Scroll through all points
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            # Check each point for matching source
            for p in points:
                meta = (p.payload or {}).get("metadata", {})
                stored = str(meta.get("source", ""))
                
                should_delete = False
                if doc_type == "pdf":
                    # For PDFs, check if filename matches (end of path)
                    if stored.endswith(source) or os.path.basename(stored) == source:
                        should_delete = True
                        deleted_count += 1
                else:  # web
                    # For web, check if URL matches
                    if stored == source or source in stored:
                        should_delete = True
                        deleted_count += 1
                
                if should_delete:
                    all_delete_ids.append(p.id)
            
            if next_offset is None:
                break
        
        # Delete from Qdrant
        if all_delete_ids:
            print(f"🗑️ Deleting {len(all_delete_ids)} chunks from Qdrant for {source}")
            client.delete(collection_name=collection, points_selector=all_delete_ids)
            return True
        
        print(f"⚠️ No documents found to delete for {source}")
        return False
        
    except Exception as e:
        print(f"❌ Error deleting documents: {e}")
        return False

# ==========================
# BUILDERS - UPDATED WITH METADATA
# ==========================
def build_vector_store_from_pdfs(user_id, uploaded_files, append=False):
    """Build vector store from uploaded PDF files and log to MongoDB"""
    print(f"📥 Starting PDF processing for user {user_id}")
    print(f"   Mode: {'Append' if append else 'Replace'}")
    print(f"   Files: {[f.name for f in uploaded_files]}")
    
    if not append:
        print("🔄 Clearing existing data...")
        clear_all_data(user_id)
        db_manager.clear_user_data(user_id)  # Clear MongoDB too
    else:
        print("➕ Adding to existing data...")
    
    # FIXED: Get vector store FIRST (this creates collection if needed)
    try:
        store = get_qdrant_vector_store(user_id)
    except Exception as e:
        print(f"❌ FAILED to get/create vector store: {e}")
        return None, "failed"
    
    # Save uploaded files to temp storage
    try:
        file_paths = save_uploaded_files(uploaded_files, user_id)
        print(f"✅ Saved {len(file_paths)} files to temp storage")
    except Exception as e:
        print(f"❌ Failed to save files: {e}")
        return None, "failed"
    
    # Get chunks from these files
    all_chunks = []
    file_stats = []  # To track file info for MongoDB
    
    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            print(f"📄 Processing: {filename}")
            
            # Extract metadata first
            metadata = extract_document_metadata(file_path)
            print(f"   Extracted metadata: {metadata['title'] or filename}, {metadata['pages']} pages")
            
            # Load documents
            documents = load_pdf_files([file_path])
            if documents:
                # Split into chunks
                chunks = split_documents_into_chunks(documents)
                
                # Add metadata to each chunk's metadata
                for chunk in chunks:
                    chunk.metadata["document_filename"] = filename
                    chunk.metadata["has_metadata"] = True
                    # Store metadata as JSON string for retrieval
                    chunk.metadata["doc_metadata"] = json.dumps({
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "pages": metadata.get("pages", 0),
                        "topics": metadata.get("topics", [])[:5]
                    })
                
                all_chunks.extend(chunks)
                
                # Create and add overview chunk
                overview_chunk = create_document_overview_chunk(file_path, metadata)
                all_chunks.append(overview_chunk)
                
                # Track stats
                pages = len(documents)
                file_stats.append({
                    'filename': filename,
                    'pages': pages,
                    'chunks': len(chunks) + 1,  # +1 for overview chunk
                    'metadata': metadata
                })
                print(f"   Created {len(chunks)} content chunks + 1 overview chunk from {pages} pages")
            else:
                print(f"⚠️ No documents loaded from {filename}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    if all_chunks:
        # Add to Qdrant
        print(f"📤 Adding {len(all_chunks)} total chunks to Qdrant...")
        try:
            store.add_documents(all_chunks)
            print(f"✅ Added {len(all_chunks)} chunks to Qdrant")
        except Exception as e:
            print(f"❌ Failed to add documents to Qdrant: {e}")
            return None, "failed"
        
        # Log to MongoDB with enhanced metadata
        print("📝 Logging files to MongoDB...")
        for file, stats in zip(uploaded_files, file_stats):
            try:
                db_manager.log_file_upload(
                    user_id=user_id,
                    filename=stats['filename'],
                    file_size=file.size,
                    pages_processed=stats['pages'],
                    metadata=stats.get('metadata', {})
                )
                print(f"   Logged: {stats['filename']} ({stats['pages']} pages)")
            except Exception as e:
                print(f"⚠️ Failed to log {stats['filename']} to MongoDB: {e}")
        
        print(f"✅ Successfully processed {len(file_stats)} files")
        return store, "added"
    
    print("❌ No chunks were created from the uploaded files")
    return None, "no_documents"

def build_vector_store_from_urls(user_id, urls, append=False):
    """Build vector store from URLs and log to MongoDB"""
    print(f"🌐 Starting URL processing for user {user_id}")
    print(f"   URLs: {urls}")
    
    if not append:
        print("🔄 Clearing existing data...")
        clear_all_data(user_id)
        db_manager.clear_user_data(user_id)
    else:
        print("➕ Adding to existing data...")
    
    # FIXED: Get vector store FIRST (this creates collection if needed)
    try:
        store = get_qdrant_vector_store(user_id)
    except Exception as e:
        print(f"❌ FAILED to get/create vector store: {e}")
        return None, "failed"
    
    chunks = scrape_urls_to_chunks(urls)

    if chunks:
        print(f"📤 Adding {len(chunks)} chunks to Qdrant...")
        try:
            store.add_documents(chunks)
            print(f"✅ Added {len(chunks)} chunks to Qdrant")
        except Exception as e:
            print(f"❌ Failed to add documents to Qdrant: {e}")
            return None, "failed"
        
        # Log to MongoDB
        print("📝 Logging URLs to MongoDB...")
        successful_urls = []
        for chunk in chunks:
            url = chunk.metadata.get('source')
            if url and url not in successful_urls:
                successful_urls.append(url)
        
        try:
            db_manager.log_web_scrape(
                user_id=user_id,
                urls=urls,
                successful_urls=successful_urls,
                total_chunks=len(chunks)
            )
            print(f"✅ Logged {len(successful_urls)} URLs to MongoDB")
        except Exception as e:
            print(f"⚠️ Failed to log URLs to MongoDB: {e}")
        
        return store, "added"
    
    print("❌ No chunks were created from the URLs")
    return None, "no_new_urls"

# ==========================
# DOCUMENT OVERVIEW FUNCTIONS
# ==========================
def get_document_overview(user_id: str) -> List[Dict]:
    """Get overview of all documents in the vector store"""
    try:
        client = get_qdrant_client()
        collection_name = get_user_collection_name(user_id)
        
        # First, try to get overview chunks
        response = client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "metadata.is_overview",
                        "match": {"value": True}
                    }
                ]
            },
            with_payload=True,
            limit=20
        )
        
        documents = []
        for point in response[0]:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            
            if metadata.get("is_overview"):
                try:
                    doc_metadata = json.loads(metadata.get("document_metadata", "{}"))
                    documents.append({
                        "filename": doc_metadata.get("filename", metadata.get("filename", "Unknown")),
                        "title": doc_metadata.get("title", ""),
                        "author": doc_metadata.get("author", ""),
                        "pages": doc_metadata.get("pages", 0),
                        "sections": doc_metadata.get("sections", [])[:10],  # Top 10 sections
                        "topics": doc_metadata.get("topics", [])[:10],      # Top 10 topics
                        "type": "pdf",
                        "has_overview": True
                    })
                except:
                    # Fallback if JSON parsing fails
                    documents.append({
                        "filename": metadata.get("filename", "Unknown"),
                        "title": metadata.get("filename", "Unknown"),
                        "type": "pdf",
                        "has_overview": False
                    })
        
        # If no overview chunks found, try to infer from regular chunks
        if not documents:
            print("⚠️ No overview chunks found, inferring from regular content...")
            # Get a sample of chunks to infer document info
            response = client.scroll(
                collection_name=collection_name,
                limit=50,
                with_payload=True
            )
            
            # Group by source
            sources = {}
            for point in response[0]:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                source = metadata.get("source", "Unknown")
                
                if source not in sources:
                    sources[source] = {
                        "chunks": [],
                        "type": metadata.get("type", "unknown")
                    }
                sources[source]["chunks"].append(payload.get("page_content", ""))
            
            # Create basic overview from sources
            for source, data in sources.items():
                doc_type = data["type"]
                if doc_type == "pdf":
                    filename = os.path.basename(source)
                else:
                    filename = source.replace('https://', '').replace('http://', '').split('/')[0]
                
                # Extract topics from sample content
                sample_content = " ".join(data["chunks"][:3]) if data["chunks"] else ""
                topics = extract_topics_from_text(sample_content, max_topics=5)
                
                documents.append({
                    "filename": filename,
                    "title": filename,
                    "type": doc_type,
                    "topics": topics,
                    "has_overview": False,
                    "estimated_from": len(data["chunks"])
                })
        
        return documents
        
    except Exception as e:
        print(f"❌ Error getting document overview: {e}")
        return []

def generate_document_summary(user_id: str) -> str:
    """Generate a textual summary of all documents"""
    documents = get_document_overview(user_id)
    
    if not documents:
        return "No documents found in the knowledge base."
    
    summary = "## 📚 Knowledge Base Summary\n\n"
    
    pdf_docs = [d for d in documents if d.get("type") == "pdf"]
    web_docs = [d for d in documents if d.get("type") == "web"]
    
    if pdf_docs:
        summary += f"### 📄 PDF Documents ({len(pdf_docs)})\n"
        for doc in pdf_docs:
            summary += f"**{doc.get('filename', 'Unknown')}**\n"
            if doc.get("title") and doc["title"] != doc["filename"]:
                summary += f"Title: {doc['title']}\n"
            if doc.get("pages"):
                summary += f"Pages: {doc['pages']}\n"
            if doc.get("topics"):
                summary += f"Topics: {', '.join(doc['topics'][:5])}\n"
            summary += "\n"
    
    if web_docs:
        summary += f"### 🌐 Web Pages ({len(web_docs)})\n"
        for doc in web_docs[:10]:  # Limit to 10 web pages
            summary += f"- {doc.get('filename', 'Unknown')}\n"
            if doc.get("topics"):
                summary += f"  Topics: {', '.join(doc['topics'][:3])}\n"
    
    # Add statistics
    total_chunks = sum(d.get("estimated_from", 1) for d in documents)
    summary += f"\n### 📊 Statistics\n"
    summary += f"- Total documents: {len(documents)}\n"
    summary += f"- Estimated content chunks: {total_chunks}\n"
    summary += f"- Document types: {len(pdf_docs)} PDFs, {len(web_docs)} web pages\n"
    
    # Add common topics across all documents
    all_topics = []
    for doc in documents:
        all_topics.extend(doc.get("topics", []))
    
    if all_topics:
        from collections import Counter
        topic_counts = Counter(all_topics)
        common_topics = [topic for topic, count in topic_counts.most_common(10)]
        summary += f"- Common topics: {', '.join(common_topics)}\n"
    
    summary += "\n*Ask specific questions about any document for detailed information.*"
    
    return summary

# ==========================
# ACCESS
# ==========================
def get_vector_store(user_id):
    return get_qdrant_vector_store(user_id)

def vector_store_exists(user_id):
    try:
        client = get_qdrant_client()
        info = client.get_collection(get_user_collection_name(user_id))
        exists = info.points_count > 0
        print(f"🔍 Vector store exists check: {exists} (points: {info.points_count})")
        return exists
    except Exception as e:
        print(f"🔍 Vector store doesn't exist: {e}")
        return False
