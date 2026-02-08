import os
import streamlit as st
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from agents.workflow import AgentWorkflow
from vector_store import get_vector_store, get_bm25_retriever

logger = logging.getLogger(__name__)

# ==========================
# GENERALIZED CONVERSATION MANAGER
# ==========================
class ConversationManager:
    def __init__(self):
        self.history = []
        self.current_topics = []  # Multiple topics can be active
        self.last_question = None
        self.last_answer_summary = None
        self.document_types = set()  # Track types of documents user has
        
    def add_interaction(self, question: str, answer: str, query_type: str = None):
        """Add a Q&A pair to conversation history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create a summary of the answer for context
        answer_summary = self._create_answer_summary(answer)
        
        interaction = {
            "timestamp": timestamp,
            "question": question,
            "answer_summary": answer_summary,
            "full_answer": answer[:500],
            "query_type": query_type,
            "detected_topics": self._extract_topics(question, answer)
        }
        
        # Keep last 5 interactions
        self.history.append(interaction)
        if len(self.history) > 5:
            self.history.pop(0)
            
        # Update current topics
        self.last_question = question
        self.last_answer_summary = answer_summary
        
        # Add new topics from this interaction
        for topic in interaction["detected_topics"]:
            if topic not in self.current_topics:
                self.current_topics.append(topic)
        
        # Keep only last 5 topics
        if len(self.current_topics) > 5:
            self.current_topics = self.current_topics[-5:]
        
    def _create_answer_summary(self, answer: str) -> str:
        """Create a concise summary of the answer"""
        # Remove code blocks for summary
        clean_answer = re.sub(r'```.*?\n', '', answer)
        clean_answer = re.sub(r'```', '', clean_answer)
        
        # Take first 2-3 sentences
        sentences = re.split(r'[.!?]+', clean_answer)
        summary = ' '.join(sentences[:3]).strip()
        
        if len(summary) > 150:
            summary = summary[:147] + "..."
            
        return summary if summary else "Provided information"
    
    def _extract_topics(self, question: str, answer: str) -> List[str]:
        """Extract topics from question and answer"""
        combined_text = f"{question} {answer}".lower()
        topics = set()
        
        # Common technical topics
        tech_keywords = {
            'programming': ['code', 'program', 'function', 'class', 'object', 'variable', 'loop'],
            'strings': ['string', 'text', 'char', 'character', 'concatenat'],
            'arrays': ['array', 'list', 'collection', 'vector'],
            'data': ['data', 'database', 'storage', 'json', 'xml'],
            'web': ['web', 'html', 'css', 'javascript', 'api', 'http'],
            'ai': ['ai', 'machine learning', 'neural', 'model', 'train'],
            'math': ['math', 'calculation', 'formula', 'equation', 'statistic'],
            'business': ['business', 'market', 'finance', 'economic', 'strategy']
        }
        
        # Extract topics based on content
        for topic, keywords in tech_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    topics.add(topic)
                    break
        
        # Also extract specific terms from the question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        common_terms = {'what', 'how', 'why', 'when', 'where', 'which', 'explain', 'example', 'code'}
        specific_terms = question_words - common_terms
        
        # Add specific technical terms as topics
        for term in specific_terms:
            if len(term) > 3 and term not in ['with', 'from', 'about', 'have', 'does']:
                topics.add(term)
        
        return list(topics)[:5]  # Return up to 5 topics
        
    def get_context(self) -> str:
        """Get formatted conversation context for prompts"""
        if not self.history:
            return "No previous conversation."
            
        context_lines = ["## Recent Conversation"]
        for i, interaction in enumerate(self.history[-3:], 1):
            context_lines.append(f"**Q{i}:** {interaction['question']}")
            if interaction['answer_summary']:
                context_lines.append(f"**A{i}:** {interaction['answer_summary']}")
            context_lines.append("---")
            
        # Add current topics if available
        if self.current_topics:
            context_lines.append(f"\n**Current Topics:** {', '.join(self.current_topics)}")
            
        return "\n".join(context_lines)
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect query intent and type - GENERALIZED"""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Check for very short queries (likely follow-ups)
        if len(words) <= 3:
            if self.last_question:  # We have context
                if any(word in query_lower for word in ['example', 'examples', 'show me', 'demonstrate']):
                    return {
                        "type": "follow_up_examples",
                        "requires_context": True,
                        "original_topic": self.current_topics[-1] if self.current_topics else "previous topic"
                    }
                elif any(word in query_lower for word in ['code', 'snippet', 'program', 'implementation']):
                    return {
                        "type": "follow_up_code",
                        "requires_context": True,
                        "original_topic": self.current_topics[-1] if self.current_topics else "previous topic"
                    }
                elif any(word in query_lower for word in ['more', 'continue', 'elaborate', 'detail']):
                    return {
                        "type": "follow_up_elaborate",
                        "requires_context": True
                    }
        
        # Document metadata queries
        metadata_patterns = [
            r'what(\'s| is) in',
            r'contains?',
            r'document content',
            r'pdf content',
            r'table of content',
            r'chapters?',
            r'sections?',
            r'topics? covered',
            r'overview'
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, query_lower):
                return {
                    "type": "document_metadata",
                    "requires_context": False
                }
        
        # General knowledge queries
        if any(word in query_lower for word in ['explain', 'what is', 'define', 'meaning of']):
            return {
                "type": "explanation",
                "requires_context": True
            }
        
        # How-to queries
        if any(word in query_lower for word in ['how to', 'how do i', 'steps to', 'process of']):
            return {
                "type": "how_to",
                "requires_context": True
            }
        
        # Comparison queries
        if any(word in query_lower for word in ['difference between', 'compare', 'vs', 'versus']):
            return {
                "type": "comparison",
                "requires_context": True
            }
        
        # Default to general query
        return {
            "type": "general",
            "requires_context": True
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        self.current_topics = []
        self.last_question = None
        self.last_answer_summary = None

# ==========================
# ENHANCED HYBRID RETRIEVER (GENERALIZED)
# ==========================
class EnhancedHybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    conversation_context: Optional[str] = None
    query_intent: Optional[Dict] = None
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = []
        seen_content = set()
        
        # Prepare queries based on intent
        queries_to_try = [query]
        
        # Add enhanced queries for specific intents
        if self.query_intent:
            intent_type = self.query_intent.get("type", "")
            
            if intent_type == "follow_up_examples" and self.conversation_context:
                # Try to find examples
                enhanced = f"examples of {self.query_intent.get('original_topic', '')} {query}"
                queries_to_try.append(enhanced)
                
            elif intent_type == "follow_up_code":
                enhanced = f"code example {query} programming implementation"
                queries_to_try.append(enhanced)
            
            elif intent_type == "document_metadata":
                # Special terms for document structure
                queries_to_try = [
                    "table of contents chapters sections overview summary",
                    "document structure",
                    "topics covered"
                ]
        
        # Try different query variations
        all_docs = []
        for query_variant in queries_to_try[:2]:  # Try first 2 variants
            variant_docs = self._retrieve_with_variant(query_variant, run_manager)
            for doc in variant_docs:
                content_hash = hash(doc.page_content[:300])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
            
            if len(all_docs) >= 8:
                break
        
        # Sort by relevance (simplified - in production use scores)
        return all_docs[:8]
    
    def _retrieve_with_variant(self, query: str, run_manager) -> List[Document]:
        """Retrieve documents using a specific query variant"""
        docs = []
        
        for retriever in self.retrievers:
            try:
                results = retriever.invoke(
                    query,
                    config={"callbacks": run_manager.get_child() if run_manager else None}
                )
                docs.extend(results)
                
                if len(docs) >= 5:
                    break
                    
            except Exception as e:
                logger.debug(f"Retriever variant failed: {e}")
                continue
        
        return docs

# ==========================
# GENERALIZED QUERY PROCESSOR
# ==========================
class GeneralizedQueryProcessor:
    def __init__(self, groq_api_key: str, user_id: str):
        self.groq_api_key = groq_api_key
        self.user_id = user_id
        self.conversation_manager = ConversationManager()
        
    def initialize_qa_chain(self):
        """Initialize QA chain components - GENERALIZED"""
        try:
            vector_store = get_vector_store(self.user_id)
            if not vector_store:
                return None
                
            # GENERALIZED system prompt
            system_prompt = """You are DocuBot, an intelligent assistant that helps users understand their documents.

IMPORTANT GUIDELINES:
1. **CONVERSATION AWARENESS**: Maintain context from previous questions. If the user asks a follow-up question 
   (like "give examples" or "show code"), relate it to the previous discussion.

2. **DOCUMENT FOCUS**: Base your answers primarily on the provided document context. 
   If information is not in the context, say: "Based on the available information, I don't have specific details about this."

3. **COMPREHENSIVE ANSWERS**: Provide complete information including:
   - Clear explanations
   - Relevant examples when helpful
   - Code snippets for programming topics
   - Step-by-step instructions for how-to questions

4. **FORMATTING**: Use markdown formatting for readability:
   - **Bold** for key terms
   - Lists for steps or features
   - Code blocks with language specification
   - Tables for comparisons when relevant

5. **HONESTY**: If you're unsure or lack information, acknowledge it rather than guessing.

{conversation_context}

Document Context:
{context}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            vector_retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.3  # Minimum relevance score
                }
            )
            
            bm25 = get_bm25_retriever(self.user_id)

            return {
                "vector_retriever": vector_retriever,
                "bm25_retriever": bm25,
                "llm": ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                    groq_api_key=self.groq_api_key,
                    max_tokens=1000
                ),
                "prompt": prompt
            }
            
        except Exception as e:
            logger.exception("Error creating QA chain")
            return None
    
    def process_query(self, query: str, use_agentic: bool = True) -> Dict[str, Any]:
        """Process a query with conversation context - GENERALIZED"""
        qa_components = self.initialize_qa_chain()
        if not qa_components:
            return {"success": False, "error": "Knowledge base not ready"}
        
        # Detect query intent
        intent = self.conversation_manager.detect_intent(query)
        
        # Get conversation context
        conversation_context = self.conversation_manager.get_context()
        
        # Build retrievers list
        retrievers = []
        if qa_components["bm25_retriever"]:
            retrievers.append(qa_components["bm25_retriever"])
        retrievers.append(qa_components["vector_retriever"])
        
        # Create retriever with context
        retriever = EnhancedHybridRetriever(
            retrievers=retrievers,
            conversation_context=conversation_context if intent.get("requires_context", True) else None,
            query_intent=intent
        )
        
        try:
            # Retrieve relevant documents
            docs = retriever.invoke(query)
            
            # Handle document metadata queries specially
            if intent["type"] == "document_metadata" and docs:
                answer = self._handle_metadata_query(query, docs)
            elif use_agentic:
                workflow = AgentWorkflow()
                result = workflow.full_pipeline(query, retriever)
                answer = result.get("draft_answer", "")
                verification_report = result.get("verification_report")
            else:
                context = "\n".join(d.page_content for d in docs[:5])
                
                # Format context for prompt
                formatted_conv_context = conversation_context if intent.get("requires_context", True) else ""
                
                message = qa_components["prompt"].format(
                    input=query,
                    context=context,
                    conversation_context=formatted_conv_context
                )
                
                response = qa_components["llm"].invoke(message)
                answer = response.content
                verification_report = None
            
            # If answer suggests no info but we have docs, try to provide a helpful response
            if "don't have" in answer.lower() and docs:
                answer = self._create_helpful_response(query, docs, answer)
            
            # Store in conversation history
            self.conversation_manager.add_interaction(query, answer, intent["type"])
            
            return {
                "success": True,
                "answer": answer,
                "sources": format_source_documents(docs[:5]),
                "verification_report": verification_report,
                "query_intent": intent
            }
            
        except Exception as e:
            logger.exception("Query processing failed")
            return {"success": False, "error": str(e)}
    
    def _handle_metadata_query(self, query: str, docs: List[Document]) -> str:
        """Handle queries about document contents/metadata"""
        # Extract document information from chunks
        document_info = {}
        
        for doc in docs:
            meta = doc.metadata or {}
            source = meta.get("source", "Unknown")
            
            if source not in document_info:
                document_info[source] = {
                    "name": os.path.basename(str(source)),
                    "type": "web" if str(source).startswith(('http://', 'https://')) else "pdf",
                    "excerpts": [],
                    "pages": set()
                }
            
            # Add excerpt
            excerpt = doc.page_content[:150].strip()
            if excerpt and len(excerpt) > 20:
                document_info[source]["excerpts"].append(excerpt)
            
            # Add page if available
            page = meta.get("page")
            if isinstance(page, int):
                document_info[source]["pages"].add(page + 1)
        
        # Generate response
        response = "## 📚 Document Overview\n\n"
        
        if not document_info:
            return "I couldn't find specific document information in the knowledge base."
        
        for source, info in document_info.items():
            doc_type_icon = "🌐" if info["type"] == "web" else "📄"
            response += f"### {doc_type_icon} {info['name']}\n"
            
            if info["pages"]:
                pages = sorted(info["pages"])
                if len(pages) > 3:
                    page_str = f"Pages {min(pages)}-{max(pages)}"
                else:
                    page_str = f"Pages {', '.join(map(str, pages))}"
                response += f"**Coverage:** {page_str}\n"
            
            if info["excerpts"]:
                response += "**Key Content:**\n"
                for excerpt in info["excerpts"][:3]:  # Show top 3 excerpts
                    response += f"- {excerpt}...\n"
            
            response += "\n"
        
        response += "\n*Ask specific questions about any of these documents for more detailed information.*"
        return response
    
    def _create_helpful_response(self, query: str, docs: List[Document], original_answer: str) -> str:
        """Create a more helpful response when docs exist but answer says no info"""
        # Extract topics from retrieved docs
        doc_topics = set()
        for doc in docs[:3]:
            content = doc.page_content.lower()
            # Look for common section headers
            lines = content.split('\n')
            for line in lines[:5]:
                if len(line) > 10 and len(line) < 100:
                    if line.endswith(':') or line.isupper():
                        doc_topics.add(line.strip(':'))
        
        if doc_topics:
            topics_list = ', '.join(list(doc_topics)[:5])
            return (
                f"While I don't have specific information about '{query}', "
                f"your documents cover related topics including: **{topics_list}**.\n\n"
                f"Would you like me to explain any of these topics in detail?"
            )
        
        return original_answer

# ==========================
# SOURCE FORMATTER (GENERALIZED)
# ==========================
def format_source_documents(docs: List[Document]) -> List[Dict]:
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        
        if isinstance(page, int):
            page += 1
        
        source_str = str(source)
        doc_type = "web" if source_str.startswith(('http://', 'https://')) else "pdf"
        
        # Extract document name
        if doc_type == "pdf":
            doc_name = os.path.basename(source_str)
        else:
            # Clean URL for display
            doc_name = source_str.replace('https://', '').replace('http://', '').split('/')[0]
        
        # Create excerpt - smarter truncation
        content = doc.page_content.strip()
        if len(content) > 200:
            # Try to truncate at sentence end
            sentences = re.split(r'[.!?]', content)
            excerpt = sentences[0] if sentences else content[:200]
            if len(excerpt) > 200:
                excerpt = excerpt[:197] + "..."
            elif len(sentences) > 1 and len(excerpt + sentences[1]) < 250:
                excerpt = excerpt + sentences[1]
                excerpt = excerpt.strip() + "..."
        else:
            excerpt = content
        
        sources.append({
            "document": doc_name,
            "page": page,
            "excerpt": excerpt,
            "type": doc_type,
            "full_source": source_str,
            "relevance_score": meta.get("score", 0) if "score" in meta else None
        })
    
    return sources

# ==========================
# COMPATIBILITY WRAPPER
# ==========================
@st.cache_resource(show_spinner=False)
def get_cached_query_processor(groq_api_key, user_id):
    """Cached QueryProcessor for Streamlit compatibility"""
    return GeneralizedQueryProcessor(groq_api_key, user_id)

def process_query(prompt, groq_api_key, user_id, use_agentic=True):
    """Wrapper for backward compatibility"""
    processor = get_cached_query_processor(groq_api_key, user_id)
    return processor.process_query(prompt, use_agentic)
