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
# CONVERSATION MANAGER
# ==========================
class ConversationManager:
    def __init__(self):
        self.history = []
        self.current_topic = None
        self.last_question = None
        self.last_answer_summary = None
        
    def add_interaction(self, question: str, answer: str, query_type: str = None):
        """Add a Q&A pair to conversation history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Create a summary of the answer for context
        answer_summary = answer[:150] + "..." if len(answer) > 150 else answer
        
        interaction = {
            "timestamp": timestamp,
            "question": question,
            "answer_summary": answer_summary,
            "full_answer": answer[:500],  # Store truncated answer
            "query_type": query_type
        }
        
        # Keep only last 5 interactions
        self.history.append(interaction)
        if len(self.history) > 5:
            self.history.pop(0)
            
        # Update current topic based on question
        self.last_question = question
        self.last_answer_summary = answer_summary
        self.current_topic = self._extract_topic(question)
        
    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question"""
        # Simple topic extraction - can be enhanced
        if "string" in question.lower():
            return "C++ strings"
        elif "array" in question.lower() or "char" in question.lower():
            return "arrays/char arrays"
        elif "convert" in question.lower():
            return "conversion"
        elif "example" in question.lower():
            return "examples"
        elif "code" in question.lower():
            return "code examples"
        return "general programming"
        
    def get_context(self) -> str:
        """Get formatted conversation context"""
        if not self.history:
            return "No previous conversation."
            
        context_lines = ["Previous conversation:"]
        for i, interaction in enumerate(self.history[-3:], 1):  # Last 3 interactions
            context_lines.append(f"{i}. Q: {interaction['question']}")
            context_lines.append(f"   A: {interaction['answer_summary']}")
            
        return "\n".join(context_lines)
        
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect query intent and type"""
        query_lower = query.lower().strip()
        
        # Check for follow-up queries
        if len(query.split()) <= 3 and self.last_question:
            # Very short queries are likely follow-ups
            if query_lower in ["examples", "give examples", "show examples"]:
                return {
                    "type": "follow_up_examples",
                    "original_topic": self.current_topic,
                    "requires_context": True
                }
            elif query_lower in ["code", "show code", "example code"]:
                return {
                    "type": "follow_up_code",
                    "original_topic": self.current_topic,
                    "requires_context": True
                }
            elif query_lower in ["more", "continue", "elaborate"]:
                return {
                    "type": "follow_up_elaborate",
                    "original_topic": self.current_topic,
                    "requires_context": True
                }
        
        # Check for metadata/document queries
        metadata_keywords = ["what's in", "contains", "document", "pdf", "topics", "chapters", "sections"]
        if any(keyword in query_lower for keyword in metadata_keywords):
            return {
                "type": "document_metadata",
                "requires_context": False
            }
        
        # Check for general document content queries
        content_keywords = ["explain", "how to", "what is", "difference between", "compare"]
        if any(keyword in query_lower for keyword in content_keywords):
            return {
                "type": "document_content",
                "requires_context": True
            }
        
        # Default to document content
        return {
            "type": "document_content",
            "requires_context": True
        }

# ==========================
# ENHANCED HYBRID RETRIEVER
# ==========================
class EnhancedHybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    conversation_context: Optional[str] = None
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        docs = []
        seen = set()
        
        # Enhance query with conversation context if available
        enhanced_query = query
        if self.conversation_context:
            enhanced_query = f"{query} [Context: {self.conversation_context}]"
        
        for retriever in self.retrievers:
            try:
                results = retriever.invoke(
                    enhanced_query,
                    config={"callbacks": run_manager.get_child() if run_manager else None}
                )
                
                for doc in results:
                    # Better deduplication using content hash
                    content_hash = hash(doc.page_content[:500])  # First 500 chars
                    if content_hash not in seen:
                        seen.add(content_hash)
                        docs.append(doc)
                        
                        # Limit to 7 results total for better coverage
                        if len(docs) >= 7:
                            break
                            
                if len(docs) >= 7:
                    break
                    
            except Exception as e:
                logger.warning(f"Retriever failed: {e}")
                continue
        
        # If we have conversation context and few results, try without context
        if len(docs) < 3 and self.conversation_context:
            for retriever in self.retrievers:
                try:
                    results = retriever.invoke(
                        query,  # Original query without context
                        config={"callbacks": run_manager.get_child() if run_manager else None}
                    )
                    
                    for doc in results:
                        content_hash = hash(doc.page_content[:500])
                        if content_hash not in seen:
                            seen.add(content_hash)
                            docs.append(doc)
                            
                            if len(docs) >= 7:
                                break
                                
                    if len(docs) >= 7:
                        break
                        
                except Exception as e:
                    continue
        
        return docs[:7]  # Return up to 7 documents

# ==========================
# QUERY PROCESSOR
# ==========================
class QueryProcessor:
    def __init__(self, groq_api_key: str, user_id: str):
        self.groq_api_key = groq_api_key
        self.user_id = user_id
        self.conversation_manager = ConversationManager()
        
    def initialize_qa_chain(self):
        """Initialize QA chain components"""
        try:
            vector_store = get_vector_store(self.user_id)
            if not vector_store:
                return None
                
            # Enhanced system prompt with conversation awareness
            system_prompt = """You are DocuBot, a helpful assistant answering questions based on the user's documents.

IMPORTANT GUIDELINES:
1. ALWAYS maintain conversation context. If the user asks a follow-up question (like "give examples" or "show code"), 
   provide examples/code related to the previous topic.
2. If asked "what's in my document?" or similar, provide a summary of document contents if available.
3. If information is not in the provided context, say: "Sorry, I don't have any information about your question."
4. Provide complete code examples with proper includes when relevant.
5. Format your answers clearly with markdown when helpful.

{conversation_context}

Context from documents:
{context}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            bm25 = get_bm25_retriever(self.user_id)

            return {
                "retriever": vector_retriever,
                "bm25": bm25,
                "llm": ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                    groq_api_key=self.groq_api_key
                ),
                "prompt": prompt
            }
            
        except Exception as e:
            logger.exception("Error creating QA chain")
            return None
    
    def process_query(self, query: str, use_agentic: bool = True) -> Dict[str, Any]:
        """Process a query with conversation context"""
        qa_components = self.initialize_qa_chain()
        if not qa_components:
            return {"success": False, "error": "Knowledge base not ready"}
        
        # Detect query intent
        intent = self.conversation_manager.detect_intent(query)
        
        # Build enhanced query based on intent
        enhanced_query = self._enhance_query(query, intent)
        
        # Get conversation context for retrieval
        conversation_context = self.conversation_manager.get_context()
        
        # Create retriever with conversation context
        retrievers = []
        if qa_components["bm25"]:
            retrievers.append(qa_components["bm25"])
        retrievers.append(qa_components["retriever"])
        
        retriever = EnhancedHybridRetriever(
            retrievers=retrievers,
            conversation_context=conversation_context if intent.get("requires_context", True) else None
        )
        
        try:
            if use_agentic:
                workflow = AgentWorkflow()
                result = workflow.full_pipeline(enhanced_query, retriever)
                docs = retriever.invoke(enhanced_query)
                answer = result.get("draft_answer")
            else:
                docs = retriever.invoke(enhanced_query)
                context = "\n".join(d.page_content for d in docs)
                
                # Format the conversation context for the prompt
                formatted_conv_context = conversation_context if intent.get("requires_context", True) else ""
                
                message = qa_components["prompt"].format(
                    input=query,
                    context=context,
                    conversation_context=formatted_conv_context
                )
                answer = qa_components["llm"].invoke(message).content
            
            # Store in conversation history
            self.conversation_manager.add_interaction(query, answer, intent["type"])
            
            return {
                "success": True,
                "answer": answer,
                "sources": format_source_documents(docs[:5]),
                "verification_report": result.get("verification_report") if use_agentic else None,
                "query_intent": intent
            }
            
        except Exception as e:
            logger.exception("Query processing failed")
            return {"success": False, "error": str(e)}
    
    def _enhance_query(self, query: str, intent: Dict[str, Any]) -> str:
        """Enhance query based on intent detection"""
        if intent["type"] == "follow_up_examples":
            topic = intent.get("original_topic", "the previous topic")
            return f"{query} Provide specific examples related to {topic}"
        
        elif intent["type"] == "follow_up_code":
            topic = intent.get("original_topic", "the previous topic")
            return f"{query} Provide complete code examples for {topic}"
        
        elif intent["type"] == "follow_up_elaborate":
            return f"{query} Provide more detailed information"
        
        elif intent["type"] == "document_metadata":
            # Special handling for document metadata queries
            return "document contents table of contents chapters sections topics overview summary"
        
        return query

# ==========================
# SOURCE FORMATTER
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
            doc_name = source_str.split('//')[1].split('/')[0] if '//' in source_str else source_str
        
        sources.append({
            "document": doc_name,
            "page": page,
            "excerpt": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            "type": doc_type,
            "full_source": source_str
        })
    return sources

# ==========================
# COMPATIBILITY WRAPPER
# ==========================
@st.cache_resource(show_spinner=False)
def get_cached_query_processor(groq_api_key, user_id):
    """Cached QueryProcessor for Streamlit compatibility"""
    return QueryProcessor(groq_api_key, user_id)

def process_query(prompt, groq_api_key, user_id, use_agentic=True):
    """Wrapper for backward compatibility"""
    processor = get_cached_query_processor(groq_api_key, user_id)
    return processor.process_query(prompt, use_agentic)
