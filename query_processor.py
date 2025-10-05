import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from vector_store import get_vector_store

@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(groq_api_key, user_id):
    """Cached QA chain - only loads once per user session"""
    try:
        db = get_vector_store(user_id)
        if db is None:
            return None

        # Improved prompt for document-specific responses
        CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant for DocuBot. Your role is to answer questions based ONLY on the provided context from the user's uploaded documents and websites.



CONTEXT FROM USER'S DOCUMENTS:
{context}

USER'S QUESTION: {question}

BASED ON THE ABOVE CONTEXT, PROVIDE A HELPFUL ANSWER:"""
        
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )

        # Optimized retriever with smaller context
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 3,  # Increased to 3 for better coverage
                'score_threshold': 0.6  # Slightly lower threshold to catch more relevant docs
            }
        )
        
        # Faster LLM configuration
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=300,  # Increased for more detailed answers
            timeout=15,  # Slightly longer timeout
            groq_api_key=groq_api_key,
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        return qa_chain
        
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

def format_source_documents(source_documents):
    """Fast source document formatting."""
    formatted_sources = []
    
    for doc in source_documents:
        try:
            # Fast metadata extraction
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            
            # Determine source type and name quickly
            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                source_type = 'web'
                source_name = source
            else:
                source_type = 'pdf'
                source_name = os.path.basename(str(source)) if source else 'Unknown'
            
            # Simple page number handling
            page_num = metadata.get('page', 'N/A')
            if isinstance(page_num, int):
                page_num += 1
            
            # Fast excerpt truncation
            excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            formatted_sources.append({
                'document': source_name,
                'page': page_num,
                'excerpt': excerpt,
                'type': source_type
            })
            
        except Exception:
            # Skip problematic documents silently
            continue
    
    return formatted_sources
def process_query(prompt, groq_api_key, user_id):
    """Fast query processing with error handling."""
    try:
        # Get cached chain (fast)
        qa_chain = get_cached_qa_chain(groq_api_key, user_id)
        
        if not qa_chain:
            return {
                'success': False,
                'error': "Knowledge base not ready. Please add documents first."
            }
        
        # Fast invocation
        response = qa_chain.invoke({'query': prompt})
        result = response.get("result", "No answer generated.")
        source_documents = response.get("source_documents", [])
        
        print(f"DEBUG: Raw source documents: {source_documents}")  # Debug line
        print(f"DEBUG: Number of source documents: {len(source_documents)}")  # Debug line
        
        # Fast formatting
        formatted_sources = format_source_documents(source_documents)
        
        print(f"DEBUG: Formatted sources: {formatted_sources}")  # Debug line
        
        return {
            'success': True,
            'answer': result,
            'sources': formatted_sources
        }
            
    except Exception as e:
        # Specific error handling for common cases
        error_msg = "Sorry, I encountered an issue processing your question. Please try again."
        
        if "timeout" in str(e).lower():
            error_msg = "Request timed out. Please try a shorter question."
        elif "rate limit" in str(e).lower():
            error_msg = "Rate limit exceeded. Please wait a moment and try again."
        elif "api key" in str(e).lower():
            error_msg = "API configuration issue. Please check your settings."
            
        print(f"Query processing error: {e}")
        return {
            'success': False,
            'error': error_msg
        }