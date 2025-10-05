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

        # Simple, effective prompt (like MediBot)
        CUSTOM_PROMPT_TEMPLATE = """You are a helpful assistant for DocuBot. Answer the question naturally and conversationally using the context provided from the user's documents and websites.

Context from your knowledge base:
{context}

Question: {question}

Provide a clear, helpful answer:"""
        
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )

        # Simple retriever (like MediBot - just k, no threshold)
        retriever = db.as_retriever(
            search_kwargs={"k": 5}  # Get top 5 relevant documents
        )
        
        # LLM config (matching MediBot's working setup)
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
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
    """Format source documents for display."""
    formatted_sources = []
    
    for doc in source_documents:
        try:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            
            # Determine source type and name
            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                source_type = 'web'
                source_name = source
            else:
                source_type = 'pdf'
                source_name = os.path.basename(str(source)) if source else 'Unknown'
            
            # Get page number
            page_num = metadata.get('page', 'N/A')
            if isinstance(page_num, int):
                page_num += 1  # Make it 1-indexed for display
            
            # Create excerpt
            excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            formatted_sources.append({
                'document': source_name,
                'page': page_num,
                'excerpt': excerpt,
                'type': source_type
            })
            
        except Exception as e:
            print(f"Error formatting source: {e}")
            continue
    
    return formatted_sources

def process_query(prompt, groq_api_key, user_id):
    """Process user query and return answer with sources."""
    try:
        # Get cached chain
        qa_chain = get_cached_qa_chain(groq_api_key, user_id)
        
        if not qa_chain:
            return {
                'success': False,
                'error': "Knowledge base not ready. Please add documents first."
            }
        
        # Process query (exactly like MediBot)
        response = qa_chain.invoke({'query': prompt})
        answer = response.get("result", "No answer generated.")
        source_documents = response.get("source_documents", [])
        
        # Format sources
        formatted_sources = format_source_documents(source_documents)
        
        return {
            'success': True,
            'answer': answer,
            'sources': formatted_sources
        }
            
    except Exception as e:
        # Error handling
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