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

        # Improved prompt for better responses
        CUSTOM_PROMPT_TEMPLATE = """You are DocuBot AI, a helpful assistant that answers questions based on the user's uploaded documents and websites.

IMPORTANT INSTRUCTIONS:
- Answer questions using ONLY the information provided in the CONTEXT below
- If the context contains relevant information, provide a clear and helpful answer
- If the context doesn't contain enough information to answer the question, say: "I don't have enough information in your knowledge base to answer this question accurately."
- Always be specific and cite what you found in the documents
- Never make up information that isn't in the context

CONTEXT FROM YOUR KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

YOUR ANSWER (based only on the context above):"""
        
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )

        # FIXED: More lenient retriever settings
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 5,  # Retrieve more documents for better coverage
                # Removed score_threshold - let the LLM decide relevance
            }
        )
        
        # Optimized LLM configuration
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.2,  # Slightly higher for more natural responses
            max_tokens=400,  # More room for detailed answers
            timeout=20,
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
    seen_sources = set()  # Prevent duplicates
    
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
            
            # Create unique identifier
            page_num = metadata.get('page', 'N/A')
            if isinstance(page_num, int):
                page_num += 1
            
            source_id = f"{source_name}_{page_num}"
            
            # Skip duplicates
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            
            # Excerpt with better truncation
            excerpt = doc.page_content.strip()
            if len(excerpt) > 250:
                excerpt = excerpt[:247] + "..."
            
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
    """Fast query processing with error handling."""
    try:
        # Get cached chain
        qa_chain = get_cached_qa_chain(groq_api_key, user_id)
        
        if not qa_chain:
            return {
                'success': False,
                'error': "Knowledge base not ready. Please add documents first."
            }
        
        # Process query
        response = qa_chain.invoke({'query': prompt})
        result = response.get("result", "No answer generated.")
        source_documents = response.get("source_documents", [])
        
        # Format sources
        formatted_sources = format_source_documents(source_documents)
        
        return {
            'success': True,
            'answer': result,
            'sources': formatted_sources
        }
            
    except Exception as e:
        # Specific error handling
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