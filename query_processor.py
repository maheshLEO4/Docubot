import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from vector_store import get_vector_store

@st.cache_resource
def get_cached_qa_chain(groq_api_key, user_id):
    """Cached QA chain - only loads once per user session"""
    db = get_vector_store(user_id)
    if db is None:
        return None

    # Simplified prompt
    CUSTOM_PROMPT_TEMPLATE = """Using the provided context, answer the question with accuracy and clarity.

    Context:
    {context}

    Question: {question}

    Provide a well-structured answer that:
    - Directly addresses the question
    - Cites specific information from the context
    - Acknowledges any limitations in the available information
    - Uses clear, professional language

    Answer:"""
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )

    try:
        # Optimized retriever
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 2}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=200,  # Even smaller for speed
                groq_api_key=groq_api_key,
            ),
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
            source = doc.metadata.get('source', 'Unknown')
            
            if isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    source_name = source
                else:
                    source_name = os.path.basename(source)
            else:
                source_name = 'Unknown'
            
            page_num = doc.metadata.get('page', 'N/A')
            if page_num != 'N/A' and isinstance(page_num, int):
                page_num += 1
            
            excerpt = doc.page_content
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."
            
            formatted_sources.append({
                'document': source_name,
                'page': page_num,
                'excerpt': excerpt,
                'type': 'web' if source.startswith(('http://', 'https://')) else 'pdf'
            })
        except Exception as e:
            print(f"Error formatting source document: {e}")
            continue
    
    return formatted_sources

def process_query(prompt, groq_api_key, user_id):
    """Process a user query and return the response with source documents."""
    try:
        # Use cached QA chain
        qa_chain = get_cached_qa_chain(groq_api_key, user_id)
        
        if qa_chain:
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]
            
            formatted_sources = format_source_documents(source_documents)
            
            return {
                'success': True,
                'answer': result,
                'sources': formatted_sources
            }
        else:
            return {
                'success': False,
                'error': "Unable to load QA system. Please make sure you have processed some documents first."
            }
            
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(f"Query processing error: {e}")
        return {
            'success': False,
            'error': error_msg
        }