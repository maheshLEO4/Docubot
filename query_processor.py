import streamlit as st
from langchain_groq import ChatGroq
from vector_store import VectorStoreManager
from config import config
import os

class QueryProcessor:
    """Optimized query processor with caching"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_store = VectorStoreManager(user_id)
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache"""
        if 'query_cache' not in st.session_state:
            st.session_state.query_cache = {}
    
    @st.cache_resource(ttl=300, show_spinner=False)
    def _get_qa_chain(_self, api_key: str):
        """Cached QA chain"""
        try:
            store = _self.vector_store.get_store()
            if not store:
                return None
            
            # Optimized retriever
            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.5
                }
            )
            
            # Efficient prompt template
            from langchain_core.prompts import PromptTemplate
            prompt_template = """Use the following context to answer the question. 
            If you don't know the answer, say you don't know. Keep answers concise.
            
            Context: {context}
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Optimized LLM
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=500,
                timeout=30,
                groq_api_key=api_key
            )
            
            # Create QA chain (updated for latest LangChain)
            from langchain.chains import RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': prompt,
                    'verbose': False
                }
            )
            
            return qa_chain
            
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return None
    
    def _format_sources(self, source_docs):
        """Format source documents for display"""
        formatted = []
        for doc in source_docs[:3]:  # Limit to top 3
            try:
                metadata = doc.metadata
                source = metadata.get('source', 'Unknown')
                
                if source.startswith(('http://', 'https://')):
                    doc_type = 'web'
                    doc_name = source.split('//')[-1].split('/')[0]
                else:
                    doc_type = 'pdf'
                    doc_name = os.path.basename(str(source))
                
                formatted.append({
                    'document': doc_name,
                    'type': doc_type,
                    'page': metadata.get('page', 'N/A'),
                    'excerpt': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content
                })
            except:
                continue
        
        return formatted
    
    def process(self, query: str, api_key: str) -> dict:
        """Process user query"""
        cache_key = f"{self.user_id}_{hash(query)}"
        
        # Check cache
        if cache_key in st.session_state.query_cache:
            return st.session_state.query_cache[cache_key]
        
        try:
            # Get QA chain
            qa_chain = self._get_qa_chain(api_key)
            if not qa_chain:
                return {
                    'success': False,
                    'error': "Knowledge base not ready. Please add documents first."
                }
            
            # Process query
            with st.spinner("🔍 Searching knowledge base..."):
                result = qa_chain.invoke({'query': query})
            
            answer = result.get('result', 'No answer generated.')
            sources = self._format_sources(result.get('source_documents', []))
            
            response = {
                'success': True,
                'answer': answer,
                'sources': sources
            }
            
            # Cache response
            st.session_state.query_cache[cache_key] = response
            return response
            
        except Exception as e:
            error_msg = "Sorry, I encountered an issue. Please try again."
            if "timeout" in str(e):
                error_msg = "Request timed out. Please try a shorter question."
            elif "rate limit" in str(e):
                error_msg = "Rate limit exceeded. Please wait a moment."
            
            print(f"Query error: {e}")
            return {
                'success': False,
                'error': error_msg
            }