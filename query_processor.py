import streamlit as st
from vector_store import VectorStoreManager
from config import config

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
                print("QA Chain: Store not available")
                return None
            
            # Check if store has documents
            try:
                test_results = store.similarity_search("test", k=1)
                if len(test_results) == 0:
                    print("QA Chain: Store has no documents")
                    return None
            except Exception as e:
                print(f"QA Chain: Error checking store contents: {e}")
                return None
            
            # Optimized retriever
            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.5
                }
            )
            
            # Import PromptTemplate
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
            
            # Import ChatGroq
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=500,
                timeout=30,
                groq_api_key=api_key
            )
            
            # Import RetrievalQA
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
    
    def process(self, query: str, api_key: str) -> dict:
        """Process user query"""
        cache_key = f"{self.user_id}_{hash(query)}"
        
        # Check cache
        if cache_key in st.session_state.query_cache:
            return st.session_state.query_cache[cache_key]
        
        try:
            # First check if vector store actually has data
            if not self.vector_store.exists():
                return {
                    'success': False,
                    'error': "Knowledge base is empty. Please add documents first."
                }
            
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
            print(f"Query error: {e}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _format_sources(self, source_docs):
        """Format source documents for display"""
        import os
        formatted = []
        for doc in source_docs[:3]:
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