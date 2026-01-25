import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from vector_store import get_vector_store, get_bm25_retriever
from agents.workflow import AgentWorkflow  # ✅ NEW IMPORT

# ==========================
# QA CHAIN (Keep for fallback if needed)
# ==========================
@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(groq_api_key, user_id):
    """Cached QA chain - only loads once per user session"""
    try:
        db = get_vector_store(user_id)
        if db is None:
            return None

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # ==========================
        # HYBRID RETRIEVER
        # ==========================
        vector_retriever = db.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = get_bm25_retriever(user_id)

        if bm25_retriever:
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
        else:
            retriever = vector_retriever

        # ==========================
        # LLM
        # ==========================
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

# ==========================
# SOURCE FORMATTER (Keep as is)
# ==========================
def format_source_documents(source_documents):
    """Format source documents for display."""
    formatted_sources = []

    for doc in source_documents:
        try:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')

            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                source_type = 'web'
                source_name = source
            else:
                source_type = 'pdf'
                source_name = os.path.basename(str(source)) if source else 'Unknown'

            page_num = metadata.get('page', 'N/A')
            if isinstance(page_num, int):
                page_num += 1

            excerpt = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )

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

# ==========================
# AGENTIC QUERY PROCESSOR (MAIN UPDATE)
# ==========================
def process_query(prompt, groq_api_key, user_id, use_agentic=True):  # ✅ Added use_agentic param
    """Process user query and return answer with sources."""
    try:
        # ==========================
        # 1. GET RETRIEVER
        # ==========================
        db = get_vector_store(user_id)
        if not db:
            return {
                'success': False,
                'error': "Knowledge base not ready. Please add documents first."
            }

        # Create hybrid retriever (same as before)
        vector_retriever = db.as_retriever(search_kwargs={"k": 8})  # More docs for agents
        bm25_retriever = get_bm25_retriever(user_id)
        
        if bm25_retriever:
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
        else:
            retriever = vector_retriever

        # ==========================
        # 2. AGENTIC WORKFLOW
        # ==========================
        if use_agentic:
            workflow = AgentWorkflow()
            agent_result = workflow.full_pipeline(
                question=prompt,
                retriever=retriever
            )
            
            # For source tracking, we still retrieve documents
            retrieved_docs = retriever.invoke(prompt)
            
            return {
                'success': True,
                'answer': agent_result.get("draft_answer", "No answer generated."),
                'sources': format_source_documents(retrieved_docs[:5]),  # Top 5 sources
                'verification_report': agent_result.get("verification_report", "")
            }
        
        # ==========================
        # 3. FALLBACK: CLASSICAL QA
        # ==========================
        else:
            qa_chain = get_cached_qa_chain(groq_api_key, user_id)
            
            if not qa_chain:
                return {
                    'success': False,
                    'error': "Knowledge base not ready. Please add documents first."
                }

            response = qa_chain.invoke({'query': prompt})

            return {
                'success': True,
                'answer': response.get("result", "No answer generated."),
                'sources': format_source_documents(
                    response.get("source_documents", [])
                ),
                'verification_report': None  # No verification in classic mode
            }

    except Exception as e:
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
