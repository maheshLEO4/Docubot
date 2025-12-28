from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from vector_store import get_vector_store
from config import get_config

def process_query(user_id, prompt):
    try:
        vectorstore = get_vector_store(user_id)
        llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            groq_api_key=get_config('GROQ_API_KEY'),
            temperature=0.1
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        result = qa_chain.invoke({"query": prompt})
        return {"answer": result["result"]}
    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}"}