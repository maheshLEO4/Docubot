import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from vector_store import get_vector_store
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def enhance_answer(original_answer, question, context_docs):
    """Post-process the answer to improve quality."""
    
    vague_phrases = ["i don't know", "the context doesn't", "not mentioned", "no information"]
    if any(phrase in original_answer.lower() for phrase in vague_phrases):
        return original_answer
    
    if len(original_answer.split()) > 50:
        sentences = [s.strip() for s in original_answer.split('. ') if s.strip()]
        if len(sentences) > 3:
            bullet_points = []
            for sentence in sentences:
                if sentence and len(sentence.split()) > 5:
                    if not sentence.endswith(('.', '!', '?')):
                        sentence += '.'
                    bullet_points.append(f"• {sentence}")
            
            if bullet_points:
                return "\n\n".join(bullet_points)
    
    return original_answer

def get_qa_chain():
    """Creates and returns the RetrievalQA chain."""
    db = get_vector_store()
    if db is None:
        return None

    CUSTOM_PROMPT_TEMPLATE = """You are an expert AI assistant analyzing document content. Use the following pieces of context to answer the question at the end.

Follow these guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information from the documents to answer this question accurately."
3. Provide comprehensive but concise answers
4. Include relevant details, numbers, and facts when available
5. Structure your answer logically
6. If referring to multiple points, use bullet points for clarity

Context Information:
{context}

Question: {question}

Please provide a helpful, accurate answer based on the context:"""

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=512,
            groq_api_key=os.getenv("GROQ_API_KEY"),  # Get from .env
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def process_query(prompt):
    """Process a user query and return the response with source documents."""
    try:
        qa_chain = get_qa_chain()
        if qa_chain:
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]
            
            enhanced_result = enhance_answer(result, prompt, source_documents)
            
            # Format source documents for display
            formatted_sources = []
            for doc in source_documents:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 'N/A')
                if page_num != 'N/A':
                    page_num += 1
                
                excerpt = doc.page_content
                if len(excerpt) > 200:
                    excerpt = excerpt[:200] + "..."
                
                formatted_sources.append({
                    'document': source_name,
                    'page': page_num,
                    'excerpt': excerpt
                })
            
            return {
                'success': True,
                'answer': enhanced_result,
                'sources': formatted_sources
            }
        else:
            return {
                'success': False,
                'error': "Unable to load QA system. Please reprocess your documents."
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"An error occurred while processing your question: {str(e)}"
        }