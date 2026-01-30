from langchain_groq import ChatGroq
from typing import Dict, List
from langchain_core.documents import Document
import os

class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with Groq LLM.
        """
        print("Initializing ResearchAgent with Groq LLM...")
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        print("LLM initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are a helpful AI assistant that answers questions based on the user's documents.

        **Instructions:**
        - Answer the following question using only the information from the documents.
        - If the information isn't in the documents, say you don't know or can't answer.
        - Be clear, concise, and factual.
        - Don't mention "context", "documents", or "provided information" in your answer.
        - Just give the answer naturally as if you know it.
        - Never say things like "based on the provided context" or "according to the documents".
        
        **Question:** {question}
        
        **Information from documents:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            print("Sending prompt to the model...")
            response = self.llm.invoke(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        # Extract and process the LLM's response
        draft_answer = self.sanitize_response(response.content) if response.content else "I cannot answer this question."

        print(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
