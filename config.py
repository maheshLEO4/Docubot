import os
import streamlit as st
from dotenv import load_dotenv

def get_api_key():
    """
    Get API key from environment variables or Streamlit secrets
    Works for both local development and Streamlit Cloud
    """
    # Load .env file for local development
    load_dotenv()
    
    # Try Streamlit secrets first (for deployment)
    try:
        if hasattr(st, 'secrets') and st.secrets:
            if 'GROQ_API_KEY' in st.secrets:
                return st.secrets.get('GROQ_API_KEY')
    except Exception:
        pass
    
    # Try environment variable (for local development)
    if 'GROQ_API_KEY' in os.environ:
        return os.environ['GROQ_API_KEY']
    
    # Return None if not found
    return None

def validate_api_key():
    """
    Validate that API key exists and return it
    """
    api_key = get_api_key()
    if not api_key:
        raise ValueError("""
        ❌ GROQ_API_KEY not found. 
        
        Please add your API key to:
        - **Streamlit Cloud**: Go to App Settings → Secrets
        - **Local development**: Create a `.env` file
        
        Example .env file:
        GROQ_API_KEY=your_actual_api_key_here
        """)
    return api_key