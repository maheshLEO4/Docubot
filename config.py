import os
import streamlit as st
from dotenv import load_dotenv

def get_api_key(api_key_name):
    """Get API key from environment or Streamlit secrets"""
    load_dotenv()
    
    try:
        if hasattr(st, 'secrets') and st.secrets:
            if api_key_name in st.secrets:
                return st.secrets.get(api_key_name)
    except Exception:
        pass
    
    if api_key_name in os.environ:
        return os.environ[api_key_name]
    
    return None

def validate_api_key():
    """Validate GROQ API key"""
    api_key = get_api_key('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return api_key

def get_qdrant_config():
    """Get Qdrant Cloud configuration"""
    return {
        'api_key': get_api_key('QDRANT_API_KEY'),
        'url': get_api_key('QDRANT_URL')
    }

def get_mongodb_uri():
    """Get MongoDB connection URI"""
    return get_api_key('MONGODB_URI')

def get_google_oauth_config():
    """Get Google OAuth configuration with YOUR EXACT URL"""
    # Your exact Streamlit Cloud URL (without username)
    redirect_uri = "https://docubotapp.streamlit.app/"
    
    return {
        'client_id': get_api_key('GOOGLE_OAUTH_CLIENT_ID'),
        'client_secret': get_api_key('GOOGLE_OAUTH_CLIENT_SECRET'),
        'redirect_uri': redirect_uri
    }