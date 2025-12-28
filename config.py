import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    @staticmethod
    def get_api_key(key_name: str) -> str:
        """Get API key from environment or Streamlit secrets"""
        # Try Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and st.secrets:
                if key_name in st.secrets:
                    return st.secrets[key_name]
        except:
            pass
        
        # Try environment variable
        return os.getenv(key_name, "")
    
    @staticmethod
    def validate_required_keys() -> tuple[bool, list]:
        """Validate all required configuration keys"""
        required_keys = [
            'GROQ_API_KEY',
            'QDRANT_API_KEY',
            'QDRANT_URL',
            'MONGODB_URI'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not Config.get_api_key(key):
                missing_keys.append(key)
        
        return len(missing_keys) == 0, missing_keys
    
    @staticmethod
    def get_groq_api_key() -> str:
        """Get GROQ API key with validation"""
        key = Config.get_api_key('GROQ_API_KEY')
        if not key:
            raise ValueError("GROQ_API_KEY not found. Please configure it in Streamlit secrets or .env file")
        return key
    
    @staticmethod
    def get_qdrant_config() -> dict:
        """Get Qdrant configuration"""
        return {
            'api_key': Config.get_api_key('QDRANT_API_KEY'),
            'url': Config.get_api_key('QDRANT_URL')
        }
    
    @staticmethod
    def get_mongodb_uri() -> str:
        """Get MongoDB URI"""
        uri = Config.get_api_key('MONGODB_URI')
        if not uri:
            raise ValueError("MONGODB_URI not found")
        return uri

# Singleton instance
config = Config()