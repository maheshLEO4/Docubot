import os
import streamlit as st
from dotenv import load_dotenv

def get_config(key):
    load_dotenv()
    # Priority: Streamlit Secrets -> Env Variables
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception: pass
    return os.environ.get(key)

def validate_config():
    required = ['GROQ_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY', 'MONGODB_URI']
    missing = [r for r in required if not get_config(r)]
    if missing:
        st.error(f"Missing Configuration: {', '.join(missing)}")
        st.stop()
    return True