import streamlit as st
import hashlib
import os
from qdrant_client import QdrantClient

class AuthManager:
    def __init__(self):
        self.config = self._get_qdrant_config()
        if self.config['api_key'] and self.config['url']:
            try:
                self.qdrant_client = QdrantClient(
                    url=self.config['url'],
                    api_key=self.config['api_key']
                )
            except Exception as e:
                print(f"Qdrant client initialization failed: {e}")
                self.qdrant_client = None
        else:
            self.qdrant_client = None
    
    def _get_qdrant_config(self):
        """Get Qdrant Cloud configuration"""
        from config import get_api_key
        return {
            'api_key': get_api_key('QDRANT_API_KEY'),
            'url': get_api_key('QDRANT_URL')
        }
    
    def get_user_id(self):
        """Get current user ID from session state"""
        if 'user' not in st.session_state:
            return None
        return st.session_state.user['id']
    
    def get_user_vectorstore_path(self):
        """Get user-specific vectorstore path"""
        user_id = self.get_user_id()
        if not user_id:
            return None
        return f"vectorstore/user_{user_id}"
    
    def get_user_data_path(self):
        """Get user-specific data path"""
        user_id = self.get_user_id()
        if not user_id:
            return None
        return f"data/user_{user_id}"
    
    def initialize_user_session(self, user_info):
        """Initialize session state for authenticated user"""
        st.session_state.user = user_info
        st.session_state.is_authenticated = True
        
        # Initialize user-specific directories
        user_data_path = self.get_user_data_path()
        user_vectorstore_path = self.get_user_vectorstore_path()
        
        os.makedirs(user_data_path, exist_ok=True)
        os.makedirs(user_vectorstore_path, exist_ok=True)

def setup_authentication():
    """Setup authentication UI"""
    st.sidebar.title("🔐 Authentication")
    
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    
    if not st.session_state.is_authenticated:
        auth_tab1, auth_tab2 = st.sidebar.tabs(["Login", "Register"])
        
        with auth_tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if email and password:
                    # Simple demo authentication - replace with your actual auth system
                    user_id = hashlib.md5(email.encode()).hexdigest()
                    if 'auth_manager' not in st.session_state:
                        st.session_state.auth_manager = AuthManager()
                    st.session_state.auth_manager.initialize_user_session({
                        'id': user_id,
                        'email': email
                    })
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Please enter email and password")
        
        with auth_tab2:
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register", use_container_width=True):
                if new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        user_id = hashlib.md5(new_email.encode()).hexdigest()
                        if 'auth_manager' not in st.session_state:
                            st.session_state.auth_manager = AuthManager()
                        st.session_state.auth_manager.initialize_user_session({
                            'id': user_id,
                            'email': new_email
                        })
                        st.success("Registration successful!")
                        st.rerun()
                    else:
                        st.error("Passwords don't match")
                else:
                    st.error("Please fill all fields")
        
        st.stop()
    else:
        # User is authenticated - show user info and logout button
        user_email = st.session_state.user.get('email', 'User')
        st.sidebar.success(f"👋 Welcome, {user_email}!")
        
        if st.sidebar.button("Logout", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()