import streamlit as st
import hashlib
import os
from google.oauth2 import id_token
from google.auth.transport import requests
from database import MongoDBManager
from config import get_google_oauth_config

class AuthManager:
    def __init__(self):
        self.db = MongoDBManager()
        self.oauth_config = get_google_oauth_config()
    
    def verify_google_token(self, token):
        """Verify Google OAuth token"""
        try:
            # Verify the token
            idinfo = id_token.verify_oauth2_token(
                token, 
                requests.Request(), 
                self.oauth_config['client_id']
            )
            
            # Check token validity
            if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')
            
            # Get user data
            user_id = hashlib.md5(idinfo['email'].encode()).hexdigest()
            user_data = {
                'user_id': user_id,
                'email': idinfo['email'],
                'name': idinfo.get('name', ''),
                'picture': idinfo.get('picture', ''),
                'google_id': idinfo['sub']
            }
            
            # Initialize/update user in database
            self.db.init_user(user_data)
            self.db.update_last_login(user_id)
            
            return user_data
            
        except ValueError as e:
            print(f"Google token verification failed: {e}")
            return None
    
    def get_user_id(self):
        """Get current user ID from session state"""
        if 'user' not in st.session_state:
            return None
        return st.session_state.user['user_id']
    
    def get_user_collection_name(self):
        """Get user-specific Qdrant collection name"""
        user_id = self.get_user_id()
        return f"docubot_user_{user_id}" if user_id else "docubot_default"
    
    def get_user_data(self):
        """Get current user data"""
        return st.session_state.get('user')

def setup_authentication():
    """Setup Google OAuth authentication"""
    st.sidebar.title("🔐 Authentication")
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if not st.session_state.user:
        st.sidebar.info("Please sign in with Google to continue")
        
        # Google OAuth button (you'll need to implement the OAuth flow)
        # For Streamlit Cloud, you can use streamlit-google-oauth library
        try:
            from streamlit_google_oauth import st_google_oauth
            oauth_url = st_google_oauth(
                client_id=get_google_oauth_config()['client_id'],
                client_secret=get_google_oauth_config()['client_secret'],
                redirect_uri=get_google_oauth_config()['redirect_uri'],
            )
            
            if oauth_url:
                st.sidebar.markdown(f'<a href="{oauth_url}" target="_self">Sign in with Google</a>', unsafe_allow_html=True)
                
        except ImportError:
            # Fallback for local development
            st.sidebar.warning("Google OAuth not configured. Using demo mode.")
            
            email = st.sidebar.text_input("Email (Demo Mode)")
            if st.sidebar.button("Sign In (Demo)"):
                if email:
                    user_id = hashlib.md5(email.encode()).hexdigest()
                    if 'auth_manager' not in st.session_state:
                        st.session_state.auth_manager = AuthManager()
                    
                    user_data = {
                        'user_id': user_id,
                        'email': email,
                        'name': 'Demo User',
                        'picture': ''
                    }
                    st.session_state.auth_manager.db.init_user(user_data)
                    st.session_state.user = user_data
                    st.rerun()
        
        st.stop()
    else:
        # User is authenticated
        user_data = st.session_state.user
        st.sidebar.success(f"👋 Welcome, {user_data['name']}!")
        st.sidebar.caption(f"Signed in as: {user_data['email']}")
        
        if st.sidebar.button("Sign Out", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()