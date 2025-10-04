import streamlit as st
import hashlib
import os
import requests
import urllib.parse
from config import get_google_oauth_config
from database import MongoDBManager

class AuthManager:
    def __init__(self):
        self.oauth_config = get_google_oauth_config()
        self.db = MongoDBManager()
    
    def get_google_oauth_url(self):
        """Generate Google OAuth URL"""
        client_id = self.oauth_config['client_id']
        redirect_uri = self.oauth_config['redirect_uri']
        
        # Google OAuth endpoint with proper encoding
        params = {
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': 'openid email profile',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
        return auth_url
    
    def exchange_code_for_token(self, code):
        """Exchange authorization code for access token"""
        try:
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                'client_id': self.oauth_config['client_id'],
                'client_secret': self.oauth_config['client_secret'],
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': self.oauth_config['redirect_uri']
            }
            
            response = requests.post(token_url, data=data)
            if response.status_code == 200:
                token_data = response.json()
                return token_data.get('access_token')
            else:
                print(f"Token exchange failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error exchanging code: {e}")
            return None
    
    def get_google_user_info(self, access_token):
        """Get user info from Google using access token"""
        try:
            userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            headers = {'Authorization': f'Bearer {access_token}'}
            
            response = requests.get(userinfo_url, headers=headers)
            if response.status_code == 200:
                user_data = response.json()
                
                # Create user record
                user_id = hashlib.md5(user_data['email'].encode()).hexdigest()
                user_record = {
                    'user_id': user_id,
                    'email': user_data['email'],
                    'name': user_data.get('name', ''),
                    'picture': user_data.get('picture', ''),
                    'google_id': user_data['sub']
                }
                
                # Initialize user in database
                self.db.init_user(user_record)
                self.db.update_last_login(user_id)
                
                return user_record
            else:
                print(f"Failed to get user info: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None

def setup_authentication():
    """Setup Google OAuth authentication"""
    st.sidebar.title("🔐 Authentication")
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    # Display current redirect URI for debugging
    oauth_config = get_google_oauth_config()
    st.sidebar.caption(f"Redirect URI: {oauth_config['redirect_uri']}")
    
    # Check for OAuth callback
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.user:
        # We have an OAuth callback
        code = query_params['code']
        st.sidebar.info("Processing OAuth callback...")
        
        access_token = st.session_state.auth_manager.exchange_code_for_token(code)
        
        if access_token:
            user_data = st.session_state.auth_manager.get_google_user_info(access_token)
            if user_data:
                st.session_state.user = user_data
                # Clear the code from URL
                st.query_params.clear()
                st.rerun()
            else:
                st.sidebar.error("Failed to get user info from Google")
        else:
            st.sidebar.error("Failed to exchange authorization code")
    
    if not st.session_state.user:
        st.sidebar.info("Please sign in with Google to continue")
        
        # Show Google OAuth button
        oauth_url = st.session_state.auth_manager.get_google_oauth_url()
        
        st.sidebar.markdown(
            f"""
            <a href="{oauth_url}" target="_self" style="
                display: inline-block;
                background: #4285F4;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                text-align: center;
                width: 100%;
                border: none;
                cursor: pointer;
            ">
            🔐 Sign in with Google
            </a>
            """,
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown("---")
        st.sidebar.caption("Troubleshooting OAuth?")
        st.sidebar.markdown("""
        1. Go to [Google Cloud Console](https://console.cloud.google.com)
        2. Add this exact URL to **Authorized redirect URIs**:
        """)
        st.sidebar.code(oauth_config['redirect_uri'])
        
        st.sidebar.markdown("---")
        st.sidebar.caption("Demo Mode (for testing)")
        
        # Demo mode fallback
        email = st.sidebar.text_input("Email (Demo Mode)")
        if st.sidebar.button("Sign In (Demo)", use_container_width=True):
            if email:
                user_id = hashlib.md5(email.encode()).hexdigest()
                user_data = {
                    'user_id': user_id,
                    'email': email,
                    'name': 'Demo User',
                    'picture': '',
                    'google_id': 'demo'
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
        
        # Show user stats
        try:
            stats = st.session_state.auth_manager.db.get_user_stats(user_data['user_id'])
            st.sidebar.metric("Files Uploaded", stats['files_uploaded'])
            st.sidebar.metric("Queries Made", stats['queries_made'])
        except:
            pass
        
        if st.sidebar.button("Sign Out", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        return user_data['user_id']