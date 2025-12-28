import streamlit as st
import hashlib
import bcrypt
from datetime import datetime
from database import MongoDBManager
import time

class AuthManager:
    """Optimized authentication manager"""
    
    def __init__(self):
        self.db = MongoDBManager()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'auth_initialized' not in st.session_state:
            st.session_state.auth_initialized = False
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        try:
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        except Exception as e:
            raise Exception(f"Password hashing failed: {str(e)}")
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except:
            return False
    
    def register(self, email: str, password: str, name: str) -> tuple[bool, str]:
        """Register new user"""
        try:
            # Validation
            if not email or '@' not in email:
                return False, "Valid email required"
            if len(password) < 6:
                return False, "Password must be at least 6 characters"
            
            # Check existing user
            if self.db.get_user_by_email(email):
                return False, "Email already registered"
            
            # Create user
            user_id = hashlib.md5(f"{email}{time.time()}".encode()).hexdigest()
            hashed_pw = self.hash_password(password)
            
            user_data = {
                'user_id': user_id,
                'email': email.lower().strip(),
                'name': name.strip(),
                'password_hash': hashed_pw,
                'created_at': datetime.utcnow(),
                'last_login': datetime.utcnow(),
                'is_active': True
            }
            
            # Save to database
            success = self.db.create_user(user_data)
            if success:
                return True, "Registration successful"
            else:
                return False, "Registration failed"
                
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def login(self, email: str, password: str) -> tuple[bool, str, dict]:
        """Authenticate user"""
        try:
            # Get user from database
            user_data = self.db.get_user_by_email(email)
            if not user_data:
                return False, "Invalid credentials", None
            
            # Verify password
            if not self.verify_password(password, user_data.get('password_hash', '')):
                return False, "Invalid credentials", None
            
            # Update last login
            self.db.update_user_last_login(user_data['user_id'])
            
            # Set session state
            st.session_state.user = user_data
            st.session_state.user_id = user_data['user_id']
            st.session_state.auth_initialized = True
            
            return True, "Login successful", user_data
            
        except Exception as e:
            return False, f"Login error: {str(e)}", None
    
    def logout(self):
        """Logout user"""
        for key in ['user', 'user_id', 'auth_initialized', 'messages', 
                    'vector_store_exists', 'cached_user_files', 'cached_user_scrapes']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    def get_current_user(self) -> dict:
        """Get current authenticated user"""
        return st.session_state.get('user')
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('user') is not None


def render_auth_ui():
    """Render authentication UI"""
    st.sidebar.title("🔐 Authentication")
    
    if st.session_state.get('user'):
        user = st.session_state.user
        st.sidebar.success(f"Welcome, {user['name']}!")
        st.sidebar.caption(f"📧 {user['email']}")
        
        # User stats
        try:
            stats = MongoDBManager().get_user_stats(user['user_id'])
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("📄 Files", stats['files_uploaded'])
            with col2:
                st.metric("🌐 Websites", stats['websites_scraped'])
        except:
            pass
        
        if st.sidebar.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            AuthManager().logout()
        return user['user_id']
    
    # Not authenticated - show login/register
    auth_tab = st.sidebar.radio("Select", ["Login", "Register"], horizontal=True)
    
    if auth_tab == "Login":
        with st.sidebar.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.form_submit_button("Login", use_container_width=True, type="primary"):
                if email and password:
                    auth = AuthManager()
                    success, message, user_data = auth.login(email, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill all fields")
    
    else:  # Register
        with st.sidebar.form("register_form"):
            st.subheader("Register")
            email = st.text_input("Email", key="reg_email")
            name = st.text_input("Full Name", key="reg_name")
            password = st.text_input("Password", type="password", key="reg_password")
            confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            if st.form_submit_button("Register", use_container_width=True):
                if all([email, name, password, confirm]):
                    if password != confirm:
                        st.error("Passwords don't match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        auth = AuthManager()
                        success, message = auth.register(email, password, name)
                        if success:
                            st.success(message + " Please login.")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill all fields")
    
    st.stop()