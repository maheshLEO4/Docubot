import streamlit as st
import hashlib
import bcrypt
from database import MongoDBManager

class AuthManager:
    def __init__(self):
        self.db = MongoDBManager()
    
    def hash_password(self, password):
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password, hashed_password):
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def register_user(self, email, password, name):
        """Register a new user with email and password"""
        if not email or not password:
            return False, "Email and password are required"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        # Check if user already exists
        existing_user = self.db.get_user_by_email(email)
        if existing_user:
            return False, "User already exists with this email"
        
        # Create user
        user_id = hashlib.md5(email.encode()).hexdigest()
        hashed_password = self.hash_password(password)
        
        user_record = {
            'user_id': user_id,
            'email': email,
            'name': name,
            'password_hash': hashed_password,
            'auth_method': 'email_password',
            'created_at': self.db.get_current_time(),
            'last_login': self.db.get_current_time()
        }
        
        # Save to database
        self.db.users.insert_one(user_record)
        return True, "Registration successful"
    
    def login_user(self, email, password):
        """Login user with email and password"""
        if not email or not password:
            return False, "Email and password are required", None
        
        # Find user in database
        user_data = self.db.get_user_by_email(email)
        if not user_data:
            return False, "Invalid email or password", None
        
        # Verify password
        if not self.verify_password(password, user_data['password_hash']):
            return False, "Invalid email or password", None
        
        # Update last login
        self.db.update_last_login(user_data['user_id'])
        
        return True, "Login successful", user_data

def setup_authentication():
    """Setup email/password authentication"""
    st.sidebar.title("🔐 DocuBot AI")
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    if not st.session_state.user:
        # Tabs for Login/Register
        tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_btn", use_container_width=True):
                if login_email and login_password:
                    success, message, user_data = st.session_state.auth_manager.login_user(login_email, login_password)
                    if success:
                        st.session_state.user = user_data
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both email and password")
        
        with tab2:
            st.subheader("Create New Account")
            reg_email = st.text_input("Email", key="reg_email")
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            if st.button("Register", key="reg_btn", use_container_width=True):
                if reg_email and reg_name and reg_password and reg_confirm:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match")
                    else:
                        success, message = st.session_state.auth_manager.register_user(reg_email, reg_password, reg_name)
                        if success:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill all fields")
        
        st.sidebar.markdown("---")
        st.sidebar.caption("Demo Accounts (Auto-login)")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Demo User 1", use_container_width=True):
                # Auto-create and login demo user
                email = "demo1@docubot.com"
                password = "demo123"
                name = "Demo User 1"
                
                # Register if not exists
                st.session_state.auth_manager.register_user(email, password, name)
                # Login
                success, message, user_data = st.session_state.auth_manager.login_user(email, password)
                if success:
                    st.session_state.user = user_data
                    st.rerun()
        
        with col2:
            if st.button("Demo User 2", use_container_width=True):
                # Auto-create and login demo user
                email = "demo2@docubot.com"
                password = "demo123"
                name = "Demo User 2"
                
                # Register if not exists
                st.session_state.auth_manager.register_user(email, password, name)
                # Login
                success, message, user_data = st.session_state.auth_manager.login_user(email, password)
                if success:
                    st.session_state.user = user_data
                    st.rerun()
        
        st.stop()
    
    else:
        # User is authenticated
        user_data = st.session_state.user
        st.sidebar.success(f"👋 Welcome, {user_data['name']}!")
        st.sidebar.caption(f"📧 {user_data['email']}")
        
        # Show user stats
        try:
            stats = st.session_state.auth_manager.db.get_user_stats(user_data['user_id'])
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("📄 Files", stats['files_uploaded'])
            with col2:
                st.metric("💬 Queries", stats['queries_made'])
        except Exception as e:
            st.sidebar.info("📊 Stats will appear after you use the app")
        
        # Sign out
        if st.sidebar.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        return user_data['user_id']