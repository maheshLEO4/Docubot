import streamlit as st
import hashlib
import bcrypt
from database import MongoDBManager

# --- Page Configuration ---
st.set_page_config(
    page_title="DocuBot AI - Login",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

class AuthManager:
    def __init__(self):
        self.db = MongoDBManager()
    
    def hash_password(self, password):
        """Hash a password using bcrypt"""
        try:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Password hashing failed: {str(e)}")
    
    def verify_password(self, password, hashed_password):
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            return False
    
    def register_user(self, email, password, name):
        """Register a new user with email and password"""
        try:
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
                'last_login': self.db.get_current_time(),
                'is_active': True
            }
            
            # Save to database
            self.db.users.insert_one(user_record)
            return True, "Registration successful"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, email, password):
        """Login user with email and password"""
        try:
            if not email or not password:
                return False, "Email and password are required", None
            
            # Find user in database
            user_data = self.db.get_user_by_email(email)
            if not user_data:
                return False, "Invalid email or password", None
            
            # Verify password
            if not self.verify_password(password, user_data.get('password_hash', '')):
                return False, "Invalid email or password", None
            
            # Update last login
            self.db.update_last_login(user_data['user_id'])
            
            return True, "Login successful", user_data
            
        except Exception as e:
            return False, f"Login failed: {str(e)}", None
    
    def create_demo_user(self, email, password, name):
        """Create demo user if doesn't exist and login"""
        try:
            # Check if user exists
            existing_user = self.db.get_user_by_email(email)
            if not existing_user:
                # Register demo user
                success, message = self.register_user(email, password, name)
                if not success:
                    return False, message, None
            
            # Login demo user
            return self.login_user(email, password)
            
        except Exception as e:
            return False, f"Demo user creation failed: {str(e)}", None

# Initialize auth manager
if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = AuthManager()

# Check if already logged in
if 'user' in st.session_state and st.session_state.user:
    st.success(f"✅ Already logged in as {st.session_state.user['name']}")
    if st.button("🚀 Go to DocuBot Chat"):
        st.switch_page("main_app.py")
    st.stop()

# --- Main Authentication UI ---
st.title("🔐 DocuBot AI")
st.markdown("### Login to access your AI document assistant")

# Create tabs for login and register
tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

with tab1:
    st.subheader("Login to Your Account")
    
    login_email = st.text_input("📧 Email", placeholder="your@email.com", key="login_email")
    login_password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_password")
    
    if st.button("🚀 Login", use_container_width=True, type="primary"):
        if login_email and login_password:
            with st.spinner("Logging in..."):
                success, message, user_data = st.session_state.auth_manager.login_user(login_email, login_password)
                if success:
                    st.session_state.user = user_data
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        else:
            st.error("⚠️ Please enter both email and password")

with tab2:
    st.subheader("Create New Account")
    
    reg_email = st.text_input("📧 Email", placeholder="your@email.com", key="reg_email")
    reg_name = st.text_input("👤 Full Name", placeholder="Your Full Name", key="reg_name")
    reg_password = st.text_input("🔒 Password", type="password", placeholder="At least 6 characters", key="reg_password")
    reg_confirm = st.text_input("✅ Confirm Password", type="password", placeholder="Re-enter your password", key="reg_confirm")
    
    if st.button("📝 Create Account", use_container_width=True):
        if reg_email and reg_name and reg_password and reg_confirm:
            if reg_password != reg_confirm:
                st.error("❌ Passwords do not match")
            elif len(reg_password) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                with st.spinner("Creating account..."):
                    success, message = st.session_state.auth_manager.register_user(reg_email, reg_password, reg_name)
                    if success:
                        st.success("✅ Registration successful! Please login.")
                    else:
                        st.error(f"❌ {message}")
        else:
            st.error("⚠️ Please fill all fields")

# Demo accounts section
st.markdown("---")
st.subheader("🎯 Quick Start")
st.markdown("Try DocuBot instantly with demo accounts:")

col1, col2 = st.columns(2)

with col1:
    if st.button("👤 Demo User 1", use_container_width=True):
        with st.spinner("Setting up demo account..."):
            success, message, user_data = st.session_state.auth_manager.create_demo_user(
                "demo1@docubot.com", "demo123", "Demo User 1"
            )
            if success:
                st.session_state.user = user_data
                st.success("✅ Demo login successful! Redirecting...")
                st.rerun()
            else:
                st.error(f"❌ {message}")

with col2:
    if st.button("👤 Demo User 2", use_container_width=True):
        with st.spinner("Setting up demo account..."):
            success, message, user_data = st.session_state.auth_manager.create_demo_user(
                "demo2@docubot.com", "demo123", "Demo User 2"
            )
            if success:
                st.session_state.user = user_data
                st.success("✅ Demo login successful! Redirecting...")
                st.rerun()
            else:
                st.error(f"❌ {message}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>DocuBot AI - Chat with your documents and websites</p>
    </div>
    """,
    unsafe_allow_html=True
)