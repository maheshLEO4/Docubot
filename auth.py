import streamlit as st
import hashlib
import bcrypt
from database import MongoDBManager

class AuthManager:
    def __init__(self):
        self.db = MongoDBManager()

    def login_user(self, email, password):
        user = self.db.get_user_by_email(email)
        if user and bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
            return True, "Success", user
        return False, "Invalid Credentials", None

    def register_user(self, email, password, name):
        if self.db.get_user_by_email(email):
            return False, "User already exists"
        
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user_id = hashlib.md5(email.encode()).hexdigest()
        self.db.create_user({
            'user_id': user_id, 'email': email, 'name': name, 'password_hash': hashed
        })
        return True, "Registered successfully"

def setup_authentication():
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if not st.session_state.user:
        auth = AuthManager()
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                success, msg, data = auth.login_user(e, p)
                if success:
                    st.session_state.user = data
                    st.rerun()
                else: st.error(msg)
        return None
    return st.session_state.user['user_id']