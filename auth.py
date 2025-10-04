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
        'last_login': self.db.get_current_time(),
        'is_active': True
    }
    
    # Save to database - FIXED: use db.users collection directly
    try:
        self.db.db.users.insert_one(user_record)  # Changed this line
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"