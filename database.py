import os
from pymongo import MongoClient
from datetime import datetime
import uuid
from config import get_mongodb_uri

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.users = None
        self.file_uploads = None
        self.web_scrapes = None
        self.query_logs = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB and initialize collections"""
        try:
            mongodb_uri = get_mongodb_uri()
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.docubot
            
            # Initialize collections
            self.users = self.db.users
            self.file_uploads = self.db.file_uploads
            self.web_scrapes = self.db.web_scrapes
            self.query_logs = self.db.query_logs
            
            print("✅ Connected to MongoDB")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
    
    def get_current_time(self):
        """Get current UTC time"""
        return datetime.utcnow()
    
    def init_user(self, user_data):
        """Initialize user record (for backward compatibility)"""
        try:
            user_record = {
                'user_id': user_data['user_id'],
                'email': user_data['email'],
                'name': user_data.get('name', ''),
                'created_at': self.get_current_time(),
                'last_login': self.get_current_time(),
                'is_active': True
            }
            
            result = self.users.update_one(
                {'user_id': user_data['user_id']},
                {'$set': user_record},
                upsert=True
            )
            return user_record
        except Exception as e:
            print(f"Error initializing user: {e}")
            return user_data
    
    def get_user_by_email(self, email):
        """Get user by email"""
        try:
            return self.users.find_one({'email': email})
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        try:
            self.users.update_one(
                {'user_id': user_id},
                {'$set': {'last_login': self.get_current_time()}}
            )
        except Exception as e:
            print(f"Error updating last login: {e}")
    
    def log_file_upload(self, user_id, filename, file_size, pages_processed):
        """Log PDF file upload"""
        try:
            upload_id = str(uuid.uuid4())
            upload_record = {
                'upload_id': upload_id,
                'user_id': user_id,
                'filename': filename,
                'file_size': file_size,
                'pages_processed': pages_processed,
                'uploaded_at': self.get_current_time(),
                'status': 'processed'
            }
            
            self.file_uploads.insert_one(upload_record)
            return upload_id
        except Exception as e:
            print(f"Error logging file upload: {e}")
            return str(uuid.uuid4())
    
    def delete_file_upload(self, upload_id):
        """Delete file upload record"""
        try:
            self.file_uploads.delete_one({'upload_id': upload_id})
            return True
        except Exception as e:
            print(f"Error deleting file upload: {e}")
            return False
    
    def log_web_scrape(self, user_id, urls, successful_urls, total_chunks):
        """Log web scraping activity"""
        try:
            scrape_id = str(uuid.uuid4())
            scrape_record = {
                'scrape_id': scrape_id,
                'user_id': user_id,
                'urls': urls,
                'successful_urls': successful_urls,
                'total_chunks': total_chunks,
                'scraped_at': self.get_current_time(),
                'status': 'completed'
            }
            
            self.web_scrapes.insert_one(scrape_record)
            return scrape_id
        except Exception as e:
            print(f"Error logging web scrape: {e}")
            return str(uuid.uuid4())
    
    def delete_web_scrape(self, scrape_id):
        """Delete web scrape record"""
        try:
            self.web_scrapes.delete_one({'scrape_id': scrape_id})
            return True
        except Exception as e:
            print(f"Error deleting web scrape: {e}")
            return False
    
    def clear_user_data(self, user_id):
        """Clear all user data from MongoDB"""
        try:
            self.file_uploads.delete_many({'user_id': user_id})
            self.web_scrapes.delete_many({'user_id': user_id})
            return True
        except Exception as e:
            print(f"Error clearing user data: {e}")
            return False
    
    def log_query(self, user_id, query, response, sources_used, processing_time):
        """Log user queries for analytics"""
        try:
            query_id = str(uuid.uuid4())
            query_record = {
                'query_id': query_id,
                'user_id': user_id,
                'query': query,
                'response_preview': response[:200] if response else '',
                'sources_count': len(sources_used),
                'processing_time': processing_time,
                'queried_at': self.get_current_time()
            }
            
            self.query_logs.insert_one(query_record)
            return query_id
        except Exception as e:
            print(f"Error logging query: {e}")
            return str(uuid.uuid4())
    
    def get_user_files(self, user_id):
        """Get all files uploaded by user"""
        try:
            return list(self.file_uploads.find(
                {'user_id': user_id},
                sort=[('uploaded_at', -1)]
            ))
        except Exception as e:
            print(f"Error getting user files: {e}")
            return []
    
    def get_user_scrapes(self, user_id):
        """Get all web scrapes by user"""
        try:
            return list(self.web_scrapes.find(
                {'user_id': user_id},
                sort=[('scraped_at', -1)]
            ))
        except Exception as e:
            print(f"Error getting user scrapes: {e}")
            return []
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        try:
            files_count = self.file_uploads.count_documents({'user_id': user_id})
            scrapes_count = self.web_scrapes.count_documents({'user_id': user_id})
            queries_count = self.query_logs.count_documents({'user_id': user_id})
            
            return {
                'files_uploaded': files_count,
                'websites_scraped': scrapes_count,
                'queries_made': queries_count
            }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {
                'files_uploaded': 0,
                'websites_scraped': 0,
                'queries_made': 0
            }
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()