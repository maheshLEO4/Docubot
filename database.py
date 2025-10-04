import os
from pymongo import MongoClient
from datetime import datetime
import uuid
from bson import ObjectId
from config import get_mongodb_uri

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            mongodb_uri = get_mongodb_uri()
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.docubot
            print("✅ Connected to MongoDB")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
    
    def get_current_time(self):
        """Get current UTC time"""
        return datetime.utcnow()
    
    def init_user(self, user_data):
        """Initialize user record (for backward compatibility)"""
        user_record = {
            'user_id': user_data['user_id'],
            'email': user_data['email'],
            'name': user_data.get('name', ''),
            'created_at': self.get_current_time(),
            'last_login': self.get_current_time(),
            'is_active': True
        }
        
        result = self.db.users.update_one(
            {'user_id': user_data['user_id']},
            {'$set': user_record},
            upsert=True
        )
        return user_record
    
    def get_user_by_email(self, email):
        """Get user by email"""
        return self.db.users.find_one({'email': email})
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        self.db.users.update_one(
            {'user_id': user_id},
            {'$set': {'last_login': self.get_current_time()}}
        )
    
    def log_file_upload(self, user_id, filename, file_size, pages_processed):
        """Log PDF file upload"""
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
        
        self.db.file_uploads.insert_one(upload_record)
        return upload_id
    
    def log_web_scrape(self, user_id, urls, successful_urls, total_chunks):
        """Log web scraping activity"""
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
        
        self.db.web_scrapes.insert_one(scrape_record)
        return scrape_id
    
    def log_query(self, user_id, query, response, sources_used, processing_time):
        """Log user queries for analytics"""
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
        
        self.db.query_logs.insert_one(query_record)
        return query_id
    
    def get_user_files(self, user_id):
        """Get all files uploaded by user"""
        return list(self.db.file_uploads.find(
            {'user_id': user_id},
            sort=[('uploaded_at', -1)]
        ))
    
    def get_user_scrapes(self, user_id):
        """Get all web scrapes by user"""
        return list(self.db.web_scrapes.find(
            {'user_id': user_id},
            sort=[('scraped_at', -1)]
        ))
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        files_count = self.db.file_uploads.count_documents({'user_id': user_id})
        scrapes_count = self.db.web_scrapes.count_documents({'user_id': user_id})
        queries_count = self.db.query_logs.count_documents({'user_id': user_id})
        
        return {
            'files_uploaded': files_count,
            'websites_scraped': scrapes_count,
            'queries_made': queries_count
        }
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()