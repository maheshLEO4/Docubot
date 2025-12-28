from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any
from config import config

class MongoDBManager:
    """Optimized MongoDB manager with connection pooling"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.client = None
            self.db = None
            self._connect()
            self._create_indexes()
            self._initialized = True
    
    def _connect(self):
        """Connect to MongoDB with error handling"""
        try:
            mongodb_uri = config.get_mongodb_uri()
            self.client = MongoClient(
                mongodb_uri,
                maxPoolSize=10,
                minPoolSize=1,
                connectTimeoutMS=5000,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client.docubot
            
            # Test connection
            self.client.admin.command('ping')
            print("✅ MongoDB connected")
            
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
    
    def _create_indexes(self):
        """Create necessary indexes"""
        try:
            # Users collection
            self.db.users.create_index([('email', ASCENDING)], unique=True)
            self.db.users.create_index([('user_id', ASCENDING)], unique=True)
            
            # File uploads
            self.db.file_uploads.create_index([('user_id', ASCENDING), ('uploaded_at', DESCENDING)])
            
            # Web scrapes
            self.db.web_scrapes.create_index([('user_id', ASCENDING), ('scraped_at', DESCENDING)])
            
            # Query logs
            self.db.query_logs.create_index([('user_id', ASCENDING), ('queried_at', DESCENDING)])
            
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
    
    def create_user(self, user_data: Dict) -> bool:
        """Create new user"""
        try:
            self.db.users.insert_one(user_data)
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            return self.db.users.find_one({'email': email.lower().strip()})
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        try:
            self.db.users.update_one(
                {'user_id': user_id},
                {'$set': {'last_login': datetime.utcnow()}}
            )
        except Exception as e:
            print(f"Error updating last login: {e}")
    
    def log_file_upload(self, user_id: str, filename: str, pages: int) -> str:
        """Log file upload"""
        try:
            upload_id = str(uuid.uuid4())
            record = {
                'upload_id': upload_id,
                'user_id': user_id,
                'filename': filename,
                'pages_processed': pages,
                'uploaded_at': datetime.utcnow(),
                'status': 'processed'
            }
            self.db.file_uploads.insert_one(record)
            return upload_id
        except Exception as e:
            print(f"Error logging file upload: {e}")
            return str(uuid.uuid4())
    
    def log_web_scrape(self, user_id: str, urls: List[str], successful: List[str]) -> str:
        """Log web scrape"""
        try:
            scrape_id = str(uuid.uuid4())
            record = {
                'scrape_id': scrape_id,
                'user_id': user_id,
                'urls': urls,
                'successful_urls': successful,
                'scraped_at': datetime.utcnow(),
                'status': 'completed'
            }
            self.db.web_scrapes.insert_one(record)
            return scrape_id
        except Exception as e:
            print(f"Error logging web scrape: {e}")
            return str(uuid.uuid4())
    
    def get_user_files(self, user_id: str) -> List[Dict]:
        """Get user's uploaded files"""
        try:
            return list(self.db.file_uploads.find(
                {'user_id': user_id},
                sort=[('uploaded_at', DESCENDING)]
            ))
        except Exception as e:
            print(f"Error getting user files: {e}")
            return []
    
    def get_user_scrapes(self, user_id: str) -> List[Dict]:
        """Get user's web scrapes"""
        try:
            return list(self.db.web_scrapes.find(
                {'user_id': user_id},
                sort=[('scraped_at', DESCENDING)]
            ))
        except Exception as e:
            print(f"Error getting user scrapes: {e}")
            return []
    
    def delete_file(self, upload_id: str) -> bool:
        """Delete file record"""
        try:
            result = self.db.file_uploads.delete_one({'upload_id': upload_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def delete_scrape(self, scrape_id: str) -> bool:
        """Delete scrape record"""
        try:
            result = self.db.web_scrapes.delete_one({'scrape_id': scrape_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting scrape: {e}")
            return False
    
    def clear_user_data(self, user_id: str) -> bool:
        """Clear all user data"""
        try:
            self.db.file_uploads.delete_many({'user_id': user_id})
            self.db.web_scrapes.delete_many({'user_id': user_id})
            self.db.query_logs.delete_many({'user_id': user_id})
            return True
        except Exception as e:
            print(f"Error clearing user data: {e}")
            return False
    
    def get_user_stats(self, user_id: str) -> Dict[str, int]:
        """Get user statistics"""
        try:
            files = self.db.file_uploads.count_documents({'user_id': user_id})
            scrapes = self.db.web_scrapes.count_documents({'user_id': user_id})
            queries = self.db.query_logs.count_documents({'user_id': user_id})
            
            return {
                'files_uploaded': files,
                'websites_scraped': scrapes,
                'queries_made': queries
            }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {'files_uploaded': 0, 'websites_scraped': 0, 'queries_made': 0}
    
    def log_query(self, user_id: str, query: str, answer: str, sources: List[Dict]) -> str:
        """Log user query"""
        try:
            query_id = str(uuid.uuid4())
            record = {
                'query_id': query_id,
                'user_id': user_id,
                'query': query,
                'answer_preview': answer[:200],
                'sources_count': len(sources),
                'queried_at': datetime.utcnow()
            }
            self.db.query_logs.insert_one(record)
            return query_id
        except Exception as e:
            print(f"Error logging query: {e}")
            return str(uuid.uuid4())
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()