from pymongo import MongoClient
from config import get_config

class MongoDBManager:
    def __init__(self):
        self.client = MongoClient(get_config('MONGODB_URI'))
        self.db = self.client.docubot_pro
        self.users = self.db.users

    def get_user_by_email(self, email):
        return self.users.find_one({"email": email})

    def create_user(self, user_data):
        return self.users.insert_one(user_data)