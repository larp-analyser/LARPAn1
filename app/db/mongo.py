from pymongo import MongoClient
import certifi
from app.core.config import settings

class MongoDB:
    client: MongoClient = None
    db = None

    @classmethod
    def connect(cls):
        if cls.client is None:
            cls.client = MongoClient(
                settings.MONGO_URI,
                tlsCAFile=certifi.where(),
                maxPoolSize=10,
                minPoolSize=2,
                maxIdleTimeMS=120000,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                retryWrites=True,
                w="majority",
            )
            cls.db = cls.client["psi09"]
    
    @classmethod
    def get_db(cls):
        if cls.client is None:
            cls.connect()
        return cls.db

    @classmethod
    def get_collection(cls, name: str):
        return cls.get_db()[name]
    
    @classmethod
    def disconnect(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
