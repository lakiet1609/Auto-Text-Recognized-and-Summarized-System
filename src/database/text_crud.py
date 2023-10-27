from fastapi import status, HTTPException
from database.text_database import TextDatabase
from utility.configuration import Configuration
from copy import deepcopy

class TextCRUD:
    def __init__(self):
        self.database_config = Configuration().init_databse()
        self.db_instance = TextDatabase()
    
    def insert_text(self, id, name):
        collection = self.db_instance.get_text_collection()
        if self.db_instance.check_text_by_id(id):
            raise HTTPException(status.HTTP_409_CONFLICT)
        text = {'id': id, 'name': name}
        collection.insert_one(deepcopy(text))
        return text
    
    def select_all_text(self, skip: int, limit: int):
        list_text = []
        collection = self.db_instance.get_text_collection()
        docs = collection.find({}, {'_id':0, 'id': 1}).skip(skip).limit(limit)
        for doc in docs:
            list_text.append(doc)
        return list_text
    
    def select_text_by_id(self, id):
        collection = self.db_instance.get_text_collection()
        if not self.db_instance.check_text_by_id(id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        doc = collection.find_one({'id': id}, {'_id': 0, 'id': 1})
        return doc
    
    def update_text_name(self, text_id, name):
        collection = self.db_instance.get_text_collection()
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection.update_one({'id': text_id}, {'$set': {'name': name}})
    
    def delete_text_by_id(self, id):
        collection = self.db_instance.get_text_collection()
        if not self.db_instance.check_text_by_id(id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection.delete_one({'id': id})
    
    def delete_all_text(self):
        collection = self.db_instance.get_text_collection()
        collection.delete_many({})


