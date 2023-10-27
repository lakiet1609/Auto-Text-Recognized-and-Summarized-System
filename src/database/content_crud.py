from src.database.text_database import TextDatabase
from src.OCR.components.predict_system import TextSystem
from src.utility.configuration import Configuration
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import os, shutil, cv2, uuid
from pathlib import Path
import numpy as np

class ContentCRUD:
    def __init__(self):
        self.db_instance = TextDatabase()
        self.database_config = Configuration().init_databse()
        self.text_rec = TextSystem()

    def insert_content(self, text_id, image):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        text_doc = collection.find_one({'id': text_id}, {'_id': 0})
        text, _ = self.text_rec(img=image)
        content_id = str(uuid.uuid4().hex)
        content_doc = {'id': content_id, 'content': list(text)}
        if 'inputs' not in text_doc.keys() or text_doc['inputs'] is None:
            collection.update_one({'id': text_id}, {'$set': {'inputs': [content_doc]}})
        else:
            collection.update_one({'id': text_id}, {'$push': {'inputs': [content_doc]}})
        status_result = status.HTTP_201_CREATED
        return JSONResponse(status_code= status_result)
    
    def select_all_content_of_text(self, text_id, skip:int, limit:int):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        text_doc = collection.find_one({'id': text_id}, {'inputs.content': 0})
        if 'inputs' not in text_doc.keys() or text_doc['inputs'] is None:
            return []
        inputs = text_doc['inputs']
        if skip < 0:
            skip = 0
        elif skip > len(inputs):
            skip = len(inputs) - 1
        
        if limit < 0:
            limit = 0
        elif limit > len(inputs):
            limit = len(inputs) - skip
        
        inputs = inputs[skip: skip+limit]
        return inputs
    
    def delete_content_by_id(self, text_id, content_id):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        collection.update_one({'id':text_id, 'faces.id': content_id}, {'$pull': {'faces': {'id': content_id}}})
    
    def delete_all_contents(self, text_id: str):
        if not self.db_instance.check_person_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_people_collection()
        collection.update_one({'id': text_id}, {'$pull': {'faces': {}}})



