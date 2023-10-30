from database.text_database import TextDatabase
from OCR.components.predict_system import TextSystem
from utility.configuration import Configuration
from utility.schema import ImageValidation
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import os, shutil, cv2, uuid
from pathlib import Path
import numpy as np

class ContentCRUD:
    def __init__(self):
        self.db_instance = TextDatabase()
        self.database_config = Configuration().init_database()
        self.text_rec = TextSystem()


    def insert_content(self, text_id, image):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        
        collection = self.db_instance.get_text_collection()
        text_doc = collection.find_one({'id': text_id}, {'_id': 0})
        validate_result = ImageValidation.IMAGE_IS_VALID
        
        if validate_result == ImageValidation.IMAGE_IS_VALID:
            text, _ = self.text_rec(img=image)
            content_id = str(uuid.uuid4().hex)
            content_doc = {'id': content_id, 'content': [text]}
            
            collection.update_one({'id': text_id}, {'$push': {'inputs': content_doc}})
            
            status_result = status.HTTP_201_CREATED
        else:
            status_result = status.HTTP_406_NOT_ACCEPTABLE
        
        return JSONResponse(status_code= status_result, content={'INFO': validate_result})


    def select_all_content_id(self, text_id, skip:int, limit:int):
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
    

    def select_content_by_id(self, text_id, content_id):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        text_doc = collection.find_one({'id': text_id}, {'_id': 0, 'inputs': {'id': content_id, 'content': 1}})
        return  text_doc['inputs'][0]['content'][0]
    

    def select_all_content_by_id(self, text_id):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        text_docs = collection.find_one({'id': text_id},  {'_id': 0, 'inputs': {'id': 1, 'content': 1}})
        documents = text_docs['inputs']
        connected_texts = ''
        for content in documents:
            new_content = content['content'][0][0]
            connected_texts += new_content
        
        connected_texts = [connected_texts]
        return connected_texts


    def delete_content_by_id(self, text_id, content_id):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        collection.update_one({'id':text_id}, {'$pull': {'inputs': {'id': content_id}}})

    
    def delete_all_contents(self, text_id: str):
        if not self.db_instance.check_text_by_id(text_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND)
        collection = self.db_instance.get_text_collection()
        collection.update_one({'id': text_id}, {'$unset': {'inputs': 1}})





