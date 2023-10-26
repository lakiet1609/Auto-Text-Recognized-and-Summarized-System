from src.database.base_database import BaseDatabase
from src.utility.common import Singleton
from src.utility.configuration import Configuration
import numpy as np

class TextDatabase(BaseDatabase, metaclass=Singleton):
    def __init__(self):
        self.config = Configuration().init_databse()
        super(TextDatabase, self).__init__(self.config)

        database_name = self.config['database_name']
        collection_name = self.config['collection_name']
        self.database = self.client[database_name]
        self.text_collection = self.database[collection_name]
    
    def get_text_collection(self):
        return self.text_collection
    
    def initialize_local_database(self):
        text_docs = self.text_collection.find()
        self.text = {}
        self.contents = {}
        for text in text_docs:
            text_id = text['id']
            if ('inputs' not in text.keys()) or (text['inputs'] is None):
                continue
            for input in text['inputs']:
                if ('content' not in input.keys()) or (input['content'] is None):
                    continue
                content_id = input['id']
                content = list(input['content'])
                self.contents[content_id] = content
                self.text[content_id] = {
                    'text_id': text_id
                }
    
    def get_local_database(self):
        return self.text, self.contents
    
    def check_text_by_id(self, text_id: str) -> bool:
        text_doc = self.text_collection.find_one({'id': text_id}, {'_id': 0})
        if text_doc is None:
            return False
        return True
    

                
