import pymongo
from datetime import datetime

class BaseDatabase(object):
    def __init__(self, config):
        self.hostname = config['hostname']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        if (self.user == None or self.password == None) or (self.user == '' or self.password == ''):
            self.url = f'mongodb://{self.hostname}:{self.port}'
        else:
            self.url = f"mongodb://{self.user}:{self.password}@{self.hostname}:{self.port}"
        
        self.initialize()
    
    def initialize(self):
        self.client = pymongo.MongoClient(self.url)
