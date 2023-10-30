from utility.common import Singleton, read_yaml
from utility.logger import Logger
from utility import *

class Configuration(object, metaclass=Singleton):
    def __init__(self, config = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
    
    def init_database(self):
        return self.config['database']