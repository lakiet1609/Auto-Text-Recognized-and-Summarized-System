from src.utility.common import Singleton, read_yaml
from src.utility.logger import Logger
from src.utility import *

class Configuration(object, metaclass=Singleton):
    def __init__(self, config = CONFIG_FILE_PATH):
        self.config = read_yaml(config)
    
    def init_databse(self):
        return self.config['database']