from TextSummarization.constant import *
from utility.common import read_yaml, create_directory

class ConfigurationManager:
    def __init__(self,
                 config_path = CONFIG_FILE_PATH,
                 param_path = PARAM_FILE_PATH):
        
        self.config = read_yaml(config_path)
        self.param = read_yaml(param_path)

        create_directory(self.config['artifacts_root'])
    
    def get_data_validation_config(self):
        config = self.config['data_validation']
        create_directory(config['root_dir'])
        return config

    def get_data_transformation_config(self):
        config = self.config['data_transformation']
        create_directory(config['root_dir'])
        return config
    
    def get_model_trainer(self):
        config = self.config['model_trainer']
        param = self.param['TrainingArguments']
        create_directory(config['root_dir'])
        return config, param
    
    def get_prediction(self):
        config = self.config['model_prediction']
        return config