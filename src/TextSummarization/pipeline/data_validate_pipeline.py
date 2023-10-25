from src.TextSummarization.components.data_validation import DataValidation
from src.TextSummarization.config.configuration import ConfigurationManager

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        validate_data = DataValidation(data_validation_config)
        validate_data.validate_existed_files()