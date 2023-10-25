from src.TextSummarization.components.data_transformation import DataTransformation
from src.TextSummarization.config.configuration import ConfigurationManager

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transform = DataTransformation(data_transformation_config)
        data_transform.convert()