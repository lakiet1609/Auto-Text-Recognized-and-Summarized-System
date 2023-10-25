import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from src.TextSummarization.pipeline.data_validate_pipeline import DataValidationPipeline
from src.TextSummarization.pipeline.data_transformation_pipeline import DataTransformationPipeline


STAGE_NAME = 'DATA_VALIDATION_PIPELINE'
data_validation = DataValidationPipeline()
data_validation.main()

STAGE_NAME = 'DATA_TRANSFORMATION_PIPELINE'
data_transformation = DataTransformationPipeline()
data_transformation.main()
