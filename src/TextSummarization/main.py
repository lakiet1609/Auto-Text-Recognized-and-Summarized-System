import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from src.TextSummarization.pipeline.data_validate_pipeline import DataValidationPipeline

STAGE_NAME = 'DATA_VALIDATION_PIPELINE'
data_validation = DataValidationPipeline()
data_validation.main()
