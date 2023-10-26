import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from src.TextSummarization.pipeline.data_validate_pipeline import DataValidationPipeline
from src.TextSummarization.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.TextSummarization.pipeline.model_trainer_pipeline import ModelTrainerPipeline
from src.TextSummarization.pipeline.prediction import Prediction


# STAGE_NAME = 'DATA_VALIDATION_PIPELINE'
# data_validation = DataValidationPipeline()
# data_validation.main()

# STAGE_NAME = 'DATA_TRANSFORMATION_PIPELINE'
# data_transformation = DataTransformationPipeline()
# data_transformation.main()

# STAGE_NAME = 'MODEL_TRAINER_PIPELINE'
# model_trainer = ModelTrainerPipeline()
# model_trainer.main()

STAGE_NAME = 'PREDICTION'
text = ''
summary = Prediction().predict(text)
print(summary)