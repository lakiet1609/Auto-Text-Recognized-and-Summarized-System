from src.TextSummarization.components.model_trainer import ModelTrainer
from src.TextSummarization.config.configuration import ConfigurationManager

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config, model_trainer_param = config.get_model_trainer()
        model_trainer = ModelTrainer(model_trainer_config, model_trainer_param)
        model_trainer.train()