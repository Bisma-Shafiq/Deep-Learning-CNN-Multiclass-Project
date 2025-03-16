from src.CNN_Classification.config.configuration import ConfigurationManager
from src.CNN_Classification.components.prepare_base_model import PrepareBaseModel                            
from src.CNN_Classification import logger

STAGE_NAME = 'Base Model Prepare Stage'

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()



if __name__=="__main__":
        try:
             logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
             prepare_base_model = PrepareBaseModelTrainingPipeline()
             prepare_base_model.main()
             logger.info(f'>>>> stage {STAGE_NAME} completed <<<<<')
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e