from src.CNN_Classification import logger
from src.CNN_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainigPipeline
from src.CNN_Classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline



#data ingestion
STAGE_NAME = 'Data Ingestion Stage'

try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    obj = DataIngestionTrainigPipeline()
    obj.main()
    logger.info(f'>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise e


#base model
STAGE_NAME = 'Base Model Prepare Stage'

try:
        logger.info(f"*******************")
        logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f'>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise e

