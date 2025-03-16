from src.CNN_Classification import logger
from src.CNN_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainigPipeline


STAGE_NAME = 'Data Ingestion Stage'


try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    obj = DataIngestionTrainigPipeline()
    obj.main()
    logger.info(f'>>>> stage {STAGE_NAME} completed <<<<<')
except Exception as e:
    logger.error(f"An error occurred: {e}")
    raise e