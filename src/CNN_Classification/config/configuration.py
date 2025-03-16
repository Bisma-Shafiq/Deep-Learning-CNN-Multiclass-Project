from src.CNN_Classification.constants import *
from src.CNN_Classification.utils.common import read_yaml , create_directories
from src.CNN_Classification.entity.config_entity import (DataIngestionConfig, 
                                                         PrepareBaseModelConfig)

import os
class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Configuration file not found: {config_filepath}")
        if not os.path.exists(params_filepath):
            raise FileNotFoundError(f"Parameters file not found: {params_filepath}")
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    # data ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URl=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir= config.unzip_dir
        )
        return data_ingestion_config
    
    # base model
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config