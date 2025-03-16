from src.CNN_Classification.constants import *
from src.CNN_Classification.utils.common import read_yaml , create_directories
from src.CNN_Classification.entity.config_entity import DataIngestionConfig

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
