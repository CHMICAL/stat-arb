from dotenv import load_dotenv
import os
import logging as logger

logger.basicConfig(level=logger.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

load_dotenv()

class Config():
    def get_path_to_data(self):
        path_to_data = os.getenv("PATH_TO_DATA")
        return path_to_data

