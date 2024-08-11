import os
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, log_dir, log_file):
        """
        Initialize the Logger with a directory and file name.
        Sets up logging to both console and file with rotation.
        """
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        # Set up the logger instance
        self.logger = logging.getLogger("TranslatorApp")
        self.logger.setLevel(logging.INFO)

        # Create file handler with rotation
        file_handler = RotatingFileHandler(log_path, maxBytes=10**6, backupCount=5)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Add handlers to the logger
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Return the logger instance.
        """
        return self.logger
