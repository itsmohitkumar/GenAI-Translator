import unittest
from src.logger import Logger

class TestLogger(unittest.TestCase):
    def test_logger_initialization(self):
        log_dir = "test_logs"
        log_file = "test.log"
        
        # Initialize Logger
        logger_instance = Logger(log_dir, log_file)
        logger = logger_instance.get_logger()
        
        # Check if logger is set up properly
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, 20)  # Check if log level is INFO

if __name__ == '__main__':
    unittest.main()
