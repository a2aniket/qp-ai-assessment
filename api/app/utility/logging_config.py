import logging
import os
from datetime import datetime

def setup_logger():
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file name with date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{current_date}.log"
    
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# Create a single logger instance
app_logger = setup_logger()
