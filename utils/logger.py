import logging
import os

# Set the logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Create logs directory if it doesn't exist
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),
    level=logging.DEBUG,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_logger(module_name):
    """
    Returns a logger instance for the given module name.
    """
    return logging.getLogger(module_name)
