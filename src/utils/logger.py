import logging
import os
import sys
import inspect

# Dictionary to store loggers for different files
_LOGGERS = {}

def get_logger(name=None):
    """
    Returns a configured logger.
    
    Args:
        name (str, optional): Override the default logger name.
                             If None, uses the filename of the calling script.
    
    Returns:
        logging.Logger: A logger that writes to both console and a dedicated file.
    """
    # If name not provided, get the calling script's filename
    if name is None:
        frame = inspect.stack()[1]
        caller_path = frame.filename
        # Extract just the filename without extension
        name = os.path.splitext(os.path.basename(caller_path))[0]
    
    # Check if a logger for this name already exists
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler
    log_file = os.path.join('logs', f"{name}.txt")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to the root logger
    logger.propagate = False
    
    # Store in dictionary for reuse
    _LOGGERS[name] = logger
    
    return logger