import logging
import sys
from typing import Optional


def setup_logger(name: Optional[str] = None, level: int = logging.INFO):
    """
    Configure and return a logger instance
    
    Args:
        name: The name of the logger
        level: The logging level
        
    Returns:
        logging.Logger: The configured logger instance
    """
    if name is None:
        # Get the caller's module name if no name is provided
        frame = sys.getframe(1)
        name = frame.f_globals['__name__']

    # Configure logger
    logger = logging.getLogger(name)

    # Only configure if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        handler = logging.Streamhandler()
        handler.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add formatter to handler
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger

# Default application logger
app_logger = setup_logger("bulls_ai")
