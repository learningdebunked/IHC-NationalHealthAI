"""Logging utility for IHC Platform."""

import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger with JSON formatting.
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        timestamp=True
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger