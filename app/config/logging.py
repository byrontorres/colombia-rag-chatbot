"""
Structured logging configuration using structlog.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.typing import FilteringBoundLogger

from app.config.settings import settings


def setup_logging() -> FilteringBoundLogger:
    """
    Setup structured logging with appropriate configuration for the environment.
    """
    
    # Determine log level
    log_level = getattr(logging, settings.log_level, logging.INFO)
    
    # Configure structlog based on environment
    if settings.is_production:
        # Production: JSON formatted logs
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Pretty formatted logs
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    return structlog.get_logger()


def log_error(error: Exception, context: Dict[str, Any] = None, **kwargs) -> None:
    """Log error with context information."""
    logger = structlog.get_logger("error")
    
    error_context = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        **(context or {}),
        **kwargs
    }
    
    logger.error("Error occurred", **error_context, exc_info=True)


# Global logger instance
logger = setup_logging()