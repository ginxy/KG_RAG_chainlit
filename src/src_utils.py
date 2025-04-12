import os
import sys
import logging
import traceback
from datetime import datetime
from functools import wraps
from typing import Callable, Any, Coroutine, Optional
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()


def setup_logger(name: str = "kg_retrieval", log_file: Optional[str] = None) -> logging.Logger:
    """Configure production-grade logging with container-friendly settings"""
    # Use existing logger if already configured
    logger = logging.getLogger(name)

    # Skip if handlers already configured
    if logger.handlers:
        return logger

    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, log_level))

    # Create detailed formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Always log to stdout for container environments
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Add file handler if specified (or use default)
    log_file = log_file or os.getenv("LOG_FILE", "/app/logs/app.log")
    try:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Don't fail if log file can't be created
        logger.warning(f"Could not set up log file at {log_file}: {str(e)}")

    # Ensure exceptions are logged
    sys.excepthook = lambda exctype, value, tb: logger.critical(f"Uncaught exception: {value}",
        exc_info=(exctype, value, tb))

    return logger

# Set up root logger
root_logger = setup_logger("root")

def log_container_startup():
    """Log container startup environment information"""
    logger = logging.getLogger("container")

    # Log basic environment info
    logger.info(f"Container starting up with Python {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Log environment variables (excluding sensitive ones)
    safe_envs = {k: "****" if any(s in k.lower() for s in ["pass", "key", "secret", "token"]) else v for k, v in
                 os.environ.items()}
    logger.info(f"Environment variables: {safe_envs}")

    # Download libraries for nltk and spacy if not available
    import nltk
    nltk.download('punkt_tab', quiet=True)

    import spacy
    spacy.cli.download("en_core_web_lg")

    # Log installed packages if possible
    try:
        import importlib.metadata as importlib_metadata

        installed = [f"{dist.name}=={dist.version}" for dist in importlib_metadata.distributions()]
        logger.info(f"Installed packages: {installed}")

    except Exception as e:
        logger.warning(f"Could not log installed packages: {str(e)}")

def async_error_handler(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable:
    """Decorator for handling async function errors with improved logging"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            logger.debug(f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
            result = await func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}")
            return result
        except Exception as e:
            # Get full stack trace
            stack_trace = traceback.format_exc()
            # Log both the exception and its context
            logger.error(f"Error in {func.__name__}: {type(e).__name__}: {str(e)}\n"
                         f"Arguments: args={args}, kwargs={kwargs}\n"
                         f"Stack trace: {stack_trace}")
            # Re-raise the exception to maintain original behavior
            raise

    return wrapper


def validate_node_properties(properties: dict) -> bool:
    """Validate node properties before insertion"""
    required_fields = {"id", "name", "description"}
    return all(field in properties for field in required_fields)


def validate_relationship(rel_type: str, properties: dict) -> bool:
    """Validate relationship structure"""
    valid_rel_types = {"RELATES_TO", "HAS_SUBFIELD", "CONTAINS"}
    return rel_type in valid_rel_types and isinstance(properties, dict)


def text_cleaner(text: str) -> str:
    """Clean and normalize input text"""
    text = text.lower().strip()
    # Remove special characters except allowed ones
    return "".join(c if c.isalnum() or c in {"-", "_", " "} else "" for c in text)


def load_config() -> dict:
    """Load and validate configuration"""
    config = {
        "neo4j_uri"     : os.getenv("NEO4J_URI"),
        "neo4j_user"    : os.getenv("NEO4J_USER"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
        "max_retries"   : int(os.getenv("MAX_RETRIES", 3)),
        "batch_size"    : int(os.getenv("BATCH_SIZE", 100))
        }

    # Log configuration values (excluding passwords)
    safe_config = {k: "****" if "password" in k.lower() else v for k, v in config.items()}
    root_logger.info(f"Loaded configuration: {safe_config}")

    # Validate required config
    missing = [k for k, v in config.items() if v is None and k.startswith("neo4j_")]
    if missing:
        root_logger.error(f"Missing required configuration values: {missing}")

    return config

def timing_decorator(func: Callable) -> Callable:
    """Track function execution time"""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} completed successfully in {duration:.4f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {str(e)}")
            raise
    return async_wrapper

def batch_processor(items: list, batch_size: int = 100):
    """Generator for processing data in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


