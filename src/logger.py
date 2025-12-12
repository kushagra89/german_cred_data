import logging
import os

def get_logger(name: str):
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger