"""@private"""

import logging


def clean_logger():
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)  # Set the desired log level
    console_handler = logging.StreamHandler()
    httpx_logger.addHandler(console_handler)
