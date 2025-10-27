### DEPRECATED ###
## unused file. initially written for json logging on proxy.
import json
import logging
import os
from logging import Formatter

from litellm import json_logs

# Set default log level to INFO
log_level = os.getenv("LITELLM_LOG", "INFO")
numeric_level: str = getattr(logging, log_level.upper())


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def format(self, record):
        json_record = {
            "message": record.getMessage(),
            "level": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
        }
        return json.dumps(json_record)


logger = logging.root
handler = logging.StreamHandler()
if json_logs:
    handler.setFormatter(JsonFormatter())
else:
    formatter = logging.Formatter(
        "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)
logger.handlers = [handler]
logger.setLevel(numeric_level)
