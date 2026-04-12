import json
import logging
import os
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        json_record = {
            "message": record.getMessage(),
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }
        if record.exc_info:
            json_record["stacktrace"] = self.formatException(record.exc_info)
        return json.dumps(json_record)


def _is_json_enabled():
    try:
        import litellm
        return getattr(litellm, 'json_logs', False)
    except (ImportError, AttributeError):
        return os.getenv("JSON_LOGS", "false").lower() == "true"


logger = logging.getLogger("litellm_proxy_extras")

if not logger.handlers:
    handler = logging.StreamHandler()
    if _is_json_enabled():
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
