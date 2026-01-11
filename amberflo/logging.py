import json
import logging
from datetime import datetime

from .utils import boolean, get_env


aflo_debug = get_env("AFLO_DEBUG", False, validate=boolean)

_level = logging.DEBUG if aflo_debug else logging.INFO

_json_logs = get_env("AFLO_JSON_LOGS", True, validate=boolean)


def get_logger(name, level=_level, json_logs=_json_logs):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(_get_handler(json_logs))
    logger.propagate = False  # Prevent double logging
    return logger


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "time": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def _get_handler(json_logs):
    handler = logging.StreamHandler()

    if json_logs:
        formatter = JsonFormatter()

    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: [%(name)s] %(message)s"
        )

    handler.setFormatter(formatter)
    return handler
