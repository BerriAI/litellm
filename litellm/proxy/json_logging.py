import logging
import traceback
from datetime import datetime, UTC
import json

from uvicorn.config import LOGGING_CONFIG

# Override uvicorn's default logging config to use our JSON formatter
def uvicorn_json_log_config():
    config = LOGGING_CONFIG.copy()
    config['formatters'] = {
        'default': {
            '()': 'litellm.proxy.json_logging.JsonFormatter'
        },
        'access': {
            '()': 'litellm.proxy.json_logging.AccessJsonFormatter'
        }
    }
    return config

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(timespec='milliseconds') + 'Z',
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
            "process": record.process,
            "thread": record.threadName,
        }
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = "".join(traceback.format_exception(*record.exc_info))
            exc_type, exc_value, exc_traceback = record.exc_info
            log_entry['exception'] = {
                'type': str(exc_type.__name__),
                'message': str(exc_value),
                'traceback': traceback.format_tb(exc_traceback)
            }

        # Add extra data
        if hasattr(record, 'extra_data') and isinstance(record.extra_data, dict):
             log_entry.update(record.extra_data) # extra データの処理例

        return json.dumps(log_entry, ensure_ascii=False)

# ref: https://github.com/encode/uvicorn/blob/0.34.1/uvicorn/logging.py#L73
class AccessJsonFormatter(logging.Formatter):
    def format(self, record):
        (
            client_addr,
            method,
            full_path,
            http_version,
            status_code,
        ) = record.args
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, UTC).isoformat(timespec='milliseconds') + 'Z',
            'level': record.levelname,
            'logger_name': record.name,
            'process': record.process,
            'thread': record.threadName,
            'client_addr': client_addr,
            'method': method,
            'full_path': full_path,
            'http_version': http_version,
            'status_code': status_code,
        }
        return json.dumps(log_entry, ensure_ascii=False)
