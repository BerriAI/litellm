from enum import Enum
from typing import Optional, TypedDict

DD_MAX_BATCH_SIZE = 1000


class DataDogStatus(str, Enum):
    INFO = "info"
    WARN = "warning"
    ERROR = "error"


class DatadogPayload(TypedDict, total=False):
    ddsource: str
    ddtags: str
    hostname: str
    message: str
    service: str
    status: str


class DD_ERRORS(Enum):
    DATADOG_413_ERROR = "Datadog API Error - Payload too large (batch is above 5MB uncompressed). If you want this logged either disable request/response logging or set `DD_BATCH_SIZE=50`"


class DatadogProxyFailureHookJsonMessage(TypedDict, total=False):
    exception: str
    error_class: str
    status_code: Optional[int]
    traceback: str
    user_api_key_dict: dict
