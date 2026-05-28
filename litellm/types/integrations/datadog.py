from enum import Enum
from typing import Optional

from typing_extensions import NotRequired, TypedDict

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams

DD_MAX_BATCH_SIZE = 1000


class DataDogStatus(str, Enum):
    INFO = "info"
    WARN = "warning"
    ERROR = "error"


DatadogPayload = TypedDict(
    "DatadogPayload",
    {
        "ddsource": str,
        "ddtags": str,
        "hostname": str,
        "message": str,
        "service": str,
        "status": str,
        "dd.trace_id": NotRequired[str],
        "dd.span_id": NotRequired[str],
    },
    total=False,
)


class DD_ERRORS(Enum):
    DATADOG_413_ERROR = (
        "Datadog API Error - Payload too large (batch is above 5MB "
        "uncompressed). The batch is split in half and retried; single "
        "events that still exceed the 5MB limit are dropped. To recover, "
        "disable request/response body logging via "
        "`litellm.turn_off_message_logging = True` or "
        "`DatadogInitParams(turn_off_message_logging=True)`."
    )


class DatadogInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing a DataDog logger on litellm
    """

    pass


class DatadogProxyFailureHookJsonMessage(TypedDict, total=False):
    exception: str
    error_class: str
    status_code: Optional[int]
    traceback: str
    user_api_key_dict: dict
