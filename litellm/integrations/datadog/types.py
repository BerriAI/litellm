from enum import Enum
from typing import TypedDict


class DatadogPayload(TypedDict, total=False):
    ddsource: str
    ddtags: str
    hostname: str
    message: str
    service: str


class DD_ERRORS(Enum):
    DATADOG_413_ERROR = "Datadog API Error - Payload too large (batch is above 5MB uncompressed). If you want this logged either disable request/response logging or set `DD_BATCH_SIZE=50`"
