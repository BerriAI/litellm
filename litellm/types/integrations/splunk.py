from typing import Any, Dict, List

from pydantic import BaseModel


class SplunkEvent(BaseModel):
    """
    Represents a single event to be sent to Splunk HEC.
    """

    source: str
    sourcetype: str
    host: str
    event: Dict[str, Any]
    index: str
    tags: List[str]


class SplunkPayload(BaseModel):
    """
    The payload structure expected by Splunk HEC.
    """

    event: SplunkEvent
