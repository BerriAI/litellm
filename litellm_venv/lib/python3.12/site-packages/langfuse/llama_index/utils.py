from datetime import datetime
from typing import Optional, Dict, Any, TypedDict
from langfuse.model import ModelUsage
from langfuse.utils import _get_timestamp

try:
    from llama_index.core.callbacks.schema import (
        CBEventType,
        CBEvent,
    )
except ImportError:
    raise ModuleNotFoundError(
        "Please install llama-index to use the Langfuse llama-index integration: 'pip install llama-index'"
    )


class CallbackEvent(CBEvent):
    time: datetime

    def __init__(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
    ):
        super().__init__(event_type, payload=payload, id_=event_id)
        self.time = _get_timestamp()


class ParsedLLMEndPayload(TypedDict):
    end_time: datetime
    input: Optional[str]
    output: Optional[dict]
    usage: Optional[ModelUsage]
    model: Optional[str]
