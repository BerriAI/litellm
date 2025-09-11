"""@private"""

import enum
import math
from asyncio import Queue
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from json import JSONEncoder
from logging import getLogger
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from langfuse.api.core import pydantic_utilities, serialize_datetime
from langfuse.media import LangfuseMedia

# Attempt to import Serializable
try:
    from langchain.load.serializable import Serializable
except ImportError:
    # If Serializable is not available, set it to NoneType
    Serializable = type(None)

# Attempt to import numpy
try:
    import numpy as np
except ImportError:
    np = None

logger = getLogger(__name__)


class EventSerializer(JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = set()  # Track seen objects to detect circular references

    def default(self, obj: Any):
        try:
            if isinstance(obj, (datetime)):
                # Timezone-awareness check
                return serialize_datetime(obj)

            if isinstance(obj, LangfuseMedia):
                return (
                    obj._reference_string
                    or f"<Upload handling failed for LangfuseMedia of type {obj._content_type}>"
                )

            # Check if numpy is available and if the object is a numpy scalar
            # If so, convert it to a Python scalar using the item() method
            if np is not None and isinstance(obj, np.generic):
                return obj.item()

            if isinstance(obj, float) and math.isnan(obj):
                return None

            if isinstance(obj, (Exception, KeyboardInterrupt)):
                return f"{type(obj).__name__}: {str(obj)}"

            # LlamaIndex StreamingAgentChatResponse and StreamingResponse is not serializable by default as it is a generator
            # Attention: These LlamaIndex objects are a also a dataclasses, so check for it first
            if "Streaming" in type(obj).__name__:
                return str(obj)

            if isinstance(obj, enum.Enum):
                return obj.value

            if isinstance(obj, Queue):
                return type(obj).__name__

            if is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, UUID):
                return str(obj)

            if isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8")
                except UnicodeDecodeError:
                    return "<not serializable bytes>"

            if isinstance(obj, (date)):
                return obj.isoformat()

            if isinstance(obj, BaseModel):
                obj.model_rebuild() if pydantic_utilities.IS_PYDANTIC_V2 else obj.update_forward_refs()  # This method forces the OpenAI model to instantiate its serializer to avoid errors when serializing

                # For LlamaIndex models, we need to rebuild the raw model as well if they include OpenAI models
                if isinstance(raw := getattr(obj, "raw", None), BaseModel):
                    raw.model_rebuild() if pydantic_utilities.IS_PYDANTIC_V2 else raw.update_forward_refs()

                return (
                    obj.model_dump()
                    if pydantic_utilities.IS_PYDANTIC_V2
                    else obj.dict()
                )

            if isinstance(obj, Path):
                return str(obj)

            # if langchain is not available, the Serializable type is NoneType
            if Serializable is not type(None) and isinstance(obj, Serializable):
                return obj.to_json()

            # 64-bit integers might overflow the JavaScript safe integer range.
            # Since Node.js is run on the server that handles the serialized value,
            # we need to ensure that integers outside the safe range are converted to strings.
            if isinstance(obj, (int)):
                return obj if self.is_js_safe_integer(obj) else str(obj)

            # Standard JSON-encodable types
            if isinstance(obj, (str, float, type(None))):
                return obj

            if isinstance(obj, (tuple, set, frozenset)):
                return list(obj)

            if isinstance(obj, dict):
                return {self.default(k): self.default(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [self.default(item) for item in obj]

            # Important: this needs to be always checked after str and bytes types
            # Useful for serializing protobuf messages
            if isinstance(obj, Sequence):
                return [self.default(item) for item in obj]

            if hasattr(obj, "__slots__"):
                return self.default(
                    {slot: getattr(obj, slot, None) for slot in obj.__slots__}
                )
            elif hasattr(obj, "__dict__"):
                obj_id = id(obj)

                if obj_id in self.seen:
                    # Break on circular references
                    return type(obj).__name__
                else:
                    self.seen.add(obj_id)
                    result = {k: self.default(v) for k, v in vars(obj).items()}
                    self.seen.remove(obj_id)

                    return result

            else:
                # Return object type rather than JSONEncoder.default(obj) which simply raises a TypeError
                return f"<{type(obj).__name__}>"

        except Exception as e:
            logger.debug(
                f"Serialization failed for object of type {type(obj).__name__}",
                exc_info=e,
            )
            return f'"<not serializable object of type: {type(obj).__name__}>"'

    def encode(self, obj: Any) -> str:
        self.seen.clear()  # Clear seen objects before each encode call

        try:
            return super().encode(self.default(obj))
        except Exception:
            return f'"<not serializable object of type: {type(obj).__name__}>"'  # escaping the string to avoid JSON parsing errors

    @staticmethod
    def is_js_safe_integer(value: int) -> bool:
        """Ensure the value is within JavaScript's safe range for integers.

        Python's 64-bit integers can exceed this range, necessitating this check.
        https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER
        """
        max_safe_int = 2**53 - 1
        min_safe_int = -(2**53) + 1

        return min_safe_int <= value <= max_safe_int
