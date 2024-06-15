import json
from typing import Any, Optional, TypedDict, Union

from pydantic import BaseModel
from typing_extensions import (
    Protocol,
    Required,
    Self,
    TypeGuard,
    get_origin,
    override,
    runtime_checkable,
)


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    is_finished: Required[bool]
    finish_reason: Required[Optional[str]]
    logprobs: Optional[BaseModel]
    original_chunk: Optional[BaseModel]
    usage: Optional[BaseModel]
