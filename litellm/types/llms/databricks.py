from typing import TypedDict, Any, Union, Optional
import json
from typing_extensions import (
    Self,
    Protocol,
    TypeGuard,
    override,
    get_origin,
    runtime_checkable,
    Required,
)
from pydantic import BaseModel


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    is_finished: Required[bool]
    finish_reason: Required[Optional[str]]
    logprobs: Optional[BaseModel]
    original_chunk: Optional[BaseModel]
    usage: Optional[BaseModel]
