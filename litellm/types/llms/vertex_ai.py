from typing import TypedDict, Any, Union, Optional, List, Literal, Dict
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


class Field(TypedDict):
    key: str
    value: Dict[str, Any]


class FunctionCallArgs(TypedDict):
    fields: Field


class FunctionResponse(TypedDict):
    name: str
    response: FunctionCallArgs


class FunctionCall(TypedDict):
    name: str
    args: FunctionCallArgs


class FileDataType(TypedDict):
    mime_type: str
    file_uri: str  # the cloud storage uri of storing this file


class BlobType(TypedDict):
    mime_type: Required[str]
    data: Required[bytes]


class PartType(TypedDict, total=False):
    text: str
    inline_data: BlobType
    file_data: FileDataType
    function_call: FunctionCall
    function_response: FunctionResponse


class ContentType(TypedDict, total=False):
    role: Literal["user", "model"]
    parts: Required[List[PartType]]
