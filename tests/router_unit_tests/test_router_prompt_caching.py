import sys
import os
import traceback
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
import pytest
import litellm
from unittest.mock import patch, MagicMock, AsyncMock
from create_mock_standard_logging_payload import create_standard_logging_payload
from litellm.types.utils import StandardLoggingPayload
import unittest
from pydantic import BaseModel
from litellm.router_utils.prompt_caching_cache import PromptCachingCache


class ExampleModel(BaseModel):
    field1: str
    field2: int


def test_serialize_pydantic_object():
    model = ExampleModel(field1="value", field2=42)
    serialized = PromptCachingCache.serialize_object(model)
    assert serialized == {"field1": "value", "field2": 42}


def test_serialize_dict():
    obj = {"b": 2, "a": 1}
    serialized = PromptCachingCache.serialize_object(obj)
    assert serialized == '{"a":1,"b":2}'  # JSON string with sorted keys


def test_serialize_nested_dict():
    obj = {"z": {"b": 2, "a": 1}, "x": [1, 2, {"c": 3}]}
    serialized = PromptCachingCache.serialize_object(obj)
    expected = '{"x":[1,2,{"c":3}],"z":{"a":1,"b":2}}'  # JSON string with sorted keys
    assert serialized == expected


def test_serialize_list():
    obj = ["item1", {"a": 1, "b": 2}, 42]
    serialized = PromptCachingCache.serialize_object(obj)
    expected = ["item1", '{"a":1,"b":2}', 42]
    assert serialized == expected


def test_serialize_fallback():
    obj = 12345  # Simple non-serializable object
    serialized = PromptCachingCache.serialize_object(obj)
    assert serialized == 12345


def test_serialize_non_serializable():
    class CustomClass:
        def __str__(self):
            return "custom_object"

    obj = CustomClass()
    serialized = PromptCachingCache.serialize_object(obj)
    assert serialized == "custom_object"  # Fallback to string conversion
