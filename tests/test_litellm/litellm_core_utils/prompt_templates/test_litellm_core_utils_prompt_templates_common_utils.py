import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_format_from_file_id,
    handle_any_messages_to_chat_completion_str_messages_conversion,
    update_messages_with_model_file_ids,
    unpack_defs
)


def test_get_format_from_file_id():
    unified_file_id = (
        "litellm_proxy:application/pdf;unified_id,cbbe3534-8bf8-4386-af00-f5f6b7e370bf"
    )

    format = get_format_from_file_id(unified_file_id)

    assert format == "application/pdf"


def test_update_messages_with_model_file_ids():
    file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg"
    model_id = "my_model_id"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file_id,
                    },
                },
            ],
        },
    ]

    model_file_id_mapping = {file_id: {"my_model_id": "provider_file_id"}}

    updated_messages = update_messages_with_model_file_ids(
        messages, model_id, model_file_id_mapping
    )

    assert updated_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "provider_file_id",
                        "format": "application/pdf",
                    },
                },
            ],
        }
    ]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list():
    # Test with list of messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list_infinite_loop():
    # Test that list handling doesn't cause infinite recursion
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    # This should complete without stack overflow
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_dict():
    # Test with single dictionary message
    message = {"role": "user", "content": "Hello"}
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_handle_any_messages_to_chat_completion_str_messages_conversion_str():
    # Test with string message
    message = "Hello"
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == message


def test_handle_any_messages_to_chat_completion_str_messages_conversion_other():
    # Test with non-string/dict/list type
    message = 123
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == "123"


def test_handle_any_messages_to_chat_completion_str_messages_conversion_complex():
    # Test with complex nested structure
    message = {
        "role": "user",
        "content": {"text": "Hello", "metadata": {"timestamp": "2024-01-01"}},
    }
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_unpack_defs_basic_reference_resolution():
    # Test basic $ref resolution to $defs
    schema = {
        "type": "object",
        "properties": {
            "user": {"$ref": "#/$defs/User"}
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"}
                },
                "required": ["id", "name"]
            }
        }
    }

    result = unpack_defs(schema)
    
    # Should resolve the reference
    assert "user" in result["properties"]
    assert result["properties"]["user"]["type"] == "object"
    assert "id" in result["properties"]["user"]["properties"]
    assert "name" in result["properties"]["user"]["properties"]
    
    # $defs should be removed
    assert "$defs" not in result


def test_unpack_defs_nested_reference_resolution():
    # Test nested $refs within resolved content
    schema = {
        "type": "object",
        "properties": {
            "user": {"$ref": "#/$defs/User"}
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "profile": {"$ref": "#/$defs/Profile"}
                }
            },
            "Profile": {
                "type": "object",
                "properties": {
                    "bio": {"type": "string"},
                    "avatar_url": {"type": "string"}
                }
            }
        }
    }
    
    result = unpack_defs(schema)
    
    # Should resolve nested reference
    user_props = result["properties"]["user"]["properties"]
    assert "profile" in user_props
    assert user_props["profile"]["type"] == "object"
    assert "bio" in user_props["profile"]["properties"]
    assert "avatar_url" in user_props["profile"]["properties"]


def test_unpack_defs_anyof_resolution():
    # Test $ref within an anyOf structure
    schema = {
        "type": "object",
        "properties": {
            "contact": {
                "anyOf": [
                    {"$ref": "#/$defs/User"},
                    {"$ref": "#/$defs/Company"}
                ]
            }
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            },
            "Company": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "website": {"type": "string"}
                }
            }
        }
    }
    
    result = unpack_defs(schema)
    
    # Should resolve refs within anyOf
    contact_anyof = result["properties"]["contact"]["anyOf"]
    assert len(contact_anyof) == 2
    assert contact_anyof[0]["type"] == "object"
    assert "name" in contact_anyof[0]["properties"]
    assert contact_anyof[1]["type"] == "object"
    assert "company_name" in contact_anyof[1]["properties"]


def test_unpack_defs_allof_resolution():
    # Test $ref within an allOf structure
    schema = {
        "type": "object",
        "properties": {
            "enhanced_user": {
                "allOf": [
                    {"$ref": "#/$defs/User"},
                    {"$ref": "#/$defs/Timestamps"}
                ]
            }
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            },
            "Timestamps": {
                "type": "object",
                "properties": {
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                }
            }
        }
    }
    
    result = unpack_defs(schema)
    
    # Should resolve refs within allOf
    enhanced_user_allof = result["properties"]["enhanced_user"]["allOf"]
    assert len(enhanced_user_allof) == 2
    assert "id" in enhanced_user_allof[0]["properties"]
    assert "created_at" in enhanced_user_allof[1]["properties"]


def test_unpack_defs_array_items_with_refs():
    # Test array items containing $refs
    schema = {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {"$ref": "#/$defs/User"}
            },
            "posts": {
                "type": "array",
                "items": {"$ref": "#/$defs/Post"}
            }
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            },
            "Post": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "author": {"$ref": "#/$defs/User"}
                }
            }
        }
    }
    
    result = unpack_defs(schema)
    
    # Should resolve refs in array items
    users_items = result["properties"]["users"]["items"]
    assert users_items["type"] == "object"
    assert "id" in users_items["properties"]
    
    posts_items = result["properties"]["posts"]["items"]
    assert posts_items["type"] == "object"
    assert "title" in posts_items["properties"]
    # Should also resolve nested ref in author
    assert posts_items["properties"]["author"]["type"] == "object"
    assert "name" in posts_items["properties"]["author"]["properties"]


def test_unpack_defs_no_properties():
    # Test schema without properties (should return as-is)
    schema = {
        "type": "string",
        "enum": ["value1", "value2"]
    }
    
    result = unpack_defs(schema)
    assert result == schema


def test_unpack_defs_empty_properties():
    # Test schema with empty properties
    schema = {
        "type": "object",
        "properties": {},
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        }
    }
    
    result = unpack_defs(schema)
    assert result["properties"] == {}
    assert "$defs" not in result

def test_unpack_defs_no_refs_in_properties():
    # Test schema with properties but no $refs
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "$defs": {
            "UnusedDef": {
                "type": "object",
                "properties": {
                    "unused": {"type": "string"}
                }
            }
        }
    }
    
    result = unpack_defs(schema)
    
    # Properties should remain unchanged
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["age"]["type"] == "integer"
    # $defs should still be removed
    assert "$defs" not in result

def test_unpack_defs_circular_refs_handling():
    # Test that circular references are handled properly.
    schema = {
        "type": "object",
        "properties": {
            "node": {"$ref": "#/$defs/TreeNode"}
        },
        "$defs": {
            "TreeNode": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/TreeNode"}
                    }
                }
            }
        }
    }
    
    # This should not raise an error (handles circular refs)
    result = unpack_defs(schema)
    
    # The structure should be resolved
    assert "node" in result["properties"]
    node_props = result["properties"]["node"]["properties"]
    assert "value" in node_props
    assert "children" in node_props

def test_unpack_defs_from_pydantic_schema():
    # Test unpacking from a Pydantic schema
    from pydantic import BaseModel
    from typing import List, Optional

    class VatAmount(BaseModel):
        vatRate: float
        vatAmount: float

    class ExpenseReceipt(BaseModel):
        vatAmounts: Optional[List[VatAmount]] = None

    schema = ExpenseReceipt.model_json_schema()

    result = unpack_defs(schema)

    # Should resolve the VatAmount reference
    assert "vatAmounts" in result["properties"]
    assert "anyOf" in result["properties"]["vatAmounts"]
    assert len(result["properties"]["vatAmounts"]["anyOf"]) == 2
    assert any(item["type"] == "null" for item in result["properties"]["vatAmounts"]["anyOf"])
    assert any(item["type"] == "array" for item in result["properties"]["vatAmounts"]["anyOf"])