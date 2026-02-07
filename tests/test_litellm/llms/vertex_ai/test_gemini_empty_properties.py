"""Test for Gemini schema handling with empty properties."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.vertex_ai.common_utils import add_object_type


def test_add_object_type_empty_properties_keeps_type():
    """Gemini requires type: object even when properties is empty."""
    schema = {"properties": {}, "type": "object"}
    add_object_type(schema)
    assert schema.get("type") == "object"
    assert "properties" not in schema
