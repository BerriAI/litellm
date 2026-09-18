"""Test for Gemini schema handling with empty properties."""



from litellm.llms.vertex_ai.common_utils import add_object_type


def test_add_object_type_empty_properties_keeps_type():
    """Gemini requires type: object even when properties is empty."""
    schema = {"properties": {}, "type": "object"}
    add_object_type(schema)
    assert schema.get("type") == "object"
    assert "properties" not in schema
