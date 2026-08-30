from litellm.litellm_core_utils.json_validation_rule import (
    normalize_json_schema_types,
    normalize_tool_schema,
)


def test_normalizes_a_plain_string_type():
    assert normalize_json_schema_types({"type": "STRING"}) == {"type": "string"}


def test_normalizes_a_list_of_types():
    """A list of types is how a nullable field is expressed.

    Regression: these entries fell through to the generic list recursion, which
    returns bare strings untouched, so they stayed uppercase.
    """
    assert normalize_json_schema_types({"type": ["STRING", "NULL"]}) == {
        "type": ["string", "null"]
    }


def test_normalizes_a_list_of_types_when_nested():
    schema = {"properties": {"a": {"type": ["INTEGER", "NULL"]}}}

    assert normalize_json_schema_types(schema) == {
        "properties": {"a": {"type": ["integer", "null"]}}
    }


def test_leaves_unknown_type_entries_alone():
    assert normalize_json_schema_types({"type": ["STRING", "custom"]}) == {
        "type": ["string", "custom"]
    }


def test_still_normalizes_properties_items_and_anyof():
    schema = {
        "type": "OBJECT",
        "properties": {"xs": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
        "anyOf": [{"type": "STRING"}],
    }

    assert normalize_json_schema_types(schema) == {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": {"type": "integer"}}},
        "anyOf": [{"type": "string"}],
    }


def test_tool_schema_normalizes_a_nullable_parameter():
    tool = {
        "function": {
            "parameters": {
                "type": "OBJECT",
                "properties": {"x": {"type": ["STRING", "NULL"]}},
            }
        }
    }

    assert normalize_tool_schema(tool) == {
        "function": {
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": ["string", "null"]}},
            }
        }
    }
