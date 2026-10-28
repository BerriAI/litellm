from typing import Final

import jsonschema
import pytest
from pydantic import JsonValue

from litellm.proxy.admin_mcp._types import JsonObject, object_value
from litellm.proxy.admin_mcp.schema import SchemaError, resolve, tool_schema


def _operation(schema: JsonValue) -> JsonObject:
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": schema}}}}


def test_schema_keywords_in_property_names_and_literal_values_are_preserved() -> None:
    literal: Final[JsonValue] = {
        "$ref": "#/components/schemas/NotAReference",
        "$dynamicRef": "https://untrusted.example/schema.json",
        "$recursiveRef": "#",
        "$id": "https://untrusted.example/",
        "nullable": True,
    }
    body: Final[JsonValue] = {
        "type": "object",
        "properties": {
            "$ref": {"type": "string"},
            "nullable": {"type": "boolean"},
            "metadata": {"const": literal, "default": literal, "examples": [literal], "enum": [literal]},
        },
        "required": ["$ref", "nullable", "metadata"],
        "additionalProperties": False,
    }
    schema: Final = tool_schema({}, {}, _operation(body))
    assert schema == {
        "type": "object",
        "properties": {"body": body},
        "additionalProperties": False,
        "required": ["body"],
    }
    validator: Final = jsonschema.Draft202012Validator(schema)
    validator.validate({"body": {"$ref": "literal", "nullable": True, "metadata": literal}})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"body": {"$ref": "literal", "nullable": "wrong", "metadata": literal}})


@pytest.mark.parametrize(
    "body,accepted,rejected",
    [
        ({"type": "object", "additionalProperties": {"$ref": "#/components/schemas/Value"}}, {"a": 3}, {"a": 2}),
        (
            {"type": "object", "patternProperties": {"^item": {"$ref": "#/components/schemas/Value"}}},
            {"item_a": 3},
            {"item_a": 2},
        ),
        ({"type": "array", "items": {"$ref": "#/components/schemas/Value"}}, [3], [2]),
        ({"type": "array", "prefixItems": [{"$ref": "#/components/schemas/Value"}]}, [3], [2]),
        ({"anyOf": [{"$ref": "#/components/schemas/Value"}, {"type": "null"}]}, 3, 2),
        ({"if": {"type": "integer"}, "then": {"$ref": "#/components/schemas/Value"}, "else": False}, 3, 2),
    ],
)
def test_references_inside_schema_keywords_are_retained(
    body: JsonValue, accepted: JsonValue, rejected: JsonValue
) -> None:
    spec: Final[JsonObject] = {"components": {"schemas": {"Value": {"type": "integer", "minimum": 3}}}}
    schema: Final = tool_schema(spec, {}, _operation(body))
    assert schema["$defs"] == {"Value": {"type": "integer", "minimum": 3}}
    validator: Final = jsonschema.Draft202012Validator(schema)
    validator.validate({"body": accepted})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"body": rejected})


def test_mutually_recursive_references_with_escaped_names_terminate_and_validate() -> None:
    spec: Final[JsonObject] = {
        "components": {
            "schemas": {
                "A/B~C": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/Other"}}},
                "Other": {"anyOf": [{"$ref": "#/components/schemas/A~1B~0C"}, {"type": "null"}]},
                "Unused": {"type": "string"},
            }
        }
    }
    schema: Final = tool_schema(spec, {}, _operation({"$ref": "#/components/schemas/A~1B~0C"}))
    assert set(object_value(schema["$defs"])) == {"A/B~C", "Other"}
    validator: Final = jsonschema.Draft202012Validator(schema)
    validator.validate({"body": {"next": {"next": None}}})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"body": {"next": "wrong"}})


def test_long_reference_graph_does_not_depend_on_python_recursion_limit() -> None:
    count: Final = 1200
    components: Final[JsonValue] = {
        **{f"Node{i}": {"$ref": f"#/components/schemas/Node{i + 1}"} for i in range(count)},
        f"Node{count}": {"type": "integer"},
    }
    spec: Final[JsonObject] = {"components": {"schemas": components}}
    schema: Final = tool_schema(spec, {}, _operation({"$ref": "#/components/schemas/Node0"}))
    definitions: Final = object_value(schema["$defs"])
    assert len(definitions) == count
    assert definitions[f"Node{count}"] == {"type": "integer"}
    assert object_value(schema["properties"])["body"] == {"$ref": "#/$defs/Node1"}


def test_long_parameter_reference_chain_is_resolved_without_recursion() -> None:
    count: Final = 1200
    parameter: Final[JsonValue] = {"name": "limit", "in": "query", "schema": {"type": "integer"}}
    parameters: Final[JsonValue] = {
        **{f"P{i}": {"$ref": f"#/components/parameters/P{i + 1}"} for i in range(count)},
        f"P{count}": parameter,
    }
    spec: Final[JsonObject] = {"components": {"parameters": parameters}}
    assert resolve(spec, {"$ref": "#/components/parameters/P0"}) == parameter


def test_cyclic_parameter_reference_chain_fails_closed() -> None:
    spec: Final[JsonObject] = {
        "components": {
            "parameters": {"A": {"$ref": "#/components/parameters/B"}, "B": {"$ref": "#/components/parameters/A"}}
        }
    }
    with pytest.raises(SchemaError, match="Unsupported OpenAPI reference"):
        resolve(spec, {"$ref": "#/components/parameters/A"})


@pytest.mark.parametrize(
    "reference",
    [
        "https://untrusted.example/schema.json",
        "#/components/schemas/Missing",
        "#/components/schemas/Parent/properties/child",
        "#/components/schemas/Bad~2Escape",
    ],
)
def test_unsupported_or_missing_schema_reference_fails_closed(reference: str) -> None:
    with pytest.raises(SchemaError):
        tool_schema({}, {}, _operation({"$ref": reference}))


def test_escaped_component_names_cannot_hide_a_linked_definition() -> None:
    spec: Final[JsonObject] = {
        "components": {
            "schemas": {
                "A~1B": {"type": "object", "properties": {"next": {"$ref": "#/components/schemas/A~1B"}}},
                "A/B": {"type": "integer"},
            }
        }
    }
    schema: Final = tool_schema(spec, {}, _operation({"$ref": "#/components/schemas/A~01B"}))
    assert schema["$defs"] == {"A/B": {"type": "integer"}}
    validator: Final = jsonschema.Draft202012Validator(schema)
    validator.validate({"body": {"next": 3}})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"body": {"next": "wrong"}})


@pytest.mark.parametrize("keyword", ["$dynamicRef", "$recursiveRef", "$id"])
def test_reference_scopes_and_dynamic_references_are_rejected(keyword: str) -> None:
    with pytest.raises(SchemaError, match="unsupported"):
        tool_schema({}, {}, _operation({"type": "string", keyword: "https://untrusted.example/schema.json"}))


@pytest.mark.parametrize(
    "body", [{"type": "integer", "nullable": True}, {"$ref": "#/components/schemas/Number", "nullable": True}]
)
def test_nullable_schemas_accept_null_without_losing_the_original_constraint(body: JsonValue) -> None:
    spec: Final[JsonObject] = {"components": {"schemas": {"Number": {"type": "integer"}}}}
    validator: Final = jsonschema.Draft202012Validator(tool_schema(spec, {}, _operation(body)))
    validator.validate({"body": None})
    validator.validate({"body": 3})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"body": "wrong"})
