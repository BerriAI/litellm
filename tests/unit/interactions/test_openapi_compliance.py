"""
OpenAPI compliance tests for Google Interactions API.

Validates that our SDK requests/responses match the OpenAPI spec at:
https://ai.google.dev/static/api/interactions.openapi.json

Run with: pytest tests/unit/interactions/test_openapi_compliance.py -v
"""

import json
import os
import re
from typing import Any, Dict, Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openapi_core import OpenAPI

OPENAPI_SPEC_URL = "https://ai.google.dev/static/api/interactions.openapi.json"


def _load_openapi_spec_dict() -> Dict[str, Any]:
    """
    Load the OpenAPI spec JSON.

    In CI or offline environments, network access may not be available.
    In that case, gracefully skip these tests instead of erroring.
    """
    try:
        response = httpx.get(OPENAPI_SPEC_URL, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:  # pragma: no cover - defensive, env-dependent
        pytest.skip(
            f"Skipping Google Interactions OpenAPI compliance tests - "
            f"unable to load spec from {OPENAPI_SPEC_URL}: {e}"
        )


def _declared_type_value(variant_schema: Dict[str, Any]) -> Any:
    """The single `type` value a union variant pins, whether spelled as a const or a 1-item enum."""
    type_property = variant_schema.get("properties", {}).get("type", {})
    enum_values = type_property.get("enum") or []
    return type_property.get("const") or (enum_values[0] if len(enum_values) == 1 else None)


def _resolve_local_ref(spec_dict: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve component references used by operations, schemas, and parameters."""
    if "$ref" not in schema:
        return schema
    reference: Final = schema["$ref"]
    assert reference.startswith("#/components/"), f"Expected a local component reference: {reference}"
    category, name = reference.removeprefix("#/components/").split("/")
    return spec_dict["components"][category][name.replace("~1", "/").replace("~0", "~")]


def _interaction_operation(
    spec_dict: dict[str, Any], method: str, *, individual: bool = False
) -> tuple[str, dict[str, Any]]:
    """Match collection or item routes exactly, independent of placeholder names."""
    pattern: Final = r"(?:/[^/]+)*/interactions" + (r"/(\{[^/{}]+\})" if individual else "")
    matches: Final = tuple(
        (path, path_item, match)
        for path, path_item in spec_dict["paths"].items()
        if (match := re.fullmatch(pattern, path)) and method in path_item
    )
    assert len(matches) == 1, f"Expected one {method.upper()} interactions endpoint, got {matches}"
    path, path_item, match = matches[0]
    operation: Final = path_item[method]
    if individual:
        parameter_name: Final = match.group(1)[1:-1]
        parameters: Final = {
            (parameter["name"], parameter["in"]): parameter
            for raw_parameter in (*path_item.get("parameters", ()), *operation.get("parameters", ()))
            for parameter in (_resolve_local_ref(spec_dict, raw_parameter),)
        }
        parameter: Final = parameters.get((parameter_name, "path"))
        assert parameter is not None, f"{path} must declare its interaction ID path parameter"
        assert parameter.get("required") is True, f"{path} must require its interaction ID"
        parameter_schema: Final = _resolve_local_ref(spec_dict, parameter["schema"])
        assert parameter_schema.get("type") == "string", f"{path} must accept a string interaction ID"
    return path, operation


def _model_request_schema(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Find the model variant of the JSON body declared by the create operation."""
    _, operation = _interaction_operation(spec_dict, "post")
    request_body: Final = _resolve_local_ref(spec_dict, operation["requestBody"])
    assert request_body.get("required") is True, "Creating an interaction must require a request body"
    schema: Final = _resolve_local_ref(spec_dict, request_body["content"]["application/json"]["schema"])
    variants: Final = tuple(_resolve_local_ref(spec_dict, variant) for variant in schema.get("oneOf", (schema,)))
    model_variants: Final = tuple(variant for variant in variants if "model" in variant.get("properties", {}))
    assert len(model_variants) == 1, f"Expected one model request variant, got {model_variants}"
    return model_variants[0]


@pytest.fixture(scope="module")
def spec_dict() -> Dict[str, Any]:
    """Load raw spec dict for manual validation."""
    return _load_openapi_spec_dict()


@pytest.fixture(scope="module")
def openapi_spec(spec_dict: Dict[str, Any]) -> OpenAPI:
    """Load the OpenAPI spec as an OpenAPI object."""
    return OpenAPI.from_dict(spec_dict)


class TestRequestCompliance:
    """Tests that our request bodies match the OpenAPI spec."""

    def test_create_model_interaction_request_schema(self, spec_dict):
        """Verify the model request schema declared by POST /interactions."""
        schema = _model_request_schema(spec_dict)

        assert "model" in schema["required"]
        for field in ("model", "input"):
            assert field in schema["properties"]
            assert schema["properties"][field].get("readOnly") is not True
            assert _resolve_local_ref(spec_dict, schema["properties"][field]).get("readOnly") is not True

        # Check our supported optional fields exist in spec
        our_optional_fields = [
            "tools",
            "system_instruction",
            "generation_config",
            "stream",
            "store",
            "background",
            "response_modalities",
            "response_format",
            "response_mime_type",
            "previous_interaction_id",
        ]

        spec_properties = schema["properties"]
        for field in our_optional_fields:
            assert field in spec_properties, f"Field '{field}' not in OpenAPI spec"
            print(f"✓ Field '{field}' exists in spec")

    def test_input_types_match_spec(self, spec_dict):
        """Verify input field supports string, Content, Content[], Turn[]."""
        schema = _model_request_schema(spec_dict)
        input_schema = _resolve_local_ref(spec_dict, schema["properties"]["input"])

        # Should be oneOf with multiple types
        assert "oneOf" in input_schema

        input_types = []
        for option in input_schema["oneOf"]:
            if option.get("type") == "string":
                input_types.append("string")
            elif option.get("type") == "array":
                input_types.append("array")
            elif "$ref" in option:
                input_types.append(option["$ref"])

        print(f"Input supports types: {input_types}")
        assert "string" in input_types, "Input should support string"
        assert "array" in input_types, "Input should support array"

    def test_content_variants_are_identified_by_their_type_field(self, spec_dict):
        """Verify a Content part can be told apart by its `type`, however the spec spells that.

        Our transformation reads `type` off each content part to route it, so what has to hold is
        that every variant of the union pins a distinct `type` value and that text is one of them.
        A spec may express that with an OpenAPI `discriminator` on the union or with a `const` on
        each member's own `type`; both are equivalent for us, so accepting only the first makes
        this test fail on a stylistic change upstream that costs us nothing.
        """
        content_schema = spec_dict["components"]["schemas"]["Content"]

        discriminator = content_schema.get("discriminator")
        if discriminator is not None:
            assert (
                discriminator.get("propertyName") == "type"
            ), f"Content is discriminated on {discriminator.get('propertyName')!r}, not 'type'"

        variant_names = [
            option["$ref"].split("/")[-1]
            for option in content_schema.get("oneOf", [])
            if "$ref" in option
        ]
        assert variant_names, f"Content is not a union of named variants: {content_schema}"

        mapping = (discriminator or {}).get("mapping") or {}
        type_values = {
            variant: mapping_value
            for mapping_value, ref in mapping.items()
            for variant in [ref.split("/")[-1]]
        } or {
            variant: _declared_type_value(spec_dict["components"]["schemas"].get(variant, {}))
            for variant in variant_names
        }

        assert set(type_values) == set(variant_names) and all(type_values.values()), (
            f"every Content variant needs a discoverable type value, "
            f"got {type_values} for variants {sorted(variant_names)}"
        )
        assert len(set(type_values.values())) == len(type_values), (
            f"Content variants must pin DISTINCT type values, got {type_values}"
        )
        assert type_values.get("TextContent") == "text", (
            f"TextContent must be reachable as type 'text', got {type_values}"
        )
        print(f"Content variants by type: {type_values}")

    def test_text_content_schema(self, spec_dict):
        """Verify TextContent schema."""
        text_schema = spec_dict["components"]["schemas"]["TextContent"]

        assert "type" in text_schema["required"]
        assert "text" in text_schema["properties"]
        assert text_schema["properties"]["type"].get("const") == "text"
        print("✓ TextContent schema is correct")

    def test_step_schema(self, spec_dict):
        """Verify step-based multi-turn input.

        Google replaced the role-carrying `Turn` schema with typed steps
        (spec update of Aug 13, 2026): conversation history is now a `Step[]`
        where `UserInputStep`/`ModelOutputStep` pin `type` values that our
        transformations read to recover the role. Assert exactly what our code
        depends on: `InteractionsInput` accepts a Step array, both step kinds
        are part of the `Step` union, each pins its `type` const, and each
        carries a `Content[]` content field.
        """
        input_schema = spec_dict["components"]["schemas"]["InteractionsInput"]
        step_array_items = [
            option["items"]["$ref"].split("/")[-1]
            for option in input_schema["oneOf"]
            if option.get("type") == "array" and "$ref" in option.get("items", {})
        ]
        assert "Step" in step_array_items, f"InteractionsInput should accept Step[], got arrays of {step_array_items}"

        step_variants = {
            option["$ref"].split("/")[-1]
            for option in spec_dict["components"]["schemas"]["Step"]["oneOf"]
            if "$ref" in option
        }
        assert {"UserInputStep", "ModelOutputStep"} <= step_variants, f"Step union is missing role steps: {step_variants}"

        for step_name, type_value in [("UserInputStep", "user_input"), ("ModelOutputStep", "model_output")]:
            step_schema = spec_dict["components"]["schemas"][step_name]
            assert step_schema["properties"]["type"].get("const") == type_value
            assert "type" in step_schema["required"]
            content_items = step_schema["properties"]["content"]["items"]
            assert content_items["$ref"].split("/")[-1] == "Content"
            print(f"✓ {step_name} pins type '{type_value}' with Content[] content")


class TestResponseCompliance:
    """Tests that our response types match the OpenAPI spec."""

    def test_interaction_response_fields(self, spec_dict):
        """Verify our InteractionsAPIResponse has correct fields."""
        # The response is the dedicated `Interaction` schema. Google moved the
        # output-only fields (notably the `steps` array, formerly `outputs`)
        # off `CreateModelInteractionParams` and onto `Interaction`; the request
        # schema no longer carries `steps`. Google later moved `role` off
        # `Interaction` onto the per-turn `Turn` schema (asserted in
        # test_turn_schema), so it is no longer a top-level output field here.
        # Keep this aligned with the live spec.
        schema = spec_dict["components"]["schemas"]["Interaction"]

        # Output fields (readOnly). `role` was removed from the `Interaction`
        # schema by Google; it now lives only on `Turn`.
        output_fields = [
            "id",
            "status",
            "created",
            "updated",
            "steps",
            "usage",
        ]

        for field in output_fields:
            assert field in schema["properties"], f"Output field '{field}' not in spec"
            print(f"✓ Output field '{field}' exists in spec")

    def test_status_enum_values(self, spec_dict):
        """Verify status enum values match spec."""
        # `status` is an output-only field; validate against the response schema.
        schema = spec_dict["components"]["schemas"]["Interaction"]
        status_prop = schema["properties"]["status"]
        # Google Interactions API uses lowercase status values (updated Feb 2026).
        # Keep this an exact match: this test intentionally breaks CI when
        # Google changes the live spec — that breakage is how we get notified
        # to review the change.
        expected_statuses = [
            "in_progress",
            "requires_action",
            "completed",
            "failed",
            "cancelled",
            "incomplete",
            "budget_exceeded",
            "queued",
        ]
        assert status_prop["enum"] == expected_statuses
        print(f"✓ Status enum values: {expected_statuses}")

    def test_usage_schema(self, spec_dict):
        """Verify Usage schema fields."""
        usage_schema = spec_dict["components"]["schemas"]["Usage"]

        # Key usage fields
        expected_fields = ["total_input_tokens", "total_output_tokens", "total_tokens"]

        for field in expected_fields:
            assert (
                field in usage_schema["properties"]
            ), f"Usage field '{field}' not in spec"
            print(f"✓ Usage field '{field}' exists")


class TestToolsCompliance:
    """Tests that our tool types match the OpenAPI spec."""

    def test_tool_schema(self, spec_dict):
        """Verify Tool schema."""
        tool_schema = spec_dict["components"]["schemas"]["Tool"]

        # Tool should be oneOf multiple tool types
        assert "oneOf" in tool_schema or "properties" in tool_schema
        print(f"✓ Tool schema found")

    def test_function_declaration_schema(self, spec_dict):
        """Verify FunctionDeclaration schema for function tools."""
        if "FunctionDeclaration" in spec_dict["components"]["schemas"]:
            func_schema = spec_dict["components"]["schemas"]["FunctionDeclaration"]
            assert "name" in func_schema.get(
                "properties", {}
            ) or "name" in func_schema.get("required", [])
            print("✓ FunctionDeclaration schema found")
        else:
            print("⚠ FunctionDeclaration schema not found (may be nested)")


class TestEndpointCompliance:
    """Tests that our endpoints match the OpenAPI spec."""

    def test_create_endpoint_exists(self, spec_dict):
        """Verify POST /interactions endpoint exists."""
        create_path, _ = _interaction_operation(spec_dict, "post")
        print(f"✓ Create endpoint: POST {create_path}")

    def test_get_endpoint_exists(self, spec_dict):
        """Verify GET /interactions/{id} endpoint exists."""
        get_path, _ = _interaction_operation(spec_dict, "get", individual=True)
        print(f"✓ Get endpoint: GET {get_path}")

    def test_delete_endpoint_exists(self, spec_dict):
        """Verify DELETE /interactions/{id} endpoint exists."""
        delete_path, _ = _interaction_operation(spec_dict, "delete", individual=True)
        print(f"✓ Delete endpoint: DELETE {delete_path}")


class TestOperationResolution:
    """Keep structural resolution strict without depending on generated names."""

    @pytest.mark.parametrize("as_union", [False, True])
    def test_model_schema_comes_from_create_operation(self, as_union):
        model_schema: Final = {"properties": {"model": {"type": "string"}}, "required": ["model"]}
        reference: Final = {"$ref": "#/components/schemas/RenamedModelRequest"}
        body_schema: Final = (
            {"oneOf": [{"properties": {"agent": {"type": "string"}}}, reference]} if as_union else reference
        )
        spec: Final = {
            "paths": {
                "/{version}/interactions": {
                    "post": {
                        "requestBody": {"required": True, "content": {"application/json": {"schema": body_schema}}}
                    }
                }
            },
            "components": {
                "schemas": {"RenamedModelRequest": model_schema, "CreateModelInteractionParams": {"properties": {}}}
            },
        }
        assert _model_request_schema(spec) is model_schema

    @pytest.mark.parametrize("method,shared", [("get", False), ("delete", True)])
    def test_item_route_accepts_a_renamed_declared_identifier(self, method, shared):
        parameter: Final = {"name": "renamedId", "in": "path", "required": True, "schema": {"type": "string"}}
        parameters: Final = [{"$ref": "#/components/parameters/Identifier"}]
        operation: Final = {"parameters": [] if shared else parameters}
        path: Final = "/{version}/interactions/{renamedId}"
        spec: Final = {
            "paths": {path: {"parameters": parameters if shared else [], method: operation}},
            "components": {"parameters": {"Identifier": parameter}},
        }
        assert _interaction_operation(spec, method, individual=True) == (path, operation)

    @pytest.mark.parametrize(
        "path,parameter,error",
        [
            (
                "/interactions/{id}/cancel",
                {"required": True, "type": "string"},
                "Expected one GET interactions endpoint",
            ),
            (
                "/other_interactions/{id}",
                {"required": True, "type": "string"},
                "Expected one GET interactions endpoint",
            ),
            ("/interactions/{id}", {"required": False, "type": "string"}, "must require its interaction ID"),
            ("/interactions/{id}", {"required": True, "type": "integer"}, "must accept a string interaction ID"),
        ],
    )
    def test_item_route_rejects_incompatible_contracts(self, path, parameter, error):
        spec: Final = {
            "paths": {
                path: {
                    "get": {
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": parameter["required"],
                                "schema": {"type": parameter["type"]},
                            }
                        ]
                    }
                }
            }
        }
        with pytest.raises(AssertionError, match=error):
            _interaction_operation(spec, "get", individual=True)


if __name__ == "__main__":
    # Quick manual test
    import httpx

    print("Loading OpenAPI spec...")
    response = httpx.get(OPENAPI_SPEC_URL)
    spec = response.json()

    print(f"\nSpec version: {spec.get('openapi')}")
    print(f"API title: {spec.get('info', {}).get('title')}")
    print(f"\nEndpoints:")
    for path, methods in spec.get("paths", {}).items():
        for method in methods:
            if method in ["get", "post", "delete", "put", "patch"]:
                print(f"  {method.upper()} {path}")

    print(
        f"\nSchemas: {list(spec.get('components', {}).get('schemas', {}).keys())[:10]}..."
    )
