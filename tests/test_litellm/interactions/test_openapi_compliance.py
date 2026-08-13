"""
OpenAPI compliance tests for Google Interactions API.

Validates that our SDK requests/responses match the OpenAPI spec at:
https://ai.google.dev/static/api/interactions.openapi.json

These assertions run against the pinned copy of that spec sitting next to this
file, so they never depend on the network and never change meaning under an
unrelated PR. Refresh the copy deliberately:

    curl -sSfo tests/test_litellm/interactions/interactions.openapi.json \
        https://ai.google.dev/static/api/interactions.openapi.json

then fix whatever the assertions catch in the same PR. That review of the diff
is the notification this suite exists to give us.

Run with: pytest tests/test_litellm/interactions/test_openapi_compliance.py -v
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from openapi_core import OpenAPI

OPENAPI_SPEC_URL = "https://ai.google.dev/static/api/interactions.openapi.json"
PINNED_SPEC_PATH = Path(__file__).parent / "interactions.openapi.json"


def _load_openapi_spec_dict() -> Dict[str, Any]:
    """Load the pinned copy of the OpenAPI spec."""
    return json.loads(PINNED_SPEC_PATH.read_text())


def _declared_type_value(variant_schema: Dict[str, Any]) -> Any:
    """The single `type` value a union variant pins, whether spelled as a const or a 1-item enum."""
    type_property = variant_schema.get("properties", {}).get("type", {})
    enum_values = type_property.get("enum") or []
    return type_property.get("const") or (enum_values[0] if len(enum_values) == 1 else None)


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
        """Verify CreateModelInteractionParams schema fields."""
        schema = spec_dict["components"]["schemas"]["CreateModelInteractionParams"]

        # Required fields per spec
        assert "model" in schema["required"]
        assert "input" in schema["required"]

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
        schema = spec_dict["components"]["schemas"]["CreateModelInteractionParams"]
        input_schema = schema["properties"]["input"]

        # The input property may be inline oneOf or a $ref to InteractionsInput
        if "$ref" in input_schema:
            ref_name = input_schema["$ref"].split("/")[-1]
            input_schema = spec_dict["components"]["schemas"][ref_name]

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

    def test_multi_turn_input_is_a_list_of_role_tagged_steps(self, spec_dict):
        """Verify a multi-turn conversation can be sent as alternating user/model steps.

        Google dropped the `Turn` schema that used to carry an explicit `role`; a turn is now
        a `Step`, and the role is the step's own `type` value. What our transformation needs is
        that the input accepts a list of steps and that user and model steps stay distinguishable
        and each carry content.
        """
        schemas = spec_dict["components"]["schemas"]

        step_list_variants = [
            variant
            for variant in schemas["InteractionsInput"]["oneOf"]
            if variant.get("type") == "array"
            and variant.get("items", {}).get("$ref", "").endswith("/Step")
        ]
        assert step_list_variants, (
            f"input no longer accepts a list of steps: {schemas['InteractionsInput']['oneOf']}"
        )

        step_variants = {
            option["$ref"].split("/")[-1]
            for option in schemas["Step"].get("oneOf", [])
            if "$ref" in option
        }
        assert {"UserInputStep", "ModelOutputStep"} <= step_variants, (
            f"Step must cover both conversation roles, got {sorted(step_variants)}"
        )

        roles_by_step = {
            step: _declared_type_value(schemas[step])
            for step in ("UserInputStep", "ModelOutputStep")
        }
        assert roles_by_step == {
            "UserInputStep": "user_input",
            "ModelOutputStep": "model_output",
        }, f"conversation roles are no longer reachable by step type, got {roles_by_step}"

        for step in roles_by_step:
            content_prop = schemas[step]["properties"]["content"]
            assert content_prop["type"] == "array", f"{step}.content is not a list"
            assert content_prop["items"]["$ref"].endswith(
                "/Content"
            ), f"{step}.content does not hold Content parts"

        print(f"✓ Multi-turn input is a list of steps, roles: {roles_by_step}")


class TestResponseCompliance:
    """Tests that our response types match the OpenAPI spec."""

    def test_interaction_response_fields(self, spec_dict):
        """Verify our InteractionsAPIResponse has correct fields."""
        # The response is the dedicated `Interaction` schema. Google moved the
        # output-only fields (notably the `steps` array, formerly `outputs`)
        # off `CreateModelInteractionParams` and onto `Interaction`; the request
        # schema no longer carries `steps`. `role` is not an output field here
        # either: it is carried by each step's own `type` (asserted in
        # test_multi_turn_input_is_a_list_of_role_tagged_steps).
        schema = spec_dict["components"]["schemas"]["Interaction"]

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
        paths = spec_dict["paths"]

        # Find the create interactions endpoint
        create_path = None
        for path, methods in paths.items():
            if "interactions" in path and "post" in methods:
                create_path = path
                break

        assert create_path is not None, "POST /interactions endpoint not found"
        print(f"✓ Create endpoint: POST {create_path}")

    def test_get_endpoint_exists(self, spec_dict):
        """Verify GET /interactions/{id} endpoint exists."""
        paths = spec_dict["paths"]

        get_path = None
        for path, methods in paths.items():
            if "{id}" in path and "interactions" in path and "get" in methods:
                get_path = path
                break

        assert get_path is not None, "GET /interactions/{id} endpoint not found"
        print(f"✓ Get endpoint: GET {get_path}")

    def test_delete_endpoint_exists(self, spec_dict):
        """Verify DELETE /interactions/{id} endpoint exists."""
        paths = spec_dict["paths"]

        delete_path = None
        for path, methods in paths.items():
            if "{id}" in path and "interactions" in path and "delete" in methods:
                delete_path = path
                break

        assert delete_path is not None, "DELETE /interactions/{id} endpoint not found"
        print(f"✓ Delete endpoint: DELETE {delete_path}")


if __name__ == "__main__":
    # Quick manual test
    print(f"Loading pinned OpenAPI spec ({PINNED_SPEC_PATH})...")
    spec = _load_openapi_spec_dict()

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
