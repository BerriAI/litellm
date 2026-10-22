"""
OpenAPI compliance tests for Google Interactions API.

Checks the supported contract against the official OpenAPI snapshot fetched on
2026-09-28 from:
https://ai.google.dev/static/api/interactions.openapi.json

Refresh fixtures/google-interactions.openapi.json explicitly when reviewing
provider contract changes; unit tests must also run offline.

Run with: pytest tests/unit/interactions/test_openapi_compliance.py -v
"""

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, Final

import httpx
import pytest
from openapi_core import OpenAPI
from pytest_socket import disable_socket, enable_socket

OPENAPI_SPEC_URL = "https://ai.google.dev/static/api/interactions.openapi.json"


def _load_openapi_spec_dict() -> Dict[str, Any]:
    spec_path: Final = Path(__file__).parent / "fixtures" / "google-interactions.openapi.json"
    return json.loads(spec_path.read_text(encoding="utf-8"))


def test_spec_loads_without_fetching_the_live_contract() -> None:
    disable_socket()
    try:
        spec: Final = _load_openapi_spec_dict()
    except pytest.skip.Exception:
        pytest.fail("The pinned contract must load offline without skipping")
    finally:
        enable_socket()

    assert "post" in spec["paths"]["/{api_version}/interactions"]


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
        """Verify the model request fields in the pinned provider contract."""
        schema: Final = spec_dict["components"]["schemas"]["ModelInteraction"]
        request_schema: Final = spec_dict["paths"]["/{api_version}/interactions"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        assert {"$ref": "#/components/schemas/ModelInteraction"} in request_schema["oneOf"]

        assert "model" in schema["required"]
        assert "model" in schema["properties"]
        assert "input" in schema["properties"]

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
        """Verify input supports strings and structured content arrays."""
        schema: Final = spec_dict["components"]["schemas"]["ModelInteraction"]
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
            assert discriminator.get("propertyName") == "type", (
                f"Content is discriminated on {discriminator.get('propertyName')!r}, not 'type'"
            )

        variant_names = [
            option["$ref"].split("/")[-1] for option in content_schema.get("oneOf", []) if "$ref" in option
        ]
        assert variant_names, f"Content is not a union of named variants: {content_schema}"

        mapping = (discriminator or {}).get("mapping") or {}
        type_values = {
            variant: mapping_value for mapping_value, ref in mapping.items() for variant in [ref.split("/")[-1]]
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
        assert {"UserInputStep", "ModelOutputStep"} <= step_variants, (
            f"Step union is missing role steps: {step_variants}"
        )

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
            assert field in usage_schema["properties"], f"Usage field '{field}' not in spec"
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
            assert "name" in func_schema.get("properties", {}) or "name" in func_schema.get("required", [])
            print("✓ FunctionDeclaration schema found")
        else:
            print("⚠ FunctionDeclaration schema not found (may be nested)")


class TestEndpointCompliance:
    """Tests that our endpoints match the OpenAPI spec."""

    def test_create_endpoint_exists(self, spec_dict):
        """Verify POST /interactions endpoint exists."""
        assert any(
            path == "/{api_version}/interactions" and "post" in methods for path, methods in spec_dict["paths"].items()
        ), "POST /interactions endpoint not found"

    def test_get_endpoint_exists(self, spec_dict):
        """Verify GET /interactions/{id} endpoint exists."""
        assert any(
            re.fullmatch(r"/\{api_version\}/interactions/\{[^/{}]+\}", path) and "get" in methods
            for path, methods in spec_dict["paths"].items()
        ), "GET /interactions/{id} endpoint not found"

    def test_delete_endpoint_exists(self, spec_dict):
        """Verify DELETE /interactions/{id} endpoint exists."""
        assert any(
            re.fullmatch(r"/\{api_version\}/interactions/\{[^/{}]+\}", path) and "delete" in methods
            for path, methods in spec_dict["paths"].items()
        ), "DELETE /interactions/{id} endpoint not found"


@pytest.mark.parametrize(
    ("check", "suffix"),
    (
        (TestEndpointCompliance().test_create_endpoint_exists, ""),
        (TestEndpointCompliance().test_get_endpoint_exists, "/{id}"),
        (TestEndpointCompliance().test_delete_endpoint_exists, "/{id}"),
    ),
)
@pytest.mark.parametrize("prefix", ("/unrelated/nested", "/other", "/{api_version}/nested", ""))
def test_endpoint_checks_reject_unrelated_paths(
    check: Callable[[Mapping[str, Mapping[str, Mapping[str, object]]]], None], suffix: str, prefix: str
) -> None:
    spec: Final = {"paths": {f"{prefix}/interactions{suffix}": {"post": {}, "get": {}, "delete": {}}}}

    with pytest.raises(AssertionError, match="endpoint not found"):
        check(spec)


@pytest.mark.parametrize("parameter", ("id", "interactionsId", "interaction_id"))
def test_resource_checks_accept_different_parameter_names(parameter: str) -> None:
    spec: Final = {"paths": {f"/{{api_version}}/interactions/{{{parameter}}}": {"get": {}, "delete": {}}}}
    checks: Final = TestEndpointCompliance()

    checks.test_get_endpoint_exists(spec)
    checks.test_delete_endpoint_exists(spec)


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

    print(f"\nSchemas: {list(spec.get('components', {}).get('schemas', {}).keys())[:10]}...")
