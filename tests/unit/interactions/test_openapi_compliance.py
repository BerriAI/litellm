"""
OpenAPI compliance tests for Google Interactions API.

Validates that our SDK requests/responses match the OpenAPI spec snapshot in
interactions.openapi.json, taken from https://ai.google.dev/static/api/interactions.openapi.json
(litellm/types/interactions/README.md says how to refresh it). The scheduled
interactions-spec-canary workflow runs the same tests against the live spec by passing
--interactions-spec-url, so a change Google makes shows up there, not in every PR.

Run with: pytest tests/unit/interactions/test_openapi_compliance.py -v
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, get_args
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openapi_core import OpenAPI
from pydantic import BaseModel, TypeAdapter

OPENAPI_SPEC_URL = "https://ai.google.dev/static/api/interactions.openapi.json"
OPENAPI_SPEC_SNAPSHOT: Final = Path(__file__).with_name("interactions.openapi.json")
SPEC_URL_OPTION: Final = TypeAdapter(str | None)


@dataclass(frozen=True, slots=True)
class LoadedSpec:
    source: str
    spec: Dict[str, Any]


def _status_enum_of(model: type[BaseModel]) -> type[Enum]:
    annotation: Final = model.model_fields["status"].annotation
    candidates: Final = (annotation, *get_args(annotation))
    return next(candidate for candidate in candidates if isinstance(candidate, type) and issubclass(candidate, Enum))


def _requested_spec_url(request: pytest.FixtureRequest) -> str | None:
    return SPEC_URL_OPTION.validate_python(request.config.getoption("interactions_spec_url"))


def _load_openapi_spec(live_url: str | None) -> LoadedSpec:
    if live_url is None:
        return LoadedSpec(source=str(OPENAPI_SPEC_SNAPSHOT), spec=json.loads(OPENAPI_SPEC_SNAPSHOT.read_text()))
    with httpx.Client(transport=httpx.HTTPTransport(retries=3), timeout=30.0) as client:
        response: Final = client.get(live_url)
    response.raise_for_status()
    return LoadedSpec(source=str(response.url), spec=response.json())


def _model_create_request_schema(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    schemas = spec_dict["components"]["schemas"]
    create_path = next(path for path in spec_dict["paths"] if path.endswith("/interactions"))
    body_schema = spec_dict["paths"][create_path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    variants = [schemas[option["$ref"].split("/")[-1]] for option in body_schema.get("oneOf", []) if "$ref" in option]
    return next(variant for variant in variants if "model" in variant.get("properties", {}))


def _interaction_resource_path(spec_dict: Dict[str, Any], method: str) -> str | None:
    return next(
        (
            path
            for path, methods in spec_dict["paths"].items()
            if re.search(r"/interactions/\{[^}]+\}$", path) and method in methods
        ),
        None,
    )


def _declared_type_value(variant_schema: Dict[str, Any]) -> Any:
    """The single `type` value a union variant pins, whether spelled as a const or a 1-item enum."""
    type_property = variant_schema.get("properties", {}).get("type", {})
    enum_values = type_property.get("enum") or []
    return type_property.get("const") or (enum_values[0] if len(enum_values) == 1 else None)


@pytest.fixture(scope="module")
def loaded_spec(request: pytest.FixtureRequest) -> LoadedSpec:
    return _load_openapi_spec(_requested_spec_url(request))


@pytest.fixture(scope="module")
def spec_dict(loaded_spec: LoadedSpec) -> Dict[str, Any]:
    return loaded_spec.spec


@pytest.fixture(scope="module")
def openapi_spec(spec_dict: Dict[str, Any]) -> OpenAPI:
    """Load the OpenAPI spec as an OpenAPI object."""
    return OpenAPI.from_dict(spec_dict)


def test_spec_comes_from_the_requested_source(loaded_spec: LoadedSpec, request: pytest.FixtureRequest) -> None:
    assert loaded_spec.source == (_requested_spec_url(request) or str(OPENAPI_SPEC_SNAPSHOT))


def test_interaction_status_export_carries_the_interaction_models_statuses() -> None:
    from litellm.types.interactions import InteractionStatus, generated

    exported: Final = frozenset(member.value for member in InteractionStatus)
    modelled: Final = frozenset(member.value for member in _status_enum_of(generated.Interaction))
    assert exported == modelled, sorted(exported ^ modelled)


class TestRequestCompliance:
    """Tests that our request bodies match the OpenAPI spec."""

    def test_create_model_interaction_request_schema(self, spec_dict):
        schema = _model_create_request_schema(spec_dict)

        assert "model" in schema["required"]
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
        """Verify input field supports string, Content, Content[], Turn[]."""
        schema = _model_create_request_schema(spec_dict)
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

    @pytest.mark.parametrize(
        "schema_name",
        [
            "Interaction",
            "InteractionStatusUpdate",
            "CreateModelInteractionParams",
            "CreateAgentInteractionParams",
            "UrlContextResult",
        ],
    )
    def test_status_enum_values(self, spec_dict, schema_name):
        from litellm.types.interactions import generated

        schema = spec_dict["components"]["schemas"][schema_name]
        spec_statuses: Final = frozenset(schema["properties"]["status"]["enum"])
        parsed_statuses: Final = frozenset(member.value for member in _status_enum_of(getattr(generated, schema_name)))
        assert spec_statuses
        assert spec_statuses <= parsed_statuses, sorted(spec_statuses - parsed_statuses)

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
        get_path = _interaction_resource_path(spec_dict, "get")

        assert get_path is not None, "GET /interactions/{id} endpoint not found"
        print(f"✓ Get endpoint: GET {get_path}")

    def test_delete_endpoint_exists(self, spec_dict):
        """Verify DELETE /interactions/{id} endpoint exists."""
        delete_path = _interaction_resource_path(spec_dict, "delete")

        assert delete_path is not None, "DELETE /interactions/{id} endpoint not found"
        print(f"✓ Delete endpoint: DELETE {delete_path}")


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
