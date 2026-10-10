"""
Simple unit tests for CustomOpenAPISpec class.

Tests basic functionality of OpenAPI schema generation.
"""

from unittest.mock import Mock, patch

import pytest

from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec


class TestCustomOpenAPISpec:
    """Test suite for CustomOpenAPISpec class."""

    @pytest.fixture
    def base_openapi_schema(self):
        """Base OpenAPI schema for testing."""
        return {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/v1/chat/completions": {"post": {"summary": "Chat completions"}},
                "/v1/embeddings": {"post": {"summary": "Embeddings"}},
                "/v1/responses": {"post": {"summary": "Responses API"}},
            },
        }

    @patch(
        "litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_request_schema"
    )
    def test_add_chat_completion_request_schema(
        self, mock_add_schema, base_openapi_schema
    ):
        """Test that chat completion schema is added correctly."""
        mock_add_schema.return_value = base_openapi_schema

        with patch("litellm.proxy._types.ProxyChatCompletionRequest") as mock_model:
            result = CustomOpenAPISpec.add_chat_completion_request_schema(
                base_openapi_schema
            )

            mock_add_schema.assert_called_once_with(
                openapi_schema=base_openapi_schema,
                model_class=mock_model,
                schema_name="ProxyChatCompletionRequest",
                paths=CustomOpenAPISpec.CHAT_COMPLETION_PATHS,
                operation_name="chat completion",
            )
            assert result == base_openapi_schema

    @patch(
        "litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_request_schema"
    )
    def test_add_embedding_request_schema(self, mock_add_schema, base_openapi_schema):
        """Test that embedding schema is added correctly."""
        mock_add_schema.return_value = base_openapi_schema

        with patch("litellm.types.embedding.EmbeddingRequest") as mock_model:
            result = CustomOpenAPISpec.add_embedding_request_schema(base_openapi_schema)

            mock_add_schema.assert_called_once_with(
                openapi_schema=base_openapi_schema,
                model_class=mock_model,
                schema_name="EmbeddingRequest",
                paths=CustomOpenAPISpec.EMBEDDING_PATHS,
                operation_name="embedding",
            )
            assert result == base_openapi_schema

    @patch(
        "litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_request_schema"
    )
    def test_add_responses_api_request_schema(
        self, mock_add_schema, base_openapi_schema
    ):
        """Test that responses API schema is added correctly."""
        mock_add_schema.return_value = base_openapi_schema

        with patch("litellm.types.llms.openai.ResponsesAPIRequestParams") as mock_model:
            result = CustomOpenAPISpec.add_responses_api_request_schema(
                base_openapi_schema
            )

            mock_add_schema.assert_called_once_with(
                openapi_schema=base_openapi_schema,
                model_class=mock_model,
                schema_name="ResponsesAPIRequestParams",
                paths=CustomOpenAPISpec.RESPONSES_API_PATHS,
                operation_name="responses API",
            )
            assert result == base_openapi_schema


def test_defs_rewritten_in_add_schema_to_components():
    """
    Test that defs are rewritten to components/schemas in add_schema_to_components.
    """

    openapi_schema = {}
    schema_name = "SchemaName"
    schema_def = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"$ref": "#/$defs/UserMessage"},
                        {"$ref": "#/$defs/AssistantMessage"},
                    ]
                },
            }
        },
        "$defs": {
            "UserMessage": {"type": "object"},
            "AssistantMessage": {"type": "object"},
        },
    }
    CustomOpenAPISpec.add_schema_to_components(
        openapi_schema=openapi_schema, schema_name=schema_name, schema_def=schema_def
    )
    assert "$defs" not in openapi_schema
    assert (
        openapi_schema["components"]["schemas"]["SchemaName"]["properties"]["messages"][
            "items"
        ]["anyOf"][0]["$ref"]
        == "#/components/schemas/UserMessage"
    )
    assert (
        openapi_schema["components"]["schemas"]["SchemaName"]["properties"]["messages"][
            "items"
        ]["anyOf"][1]["$ref"]
        == "#/components/schemas/AssistantMessage"
    )


def test_move_defs_to_components():
    """
    Test that $defs from Pydantic v2 schemas are moved to components/schemas.
    """
    openapi_schema = {}

    defs = {
        "UserMessage": {
            "type": "object",
            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
        },
        "AssistantMessage": {
            "type": "object",
            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
        },
    }

    CustomOpenAPISpec._move_defs_to_components(openapi_schema=openapi_schema, defs=defs, namespace="Req")

    assert "components" in openapi_schema
    assert "schemas" in openapi_schema["components"]
    assert "UserMessage" in openapi_schema["components"]["schemas"]
    assert "AssistantMessage" in openapi_schema["components"]["schemas"]
    assert openapi_schema["components"]["schemas"]["UserMessage"]["type"] == "object"


def test_rewrite_defs_refs():
    """
    Test that $ref values are rewritten from #/$defs/ to #/components/schemas/.
    """
    schema = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"$ref": "#/$defs/UserMessage"},
                        {"$ref": "#/$defs/AssistantMessage"},
                    ]
                },
            }
        },
        "$defs": {
            "UserMessage": {"type": "object"},
            "AssistantMessage": {"type": "object"},
        },
    }

    rewritten = CustomOpenAPISpec._rewrite_defs_refs(schema=schema, renames={})

    assert "$defs" not in rewritten
    assert (
        rewritten["properties"]["messages"]["items"]["anyOf"][0]["$ref"]
        == "#/components/schemas/UserMessage"
    )
    assert (
        rewritten["properties"]["messages"]["items"]["anyOf"][1]["$ref"]
        == "#/components/schemas/AssistantMessage"
    )


def test_get_pydantic_schema_generates_schema_for_responses_request_typed_dict():
    from litellm.types.llms.openai import ResponsesAPIRequestParams

    schema = CustomOpenAPISpec.get_pydantic_schema(ResponsesAPIRequestParams)

    assert schema is not None
    properties = schema["properties"]
    assert isinstance(properties, dict)
    for field in (
        "model",
        "input",
        "instructions",
        "tools",
        "previous_response_id",
        "background",
        "stream",
    ):
        assert field in properties


def test_responses_api_paths_covers_all_three_routes():
    assert CustomOpenAPISpec.RESPONSES_API_PATHS == [
        "/v1/responses",
        "/responses",
        "/openai/v1/responses",
    ]


def test_add_schema_to_components_renames_colliding_def_instead_of_overwriting():
    openapi = {
        "components": {
            "schemas": {
                "Message": {"type": "object", "properties": {"content": {"type": "string"}}},
            }
        }
    }

    CustomOpenAPISpec.add_schema_to_components(
        openapi,
        "Req",
        {
            "type": "object",
            "properties": {"m": {"$ref": "#/$defs/Message"}, "n": {"$ref": "#/$defs/Other"}},
            "$defs": {
                "Message": {"type": "object", "properties": {"role": {"type": "string"}}},
                "Other": {"type": "integer"},
            },
        },
    )

    schemas = openapi["components"]["schemas"]
    assert schemas["Message"] == {"type": "object", "properties": {"content": {"type": "string"}}}
    assert schemas["Req_Message"] == {"type": "object", "properties": {"role": {"type": "string"}}}
    assert schemas["Other"] == {"type": "integer"}
    assert schemas["Req"]["properties"]["m"]["$ref"] == "#/components/schemas/Req_Message"
    assert schemas["Req"]["properties"]["n"]["$ref"] == "#/components/schemas/Other"
    assert "$defs" not in schemas["Req"]


def test_add_schema_to_components_keeps_name_for_identical_existing_def():
    openapi = {"components": {"schemas": {"Same": {"type": "integer"}}}}

    CustomOpenAPISpec.add_schema_to_components(
        openapi,
        "Req",
        {
            "type": "object",
            "properties": {"s": {"$ref": "#/$defs/Same"}},
            "$defs": {"Same": {"type": "integer"}},
        },
    )

    schemas = openapi["components"]["schemas"]
    assert "Req_Same" not in schemas
    assert schemas["Req"]["properties"]["s"]["$ref"] == "#/components/schemas/Same"


def test_responses_request_params_schema_requires_model_and_input():
    from litellm.types.llms.openai import ResponsesAPIRequestParams

    schema = CustomOpenAPISpec.get_pydantic_schema(ResponsesAPIRequestParams)

    assert schema is not None
    assert set(schema["required"]) == {"model", "input"}


def test_move_defs_to_components_renames_defs_whose_refs_point_at_renamed_defs():
    openapi = {
        "components": {
            "schemas": {
                "Inner": {"type": "string"},
                "Wrapper": {"$ref": "#/components/schemas/Inner"},
            }
        }
    }

    renames = CustomOpenAPISpec._move_defs_to_components(
        openapi,
        {
            "Inner": {"type": "integer"},
            "Wrapper": {"$ref": "#/$defs/Inner"},
        },
        "NS",
    )

    schemas = openapi["components"]["schemas"]
    assert renames == {"Inner": "NS_Inner", "Wrapper": "NS_Wrapper"}
    assert schemas["Inner"] == {"type": "string"}
    assert schemas["NS_Inner"] == {"type": "integer"}
    assert schemas["Wrapper"] == {"$ref": "#/components/schemas/Inner"}
    assert schemas["NS_Wrapper"] == {"$ref": "#/components/schemas/NS_Inner"}


def test_add_schema_to_components_keeps_name_for_same_shape_existing_def():
    openapi = {
        "components": {
            "schemas": {
                "Block": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "x": {"type": "string"}},
                    "required": ["type", "x"],
                    "additionalProperties": True,
                },
            }
        }
    }

    CustomOpenAPISpec.add_schema_to_components(
        openapi,
        "Req",
        {
            "type": "object",
            "properties": {"b": {"$ref": "#/$defs/Block"}},
            "$defs": {
                "Block": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "x": {"type": "string"}},
                    "required": ["type", "x"],
                },
            },
        },
    )

    schemas = openapi["components"]["schemas"]
    assert "Req_Block" not in schemas
    assert schemas["Block"]["additionalProperties"] is True
    assert schemas["Req"]["properties"]["b"]["$ref"] == "#/components/schemas/Block"


def test_add_schema_to_components_renames_def_with_different_required_set():
    openapi = {
        "components": {
            "schemas": {
                "Block": {
                    "type": "object",
                    "properties": {
                        "keys": {"type": "array"},
                        "type": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"},
                    },
                    "required": ["type", "x", "y"],
                },
            }
        }
    }

    CustomOpenAPISpec.add_schema_to_components(
        openapi,
        "Req",
        {
            "type": "object",
            "properties": {"b": {"$ref": "#/$defs/Block"}},
            "$defs": {
                "Block": {
                    "type": "object",
                    "properties": {
                        "keys": {"type": "array"},
                        "type": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"},
                    },
                    "required": ["keys", "type", "x", "y"],
                },
            },
        },
    )

    schemas = openapi["components"]["schemas"]
    assert schemas["Block"]["required"] == ["type", "x", "y"]
    assert schemas["Req_Block"]["required"] == ["keys", "type", "x", "y"]
    assert schemas["Req"]["properties"]["b"]["$ref"] == "#/components/schemas/Req_Block"
