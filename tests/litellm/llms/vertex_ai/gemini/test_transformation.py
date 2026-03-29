import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.vertex_ai.gemini import transformation
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms import openai
from litellm.types import completion
from litellm.types.llms.vertex_ai import RequestBody

@pytest.mark.asyncio
async def test__transform_request_body_labels():
    """
    Test that Vertex AI requests use the optional Vertex AI
    "labels" parameters sent by client.
    """

    # Set up the test parameters
    model = "vertex_ai/gemini-1.5-pro"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {
        "labels": {"lparam1": "lvalue1", "lparam2": "lvalue2"}
    }
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "vertex_ai",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    # Check URL
    assert rb["contents"] == [{'parts': [{'text': 'hi'}], 'role': 'user'}, {'parts': [{'text': 'Hello! How can I assist you today?'}], 'role': 'model'}, {'parts': [{'text': 'hi'}], 'role': 'user'}]
    assert "labels" in rb and rb["labels"] == {"lparam1": "lvalue1", "lparam2": "lvalue2"}

@pytest.mark.asyncio
async def test__transform_request_body_metadata():
    """
    Test that Vertex AI requests use the optional Open AI
    "metadata" parameters sent by client.
    """

    # Set up the test parameters
    model = "vertex_ai/gemini-1.5-pro"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {}
    litellm_params = {
        "metadata": {
            "requester_metadata": {"rparam1": "rvalue1", "rparam2": "rvalue2"}
        }
    }
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "vertex_ai",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    # Check URL
    assert rb["contents"] == [{'parts': [{'text': 'hi'}], 'role': 'user'}, {'parts': [{'text': 'Hello! How can I assist you today?'}], 'role': 'model'}, {'parts': [{'text': 'hi'}], 'role': 'user'}]
    assert "labels" in rb and rb["labels"] == {"rparam1": "rvalue1", "rparam2": "rvalue2"}

@pytest.mark.asyncio
async def test__transform_request_body_labels_and_metadata():
    """
    Test that Vertex AI requests use the optional Vertex AI
    "labels" parameters sent by client and that the "metadata"
    optional Open AI parameters are ignored if the client uses
    "labels" parameters.
    """

    # Set up the test parameters
    model = "vertex_ai/gemini-1.5-pro"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]
    optional_params = {
        "labels": {"lparam1": "lvalue1", "lparam2": "lvalue2"}
    }
    litellm_params = {
        "metadata": {
            "requester_metadata": {"rparam1": "rvalue1", "rparam2": "rvalue2"}
        }
    }
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "vertex_ai",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    # Check URL
    assert rb["contents"] == [{'parts': [{'text': 'hi'}], 'role': 'user'}, {'parts': [{'text': 'Hello! How can I assist you today?'}], 'role': 'model'}, {'parts': [{'text': 'hi'}], 'role': 'user'}]
    assert "labels" in rb and rb["labels"] == {"lparam1": "lvalue1", "lparam2": "lvalue2"}

@pytest.mark.asyncio
async def test__transform_request_body_image_config():
    """
    Test that Vertex AI Gemini supports the imageConfig parameter for gemini-2.5-flash-image model.
    """
    model = "gemini-2.5-flash-image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme"
                }
            ]
        }
    ]
    optional_params = {
        "imageConfig": {"aspectRatio": "16:9"},
        "responseModalities": ["Image"]
    }
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    assert "generationConfig" in rb
    assert "imageConfig" in rb["generationConfig"]
    assert rb["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}


@pytest.mark.asyncio
async def test__transform_request_body_image_config_snake_case():
    """
    Test that Vertex AI Gemini supports the image_config parameter (snake_case) for gemini-2.5-flash-image model.
    This should be transformed to imageConfig with aspectRatio.
    """
    model = "gemini-2.5-flash-image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme"
                }
            ]
        }
    ]
    optional_params = {
        "image_config": {"aspect_ratio": "16:9"}
    }
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    assert "generationConfig" in rb
    assert "image_config" in rb["generationConfig"]
    assert rb["generationConfig"]["image_config"] == {"aspect_ratio": "16:9"}


@pytest.mark.asyncio
async def test__transform_request_body_image_config_with_image_size():
    """Test imageSize parameter support in imageConfig"""
    model = "gemini-3-pro-image-preview"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate a 4K image of Tokyo skyline"}
            ]
        }
    ]
    optional_params = {
        "imageConfig": {"aspectRatio": "16:9", "imageSize": "4K"},
        "responseModalities": ["Image"]
    }
    litellm_params = {}
    transform_request_params = {
        "messages": messages,
        "model": model,
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": litellm_params,
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    assert "generationConfig" in rb
    assert "imageConfig" in rb["generationConfig"]
    assert rb["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    assert rb["generationConfig"]["imageConfig"]["imageSize"] == "4K"


def test_map_function_google_search_snake_case():
    """
    Test that google_search tool (snake_case) is properly mapped to googleSearch.
    Fixes issue where tools=[{"google_search": {}}] was being stripped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    # Test snake_case google_search
    tools = [{"google_search": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearch" in result[0]
    assert result[0]["googleSearch"] == {}


def test_map_function_google_search_camel_case():
    """
    Test that googleSearch tool (camelCase) still works.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    # Test camelCase googleSearch
    tools = [{"googleSearch": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearch" in result[0]
    assert result[0]["googleSearch"] == {}


def test_map_function_google_search_retrieval_snake_case():
    """
    Test that google_search_retrieval tool (snake_case) is properly mapped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearchRetrieval" in result[0]


def test_map_function_enterprise_web_search_snake_case():
    """
    Test that enterprise_web_search tool (snake_case) is properly mapped.
    """
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"enterprise_web_search": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "enterpriseWebSearch" in result[0]


# ---------------------------------------------------------------------------
# Tests for fix of issue #24399:
#   LiteLLM Vertex AI Gemini Multimodal Data Drop and Schema Mismatch
#
# The Vertex AI REST API requires camelCase field names in the JSON payload
# (e.g. ``inlineData``, ``functionResponse``, ``mimeType``), but LiteLLM's
# internal PartType TypedDict uses snake_case (``inline_data``,
# ``function_response``, ``mime_type``).  Without conversion the API silently
# drops multimodal parts and returns schema-mismatch errors.
#
# A blanket recursive camelCase conversion (the previous workaround) broke
# user-defined identifiers inside tool arguments and JSON-schema properties
# (e.g. ``security_risk`` was incorrectly renamed to ``securityRisk``).
#
# The fix introduces ``_convert_part_to_vertex_httpx_format`` which converts
# only the known structural Gemini field names and leaves user content alone.
# ---------------------------------------------------------------------------

from litellm.llms.vertex_ai.gemini.transformation import (
    _convert_part_to_vertex_httpx_format,
    _convert_contents_to_vertex_format,
)


class TestConvertPartToVertexHttpxFormat:
    """Unit-level coverage for the part-level field-name converter."""

    def test_inline_data_renamed(self):
        """inline_data key becomes inlineData and mime_type becomes mimeType."""
        part = {
            "inline_data": {
                "mime_type": "image/png",
                "data": "base64encodeddata",
            }
        }
        result = _convert_part_to_vertex_httpx_format(part)

        assert "inlineData" in result, "inline_data must be renamed to inlineData"
        assert "inline_data" not in result, "old snake_case key must be absent"
        assert result["inlineData"]["mimeType"] == "image/png", (
            "mime_type inside blob must become mimeType"
        )
        assert result["inlineData"]["data"] == "base64encodeddata"
        assert "mime_type" not in result["inlineData"]

    def test_file_data_renamed(self):
        """file_data key becomes fileData with mimeType and fileUri."""
        part = {
            "file_data": {
                "mime_type": "application/pdf",
                "file_uri": "gs://bucket/doc.pdf",
            }
        }
        result = _convert_part_to_vertex_httpx_format(part)

        assert "fileData" in result
        assert "file_data" not in result
        assert result["fileData"]["mimeType"] == "application/pdf"
        assert result["fileData"]["fileUri"] == "gs://bucket/doc.pdf"
        assert "mime_type" not in result["fileData"]
        assert "file_uri" not in result["fileData"]

    def test_function_call_renamed(self):
        """function_call key becomes functionCall; args payload is untouched."""
        user_args = {"security_risk": "high", "count": 3}
        part = {"function_call": {"name": "assess_risk", "args": user_args}}
        result = _convert_part_to_vertex_httpx_format(part)

        assert "functionCall" in result
        assert "function_call" not in result
        # User-defined arg names must NOT be camelCased
        assert result["functionCall"]["args"]["security_risk"] == "high", (
            "user-defined arg key security_risk must not be renamed to securityRisk"
        )

    def test_function_response_renamed(self):
        """function_response key becomes functionResponse; response payload is untouched."""
        user_response = {"security_risk": "low", "details": "all clear"}
        part = {
            "function_response": {
                "name": "assess_risk",
                "response": user_response,
            }
        }
        result = _convert_part_to_vertex_httpx_format(part)

        assert "functionResponse" in result
        assert "function_response" not in result
        assert result["functionResponse"]["response"]["security_risk"] == "low", (
            "user-defined response key security_risk must not be renamed"
        )

    def test_media_resolution_renamed(self):
        """media_resolution becomes mediaResolution."""
        part = {"text": "describe this", "media_resolution": "high"}
        result = _convert_part_to_vertex_httpx_format(part)

        assert "mediaResolution" in result
        assert "media_resolution" not in result
        assert result["mediaResolution"] == "high"

    def test_text_part_passthrough(self):
        """Text-only parts are passed through unchanged."""
        part = {"text": "Hello, world!"}
        result = _convert_part_to_vertex_httpx_format(part)
        assert result == {"text": "Hello, world!"}

    def test_thought_and_thought_signature_passthrough(self):
        """Thinking / thought fields that are already camelCase pass through."""
        part = {"thought": True, "thoughtSignature": "abc123", "text": "thinking..."}
        result = _convert_part_to_vertex_httpx_format(part)
        assert result["thought"] is True
        assert result["thoughtSignature"] == "abc123"
        assert result["text"] == "thinking..."

    def test_schema_properties_not_renamed(self):
        """
        User-defined JSON schema field names (like security_risk) must NOT be
        camelCased.  This guards against the regression described in issue #24399
        where a blanket camelCase conversion broke schema validation because the
        ``required`` list (plain strings) no longer matched the renamed
        ``properties`` keys.
        """
        schema_payload = {
            "properties": {
                "security_risk": {"type": "string"},
                "user_id": {"type": "integer"},
            },
            "required": ["security_risk", "user_id"],
        }
        part = {
            "function_response": {
                "name": "validate_schema",
                "response": schema_payload,
            }
        }
        result = _convert_part_to_vertex_httpx_format(part)
        resp = result["functionResponse"]["response"]
        assert "security_risk" in resp["properties"], (
            "user-defined schema property security_risk must not become securityRisk"
        )
        assert "user_id" in resp["properties"]
        assert resp["required"] == ["security_risk", "user_id"]


class TestConvertContentsToVertexFormat:
    def test_multimodal_user_message(self):
        """
        A user message containing both text and an image inline_data part
        should produce a contents list with camelCase structural keys.
        """
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": "What is in this image?"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "base64data",
                        }
                    },
                ],
            }
        ]
        result = _convert_contents_to_vertex_format(contents)

        assert len(result) == 1
        parts = result[0]["parts"]
        assert len(parts) == 2

        assert parts[0] == {"text": "What is in this image?"}

        image_part = parts[1]
        assert "inlineData" in image_part, "inline_data must become inlineData"
        assert image_part["inlineData"]["mimeType"] == "image/png"
        assert image_part["inlineData"]["data"] == "base64data"

    def test_tool_result_with_image(self):
        """
        A tool result message with a function_response and a separate
        inline_data image part must have both parts correctly renamed.
        """
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "take_screenshot",
                            "response": {"url": "https://example.com"},
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "screenshotbase64",
                        }
                    },
                ],
            }
        ]
        result = _convert_contents_to_vertex_format(contents)

        parts = result[0]["parts"]
        assert "functionResponse" in parts[0]
        assert "function_response" not in parts[0]
        assert "inlineData" in parts[1]
        assert parts[1]["inlineData"]["mimeType"] == "image/png"

    def test_role_preserved(self):
        """The role field on each content block must be preserved."""
        contents = [
            {"role": "model", "parts": [{"text": "Sure!"}]},
        ]
        result = _convert_contents_to_vertex_format(contents)
        assert result[0]["role"] == "model"