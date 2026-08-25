
import json
from collections.abc import Mapping
from typing import Optional

import pytest

import litellm
from litellm.llms.vertex_ai.gemini import transformation
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
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
    optional_params = {"labels": {"lparam1": "lvalue1", "lparam2": "lvalue2"}}
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
    assert rb["contents"] == [
        {"parts": [{"text": "hi"}], "role": "user"},
        {"parts": [{"text": "Hello! How can I assist you today?"}], "role": "model"},
        {"parts": [{"text": "hi"}], "role": "user"},
    ]
    assert "labels" in rb and rb["labels"] == {
        "lparam1": "lvalue1",
        "lparam2": "lvalue2",
    }


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
        "metadata": {"requester_metadata": {"rparam1": "rvalue1", "rparam2": "rvalue2"}}
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
    assert rb["contents"] == [
        {"parts": [{"text": "hi"}], "role": "user"},
        {"parts": [{"text": "Hello! How can I assist you today?"}], "role": "model"},
        {"parts": [{"text": "hi"}], "role": "user"},
    ]
    assert "labels" in rb and rb["labels"] == {
        "rparam1": "rvalue1",
        "rparam2": "rvalue2",
    }


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
    optional_params = {"labels": {"lparam1": "lvalue1", "lparam2": "lvalue2"}}
    litellm_params = {
        "metadata": {"requester_metadata": {"rparam1": "rvalue1", "rparam2": "rvalue2"}}
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
    assert rb["contents"] == [
        {"parts": [{"text": "hi"}], "role": "user"},
        {"parts": [{"text": "Hello! How can I assist you today?"}], "role": "model"},
        {"parts": [{"text": "hi"}], "role": "user"},
    ]
    assert "labels" in rb and rb["labels"] == {
        "lparam1": "lvalue1",
        "lparam2": "lvalue2",
    }


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
                    "text": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
                }
            ],
        }
    ]
    optional_params = {
        "imageConfig": {"aspectRatio": "16:9"},
        "responseModalities": ["Image"],
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
                    "text": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
                }
            ],
        }
    ]
    optional_params = {"image_config": {"aspect_ratio": "16:9"}}
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
            ],
        }
    ]
    optional_params = {
        "imageConfig": {"aspectRatio": "16:9", "imageSize": "4K"},
        "responseModalities": ["Image"],
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


def test__transform_request_body_google_maps_json_schema_uses_response_format():
    """googleMaps + JSON schema must use responseFormat, not response_mime_type."""
    messages = [{"role": "user", "content": "Find restaurants in Mumbai"}]
    schema = {
        "type": "object",
        "properties": {"places": {"type": "array"}},
        "required": ["places"],
    }
    optional_params = {
        "tools": [{"googleMaps": {}}],
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    transform_request_params = {
        "messages": messages,
        "model": "gemini/gemini-3.1-flash-lite",
        "optional_params": optional_params,
        "custom_llm_provider": "gemini",
        "litellm_params": {},
        "cached_content": None,
    }

    rb: RequestBody = transformation._transform_request_body(**transform_request_params)

    gen = rb["generationConfig"]
    assert "responseFormat" in gen
    assert gen["responseFormat"]["text"]["mimeType"] == "APPLICATION_JSON"
    assert gen["responseFormat"]["text"]["schema"] == schema
    assert "response_mime_type" not in gen
    assert "response_json_schema" not in gen


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

    tools = [
        {
            "google_search_retrieval": {
                "dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}
            }
        }
    ]
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


RESPONSE_SCHEMA_CHANNEL_CLIENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "total": {"type": "number"},
        "barcode": {"anyOf": [{"type": "string", "maxLength": 10}, {"type": "null"}]},
    },
    "required": ["total", "barcode"],
}


def _gemini_request_body(
    model: str,
    litellm_params: Mapping[str, bool],
    request_override: Optional[bool] = None,
) -> RequestBody:
    override_kwargs: dict[str, bool] = (
        {} if request_override is None else {"vertex_ai_use_response_json_schema": request_override}
    )
    optional_params = litellm.utils.get_optional_params(
        model=model,
        custom_llm_provider="vertex_ai",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice",
                "schema": json.loads(json.dumps(RESPONSE_SCHEMA_CHANNEL_CLIENT_SCHEMA)),
            },
        },
        **override_kwargs,
    )
    return transformation._transform_request_body(
        messages=[{"role": "user", "content": "extract it"}],
        model=model,
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=dict(litellm_params),
        cached_content=None,
    )


def test__transform_request_body_per_request_response_json_schema_opt_out():
    """
    vertex_ai_use_response_json_schema=False on the request puts the schema on Vertex's native
    responseSchema channel, and the knob itself never reaches the provider body
    """
    body = _gemini_request_body("gemini-2.5-flash", {}, request_override=False)

    generation_config = body["generationConfig"]
    assert "response_json_schema" not in generation_config
    assert generation_config["response_schema"]["propertyOrdering"] == ["total", "barcode"]
    assert generation_config["response_schema"]["properties"]["barcode"]["anyOf"] == [
        {"type": "string", "maxLength": 10, "nullable": True}
    ]
    assert "vertex_ai_use_response_json_schema" not in json.dumps(body)
    assert "litellm_param" not in json.dumps(body)


def test__transform_request_body_deployment_response_json_schema_opt_out():
    """A deployment's litellm_params opts every request routed to it out of responseJsonSchema"""
    body = _gemini_request_body("gemini-2.5-flash", {"vertex_ai_use_response_json_schema": False})

    generation_config = body["generationConfig"]
    assert "response_json_schema" not in generation_config
    assert generation_config["response_schema"]["propertyOrdering"] == ["total", "barcode"]


def test__transform_request_body_per_request_opt_in_beats_global_opt_out(monkeypatch):
    """
    With the global opted out, vertex_ai_use_response_json_schema=True on the request sends the
    client schema verbatim again, additionalProperties included
    """
    monkeypatch.setattr(litellm, "vertex_ai_use_response_json_schema", False)

    body = _gemini_request_body("gemini-2.5-flash", {}, request_override=True)

    generation_config = body["generationConfig"]
    assert "response_schema" not in generation_config
    assert generation_config["response_json_schema"] == RESPONSE_SCHEMA_CHANNEL_CLIENT_SCHEMA
    assert "litellm_param" not in json.dumps(body)


def test__transform_request_body_keeps_response_json_schema_by_default():
    """Without any override, Gemini 2.x keeps sending the verbatim responseJsonSchema"""
    body = _gemini_request_body("gemini-2.5-flash", {})

    generation_config = body["generationConfig"]
    assert "response_schema" not in generation_config
    assert generation_config["response_json_schema"] == RESPONSE_SCHEMA_CHANNEL_CLIENT_SCHEMA
