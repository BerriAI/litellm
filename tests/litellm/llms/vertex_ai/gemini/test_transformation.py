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
@pytest.mark.asyncio
async def test_vertex_transformation_field_casing():
    """
    Tests that structural fields are camelCased for Vertex AI,
    while user-defined logic (args, properties, labels) preserves snake_case.
    """
    from litellm.llms.vertex_ai.gemini.transformation import _transform_part_to_httpx_format

    # 1. Test Multimodal field naming
    part = {
        "inline_data": {
            "mime_type": "image/png",
            "data": "base64data"
        }
    }
    transformed = _transform_part_to_httpx_format(part)
    assert "inlineData" in transformed
    assert transformed["inlineData"]["mimeType"] == "image/png"

    # 2. Test functionCall with args preservation
    part = {
        "function_call": {
            "name": "my_func",
            "args": {
                "security_risk": "high",
                "other_param": 123
            }
        }
    }
    transformed = _transform_part_to_httpx_format(part)
    assert "functionCall" in transformed
    assert "args" in transformed["functionCall"]
    # Keys in args should NOT be camelCased
    assert "security_risk" in transformed["functionCall"]["args"]
    assert "other_param" in transformed["functionCall"]["args"]

    # 3. Test tools with functionDeclarations and parameters schema preservation
    part = {
        "tools": [
            {
                "function_declarations": [
                    {
                        "name": "my_func",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "security_risk": {"type": "string", "mime_type": "text/plain"}
                            },
                            "required": ["security_risk"]
                        }
                    }
                ]
            }
        ]
    }
    transformed = _transform_part_to_httpx_format(part)
    schema = transformed["tools"][0]["functionDeclarations"][0]["parameters"]
    # Property name should stay snake_case
    assert "security_risk" in schema["properties"]
    # Internal schema keyword 'mime_type' should become 'mimeType'
    assert "mimeType" in schema["properties"]["security_risk"]
    # 'required' list strings should stay snake_case
    assert "security_risk" in schema["required"]
