import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.vertex_ai.gemini import transformation
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

    assert "imageConfig" in rb
    assert rb["imageConfig"] == {"aspectRatio": "16:9"}