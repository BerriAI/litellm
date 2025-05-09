import copy
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

import litellm


def test_update_kwargs_does_not_mutate_defaults_and_merges_metadata():
    # initialize a real Router (env‑vars can be empty)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-3",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
    )

    # override to known defaults for the test
    router.default_litellm_params = {
        "foo": "bar",
        "metadata": {"baz": 123},
    }
    original = copy.deepcopy(router.default_litellm_params)
    kwargs = {}

    # invoke the helper
    router._update_kwargs_with_default_litellm_params(
        kwargs=kwargs,
        metadata_variable_name="litellm_metadata",
    )

    # 1) router.defaults must be unchanged
    assert router.default_litellm_params == original

    # 2) non‑metadata keys get merged
    assert kwargs["foo"] == "bar"

    # 3) metadata lands under "metadata"
    assert kwargs["litellm_metadata"] == {"baz": 123}


def test_router_with_model_info_and_model_group():
    """
    Test edge case where user specifies model_group in model_info
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
                "model_info": {
                    "tpm": 1000,
                    "rpm": 1000,
                    "model_group": "gpt-3.5-turbo",
                },
            }
        ],
    )

    router._set_model_group_info(
        model_group="gpt-3.5-turbo",
        user_facing_model_group_name="gpt-3.5-turbo",
    )


@pytest.mark.asyncio
async def test_router_with_tags_and_fallbacks():
    """
    If fallback model missing tag, raise error
    """
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hello, world!",
                    "tags": ["test"],
                },
            },
            {
                "model_name": "anthropic-claude-3-5-sonnet",
                "litellm_params": {
                    "model": "claude-3-5-sonnet-latest",
                    "mock_response": "Hello, world 2!",
                },
            },
        ],
        fallbacks=[
            {"gpt-3.5-turbo": ["anthropic-claude-3-5-sonnet"]},
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception):
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_testing_fallbacks=True,
            metadata={"tags": ["test"]},
        )

@pytest.mark.asyncio 
async def test_router_with_tags_and_fallbacks_with_image_url():
    """
    Test if the router 'async_function_with_fallbacks' with image_url are working correctly especially with fallbacks
    For details, please refer to the following issue: https://github.com/BerriAI/litellm/issues/9816
    """
    from litellm import Router
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "azure/gpt-4o",
                    "api_key": "fake-azure-api-key",
                    "api_version": "2023-03-15-preview",
                    "api_base": "https://fake-api-base.azure-api.net/",
                },
            },
            {
                "model_name": "gemini-2.0-flash",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                    "api_key": "fake-gemini-api-key",
                },
            },
        ],
        fallbacks=[
            {"gemini-2.0-flash": ["gpt-4o"]},
        ],
    )

    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    data = {
        "model": "gemini-2.0-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": test_image_url,
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
        "num_retries": 0,
    }

    # to avoid KeyError in router.py::log_retry
    data.setdefault("metadata", {}).update({"model_group": "gemini-2.0-flash"})

    client = AsyncHTTPHandler()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-xxxxxxxxxx",
        "model": "gemini-2.0-flash",
        "choices": [
            {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "test response",
            },
            }
        ],
    }
    mock_response.status_code = 200

    first_call_error = Exception("mock gemini-2.0-flash request failed")

    with patch.object(client, "post", side_effect=[first_call_error, mock_response]) as mock_client:
        try:
            response = await router.async_function_with_fallbacks(
                original_function=router._acompletion,
                **data,
                client=client,
            )
        except Exception as e:
            print(e)

        # verify mock is called
        assert mock_client.call_count > 0, "Mock client was not called"

        json_data = mock_client.call_args_list[0].kwargs.get('json', {})
        image_data = json_data.get('contents', [{}])[0].get('parts', [{}])[0].get('inline_data', {}).get('data')

        # to verify the image data of the first request is changed
        assert image_data != test_image_url

        # to verify th
        assert data.get('messages')[0].get('content')[0].get('image_url').get('url') == test_image_url
