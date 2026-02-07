import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from respx import MockRouter

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.router import Router


@pytest.mark.asyncio()
@pytest.mark.respx()
async def test_aaaaazure_tenant_id_auth(respx_mock: MockRouter):
    """

    Tests when we set  tenant_id, client_id, client_secret they don't get sent with the request

    PROD Test
    """
    litellm.disable_aiohttp_transport = True # since this uses respx, we need to set use_aiohttp_transport to False
    
    # Clear the HTTP client cache to ensure respx mocking works
    # This is critical because respx only intercepts clients created AFTER mocking is active
    if hasattr(litellm, 'in_memory_llm_clients_cache'):
        litellm.in_memory_llm_clients_cache.flush_cache()

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "tenant_id": os.getenv("AZURE_TENANT_ID"),
                    "client_id": os.getenv("AZURE_CLIENT_ID"),
                    "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
                },
            },
        ],
    )

    mock_response = AsyncMock()
    obj = ChatCompletion(
        id="foo",
        model="gpt-4",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Hello world!",
                    role="assistant",
                ),
            )
        ],
        created=int(datetime.now().timestamp()),
    )
    litellm.set_verbose = True

    mock_request = respx_mock.post(url__regex=r".*/chat/completions.*").mock(
        return_value=httpx.Response(200, json=obj.model_dump(mode="json"))
    )

    await router.acompletion(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}]
    )

    # Ensure all mocks were called
    respx_mock.assert_all_called()

    for call in mock_request.calls:
        print(call)
        print(call.request.content)

        json_body = json.loads(call.request.content)
        print(json_body)

        assert json_body == {
            "messages": [{"role": "user", "content": "Hello world!"}],
            "model": "gpt-4.1-mini",
            "stream": False,
        }
