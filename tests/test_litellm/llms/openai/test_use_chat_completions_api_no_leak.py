"""
Regression test for issue #28146.

`use_chat_completions_api` is a LiteLLM-internal control flag (it forces the
/responses -> /chat/completions bridge). When set as a model-level param in the
proxy config, it must never be forwarded to the upstream provider's request
body. OpenAI/Anthropic reject unknown body params with HTTP 400.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.types.utils import all_litellm_params
from litellm.utils import get_non_default_completion_params


def test_use_chat_completions_api_is_a_known_litellm_param():
    assert "use_chat_completions_api" in all_litellm_params


def test_use_chat_completions_api_not_forwarded_as_provider_param():
    forwarded = get_non_default_completion_params(
        {"use_chat_completions_api": True, "temperature": 0.5}
    )
    assert "use_chat_completions_api" not in forwarded


def test_completion_does_not_leak_flag_into_provider_request_body():
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {}
    mock_raw_response.parse.return_value = mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = (
        mock_raw_response
    )

    litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        use_chat_completions_api=True,
        api_key="sk-test",
        client=mock_client,
    )

    create_kwargs = (
        mock_client.chat.completions.with_raw_response.create.call_args.kwargs
    )
    assert "use_chat_completions_api" not in create_kwargs
    assert "use_chat_completions_api" not in (create_kwargs.get("extra_body") or {})
