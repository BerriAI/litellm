import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.groq.chat.transformation import GroqChatConfig


@pytest.mark.parametrize(
    "model, expected",
    [
        ("compound", "groq/compound"),
        ("compound-mini", "groq/compound-mini"),
        ("llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
        ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    ],
)
def test_get_groq_model_name(model, expected):
    assert GroqChatConfig._get_groq_model_name(model) == expected


@pytest.mark.parametrize("model", ["groq/compound", "groq/compound-mini"])
def test_compound_models_in_backup_cost_map(model):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/32467

    The Groq compound systems must be registered so they route through the groq
    provider and expose metadata.
    """
    json_path = Path(__file__).parents[5] / "litellm" / "model_prices_and_context_window_backup.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} missing from backup JSON"
    assert info["litellm_provider"] == "groq"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 131072
    assert info["max_output_tokens"] == 8192


@pytest.mark.parametrize(
    "requested_model, expected_api_model",
    [
        ("groq/compound", "groq/compound"),
        ("groq/compound-mini", "groq/compound-mini"),
        ("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
    ],
)
def test_compound_model_name_sent_to_groq(requested_model, expected_api_model):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/32467

    Groq exposes the compound systems with the "groq/" prefix as part of the
    actual model id, so LiteLLM must send "groq/compound(-mini)" instead of the
    prefix-stripped "compound(-mini)", which Groq rejects with model_not_found.
    """
    client = HTTPHandler()
    fake_response = httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": expected_api_model,
            "service_tier": "auto",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(client, "post", return_value=fake_response) as mock_post:
        litellm.completion(
            model=requested_model,
            messages=[{"role": "user", "content": "hi"}],
            api_key="fake-key",
            client=client,
        )

    sent_body = json.loads(mock_post.call_args.kwargs["data"])
    assert sent_body["model"] == expected_api_model
