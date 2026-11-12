import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm import completion, cost_per_token, get_llm_provider, model_cost, responses
from litellm.llms.openai_like.dynamic_config import create_config_class, create_responses_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.utils import PromptTokensDetails, Usage


def test_acedatacloud_prefix_resolves_key_and_preserves_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-test-key")
    assert get_llm_provider("acedatacloud/gpt-6-luna") == (
        "gpt-6-luna",
        "acedatacloud",
        "ace-test-key",
        "https://api.acedata.cloud/openai",
    )


def test_acedatacloud_explicit_credentials_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "environment-key")
    assert get_llm_provider(
        "acedatacloud/gpt-6-luna", api_key="explicit-key", api_base="https://example.test/openai"
    ) == ("gpt-6-luna", "acedatacloud", "explicit-key", "https://example.test/openai")


def test_acedatacloud_chat_url_and_reasoning_tools_are_preserved() -> None:
    provider: Final = JSONProviderRegistry.get("acedatacloud")
    assert provider is not None
    config: Final = create_config_class(provider)()
    assert config.get_complete_url(None, "test-key", "gpt-6-luna", {}, {}) == (
        "https://api.acedata.cloud/openai/chat/completions"
    )
    supported: Final = config.get_supported_openai_params("gpt-6-luna")
    assert {"tools", "tool_choice", "reasoning_effort", "stream_options"}.issubset(supported)


def test_acedatacloud_responses_keeps_openai_base_path() -> None:
    provider: Final = JSONProviderRegistry.get("acedatacloud")
    assert provider is not None
    config: Final = create_responses_config_class(provider)()
    assert config.get_complete_url(None, {}) == "https://api.acedata.cloud/openai/responses"


def test_acedatacloud_add_model_fields() -> None:
    fields: Final = json.loads(
        (Path(litellm.__file__).parent / "proxy/public_endpoints/provider_create_fields.json").read_text()
    )
    matches: Final = tuple(field for field in fields if field["litellm_provider"] == "acedatacloud")
    assert len(matches) == 1
    provider: Final = matches[0]
    assert provider["provider"] == "ACEDATACLOUD"
    assert provider["provider_display_name"] == "Ace Data Cloud"
    assert provider["default_model_placeholder"] == "acedatacloud/gpt-6-luna"
    assert tuple((field["key"], field["required"], field["field_type"]) for field in provider["credential_fields"]) == (
        ("api_key", True, "password"),
    )


def test_acedatacloud_chat_request_and_credit_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-test-key")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/openai/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-ace",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6-luna",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                    "cost": {"currency": "credit", "amount": 7},
                },
            },
        )
        response: Final = completion(
            model="acedatacloud/gpt-6-luna",
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            reasoning_effort="none",
            max_completion_tokens=64,
            num_retries=0,
        )
    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer ace-test-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "reasoning_effort": "none",
        "max_completion_tokens": 64,
    }
    assert response.choices[0].message.content == "OK"
    assert "cost" not in response.model_dump(warnings="error")["usage"]
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 5)


def test_acedatacloud_responses_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-test-key")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/openai/responses").respond(
            200,
            json={
                "id": "resp_ace",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-luna",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_ace",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
            },
        )
        response: Final = responses(
            model="acedatacloud/gpt-6-luna",
            input="Reply with exactly OK.",
            reasoning={"effort": "none"},
            max_output_tokens=64,
            num_retries=0,
        )
    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer ace-test-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": "Reply with exactly OK.",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 64,
    }
    assert response.output[0].content[0].text == "OK"
    assert response.model_dump(warnings="error")["usage"]["total_tokens"] == 16


@pytest.mark.parametrize(
    ("prompt_tokens", "suffix"),
    [(272000, ""), (272001, "_above_272k_tokens")],
)
def test_acedatacloud_cost_uses_its_catalog_and_long_prompt_rates(prompt_tokens: int, suffix: str) -> None:
    """272K boundary: public Ace Data Cloud model catalog, verified 2026-10-06.

    https://platform.acedata.cloud/api/v1/models/catalog/
    """
    rates: Final = model_cost["acedatacloud/gpt-6-luna"]
    usage: Final = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
        total_tokens=prompt_tokens + 5,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=10),
    )
    input_cost, output_cost = cost_per_token(
        model="acedatacloud/gpt-6-luna",
        custom_llm_provider="acedatacloud",
        usage_object=usage,
    )
    expected_input: Final = (prompt_tokens - 10) * rates[f"input_cost_per_token{suffix}"] + 10 * rates[
        f"cache_read_input_token_cost{suffix}"
    ]
    assert input_cost == pytest.approx(expected_input)
    assert output_cost == pytest.approx(5 * rates[f"output_cost_per_token{suffix}"])
