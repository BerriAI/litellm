"""
Tests for the Sarvam provider configuration and cost map entries.
"""

import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

SARVAM_MODELS: Final = (
    "sarvam/sarvam-105b",
    "sarvam/sarvam-105b-conversations",
    "sarvam/glm5.3",
    "sarvam/glm5.3-flash",
    "sarvam/gemma4",
    "sarvam/deepseekv4.1-flash-1017",
    "sarvam/deepseekv4.1-flash-1017-interactive",
)


def _sarvam_response(model: str, cached_tokens: int) -> ModelResponse:
    return ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=1000,
            completion_tokens=5,
            total_tokens=1005,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
        ),
    )


def test_sarvam_provider_resolution() -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="sarvam/sarvam-105b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "sarvam-105b"
    assert provider == "sarvam"
    assert api_base == "https://api.sarvam.ai/v2"


def test_sarvam_listed_in_add_model_provider_fields() -> None:
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    entries: Final = json.loads(fields_path.read_text())
    sarvam_entries: Final = [entry for entry in entries if entry["litellm_provider"] == "sarvam"]

    assert len(sarvam_entries) == 1
    entry: Final = sarvam_entries[0]
    assert entry["provider"] == "SARVAM"
    assert entry["provider_display_name"] == "Sarvam"
    assert entry["default_model_placeholder"] == "sarvam/sarvam-105b"
    field_keys: Final = {field["key"] for field in entry["credential_fields"]}
    assert "api_key" in field_keys
    assert "api_base" in field_keys


def test_sarvam_chat_completion_request() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_sarvam",
                "object": "chat.completion",
                "created": 1,
                "model": "sarvam-105b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="sarvam/sarvam-105b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sarvam-test-key",
            max_completion_tokens=200,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.sarvam.ai/v2/chat/completions"
    assert request.headers["authorization"] == "Bearer sarvam-test-key"
    assert body == {
        "model": "sarvam-105b",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 200,
    }
    assert response.choices[0].message.content == "Hello"


def test_sarvam_responses_request_is_bridged_to_chat_completions() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_sarvam_bridge",
                "object": "chat.completion",
                "created": 1,
                "model": "sarvam-105b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="sarvam/sarvam-105b",
            input="Say hello",
            api_key="sarvam-test-key",
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer sarvam-test-key"
    assert json.loads(request.content) == {
        "model": "sarvam-105b",
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    assert response.output_text == "Hello"


@pytest.mark.usefixtures("local_model_cost_map")
class TestSarvamCostMap:
    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_cached_prompt_tokens_cost_less_than_uncached(self, model: str) -> None:
        uncached_cost: Final = litellm.completion_cost(completion_response=_sarvam_response(model, cached_tokens=0))
        cached_cost: Final = litellm.completion_cost(completion_response=_sarvam_response(model, cached_tokens=800))

        assert 0 < cached_cost < uncached_cost

    def test_cached_tokens_bill_at_the_cached_rate(self) -> None:
        cached_tokens: Final = 800
        with respx.mock() as upstream:
            route: Final = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl_sarvam_cache",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "sarvam-105b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 5,
                        "total_tokens": 1005,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
            )
            response: Final = litellm.completion(
                model="sarvam/sarvam-105b",
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="sarvam-test-key",
            )

        rates: Final = litellm.model_cost["sarvam/sarvam-105b"]
        expected: Final = (
            (1000 - cached_tokens) * rates["input_cost_per_token"]
            + cached_tokens * rates["cache_read_input_token_cost"]
            + 5 * rates["output_cost_per_token"]
        )
        assert route.call_count == 1
        assert response.usage.prompt_tokens_details.cached_tokens == cached_tokens
        assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "model",
        (
            "sarvam/deepseekv4.1-flash-1017",
            "sarvam/deepseekv4.1-flash-1017-interactive",
        ),
    )
    def test_deepseek_skus_bill_at_their_own_rates(self, model: str) -> None:
        cached_tokens: Final = 800
        with respx.mock() as upstream:
            route: Final = upstream.post("https://api.sarvam.ai/v2/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl_sarvam_deepseek",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model.split("/", 1)[1],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 5,
                        "total_tokens": 1005,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
            )
            response: Final = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="sarvam-test-key",
            )

        rates: Final = litellm.model_cost[model]
        expected: Final = (
            (1000 - cached_tokens) * rates["input_cost_per_token"]
            + cached_tokens * rates["cache_read_input_token_cost"]
            + 5 * rates["output_cost_per_token"]
        )
        assert route.call_count == 1
        assert response.usage.prompt_tokens_details.cached_tokens == cached_tokens
        assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)

    def test_sarvam_m_kept_for_deprecation(self) -> None:
        # Source: https://docs.sarvam.ai/changelog/2026/6/5.md (2026-06-05)
        entry: Final = litellm.model_cost["sarvam/sarvam-m"]
        assert entry["deprecation_date"] == "2026-06-05"
        assert entry["cache_creation_input_token_cost_above_1hr"] == 0
