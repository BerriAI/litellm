import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from openai import OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy.auth.model_checks import get_provider_models

OPTIQ_CLOUD_API_BASE: Final = "https://cloud.mlx-optiq.com/api/v1"

OPTIQ_CLOUD_MODELS: Final = (
    "deepseek-v4.1-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "minimax-m3.1-flash",
)

WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


class _RecordingOptiQCloud:
    def __init__(self, usage: Mapping[str, object]) -> None:
        self.usage: Final = usage
        self.requests: tuple[httpx.Request, ...] = ()

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-optiq-cloud",
                "object": "chat.completion",
                "created": 1234567890,
                "model": json.loads(request.content)["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": dict(self.usage),
            },
        )

    def client(self) -> OpenAI:
        return OpenAI(
            api_key="optiq-cloud-test-key",
            base_url=OPTIQ_CLOUD_API_BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )


def test_prefixed_model_reaches_optiq_cloud_chat_completions_without_the_prefix():
    optiq: Final = _RecordingOptiQCloud({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with optiq.client() as client:
        litellm.completion(
            model="optiq_cloud/glm-5.3-flash",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    assert len(optiq.requests) == 1
    assert str(optiq.requests[0].url) == f"{OPTIQ_CLOUD_API_BASE}/chat/completions"
    assert json.loads(optiq.requests[0].content) == {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_anthropic_messages_request_is_bridged_to_optiq_cloud_chat_completions():
    optiq: Final = _RecordingOptiQCloud({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with optiq.client() as client:
        response: Final = litellm.anthropic.messages.create(
            model="optiq_cloud/glm-5.3-flash",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            client=client,
        )

    assert [str(request.url) for request in optiq.requests] == [f"{OPTIQ_CLOUD_API_BASE}/chat/completions"]
    assert response["content"] == [{"type": "text", "text": "hi"}]


@pytest.mark.parametrize("model", OPTIQ_CLOUD_MODELS)
def test_optiq_cloud_completion_is_charged_at_the_cost_map_rates(model: str):
    usage: Final = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    optiq: Final = _RecordingOptiQCloud(usage)

    with optiq.client() as client:
        response: Final = litellm.completion(
            model=f"optiq_cloud/{model}",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    rates: Final = litellm.model_cost[f"optiq_cloud/{model}"]
    cached_rate: Final = rates.get("cache_read_input_token_cost", rates["input_cost_per_token"])
    expected_cost: Final = (
        600 * rates["input_cost_per_token"] + 400 * cached_rate + 200 * rates["output_cost_per_token"]
    )
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)
    assert expected_cost > 0


@pytest.mark.parametrize("model", OPTIQ_CLOUD_MODELS)
@pytest.mark.parametrize("effort", ("minimal", "low", "xhigh"))
def test_tools_and_reasoning_effort_reach_optiq_cloud_unchanged(model: str, effort: str):
    optiq: Final = _RecordingOptiQCloud({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with optiq.client() as client:
        litellm.completion(
            model=f"optiq_cloud/{model}",
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            reasoning_effort=effort,
            drop_params=True,
            client=client,
        )

    assert json.loads(optiq.requests[0].content) == {
        "model": model,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "reasoning_effort": effort,
    }


@pytest.mark.parametrize("model", OPTIQ_CLOUD_MODELS)
def test_model_info_reports_the_reasoning_support_litellm_routes_with(model: str):
    model_info: Final = litellm.get_model_info(f"optiq_cloud/{model}")

    assert bool(model_info.get("supports_reasoning")) == litellm.utils.supports_reasoning(f"optiq_cloud/{model}")


def test_optiq_cloud_key_and_base_come_from_optiq_cloud_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPTIQ_CLOUD_API_KEY", "optiq-cloud-env-key")
    monkeypatch.delenv("OPTIQ_CLOUD_API_BASE", raising=False)

    assert get_llm_provider(model="optiq_cloud/glm-5.3-flash") == (
        "glm-5.3-flash",
        "optiq_cloud",
        "optiq-cloud-env-key",
        OPTIQ_CLOUD_API_BASE,
    )

    monkeypatch.setenv("OPTIQ_CLOUD_API_BASE", "https://optiq-cloud.internal.example/v1")

    assert get_llm_provider(model="optiq_cloud/glm-5.3-flash")[3] == "https://optiq-cloud.internal.example/v1"


def test_optiq_cloud_wildcard_lists_every_optiq_cloud_catalog_model():
    catalog_models: Final = sorted(
        key for key, row in litellm.model_cost.items() if row.get("litellm_provider") == "optiq_cloud"
    )

    assert catalog_models
    assert sorted(get_provider_models("optiq_cloud") or []) == catalog_models
