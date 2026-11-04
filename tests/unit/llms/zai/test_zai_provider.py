"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math
from collections.abc import Mapping
from typing import Final

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token
from litellm.llms.zai.chat.transformation import ZAIChatConfig
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

JSON_SCHEMA_RESPONSE_FORMAT: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "profile",
        "schema": {"type": "object", "properties": {"favorite_number": {"type": "integer"}}},
    },
}


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.fixture
def zai_response():
    """Mock response from Z.AI API"""
    return {
        "id": "chatcmpl-zai-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_zai():
    """Test that get_llm_provider correctly identifies zai provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("zai/glm-4.6")
    assert model == "glm-4.6"
    assert provider == "zai"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_zai_in_provider_lists():
    """Test that zai is registered in all necessary provider lists"""
    assert "zai" in litellm.openai_compatible_providers
    assert "zai" in litellm.provider_list


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = await litellm.acompletion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.z.ai" in str(request.url)
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


def test_zai_sync_completion(respx_mock, zai_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


def test_zai_sends_response_format_and_maps_max_completion_tokens(
    respx_mock: respx.MockRouter, zai_response: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-5.3",
        messages=[{"role": "user", "content": "Reply in JSON"}],
        response_format={"type": "json_object"},
        max_completion_tokens=64,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body


@pytest.mark.parametrize(
    "token_limits",
    [{"max_tokens": 300, "max_completion_tokens": 15}, {"max_completion_tokens": 15, "max_tokens": 300}],
)
def test_zai_max_completion_tokens_wins_over_max_tokens(token_limits: dict[str, int]) -> None:
    mapped: Final = ZAIChatConfig().map_openai_params(token_limits, {}, "glm-5.3", drop_params=False)

    assert mapped["max_tokens"] == 15
    assert "max_completion_tokens" not in mapped


@pytest.mark.parametrize(
    ("over_limit_by", "drop_params", "forwarded"),
    [(1, True, False), (1, False, True), (0, True, True)],
)
def test_zai_drop_params_drops_max_completion_tokens_above_model_limit(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    over_limit_by: int,
    drop_params: bool,
    forwarded: bool,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)
    requested: Final = litellm.model_cost["zai/glm-4.5-flash"]["max_output_tokens"] + over_limit_by

    completion(
        model="zai/glm-4.5-flash",
        messages=[{"role": "user", "content": "Count"}],
        max_completion_tokens=requested,
        drop_params=drop_params,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body.get("max_tokens") == (requested if forwarded else None)


@pytest.mark.parametrize("with_over_limit_max_completion_tokens", [False, True])
def test_zai_drop_params_keeps_max_tokens_above_model_limit(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    with_over_limit_max_completion_tokens: bool,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)
    over_limit: Final = litellm.model_cost["zai/glm-4.5-flash"]["max_output_tokens"] + 1
    over_limit_max_completion_tokens: Final = (
        {"max_completion_tokens": over_limit + 1} if with_over_limit_max_completion_tokens else {}
    )

    completion(
        model="zai/glm-4.5-flash",
        messages=[{"role": "user", "content": "Count"}],
        max_tokens=over_limit,
        drop_params=True,
        **over_limit_max_completion_tokens,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["max_tokens"] == over_limit


@pytest.mark.parametrize("pass_through", [{}, {"allowed_openai_params": ["response_format"]}])
def test_zai_forwards_json_schema_to_models_that_enforce_it(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    pass_through: dict[str, list[str]],
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)
    messages: Final = [{"role": "user", "content": "Ann likes 7"}]

    completion(model="zai/glm-5.3", messages=messages, response_format=JSON_SCHEMA_RESPONSE_FORMAT, **pass_through)

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["response_format"] == JSON_SCHEMA_RESPONSE_FORMAT
    assert body["messages"] == messages


@pytest.mark.parametrize("pass_through", [{}, {"allowed_openai_params": ["response_format"]}])
def test_zai_puts_json_schema_in_prompt_for_models_that_ignore_it(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    pass_through: dict[str, list[str]],
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Ann likes 7"}],
        response_format=JSON_SCHEMA_RESPONSE_FORMAT,
        **pass_through,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "user", "content": "Ann likes 7"}
    assert body["messages"][-1]["role"] == "user"
    prompt: Final = body["messages"][-1]["content"]
    rendered_schema: Final = prompt[prompt.index("{") : prompt.rindex("}") + 1]
    assert json.loads(rendered_schema) == JSON_SCHEMA_RESPONSE_FORMAT["json_schema"]["schema"]


def test_zai_glm_5v_turbo_bills_cached_input_at_cache_read_price(local_model_cost_map: None) -> None:
    usage: Final = Usage(
        prompt_tokens=1000,
        completion_tokens=0,
        total_tokens=1000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800),
    )

    prompt_cost, _ = cost_per_token(model="glm-5v-turbo", custom_llm_provider="zai", usage_object=usage)

    entry: Final = litellm.model_cost["zai/glm-5v-turbo"]
    assert math.isclose(
        prompt_cost, 200 * entry["input_cost_per_token"] + 800 * entry["cache_read_input_token_cost"]
    )


@pytest.mark.parametrize("model", ["glm-5v-turbo", "GLM-5V-Turbo", "GLM-5.3", "GLM-5.3-Flash"])
def test_zai_cost_resolves_case_insensitively(local_model_cost_map: None, model: str) -> None:
    prompt_cost, completion_cost = cost_per_token(
        model=model, custom_llm_provider="zai", prompt_tokens=1000, completion_tokens=1000
    )

    entry: Final = litellm.model_cost[f"zai/{model.lower()}"]
    assert math.isclose(prompt_cost, 1000 * entry["input_cost_per_token"])
    assert math.isclose(completion_cost, 1000 * entry["output_cost_per_token"])


def test_zai_glm_5v_turbo_accepts_thinking(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-5v-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "disabled"},
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["thinking"] == {"type": "disabled"}
