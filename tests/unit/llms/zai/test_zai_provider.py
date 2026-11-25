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
from litellm import Router, completion
from litellm.cost_calculator import cost_per_token


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


@pytest.mark.asyncio
async def test_zai_messages_thinking_reaches_glm_as_reasoning_effort(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response: Final = await litellm.anthropic_messages(
        model="zai/glm-5.3",
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "Is 391 prime?"}],
        api_key="test-api-key",
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["reasoning_effort"] in litellm.model_cost["zai/glm-5.3"]["reasoning_effort_levels"]
    assert response["content"][-1]["text"] == "Hello! How can I help you today?"


@pytest.mark.parametrize(
    ("requested", "sent"),
    [("minimal", "low"), ("low", "low"), ("medium", "high"), ("high", "high"), ("xhigh", "max"), ("max", "max")],
)
def test_zai_rounds_reasoning_effort_up_to_a_level_the_model_declares(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    sent: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model="zai/glm-5.3", messages=[{"role": "user", "content": "Hello"}], reasoning_effort=requested)

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == sent


@pytest.mark.parametrize("requested", ["minimal", "medium", "xhigh"])
def test_zai_forwards_reasoning_effort_as_sent_to_models_declaring_no_levels(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model="zai/glm-5.2", messages=[{"role": "user", "content": "Hello"}], reasoning_effort=requested)

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == requested


@pytest.mark.parametrize("drop_params", [True, False])
@pytest.mark.parametrize("model", ["glm-5.1", "glm-5.2"])
def test_zai_turns_thinking_off_for_reasoning_effort_none(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    drop_params: bool,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model=f"zai/{model}",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="none",
        drop_params=drop_params,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert (body.get("thinking"), body.get("reasoning_effort")) == ({"type": "disabled"}, None)


@pytest.mark.parametrize(("drop_params", "sent"), [(True, None), (False, "none")])
@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash"])
def test_zai_drops_reasoning_effort_none_for_models_that_always_think_only_under_drop_params(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    drop_params: bool,
    sent: str | None,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model=f"zai/{model}",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="none",
        drop_params=drop_params,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert (body.get("thinking"), body.get("reasoning_effort")) == (None, sent)


@pytest.mark.parametrize(("drop_params", "sent"), [(True, None), (False, {"type": "disabled"})])
@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash"])
def test_zai_drops_disabled_thinking_for_models_that_always_think_only_under_drop_params(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    drop_params: bool,
    sent: Mapping[str, str] | None,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model=f"zai/{model}",
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "disabled"},
        drop_params=drop_params,
    )

    assert json.loads(respx_mock.calls[0].request.content).get("thinking") == sent


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.2"])
def test_zai_leaves_reasoning_effort_default_to_the_model(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model=f"zai/{model}", messages=[{"role": "user", "content": "Hello"}], reasoning_effort="default")

    assert "reasoning_effort" not in json.loads(respx_mock.calls[0].request.content)


@pytest.mark.parametrize("model", ["zai/glm-registered-levels-test", "zai/GLM-Registered-Levels-Test"])
def test_zai_rounds_reasoning_effort_to_the_levels_register_model_declares(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.register_model(
        {
            "zai/glm-registered-levels-test": {
                "litellm_provider": "zai",
                "mode": "chat",
                "supports_reasoning": True,
                "reasoning_effort_levels": ["low", "high"],
            }
        }
    )
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model=model, messages=[{"role": "user", "content": "Hello"}], reasoning_effort="medium")

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == "high"


@pytest.mark.parametrize(("drop_params", "sent"), [(True, None), (False, "none")])
def test_zai_keeps_the_callers_own_thinking_next_to_reasoning_effort_none(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    drop_params: bool,
    sent: str | None,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-5.3",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="none",
        thinking={"type": "enabled"},
        drop_params=drop_params,
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert (body.get("thinking"), body.get("reasoning_effort")) == ({"type": "enabled"}, sent)


@pytest.mark.parametrize(
    ("model", "requested", "drop_params", "sent"),
    [
        ("glm-5.3", 5, True, None),
        ("glm-5.3", True, True, None),
        ("glm-5.3", {"effort": 5}, True, None),
        ("glm-5.3", 5, False, 5),
        ("glm-5.1", "turbo", True, None),
        ("glm-5.1", "turbo", False, "turbo"),
        ("glm-5.1", "xhigh", True, "xhigh"),
    ],
)
def test_zai_drops_a_reasoning_effort_that_names_no_level_only_under_drop_params(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    requested: object,
    drop_params: bool,
    sent: object,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model=f"zai/{model}",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort=requested,
        drop_params=drop_params,
    )

    assert json.loads(respx_mock.calls[0].request.content).get("reasoning_effort") == sent


def test_zai_sends_each_deployment_only_the_levels_it_declares_itself(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)
    router: Final = Router(
        model_list=[
            {
                "model_name": "glm-max-only",
                "litellm_params": {"model": "zai/glm-5.3"},
                "model_info": {"id": "glm-max-only", "reasoning_effort_levels": ["max"]},
            },
            {
                "model_name": "glm-catalog-levels",
                "litellm_params": {"model": "zai/glm-5.3"},
                "model_info": {"id": "glm-catalog-levels"},
            },
        ]
    )

    for model_name in ("glm-catalog-levels", "glm-max-only"):
        router.completion(model=model_name, messages=[{"role": "user", "content": "Hello"}], reasoning_effort="medium")

    assert [json.loads(call.request.content)["reasoning_effort"] for call in respx_mock.calls] == ["high", "max"]


@pytest.mark.parametrize(
    ("model", "requested", "sent"),
    [
        ("glm-5.3", {"effort": "medium", "summary": "detailed"}, "high"),
        ("glm-5.2", {"effort": "medium", "summary": "auto"}, "medium"),
        ("glm-5.3", {"summary": "detailed"}, None),
    ],
)
def test_zai_sends_only_the_level_of_a_reasoning_effort_with_a_summary(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    requested: Mapping[str, str],
    sent: str | None,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model=f"zai/{model}", messages=[{"role": "user", "content": "Hello"}], reasoning_effort=requested)

    assert json.loads(respx_mock.calls[0].request.content).get("reasoning_effort") == sent


def test_zai_keeps_disabled_thinking_when_only_the_deployment_levels_leave_out_none(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)
    router: Final = Router(
        model_list=[
            {
                "model_name": "glm-5-1-levels",
                "litellm_params": {"model": "zai/glm-5.1", "drop_params": True},
                "model_info": {"id": "glm-5-1-levels", "reasoning_effort_levels": ["low", "medium", "high"]},
            }
        ]
    )

    router.completion(
        model="glm-5-1-levels", messages=[{"role": "user", "content": "Hello"}], thinking={"type": "disabled"}
    )

    assert json.loads(respx_mock.calls[0].request.content)["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "effort", "sent"), [("glm-5.3", "xhigh", "max"), ("glm-4.7", "max", "max")])
async def test_zai_messages_effort_reaches_glm_rounded_only_by_zai(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    effort: str,
    sent: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    await litellm.anthropic_messages(
        model=f"zai/{model}",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": "Is 391 prime?"}],
        api_key="test-api-key",
    )

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == sent


@pytest.mark.parametrize("model", ["GLM-5.3", "Glm-5.3-Flash"])
def test_zai_rounds_reasoning_effort_for_a_model_named_in_another_case(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model=f"zai/{model}", messages=[{"role": "user", "content": "Hello"}], reasoning_effort="medium")

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == "high"
