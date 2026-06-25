"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


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


def test_zai_models_in_model_cost():
    """Test that ZAI models are in the model cost map"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    zai_models = [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
        "zai/glm-4.5v",
        "zai/glm-4.5-x",
        "zai/glm-4.5-air",
        "zai/glm-4.5-airx",
        "zai/glm-4-32b-0414-128k",
        "zai/glm-4.5-flash",
    ]

    for model in zai_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "zai"


def test_zai_glm46_cost_calculation():
    """Test the cost calculation for glm-4.6"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.6"
    info = litellm.model_cost[key]

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.6",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.6: $0.6/M input, $2.2/M output
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_flash_model_is_free():
    """Test that glm-4.5-flash has zero cost"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.5-flash"
    info = litellm.model_cost[key]

    assert info["input_cost_per_token"] == 0
    assert info["output_cost_per_token"] == 0


def test_glm47_supports_reasoning():
    """Test that GLM-4.7 supports reasoning"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.7"
    assert key in litellm.model_cost, f"Model {key} not found in model_cost"

    info = litellm.model_cost[key]
    assert info["supports_reasoning"] is True


def test_glm47_cost_calculation():
    """Test cost calculation for GLM-4.7"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.7",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.7: $0.6/M input, $2.2/M output (same as GLM-4.6)
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


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


@pytest.fixture
def zai_thinking_response():
    return {
        "id": "chatcmpl-zai-thinking",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _captured_body(respx_mock):
    assert len(respx_mock.calls) == 1
    return json.loads(respx_mock.calls[0].request.content.decode("utf-8"))


class TestZaiSupportedParamsWhitelistReasoning:
    """`thinking` and `reasoning_effort` enter the whitelist only when the
    registry marks the model `supports_reasoning: true`. The registry
    update in this PR adds the flag to the entire GLM-4.5 family
    (previously every GLM-4.5 entry in
    `model_prices_and_context_window_backup.json` was missing the flag
    despite docs.z.ai listing GLM-4.5 as the first model with `thinking`
    support), so the gate now unlocks reasoning params for all of them.
    """

    @pytest.fixture(autouse=True)
    def _use_local_model_cost(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")

    @pytest.mark.parametrize(
        "model",
        [
            "glm-4.5",
            "glm-4.5v",
            "glm-4.5-air",
            "glm-4.5-airx",
            "glm-4.5-x",
            "glm-4.5-flash",
            "glm-4.6",
            "glm-4.7",
        ],
    )
    def test_reasoning_params_in_whitelist(self, model):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        params = ZAIChatConfig().get_supported_openai_params(model=model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_reasoning_params_excluded_when_registry_flag_missing(self, monkeypatch):
        """Regression guard for the gate. A model whose registry entry
        does NOT mark `supports_reasoning: true` must keep `thinking`
        and `reasoning_effort` OUT of the whitelist — otherwise a new
        ZAI model added to the registry without the flag silently
        accepts reasoning kwargs that the upstream API will reject.
        """
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        # Synthetic model that won't match any registry entry or
        # wildcard pattern.
        params = ZAIChatConfig().get_supported_openai_params(
            model="glm-no-such-future-model-xyz"
        )
        assert "thinking" not in params
        assert "reasoning_effort" not in params


class TestZaiReasoningParamsLandInExtraBody:
    """The OpenAI Python SDK rejects unknown top-level kwargs (e.g.
    `AsyncCompletions.create() got an unexpected keyword argument 'thinking'`),
    so ZAI-specific reasoning fields must travel inside `extra_body` and
    let the SDK flatten them into the HTTP body. Without this wrapping a
    chained-proxy topology (LiteLLM A -> LiteLLM B -> ZAI) drops the
    field on hop 2: hop 1's SDK flattens `extra_body` into a top-level
    `thinking` kwarg, hop 2 re-emits that as a top-level kwarg, and the
    SDK rejects it
    """

    def test_thinking_wraps_into_extra_body(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="glm-4.6",
            drop_params=False,
        )
        assert "thinking" not in result
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_reasoning_effort_wraps_into_extra_body(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="glm-5",
            drop_params=False,
        )
        assert "reasoning_effort" not in result
        assert result["extra_body"]["reasoning_effort"] == "none"

    def test_thinking_merges_into_existing_extra_body(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={"extra_body": {"already_here": True}},
            model="glm-4.6",
            drop_params=False,
        )
        assert result["extra_body"]["already_here"] is True
        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_standard_params_stay_top_level_alongside_thinking(self):
        from litellm.llms.zai.chat.transformation import ZAIChatConfig

        result = ZAIChatConfig()._map_openai_params(
            non_default_params={
                "max_tokens": 100,
                "temperature": 0.7,
                "thinking": {"type": "enabled"},
            },
            optional_params={},
            model="glm-4.6",
            drop_params=False,
        )
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.7
        assert result["extra_body"]["thinking"] == {"type": "enabled"}
        assert "thinking" not in result

    @pytest.mark.asyncio
    async def test_top_level_thinking_kwarg_reaches_http_body(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        """Regression for the hop-2 SDK crash. Pre-fix this raised
        `AsyncCompletions.create() got an unexpected keyword argument
        'thinking'`. Post-fix the boundary HTTP body carries `thinking`
        as a top-level field (the OpenAI SDK flattened `extra_body`)
        """
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )

        body = _captured_body(respx_mock)
        assert body["thinking"] == {"type": "disabled"}
        assert "extra_body" not in body

    @pytest.mark.asyncio
    async def test_extra_body_thinking_reaches_http_body(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-4.6",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"thinking": {"type": "disabled"}},
        )

        body = _captured_body(respx_mock)
        assert body["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_top_level_reasoning_effort_reaches_http_body(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-5",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="none",
        )

        body = _captured_body(respx_mock)
        assert body["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_thinking_works_on_glm_4_5_via_registry_flag(
        self, respx_mock, zai_thinking_response, monkeypatch
    ):
        """End-to-end: GLM-4.5 carries `supports_reasoning: true` in the
        registry, so the gate in `get_supported_openai_params` lets
        `thinking` through and the SDK boundary lands it in the HTTP
        body. Without the registry update this test would fail with the
        SDK rejecting `thinking` as an unsupported param.
        """
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        litellm.model_cost = litellm.get_model_cost_map(url="")
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
            json=zai_thinking_response
        )

        await litellm.acompletion(
            model="zai/glm-4.5",
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )

        body = _captured_body(respx_mock)
        assert body["thinking"] == {"type": "disabled"}


class TestGlm45FamilyRegistrySupportsReasoning:
    """docs.z.ai lists GLM-4.5 as the first model family that supports
    `thinking`. The registry had every GLM-4.5 entry marked
    `supports_reasoning: false`, which is the source-of-truth bug that
    let the SDK-kwarg crash hide for so long
    """

    @pytest.mark.parametrize(
        "model_key",
        [
            "zai/glm-4.5",
            "zai/glm-4.5v",
            "zai/glm-4.5-x",
            "zai/glm-4.5-air",
            "zai/glm-4.5-airx",
            "zai/glm-4.5-flash",
        ],
    )
    def test_supports_reasoning_true(self, model_key):
        import os

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        assert model_key in litellm.model_cost
        assert litellm.model_cost[model_key].get("supports_reasoning") is True
