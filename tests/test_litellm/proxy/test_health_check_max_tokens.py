import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx

import litellm
from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from litellm.proxy import health_check as hc_module
from litellm.proxy.health_check import (
    _is_strategy_router_deployment,
    _resolve_health_check_max_tokens,
    _resolve_health_check_mode,
    _update_litellm_params_for_health_check,
)


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_default(monkeypatch):
    """
    Test that max_tokens defaults to 16 for non-wildcard models.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", None)
    model_info = {}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 16


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_custom():
    """
    Test that max_tokens respects health_check_max_tokens from model_info.
    """
    model_info = {"health_check_max_tokens": 5}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 5


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_wildcard():
    """
    Test that max_tokens does NOT default to 1 for wildcard models.
    """
    model_info = {}
    litellm_params = {"model": "openai/*"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert "max_tokens" not in updated_params


@pytest.mark.asyncio
async def test_ahealth_check_wildcard_models_respects_max_tokens():
    """
    Test that ahealth_check_wildcard_models respects max_tokens if passed,
    otherwise defaults to 16.
    """
    with (
        patch(
            "litellm.litellm_core_utils.llm_request_utils.pick_cheapest_chat_models_from_llm_provider",
            return_value=["gpt-4o-mini"],
        ),
        patch("litellm.acompletion", new_callable=AsyncMock),
    ):
        # Test Case 1: No max_tokens passed, should default to 16
        model_params = {}
        await HealthCheckHelpers.ahealth_check_wildcard_models(
            model="openai/*",
            custom_llm_provider="openai",
            model_params=model_params,
            litellm_logging_obj=MagicMock(),
        )
        assert model_params["max_tokens"] == 16

        # Test Case 2: Custom health_check_max_tokens passed via model_params, should be respected
        model_params = {"max_tokens": 3}
        await HealthCheckHelpers.ahealth_check_wildcard_models(
            model="openai/*",
            custom_llm_provider="openai",
            model_params=model_params,
            litellm_logging_obj=MagicMock(),
        )
        assert model_params["max_tokens"] == 3


@pytest.mark.asyncio
async def test_background_health_check_max_tokens_env_var(monkeypatch):
    """
    Test that BACKGROUND_HEALTH_CHECK_MAX_TOKENS env var is used as global default
    for explicit (non-wildcard) models.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)

    model_info = {}
    litellm_params = {"model": "azure/gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 10


@pytest.mark.asyncio
async def test_per_model_overrides_global_env_var(monkeypatch):
    """
    Test that per-model health_check_max_tokens takes priority over
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS env var.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)

    model_info = {"health_check_max_tokens": 5}
    litellm_params = {"model": "azure/gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 5


@pytest.mark.asyncio
async def test_global_env_var_applies_to_wildcard_models(monkeypatch):
    """
    Test that BACKGROUND_HEALTH_CHECK_MAX_TOKENS env var also applies to wildcard models.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 15)

    model_info = {}
    litellm_params = {"model": "openai/*"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 15


def test_resolve_health_check_max_tokens_reasoning_specific_model_info():
    model_info = {
        "health_check_max_tokens_reasoning": 64,
        "health_check_max_tokens_non_reasoning": 2,
    }
    litellm_params = {"model": "openai/gpt-4o"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=False):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 2

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 64


def test_explicit_health_check_max_tokens_beats_reasoning_specific():
    model_info = {
        "health_check_max_tokens": 9,
        "health_check_max_tokens_reasoning": 64,
        "health_check_max_tokens_non_reasoning": 2,
    }
    litellm_params = {"model": "openai/gpt-4o"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 9


def test_reasoning_specific_falls_through_when_wrong_branch_only(monkeypatch):
    """Only non-reasoning key set but model is reasoning → fall back to default 16."""
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", None)
    model_info = {"health_check_max_tokens_non_reasoning": 3}
    litellm_params = {"model": "openai/o1"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 16


@pytest.mark.asyncio
async def test_background_split_env_reasoning_vs_non_reasoning(monkeypatch):
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", 50)

    model_info = {}
    litellm_params = {"model": "azure/gpt-4"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=False):
        updated = _update_litellm_params_for_health_check(model_info, litellm_params)
        assert updated["max_tokens"] == 16

    litellm_params2 = {"model": "openai/o1"}
    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        updated2 = _update_litellm_params_for_health_check(model_info, litellm_params2)
        assert updated2["max_tokens"] == 50


@pytest.mark.asyncio
async def test_reasoning_env_precedence_over_global(monkeypatch):
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", 20)

    model_info = {}
    litellm_params = {"model": "openai/gpt-5.4"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        updated = _update_litellm_params_for_health_check(model_info, litellm_params)
        assert updated["max_tokens"] == 20


@pytest.mark.asyncio
async def test_non_reasoning_uses_global_when_reasoning_env_set(monkeypatch):
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", 20)

    model_info = {}
    litellm_params = {"model": "azure/gpt-4"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=False):
        updated = _update_litellm_params_for_health_check(model_info, litellm_params)
        assert updated["max_tokens"] == 10


def test_wildcard_ignores_reasoning_split_model_info(monkeypatch):
    """Wildcard routes do not use reasoning/non-reasoning model_info split."""
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", None)
    model_info = {
        "health_check_max_tokens_reasoning": 99,
        "health_check_max_tokens_non_reasoning": 7,
    }
    litellm_params = {"model": "openai/*"}

    assert _resolve_health_check_max_tokens(model_info, litellm_params) is None


# ---------------------------------------------------------------------------
# image_generation must not receive max_tokens.
#
# _update_litellm_params_for_health_check injected `max_tokens` for every
# deployment. For `mode: image_generation` that leaked into OpenAI
# `/v1/images/generations`, which strictly rejects unknown fields with
# `400 "Unknown parameter: 'max_tokens'"`, marking dall-e-* and
# gpt-image-1 as permanently unhealthy even though their actual image
# calls succeed. `messages` still gets injected (downstream
# `_filter_model_params` already strips it for non-chat handlers).
# ---------------------------------------------------------------------------


def test_image_generation_mode_skips_max_tokens():
    """image_generation must not receive max_tokens."""
    model_info = {"mode": "image_generation"}
    litellm_params = {"model": "openai/dall-e-3", "api_key": "sk-test"}

    updated = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert "max_tokens" not in updated
    # connection-level params must still pass through unchanged
    assert updated["api_key"] == "sk-test"


def test_health_check_max_tokens_value_is_ignored_for_non_chat_modes():
    """A configured `health_check_max_tokens` *value* (the int that controls
    how many tokens to inject) is still skipped when the mode is outside the
    allow-list — the inject decision runs before value resolution, so the
    value never reaches `_resolve_health_check_max_tokens`. Note this is
    distinct from `health_check_supports_max_tokens` (the bool that toggles
    injection on/off per deployment)."""
    model_info = {"mode": "image_generation", "health_check_max_tokens": 50}
    litellm_params = {"model": "openai/dall-e-3"}

    updated = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert "max_tokens" not in updated


def test_chat_mode_still_injects_max_tokens():
    """Regression guard: the chat-style probe payload is unchanged."""
    model_info = {"mode": "chat"}
    litellm_params = {"model": "gpt-4"}

    updated = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated["max_tokens"] == 16


def test_no_mode_still_injects_max_tokens():
    """Regression guard: model_info without `mode` keeps the legacy path."""
    model_info: dict = {}
    litellm_params = {"model": "gpt-4"}

    updated = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated["max_tokens"] == 16


# ---------------------------------------------------------------------------
# Allow-list behavior: only chat-style modes (chat / completion / responses)
# receive max_tokens. Every other mode is skipped by default.
#
# Per-deployment override via `health_check_supports_max_tokens` lets the
# operator force injection on (e.g. a non-listed but max_tokens-capable
# endpoint where they want to bound probe token usage) or off (e.g. a
# chat-style provider with a strict schema that rejects unknown fields).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["chat", "completion", "responses"])
def test_chat_style_modes_inject_max_tokens(mode):
    updated = _update_litellm_params_for_health_check({"mode": mode}, {"model": f"openai/dummy-{mode}"})

    assert updated["max_tokens"] == 16


@pytest.mark.parametrize(
    "mode",
    [
        "embedding",
        "image_generation",
        "image_edit",
        "audio_speech",
        "audio_transcription",
        "rerank",
        "video_generation",
        "ocr",
        "search",
        "moderation",
    ],
)
def test_non_chat_modes_skip_max_tokens(mode):
    updated = _update_litellm_params_for_health_check({"mode": mode}, {"model": f"openai/dummy-{mode}"})

    assert "max_tokens" not in updated


def test_explicit_override_true_forces_injection_outside_allowlist():
    """Operator opts a non-listed deployment in to bound probe token usage."""
    model_info = {
        "mode": "image_generation",
        "health_check_supports_max_tokens": True,
    }
    litellm_params = {"model": "openai/some-future-image-model"}

    updated = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated["max_tokens"] == 16


def test_explicit_override_false_suppresses_injection_inside_allowlist():
    """Operator opts a chat-style deployment out (strict-schema provider)."""
    model_info = {"mode": "chat", "health_check_supports_max_tokens": False}
    litellm_params = {"model": "openai/strict-schema-chat"}

    updated = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert "max_tokens" not in updated


def test_update_litellm_params_health_check_reasoning_effort():
    """model_info.health_check_reasoning_effort sets reasoning_effort for chat-style health checks."""
    model_info = {"health_check_reasoning_effort": "low"}
    litellm_params = {"model": "openai/gpt-5", "api_key": "x"}
    out = _update_litellm_params_for_health_check(model_info, dict(litellm_params))
    assert out.get("reasoning_effort") == "low"

    model_info = {"mode": "chat", "health_check_reasoning_effort": "none"}
    out = _update_litellm_params_for_health_check(model_info, {"model": "openai/gpt-5", "api_key": "x"})
    assert out.get("reasoning_effort") == "none"

    model_info = {"mode": "completion", "health_check_reasoning_effort": "low"}
    out = _update_litellm_params_for_health_check(model_info, {"model": "openai/gpt-5", "api_key": "x"})
    assert out.get("reasoning_effort") == "low"

    model_info = {
        "health_check_reasoning_effort": {"effort": "none", "summary": "auto"},
    }
    out = _update_litellm_params_for_health_check(model_info, {"model": "openai/gpt-5.1", "api_key": "x"})
    assert out.get("reasoning_effort") == {"effort": "none", "summary": "auto"}

    model_info = {"mode": "embedding", "health_check_reasoning_effort": "low"}
    out = _update_litellm_params_for_health_check(model_info, {"model": "text-embedding-3-small", "api_key": "x"})
    assert "reasoning_effort" not in out

    model_info = {}
    out = _update_litellm_params_for_health_check(model_info, {"model": "openai/gpt-4o", "api_key": "x"})
    assert "reasoning_effort" not in out


# ---------------------------------------------------------------------------
# Bedrock embedding deployments declared without an explicit `model_info.mode`.
#
# The health-check builder used to treat a missing mode as `chat`, so it
# injected `max_tokens` into the embedding probe. Bedrock embeddings reject it
# with 400 "extraneous key [max_tokens]". It also stripped the `bedrock/`
# routing prefix without pinning the provider, so a cross-region id like
# `us.cohere.embed-v4:0` failed downstream with "LLM Provider NOT provided".
# Mode is now resolved from the model cost map (which understands `bedrock/`
# and `us.`/`eu.`/`apac.` prefixes) and the provider is pinned to `bedrock`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deployment_model, expected_request_model",
    [
        ("bedrock/amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v2:0"),
        ("bedrock/us.cohere.embed-v4:0", "us.cohere.embed-v4:0"),
    ],
)
def test_bedrock_embedding_without_explicit_mode_skips_max_tokens(deployment_model, expected_request_model):
    """Embedding mode auto-detected from model cost map -> no max_tokens, provider pinned."""
    assert _resolve_health_check_mode({}, {"model": deployment_model}) == "embedding"

    updated = _update_litellm_params_for_health_check({}, {"model": deployment_model})

    assert "max_tokens" not in updated
    assert updated["custom_llm_provider"] == "bedrock"
    assert updated["model"] == expected_request_model


def test_resolve_health_check_mode_prefers_explicit_model_info_mode():
    """An operator-set mode wins over model-cost lookup."""
    assert _resolve_health_check_mode({"mode": "chat"}, {"model": "bedrock/amazon.titan-embed-text-v2:0"}) == "chat"


def test_resolve_health_check_mode_unknown_model_returns_none():
    assert _resolve_health_check_mode({}, {"model": "bedrock/not-a-real-model-xyz"}) is None
    assert _resolve_health_check_mode({}, {}) is None


def test_bedrock_chat_without_mode_still_injects_max_tokens_and_pins_provider():
    """Regression guard: chat-style Bedrock deployments keep max_tokens and get the provider pin."""
    updated = _update_litellm_params_for_health_check(
        {}, {"model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"}
    )

    assert updated["max_tokens"] == 16
    assert updated["custom_llm_provider"] == "bedrock"
    assert updated["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_bedrock_prefix_strip_preserves_explicit_custom_llm_provider():
    """An operator-set provider (e.g. bedrock_converse) must survive the prefix strip.

    The pin only fills in a provider when the deployment left it blank; it must
    not clobber a more specific one, otherwise a converse deployment would be
    probed against the Invoke endpoint and report a spurious failure.
    """
    updated = _update_litellm_params_for_health_check(
        {},
        {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "custom_llm_provider": "bedrock_converse",
        },
    )

    assert updated["custom_llm_provider"] == "bedrock_converse"
    assert updated["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.mark.asyncio
async def test_run_model_health_check_threads_resolved_mode_to_ahealth_check():
    """The resolved mode must reach `ahealth_check`, not just the params builder.

    A Bedrock embedding deployment declared without an explicit `model_info.mode`
    has to be probed with `mode="embedding"` so the call routes to the embedding
    handler; if the resolution were dropped it would fall back to `chat`. This
    also guards that the embedding params (no `max_tokens`, provider pinned) are
    the ones actually handed to the probe.
    """
    fake_ahealth_check = AsyncMock(return_value={})
    model = {
        "litellm_params": {"model": "bedrock/amazon.titan-embed-text-v2:0"},
        "model_info": {},
    }

    with patch.object(hc_module.litellm, "ahealth_check", fake_ahealth_check):
        await hc_module._run_model_health_check(model)

    assert fake_ahealth_check.call_args.kwargs["mode"] == "embedding"
    probed_params = fake_ahealth_check.call_args.args[0]
    assert "max_tokens" not in probed_params
    assert probed_params["custom_llm_provider"] == "bedrock"
    assert probed_params["model"] == "amazon.titan-embed-text-v2:0"


def test_autodetected_embedding_skips_reasoning_effort():
    """reasoning_effort must not leak into an embedding probe whose mode is auto-detected.

    Same bug class as the max_tokens fix: with no explicit `model_info.mode`, the
    reasoning-effort gate used to read the raw (missing) mode and treat it as
    chat-like, so a configured `health_check_reasoning_effort` was injected into a
    Bedrock embedding probe, which embeddings reject as an unknown field. The mode
    is now resolved from the cost map, so embeddings are excluded.
    """
    updated = _update_litellm_params_for_health_check(
        {"health_check_reasoning_effort": "low"},
        {"model": "bedrock/amazon.titan-embed-text-v2:0"},
    )

    assert "reasoning_effort" not in updated
    assert "max_tokens" not in updated


@pytest.mark.parametrize(
    "model, expected",
    [
        ("auto_router/router_1", True),
        ("auto_router/my_router", True),
        ("auto_router/complexity_router", True),
        ("auto_router/adaptive_router", True),
        ("auto_router/quality_router", True),
        ("auto_router/adaptive_router/subpath", True),
        ("gpt-4", False),
        ("openai/gpt-4", False),
        ("bedrock/claude", False),
    ],
)
def test_is_strategy_router_deployment(model, expected):
    assert _is_strategy_router_deployment({"model": model}) == expected


@pytest.mark.asyncio
async def test_run_model_health_check_skips_auto_router_deployment():
    """auto_router deployments return {} (healthy) without calling ahealth_check."""
    fake_ahealth_check = AsyncMock(return_value={})
    model = {
        "litellm_params": {
            "model": "auto_router/router_1",
            "auto_router_config": '{"routes": []}',
            "auto_router_default_model": "gpt-4o-mini",
            "auto_router_embedding_model": "text-embedding-3-small",
        },
        "model_info": {},
    }

    with patch.object(hc_module.litellm, "ahealth_check", fake_ahealth_check):
        result = await hc_module._run_model_health_check(model)

    fake_ahealth_check.assert_not_called()
    assert result == {}


def test_health_check_params_merge_into_probe_params():
    """health_check_params reach the probe request for the deployment that declares them."""
    media_source = {"s3Location": {"uri": "s3://my-bucket/clip.mp4"}}

    updated = _update_litellm_params_for_health_check(
        {"mode": "chat", "health_check_params": {"mediaSource": media_source}},
        {"model": "bedrock/us.twelvelabs.pegasus-1-2-v1:0"},
    )

    assert updated["mediaSource"] == media_source
    assert updated["model"] == "us.twelvelabs.pegasus-1-2-v1:0"
    assert updated["custom_llm_provider"] == "bedrock"


def test_health_check_params_lose_to_dedicated_health_check_knobs():
    """The dedicated knobs are applied after the merge, so they win on conflict."""
    model_info = {
        "mode": "chat",
        "health_check_params": {
            "max_tokens": 4096,
            "model": "openai/expensive-model",
            "messages": [{"role": "user", "content": "from health_check_params"}],
            "reasoning_effort": "high",
        },
        "health_check_max_tokens": 5,
        "health_check_model": "openai/cheap-model",
        "health_check_reasoning_effort": "none",
    }

    updated = _update_litellm_params_for_health_check(model_info, {"model": "openai/dummy"})

    assert updated["max_tokens"] == 5
    assert updated["model"] == "openai/cheap-model"
    assert updated["reasoning_effort"] == "none"
    assert updated["messages"] != model_info["health_check_params"]["messages"]


def test_health_check_params_lose_to_the_audio_speech_voice_knob():
    """health_check_voice still wins for audio_speech deployments."""
    updated = _update_litellm_params_for_health_check(
        {
            "mode": "audio_speech",
            "health_check_params": {"voice": "sage", "response_format": "wav"},
            "health_check_voice": "shimmer",
        },
        {"model": "openai/tts-1"},
    )

    assert updated["voice"] == "shimmer"
    assert updated["response_format"] == "wav"


@pytest.mark.parametrize(
    "bad_value",
    ["mediaSource", ["mediaSource"], 5, True],
)
def test_health_check_params_ignored_when_not_a_dict(bad_value, caplog):
    """A misconfigured health_check_params is skipped with a warning instead of breaking the probe."""
    with caplog.at_level(logging.WARNING, logger="litellm.proxy.health_check"):
        updated = _update_litellm_params_for_health_check(
            {"mode": "chat", "health_check_params": bad_value},
            {"model": "openai/dummy"},
        )

    assert updated["model"] == "openai/dummy"
    assert updated["max_tokens"] == 16
    assert "health_check_params" in caplog.text


def test_health_check_params_apply_to_non_chat_modes():
    """Non-chat probes get health_check_params too, and still no max_tokens."""
    updated = _update_litellm_params_for_health_check(
        {"mode": "embedding", "health_check_params": {"dimensions": 8}},
        {"model": "bedrock/amazon.titan-embed-text-v2:0"},
    )

    assert updated["dimensions"] == 8
    assert "max_tokens" not in updated


async def _pegasus_health_check_request_body(
    model_info: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()

    litellm_params = _update_litellm_params_for_health_check(
        model_info,
        {
            "model": "bedrock/us.twelvelabs.pegasus-1-2-v1:0",
            "aws_access_key_id": "fake-access-key",
            "aws_secret_access_key": "fake-secret-key",
            "aws_region_name": "us-east-1",
        },
    )

    with respx.mock(assert_all_called=True) as respx_mock:
        invoke_route = respx_mock.post(
            host="bedrock-runtime.us-east-1.amazonaws.com",
            path__regex=r"/model/.+/invoke",
        ).respond(json={"message": "a person walks a dog", "finishReason": "stop"})
        result = await litellm.ahealth_check(litellm_params, mode="chat")

    assert "error" not in result, result
    return json.loads(invoke_route.calls.last.request.content)


@pytest.mark.asyncio
async def test_health_check_params_reach_the_bedrock_invoke_body(monkeypatch):
    """The probe Bedrock actually receives carries mediaSource, which is what unblocks Pegasus."""
    media_source = {"s3Location": {"uri": "s3://my-bucket/clip.mp4"}}

    body = await _pegasus_health_check_request_body(
        {"mode": "chat", "health_check_params": {"mediaSource": media_source}}, monkeypatch
    )

    assert body["mediaSource"] == media_source
    assert body["maxOutputTokens"] == 16
    assert body["inputPrompt"]


@pytest.mark.asyncio
async def test_bedrock_invoke_body_has_no_media_source_without_health_check_params(monkeypatch):
    """Negative control: the field only appears because the deployment asked for it."""
    body = await _pegasus_health_check_request_body({"mode": "chat"}, monkeypatch)

    assert "mediaSource" not in body


@pytest.mark.asyncio
async def test_run_model_health_check_skips_complexity_router_deployment():
    fake_ahealth_check = AsyncMock(return_value={})
    model = {
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {"simple": "gpt-4o-mini"}},
            "complexity_router_default_model": "gpt-4o-mini",
        },
        "model_info": {},
    }

    with patch.object(hc_module.litellm, "ahealth_check", fake_ahealth_check):
        result = await hc_module._run_model_health_check(model)

    fake_ahealth_check.assert_not_called()
    assert result == {}


def _router_health_fixture():
    """A real Router whose SIMPLE tier, default and classifier can each be pointed at a dead
    group. That group has two replicas, so a verdict reached on only one of them is visible."""
    return litellm.Router(
        model_list=[
            {
                "model_name": "live-group",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
                "model_info": {"id": "live-1"},
            },
            {
                "model_name": "dead-group",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
                "model_info": {"id": "dead-1"},
            },
            {
                "model_name": "dead-group",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
                "model_info": {"id": "dead-2"},
            },
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "dead-group", "MEDIUM": "live-group"}},
                    "complexity_router_default_model": "live-group",
                },
                "model_info": {"id": "router-1"},
            },
        ],
        ignore_invalid_deployments=True,
    )


def _marker_deployment(router):
    return next(d for d in router.model_list if d["model_info"]["id"] == "router-1")


def test_strategy_router_reds_when_a_tier_group_has_no_healthy_deployment():
    """LIT-6073: the marker is filed healthy by the {} placeholder; the verdict must override it."""
    router = _router_health_fixture()
    healthy = [{"model_id": "router-1"}, {"model_id": "live-1"}]
    unhealthy = [{"model_id": "dead-1", "error": "boom"}, {"model_id": "dead-2", "error": "boom"}]

    new_healthy, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        healthy, unhealthy, router.model_list, router, ()
    )

    assert [e["model_id"] for e in new_healthy] == ["live-1"]
    moved = next(e for e in new_unhealthy if e["model_id"] == "router-1")
    assert moved["error"] == "tier model 'dead-group' has no healthy deployment"


def test_strategy_router_stays_green_when_every_dependency_has_a_healthy_deployment():
    """The negative class: same router, same code path, nothing unhealthy behind it."""
    router = _router_health_fixture()
    healthy = [{"model_id": "router-1"}, {"model_id": "live-1"}, {"model_id": "dead-1"}, {"model_id": "dead-2"}]

    new_healthy, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        healthy, [], router.model_list, router, ()
    )

    assert {e["model_id"] for e in new_healthy} == {"router-1", "live-1", "dead-1", "dead-2"}
    assert new_unhealthy == ()


def test_strategy_router_reds_when_a_dependency_name_matches_no_deployment():
    """An unresolvable tier name is a different fault from an unhealthy one, and says so."""
    router = _router_health_fixture()
    marker = _marker_deployment(router)
    marker["litellm_params"]["complexity_router_config"]["tiers"]["SIMPLE"] = "typo-group"

    _, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        [{"model_id": "router-1"}], [], router.model_list, router, ()
    )

    assert new_unhealthy[0]["error"] == "tier model 'typo-group' matches no deployment on this proxy"


@pytest.mark.parametrize("judged", [("router-1", "live-1"), ("router-1", "live-1", "dead-1")])
def test_strategy_router_verdict_is_silent_when_part_of_a_group_went_unjudged(judged):
    """Absent information never reds a router, whether the whole group went unjudged (hidden
    from the caller) or only a replica did (opted out of health checks). The replica this run
    never contacted can still serve every request the dead one drops."""
    router = _router_health_fixture()
    scope = [d for d in router.model_list if d["model_info"]["id"] in judged]

    new_healthy, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        [{"model_id": "router-1"}], [{"model_id": "dead-1", "error": "boom"}], scope, router, ()
    )

    assert [e["model_id"] for e in new_healthy] == ["router-1"]
    assert new_unhealthy == ({"model_id": "dead-1", "error": "boom"},)


def test_dependency_probe_expansion_is_a_no_op_when_every_dependency_is_already_checked():
    """The full-list run must gain no extra probe, or /health doubles its provider spend."""
    router = _router_health_fixture()

    assert hc_module._dependency_deployments_to_probe(router.model_list, router.model_list, router) == ()


def test_dependency_probe_expansion_adds_dependencies_for_a_targeted_router_check():
    """GET /health?model_id=<router> narrows to the marker, so the deps must be pulled back in."""
    router = _router_health_fixture()
    marker_only = [_marker_deployment(router)]

    probes = hc_module._dependency_deployments_to_probe(marker_only, router.model_list, router)

    assert {d["model_info"]["id"] for d in probes} == {"dead-1", "dead-2", "live-1"}


def test_dependency_probes_carry_one_row_per_id():
    """An alias can put the same deployment in the list twice, which is what
    filter_deployments_by_id exists for. Probing it twice doubles the provider spend, and two
    results for one id can disagree, reding the router on whichever landed in the loser."""
    router = _router_health_fixture()
    duplicated = tuple(router.model_list) + tuple(d for d in router.model_list if d["model_info"]["id"] == "dead-1")

    probes = hc_module._dependency_deployments_to_probe([_marker_deployment(router)], duplicated, router)

    assert [d["model_info"]["id"] for d in probes].count("dead-1") == 1


def test_a_dependency_alias_whose_target_is_gone_reds_the_router():
    """An alias resolving to nothing fails a request exactly like an unknown name, so the
    health check must not read the empty resolution as "no information" and stay green."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "broken-alias"}},
                    "complexity_router_default_model": "broken-alias",
                },
                "model_info": {"id": "router-1"},
            },
        ],
        model_group_alias={"broken-alias": "target-that-no-longer-exists"},
        ignore_invalid_deployments=True,
    )

    _, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        [{"model_id": "router-1"}], [], router.model_list, router, ()
    )

    assert new_unhealthy[0]["error"] == "tier model 'broken-alias' matches no deployment on this proxy"


def test_a_dependency_that_opted_out_of_health_checks_is_never_probed():
    """skip-disabled is an operator opt-out. A router depending on that deployment must not
    pull it back in and spend the proxy's provider credentials probing it."""
    disabled_dep = {
        "model_name": "dead-group",
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
        "model_info": {"id": "dead-1", "disable_background_health_check": True},
    }
    router = litellm.Router(
        model_list=[
            disabled_dep,
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "dead-group"}},
                    "complexity_router_default_model": "dead-group",
                },
                "model_info": {"id": "router-1"},
            },
        ],
        ignore_invalid_deployments=True,
    )
    marker = [d for d in router.model_list if d["model_info"]["id"] == "router-1"]

    eligible = hc_module._health_check_eligible(router.model_list, skip_disabled=True)
    probes = hc_module._dependency_deployments_to_probe(marker, eligible, router)

    assert probes == ()
    assert [d["model_info"]["id"] for d in eligible] == ["router-1"]


def test_narrowing_by_an_id_that_matches_nothing_keeps_the_whole_list():
    """Pinned because the disabled-dependency fix moved this filter into its own helper."""
    deployments = [{"model_name": "a", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a-1"}}]

    assert hc_module._narrow_to_target(deployments, None, "no-such-id") == tuple(deployments)
    assert hc_module._narrow_to_target(deployments, None, "a-1") == tuple(deployments)
    assert hc_module._narrow_to_target(deployments, "a", None) == tuple(deployments)


def _nested_router_fixture(parent_tier: str):
    return litellm.Router(
        model_list=[
            {
                "model_name": "dead-group",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-x"},
                "model_info": {"id": "dead-1"},
            },
            {
                "model_name": "child",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "dead-group"}},
                    "complexity_router_default_model": "dead-group",
                },
                "model_info": {"id": "child-1"},
            },
            {
                "model_name": "parent",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": parent_tier}},
                    "complexity_router_default_model": parent_tier,
                },
                "model_info": {"id": "parent-1"},
            },
        ],
        ignore_invalid_deployments=True,
    )


def test_a_router_routing_to_a_red_router_is_itself_red():
    """A marker never fails a probe of its own, so a single pass sees only probe failures and
    leaves the parent of a dead child green while every request through it fails."""
    router = _nested_router_fixture("child")

    new_healthy, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        [{"model_id": "parent-1"}, {"model_id": "child-1"}],
        [{"model_id": "dead-1", "error": "boom"}],
        router.model_list,
        router,
        (),
    )

    errors = {e["model_id"]: e["error"] for e in new_unhealthy if e["model_id"] != "dead-1"}
    assert errors["child-1"] == "tier model 'dead-group' has no healthy deployment"
    assert errors["parent-1"] == "tier model 'child' has no healthy deployment"
    assert new_healthy == ()


def test_a_router_routing_to_a_healthy_router_stays_green():
    """The negative class for nested propagation: the child serves, so the parent must not
    inherit a red merely for depending on another router."""
    router = _nested_router_fixture("child")
    child = next(d for d in router.model_list if d["model_info"]["id"] == "child-1")
    child["litellm_params"]["complexity_router_config"]["tiers"]["SIMPLE"] = "dead-group"

    new_healthy, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        [{"model_id": "parent-1"}, {"model_id": "child-1"}, {"model_id": "dead-1"}],
        [],
        router.model_list,
        router,
        (),
    )

    assert {e["model_id"] for e in new_healthy} == {"parent-1", "child-1", "dead-1"}
    assert new_unhealthy == ()


def test_two_routers_pointing_at_each_other_terminate_instead_of_recursing():
    """The round bound is what makes a cycle finish. Neither has a failing dependency, so
    neither reds, and the walk must not recurse forever proving it."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": other}},
                    "complexity_router_default_model": other,
                },
                "model_info": {"id": f"{name}-1"},
            }
            for name, other in (("a", "b"), ("b", "a"))
        ],
        ignore_invalid_deployments=True,
    )

    new_healthy, new_unhealthy = hc_module._finalize_strategy_router_endpoints(
        [{"model_id": "a-1"}, {"model_id": "b-1"}], [], router.model_list, router, ()
    )

    assert {e["model_id"] for e in new_healthy} == {"a-1", "b-1"}
    assert new_unhealthy == ()


def test_a_targeted_check_on_a_nested_router_probes_the_grandchild_models():
    """One hop is not enough. GET /health?model_id=<parent> narrows to the parent, and pulling
    in only the child marker leaves the child's own models unprobed, so nothing ever fails and
    both settle green on the exact path the Admin UI uses."""
    router = _nested_router_fixture("child")
    parent_only = [d for d in router.model_list if d["model_info"]["id"] == "parent-1"]

    probes = hc_module._dependency_deployments_to_probe(parent_only, router.model_list, router)

    assert {d["model_info"]["id"] for d in probes} == {"child-1", "dead-1"}


def test_transitive_probe_expansion_terminates_on_a_router_cycle():
    """Expansion follows routers through routers, so a cycle must stop rather than recurse."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": other}},
                    "complexity_router_default_model": other,
                },
                "model_info": {"id": f"{name}-1"},
            }
            for name, other in (("a", "b"), ("b", "a"))
        ],
        ignore_invalid_deployments=True,
    )
    a_only = [d for d in router.model_list if d["model_info"]["id"] == "a-1"]

    probes = hc_module._dependency_deployments_to_probe(a_only, router.model_list, router)

    assert {d["model_info"]["id"] for d in probes} == {"b-1"}
