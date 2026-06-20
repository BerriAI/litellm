from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from litellm.proxy import health_check as hc_module
from litellm.proxy.health_check import (
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
    updated = _update_litellm_params_for_health_check(
        {"mode": mode}, {"model": f"openai/dummy-{mode}"}
    )

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
    updated = _update_litellm_params_for_health_check(
        {"mode": mode}, {"model": f"openai/dummy-{mode}"}
    )

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
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-5", "api_key": "x"}
    )
    assert out.get("reasoning_effort") == "none"

    model_info = {"mode": "completion", "health_check_reasoning_effort": "low"}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-5", "api_key": "x"}
    )
    assert out.get("reasoning_effort") == "low"

    model_info = {
        "health_check_reasoning_effort": {"effort": "none", "summary": "auto"},
    }
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-5.1", "api_key": "x"}
    )
    assert out.get("reasoning_effort") == {"effort": "none", "summary": "auto"}

    model_info = {"mode": "embedding", "health_check_reasoning_effort": "low"}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "text-embedding-3-small", "api_key": "x"}
    )
    assert "reasoning_effort" not in out

    model_info = {}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-4o", "api_key": "x"}
    )
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
def test_bedrock_embedding_without_explicit_mode_skips_max_tokens(
    deployment_model, expected_request_model
):
    """Embedding mode auto-detected from model cost map -> no max_tokens, provider pinned."""
    assert _resolve_health_check_mode({}, {"model": deployment_model}) == "embedding"

    updated = _update_litellm_params_for_health_check({}, {"model": deployment_model})

    assert "max_tokens" not in updated
    assert updated["custom_llm_provider"] == "bedrock"
    assert updated["model"] == expected_request_model


def test_resolve_health_check_mode_prefers_explicit_model_info_mode():
    """An operator-set mode wins over model-cost lookup."""
    assert (
        _resolve_health_check_mode(
            {"mode": "chat"}, {"model": "bedrock/amazon.titan-embed-text-v2:0"}
        )
        == "chat"
    )


def test_resolve_health_check_mode_unknown_model_returns_none():
    assert (
        _resolve_health_check_mode({}, {"model": "bedrock/not-a-real-model-xyz"})
        is None
    )
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
