from typing import Final

import pytest

from litellm.llms.prompt_cache_policy import cache_history_policy
from litellm.types.utils import ModelInfo

_PRICES: Final[ModelInfo] = {"supports_prompt_cache_breakpoint": True}


@pytest.mark.parametrize("provider", ("openai", "azure"))
def test_modern_provider_reuses_user_tool_group_and_initial_boundaries(provider: str) -> None:
    policy: Final = cache_history_policy("test-model", provider, _PRICES, {}, None)
    assert policy.implicit
    assert policy.eligible("user", "assistant", False)
    assert policy.eligible("tool", "user", False)
    assert policy.eligible("developer", "user", True)
    assert not policy.eligible("tool", "tool", False)
    assert not policy.eligible("assistant", "user", False)
    assert not policy.eligible("developer", "developer", False)


@pytest.mark.parametrize("provider,model", (("anthropic", "test-model"), ("bedrock", "anthropic.claude-test")))
def test_explicit_provider_defaults_do_not_invent_implicit_cache_writes(provider: str, model: str) -> None:
    prices: Final[ModelInfo] = {"prompt_cache_mode": "explicit", "prompt_cache_default_ttl_seconds": 300}
    policy: Final = cache_history_policy(model, provider, prices, {}, None)
    override: Final = cache_history_policy(model, provider, prices, {"ttl": "1h"}, None)
    assert not policy.implicit
    assert policy.lifetime == 300 and policy.assumptions == ()
    assert override.lifetime == 3600 and override.assumptions == ()


@pytest.mark.parametrize(
    "retention,lifetime,assumptions",
    (
        (None, 1800, ("cache_lifetime_assumed_30m",)),
        ("in_memory", 600, ("cache_lifetime_assumed_10m",)),
        ("24h", 86400, ()),
    ),
)
def test_provider_retention_and_explicit_ttl_override(
    retention: str | None, lifetime: int, assumptions: tuple[str, ...]
) -> None:
    policy: Final = cache_history_policy("test-model", "openai", None, {}, retention)
    explicit: Final = cache_history_policy("test-model", "openai", None, {"ttl": "5m", "mode": "explicit"}, retention)
    assert (policy.lifetime, policy.assumptions) == (lifetime, assumptions)
    assert (explicit.lifetime, explicit.assumptions, explicit.implicit) == (300, (), False)


def test_unknown_provider_retains_labeled_fallback_and_all_message_boundaries() -> None:
    policy: Final = cache_history_policy("test-model", "other", _PRICES, {}, None)
    assert (policy.lifetime, policy.assumptions) == (1800, ("cache_lifetime_assumed_30m",))
    assert policy.implicit and policy.eligible("assistant", "user", False)


@pytest.mark.parametrize("model", ("custom-assistant", "custom-claude-evaluator"))
def test_custom_model_names_do_not_select_cache_policy(model: str) -> None:
    policy: Final = cache_history_policy(model, "openai", {"supports_prompt_caching": True}, {}, None)
    assert policy.implicit and policy.lifetime == 1800
    assert policy.assumptions == ("cache_lifetime_assumed_30m",)


def test_model_metadata_controls_cache_mode_and_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setitem(
        litellm.model_cost,
        "openai/custom-model",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_prompt_caching": True,
            "prompt_cache_mode": "explicit",
            "prompt_cache_default_ttl_seconds": 7200,
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
        },
    )
    prices: Final = litellm.get_model_info("openai/custom-model")
    policy: Final = cache_history_policy("custom-model", "openai", prices, {}, None)
    assert not policy.implicit and policy.lifetime == 7200 and not policy.assumptions
    explicit: Final = cache_history_policy("custom-model", "openai", prices, {"ttl": "1h"}, None)
    assert explicit.lifetime == 3600
