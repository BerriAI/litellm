import pytest

import litellm

from litellm.llms.anthropic.pass_through.messages.transformation import AnthropicMessagesConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig
from litellm.native.route_plan import (
    NativeChat,
    NativeMessages,
    NativeResponses,
    RoutePlan,
    plan_chat_route,
    plan_messages_route,
    plan_responses_route,
)

MESSAGES_PASSTHROUGH: dict[str, object] = {"supported_endpoints": ["/v1/messages"]}


def _plan(
    model: str,
    provider: str,
    *,
    model_info: object = None,
    compaction_requested: bool = False,
    force_chat_completions: bool = False,
    thinking_enabled: bool = False,
) -> RoutePlan:
    return plan_messages_route(
        requested_model=f"{provider}/{model}",
        resolved_model=model,
        custom_llm_provider=provider,
        model_info=model_info,
        compaction_requested=compaction_requested,
        force_chat_completions=force_chat_completions,
        thinking_enabled=thinking_enabled,
    )


def test_a_provider_that_speaks_messages_gets_its_own_messages_config():
    plan = _plan("claude-sonnet-4-5", "anthropic")

    assert isinstance(plan, NativeMessages)
    assert isinstance(plan.config, AnthropicMessagesConfig)


@pytest.mark.parametrize(
    ("model", "provider", "kwargs", "expected"),
    [
        ("gpt-5.5", "openai", {}, NativeResponses()),
        ("gpt-5.5", "openai", {"force_chat_completions": True}, NativeChat()),
        ("gpt-5.5", "openai", {"compaction_requested": True}, NativeChat()),
        ("my-model", "hosted_vllm", {}, NativeChat()),
        ("gpt-5.5", "openai", {"force_chat_completions": True, "thinking_enabled": True}, NativeResponses()),
        ("gpt-5.5", "openai", {"compaction_requested": True, "thinking_enabled": True}, NativeChat()),
        ("my-model", "hosted_vllm", {"force_chat_completions": True, "thinking_enabled": True}, NativeChat()),
        ("lit-unmapped-model", "openai", {"force_chat_completions": True, "thinking_enabled": True}, NativeResponses()),
    ],
)
def test_messages_without_a_native_config_bridge_to_responses_only_for_openai(
    model: str, provider: str, kwargs: dict[str, bool], expected: RoutePlan
):
    assert _plan(model, provider, **kwargs) == expected


def test_forced_chat_keeps_thinking_on_chat_for_a_model_that_cannot_reason(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    litellm.register_model({"openai/lit-plain-model": {"litellm_provider": "openai", "supports_reasoning": False}})

    plan = _plan("lit-plain-model", "openai", force_chat_completions=True, thinking_enabled=True)

    assert plan == NativeChat()


@pytest.mark.parametrize("cache_control_ttl", [True, False])
def test_a_deployment_that_opts_into_messages_passthrough_skips_translation(cache_control_ttl: bool):
    plan = _plan("my-model", "hosted_vllm", model_info={**MESSAGES_PASSTHROUGH, "cache_control_ttl": cache_control_ttl})

    assert isinstance(plan, NativeMessages)
    assert isinstance(plan.config, OpenAILikeAnthropicMessagesConfig)
    assert plan.config.supports_cache_control_ttl() is cache_control_ttl


@pytest.mark.parametrize(
    ("has_config", "use_chat_completions_api", "expected"),
    [
        (True, False, NativeResponses()),
        (True, True, NativeChat()),
        (False, False, NativeChat()),
        (False, True, NativeChat()),
    ],
)
def test_responses_reach_chat_without_a_provider_config_or_when_asked(
    has_config: bool, use_chat_completions_api: bool, expected: RoutePlan
):
    config = OpenAIResponsesAPIConfig() if has_config else None

    assert plan_responses_route(config, use_chat_completions_api) == expected


@pytest.mark.parametrize(
    ("has_config", "expected"),
    [
        (True, NativeResponses()),
        (False, NativeChat()),
    ],
)
def test_chat_bridges_to_responses_only_when_the_provider_serves_it(has_config: bool, expected: RoutePlan):
    assert plan_chat_route(OpenAIResponsesAPIConfig() if has_config else None) == expected
