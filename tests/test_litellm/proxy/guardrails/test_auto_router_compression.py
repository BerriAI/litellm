"""
Unit tests for litellm.proxy.guardrails.auto_router_compression.

Covers:
- policy_from_litellm_params: absent keys mean no policy; the "none" sentinel
  normalizes to explicit no-compression within an active policy; is_same
- policy_for_model: finds the auto-router marker deployment for an alias
- arm_pre_call: no-op without a policy; suppresses active compression guardrails;
  arms the model-side guardrail even when it isn't default_on; snapshots messages
- messages_for_routing: no-op without a policy or an unset routing side; compresses
  via the named guardrail's apply_guardrail; never writes stats onto the caller's
  own request_kwargs (regression for double-counted compression savings)
"""

from typing import Any

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.auto_router_compression import (
    AUTO_ROUTER_ROUTING_MESSAGES_SNAPSHOT_KEY,
    AutoRouterCompressionPolicy,
    arm_pre_call,
    messages_for_routing,
    policy_for_model,
    policy_from_litellm_params,
)
from litellm.constants import AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY
from litellm.types.utils import GenericGuardrailAPIInputs


class TestPolicyFromLitellmParams:
    def test_neither_key_set_is_no_policy(self):
        assert policy_from_litellm_params({}) is None

    def test_routing_only(self):
        policy = policy_from_litellm_params({"auto_router_routing_compression": "headroom-a"})
        assert policy == AutoRouterCompressionPolicy(routing="headroom-a", model=None)

    def test_none_sentinel_normalizes_to_no_compression(self):
        policy = policy_from_litellm_params(
            {"auto_router_routing_compression": "headroom-a", "auto_router_model_compression": "none"}
        )
        assert policy == AutoRouterCompressionPolicy(routing="headroom-a", model=None)

    def test_none_sentinel_is_case_insensitive(self):
        policy = policy_from_litellm_params({"auto_router_routing_compression": "NONE"})
        assert policy == AutoRouterCompressionPolicy(routing=None, model=None)

    def test_is_same_true_for_matching_names(self):
        policy = policy_from_litellm_params(
            {"auto_router_routing_compression": "x", "auto_router_model_compression": "x"}
        )
        assert policy.is_same is True

    def test_is_same_false_for_different_names(self):
        policy = policy_from_litellm_params(
            {"auto_router_routing_compression": "x", "auto_router_model_compression": "y"}
        )
        assert policy.is_same is False

    def test_is_same_true_when_both_no_compression(self):
        policy = policy_from_litellm_params(
            {"auto_router_routing_compression": "none", "auto_router_model_compression": "none"}
        )
        assert policy.is_same is True


class _FakeRouter:
    """Minimal stand-in for litellm.Router.get_model_list, for policy_for_model."""

    def __init__(self, deployments: list[dict[str, Any]]):
        self._deployments = deployments

    def get_model_list(self, model_name, team_id=None):
        return [d for d in self._deployments if d.get("model_name") == model_name]


class TestPolicyForModel:
    def test_no_router_returns_none(self):
        assert policy_for_model(llm_router=None, model_alias="smart-router", team_id=None) is None

    def test_no_marker_deployment_returns_none(self):
        router = _FakeRouter(
            [{"model_name": "smart-router", "litellm_params": {"model": "openai/gpt-4o-mini"}}]
        )
        assert policy_for_model(llm_router=router, model_alias="smart-router", team_id=None) is None

    def test_marker_deployment_without_policy_returns_none(self):
        router = _FakeRouter(
            [{"model_name": "smart-router", "litellm_params": {"model": "auto_router/complexity_router"}}]
        )
        assert policy_for_model(llm_router=router, model_alias="smart-router", team_id=None) is None

    def test_marker_deployment_with_policy_is_found(self):
        router = _FakeRouter(
            [
                {
                    "model_name": "smart-router",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "auto_router_routing_compression": "headroom-a",
                        "auto_router_model_compression": "none",
                    },
                }
            ]
        )
        policy = policy_for_model(llm_router=router, model_alias="smart-router", team_id=None)
        assert policy == AutoRouterCompressionPolicy(routing="headroom-a", model=None)


class _RecordingCompressionGuardrail(CustomGuardrail):
    """A guardrail whose apply_guardrail marks every text message as compressed."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.request_data_seen: list[dict] = []

    async def apply_guardrail(
        self, inputs: GenericGuardrailAPIInputs, request_data: dict, input_type: str, logging_obj=None
    ) -> GenericGuardrailAPIInputs:
        self.request_data_seen.append(request_data)
        structured_messages = inputs.get("structured_messages") or []
        compressed = [
            {**m, "content": f"[COMPRESSED] {m.get('content')}"} for m in structured_messages
        ]
        return {**inputs, "structured_messages": compressed}


@pytest.fixture
def registered_guardrail():
    import litellm

    guardrail = _RecordingCompressionGuardrail(guardrail_name="fake-compress")
    litellm.logging_callback_manager.add_litellm_callback(guardrail)
    yield guardrail
    litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)


class TestArmPreCall:
    @pytest.mark.asyncio
    async def test_no_router_is_noop(self):
        data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
        result = await arm_pre_call(data=data, llm_router=None)
        assert result == data
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_no_policy_does_not_create_metadata_bucket(self):
        router = _FakeRouter(
            [{"model_name": "smart-router", "litellm_params": {"model": "openai/gpt-4o-mini"}}]
        )
        data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
        result = await arm_pre_call(data=data, llm_router=router)
        assert "metadata" not in result
        assert "litellm_metadata" not in result

    @pytest.mark.asyncio
    async def test_policy_suppresses_active_compression_guardrails(self, monkeypatch):
        from litellm.proxy.guardrails import guardrail_registry

        monkeypatch.setitem(
            guardrail_registry.guardrail_class_registry, "fake-provider", _RecordingCompressionGuardrail
        )
        monkeypatch.setattr(
            "litellm.proxy.guardrails.auto_router_compression.COMPRESSION_GUARDRAIL_PROVIDERS",
            frozenset({"fake-provider"}),
        )
        import litellm

        always_on = _RecordingCompressionGuardrail(guardrail_name="always-on-compression")
        litellm.logging_callback_manager.add_litellm_callback(always_on)
        try:
            router = _FakeRouter(
                [
                    {
                        "model_name": "smart-router",
                        "litellm_params": {
                            "model": "auto_router/complexity_router",
                            "auto_router_routing_compression": "headroom-a",
                            "auto_router_model_compression": "none",
                        },
                    }
                ]
            )
            data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
            result = await arm_pre_call(data=data, llm_router=router)
            suppressed = result["metadata"][AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY]
            assert "always-on-compression" in suppressed
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(always_on)

    @pytest.mark.asyncio
    async def test_model_side_guardrail_is_requested_even_when_not_default_on(self):
        router = _FakeRouter(
            [
                {
                    "model_name": "smart-router",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "auto_router_routing_compression": "none",
                        "auto_router_model_compression": "headroom-b",
                    },
                }
            ]
        )
        data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
        result = await arm_pre_call(data=data, llm_router=router)
        assert result["metadata"]["guardrails"] == ["headroom-b"]

    @pytest.mark.asyncio
    async def test_snapshots_original_messages(self):
        router = _FakeRouter(
            [
                {
                    "model_name": "smart-router",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "auto_router_routing_compression": "headroom-a",
                        "auto_router_model_compression": "none",
                    },
                }
            ]
        )
        original_messages = [{"role": "user", "content": "hi"}]
        data = {"model": "smart-router", "messages": original_messages}
        result = await arm_pre_call(data=data, llm_router=router)
        snapshot = result["metadata"][AUTO_ROUTER_ROUTING_MESSAGES_SNAPSHOT_KEY]
        assert snapshot == original_messages
        assert snapshot is not original_messages  # a copy, not the live reference


class TestMessagesForRouting:
    @pytest.mark.asyncio
    async def test_no_policy_returns_none(self):
        assert await messages_for_routing(policy=None, messages=[], request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_routing_side_unset_returns_none(self):
        policy = AutoRouterCompressionPolicy(routing=None, model="headroom-a")
        assert await messages_for_routing(policy=policy, messages=[], request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_unknown_guardrail_name_returns_none(self):
        policy = AutoRouterCompressionPolicy(routing="does-not-exist", model=None)
        messages = [{"role": "user", "content": "hi"}]
        result = await messages_for_routing(policy=policy, messages=messages, request_kwargs={})
        assert result is None

    @pytest.mark.asyncio
    async def test_compresses_via_the_named_guardrail(self, registered_guardrail):
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model=None)
        messages = [{"role": "user", "content": "hello world"}]
        result = await messages_for_routing(policy=policy, messages=messages, request_kwargs={})
        assert result == [{"role": "user", "content": "[COMPRESSED] hello world"}]

    @pytest.mark.asyncio
    async def test_uses_the_snapshot_when_present(self, registered_guardrail):
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model=None)
        snapshot = [{"role": "user", "content": "original"}]
        request_kwargs = {"metadata": {AUTO_ROUTER_ROUTING_MESSAGES_SNAPSHOT_KEY: snapshot}}
        # `messages` here stands in for whatever a model-side guardrail already
        # rewrote `data["messages"]` to -- routing must ignore it and compress the
        # pristine snapshot instead.
        already_rewritten = [{"role": "user", "content": "rewritten by another guardrail"}]
        result = await messages_for_routing(
            policy=policy, messages=already_rewritten, request_kwargs=request_kwargs
        )
        assert result == [{"role": "user", "content": "[COMPRESSED] original"}]

    @pytest.mark.asyncio
    async def test_guardrail_receives_a_throwaway_request_data_not_the_real_request_kwargs(
        self, registered_guardrail
    ):
        """Regression: a real compression guardrail writes its stats onto whatever
        `request_data` dict it's given (`add_standard_logging_guardrail_information_to_
        request_data`). If that were the caller's own `request_kwargs`, routing-side
        compression would double-count into extract_compression_saved_tokens, which
        sums every guardrail_information entry on the real request's metadata."""
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model=None)
        messages = [{"role": "user", "content": "hi"}]
        request_kwargs = {"metadata": {}}
        await messages_for_routing(policy=policy, messages=messages, request_kwargs=request_kwargs)
        assert registered_guardrail.request_data_seen[0] is not request_kwargs
        assert request_kwargs == {"metadata": {}}
