"""Unit tests for litellm.proxy.guardrails.auto_router_compression."""

import json
from typing import Any

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails import auto_router_compression
from litellm.proxy.guardrails.auto_router_compression import (
    AutoRouterCompressionPolicy,
    arm_pre_call,
    messages_for_routing,
    policy_for_model,
    policy_from_litellm_params,
)
from litellm.types.guardrails import GuardrailEventHooks
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


def _marker(compression: dict[str, str], tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_name": "smart-router",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            **compression,
            **({"tags": tags} if tags is not None else {}),
        },
    }


class TestPolicyForModel:
    def test_no_router_returns_none(self):
        assert policy_for_model(llm_router=None, model_alias="smart-router", team_id=None, request_tags=()) is None

    def test_no_marker_deployment_returns_none(self):
        router = _FakeRouter([{"model_name": "smart-router", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
        assert policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=()) is None

    def test_marker_deployment_without_policy_returns_none(self):
        router = _FakeRouter(
            [{"model_name": "smart-router", "litellm_params": {"model": "auto_router/complexity_router"}}]
        )
        assert policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=()) is None

    def test_marker_deployment_with_policy_is_found(self):
        router = _FakeRouter(
            [_marker({"auto_router_routing_compression": "headroom-a", "auto_router_model_compression": "none"})]
        )
        policy = policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=())
        assert policy == AutoRouterCompressionPolicy(routing="headroom-a", model=None)

    def test_picks_the_marker_whose_tags_the_request_carries(self):
        """Regression: an alias with several tag-scoped markers must not suppress one
        marker's guardrail and then route under a different marker's policy."""
        router = _FakeRouter(
            [
                _marker({"auto_router_routing_compression": "headroom-eu"}, tags=["eu"]),
                _marker({"auto_router_routing_compression": "headroom-us"}, tags=["us"]),
            ]
        )

        eu = policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=("eu",))
        us = policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=("us",))

        assert eu == AutoRouterCompressionPolicy(routing="headroom-eu", model=None)
        assert us == AutoRouterCompressionPolicy(routing="headroom-us", model=None)

    def test_untagged_marker_matches_any_request(self):
        router = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-a"})])
        policy = policy_for_model(
            llm_router=router, model_alias="smart-router", team_id=None, request_tags=("anything",)
        )
        assert policy == AutoRouterCompressionPolicy(routing="headroom-a", model=None)

    def test_a_marker_scoped_to_other_tags_is_never_the_fallback(self):
        """Regression: a "us" request must not fall back to an "eu" marker's policy."""
        router = _FakeRouter(
            [
                _marker({"auto_router_routing_compression": "headroom-eu"}, tags=["eu"]),
                _marker({"auto_router_routing_compression": "headroom-default"}),
            ]
        )
        policy = policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=("us",))
        assert policy == AutoRouterCompressionPolicy(routing="headroom-default", model=None)

    def test_no_untagged_fallback_means_no_policy(self):
        """No matching marker means no policy, not an unrelated slice's compression."""
        router = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-eu"}, tags=["eu"])])
        assert (
            policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=("us",)) is None
        )

    def test_tag_scoped_marker_takes_precedence_over_untagged(self):
        """Regression: when multiple markers exist, the tag-scoped one the request
        actually matches should be used, not the first untagged one."""
        router = _FakeRouter(
            [
                _marker({"auto_router_routing_compression": "headroom-untagged"}),
                _marker({"auto_router_routing_compression": "headroom-eu"}, tags=["eu"]),
            ]
        )
        policy = policy_for_model(llm_router=router, model_alias="smart-router", team_id=None, request_tags=("eu",))
        assert policy == AutoRouterCompressionPolicy(routing="headroom-eu", model=None)


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
        compressed = [{**m, "content": f"[COMPRESSED] {m.get('content')}"} for m in structured_messages]
        return {**inputs, "structured_messages": compressed}


@pytest.fixture
def registered_guardrail(monkeypatch):
    import litellm
    from litellm.proxy.guardrails import guardrail_registry

    # Registered under a compression provider name: both hops refuse a name that does
    # not resolve to one, so a bare callback would (correctly) never be used.
    monkeypatch.setitem(guardrail_registry.guardrail_class_registry, "headroom", _RecordingCompressionGuardrail)
    guardrail = _RecordingCompressionGuardrail(guardrail_name="fake-compress")
    litellm.logging_callback_manager.add_litellm_callback(guardrail)
    yield guardrail
    litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)


class _NonCompressionGuardrail(CustomGuardrail):
    """A guardrail that is not a compression provider, e.g. a PII or content filter."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.called = False

    async def apply_guardrail(
        self, inputs: GenericGuardrailAPIInputs, request_data: dict, input_type: str, logging_obj=None
    ) -> GenericGuardrailAPIInputs:
        self.called = True
        return inputs


class TestArmPreCall:
    @pytest.mark.asyncio
    async def test_no_router_is_noop(self):
        data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
        await arm_pre_call(data=data, llm_router=None)
        assert "metadata" not in data

    @pytest.mark.asyncio
    async def test_no_policy_does_not_create_metadata_bucket(self):
        router = _FakeRouter([{"model_name": "smart-router", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
        data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
        await arm_pre_call(data=data, llm_router=router)
        assert "metadata" not in data
        assert "litellm_metadata" not in data

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
            await arm_pre_call(data=data, llm_router=router)
            assert auto_router_compression.suppressed_compression_guardrails() == frozenset({"always-on-compression"})
            # Suppression state must never ride along in metadata: that reaches spend
            # logs the caller can read, and anything there is replayable.
            assert "always-on-compression" not in json.dumps(data.get("metadata", {}))
            assert always_on.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is False
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(always_on)

    @pytest.mark.asyncio
    async def test_suppression_state_never_enters_request_metadata(self):
        """Regression (security): metadata reaches spend logs, so a suppression list
        there is one a caller could read back and replay to disable a guardrail."""
        guardrail = _RecordingCompressionGuardrail(guardrail_name="always-on-compression")
        import litellm

        litellm.logging_callback_manager.add_litellm_callback(guardrail)
        try:
            router = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-a"})])
            data = {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}]}
            await arm_pre_call(data=data, llm_router=router)
            assert "suppress" not in json.dumps(data).lower()
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(guardrail)

    @pytest.mark.asyncio
    async def test_model_side_guardrail_is_requested_even_when_not_default_on(self, monkeypatch):
        import litellm
        from litellm.proxy.guardrails import guardrail_registry

        monkeypatch.setitem(guardrail_registry.guardrail_class_registry, "headroom", _RecordingCompressionGuardrail)
        active = _RecordingCompressionGuardrail(guardrail_name="headroom-b")
        litellm.logging_callback_manager.add_litellm_callback(active)
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
        try:
            await arm_pre_call(data=data, llm_router=router)
            assert data["metadata"]["guardrails"] == ["headroom-b"]
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(active)

    @pytest.mark.asyncio
    async def test_arm_pre_call_keeps_no_copy_of_the_prompt(self):
        """Regression (security): arm_pre_call runs before the guardrails, so any copy it
        kept would be pre-masking text that routing then POSTs to an external service."""
        router = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-a"})])
        data = {"model": "smart-router", "messages": [{"role": "user", "content": "my ssn is 123-45-6789"}]}

        await arm_pre_call(data=data, llm_router=router)

        assert "123-45-6789" not in json.dumps(data.get("metadata", {}))
        assert not hasattr(auto_router_compression, "_routing_messages_snapshot")


class TestMessagesForRouting:
    @pytest.mark.asyncio
    async def test_no_policy_returns_none(self):
        assert await messages_for_routing(policy=None, messages=[], request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_routing_none_with_no_model_compression_returns_none(self):
        """Nothing compressed either hop, so the caller's own messages are already right."""
        policy = AutoRouterCompressionPolicy(routing=None, model=None)
        assert await messages_for_routing(policy=policy, messages=[], request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_routing_none_never_reaches_for_a_pre_guardrail_copy(self):
        """No uncompressed copy survives the model hop, and keeping one would mean
        retaining the pre-masking text. Routing reads what it has."""
        policy = AutoRouterCompressionPolicy(routing=None, model="headroom-a")
        model_compressed = [{"role": "user", "content": "[COMPRESSED] the full original conversation"}]

        assert await messages_for_routing(policy=policy, messages=model_compressed, request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_unknown_guardrail_name_routes_on_the_uncompressed_messages(self):
        policy = AutoRouterCompressionPolicy(routing="does-not-exist", model=None)
        messages = [{"role": "user", "content": "hi"}]
        result = await messages_for_routing(policy=policy, messages=messages, request_kwargs={})
        assert result == messages

    @pytest.mark.asyncio
    async def test_compresses_via_the_named_guardrail(self, registered_guardrail):
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model=None)
        messages = [{"role": "user", "content": "hello world"}]
        result = await messages_for_routing(policy=policy, messages=messages, request_kwargs={})
        assert result == [{"role": "user", "content": "[COMPRESSED] hello world"}]

    @pytest.mark.asyncio
    async def test_routing_compresses_what_the_other_guardrails_left_behind(self, registered_guardrail):
        """Regression (security): routing POSTs its input out, so it must read what the
        earlier guardrails left behind, not a pre-masking copy."""
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model="headroom-b")
        masked = [{"role": "user", "content": "my ssn is [REDACTED]"}]

        result = await messages_for_routing(policy=policy, messages=masked, request_kwargs={})

        assert result == [{"role": "user", "content": "[COMPRESSED] my ssn is [REDACTED]"}]
        assert registered_guardrail.request_data_seen[0]["messages"] == masked

    @pytest.mark.asyncio
    async def test_a_non_compression_guardrail_is_never_invoked_for_routing(self, monkeypatch):
        """Regression (security): naming an ordinary guardrail must not turn the routing
        hop into a way to ship prompts to whatever service backs it."""
        import litellm

        other = _NonCompressionGuardrail(guardrail_name="pii-filter")
        litellm.logging_callback_manager.add_litellm_callback(other)
        try:
            policy = AutoRouterCompressionPolicy(routing="pii-filter", model=None)
            messages = [{"role": "user", "content": "my ssn is 123-45-6789"}]

            result = await messages_for_routing(policy=policy, messages=messages, request_kwargs={})

            assert other.called is False
            assert result == messages
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(other)

    @pytest.mark.asyncio
    async def test_guardrail_receives_a_throwaway_request_data_not_the_real_request_kwargs(self, registered_guardrail):
        """Regression: a guardrail writes stats onto the request_data it is given, so
        passing the caller's own would double-count into extract_compression_saved_tokens."""
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model=None)
        messages = [{"role": "user", "content": "hi"}]
        request_kwargs = {"metadata": {}}
        await messages_for_routing(policy=policy, messages=messages, request_kwargs=request_kwargs)
        assert registered_guardrail.request_data_seen[0] is not request_kwargs
        assert request_kwargs == {"metadata": {}}
