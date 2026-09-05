"""
Unit tests for litellm.proxy.guardrails.auto_router_compression.

Covers:
- policy_from_litellm_params: absent keys mean no policy; the "none" sentinel
  normalizes to explicit no-compression within an active policy; is_same
- policy_for_model: finds the auto-router marker deployment for an alias, and
  picks the tag-scoped marker the request's tags actually match
- arm_pre_call: no-op without a policy; suppresses active compression guardrails
  with a forgery-proof marker; arms the model-side guardrail even when it isn't
  default_on; keeps the pre-compression snapshot out of persisted metadata
- messages_for_routing: no-op without a policy; routes on the pre-compression
  snapshot when the two hops differ; compresses via the named guardrail's
  apply_guardrail; never writes stats onto the caller's own request_kwargs
  (regression for double-counted compression savings)
"""

import json
from typing import Any

import pytest

from litellm.constants import AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY
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
            suppressed = data["metadata"][AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY]
            assert tuple(suppressed) == (always_on.auto_router_suppression_marker(),)
            # The bare name alone must never suppress: that is what a caller could forge.
            assert "always-on-compression" not in suppressed
            assert always_on.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is False
        finally:
            litellm.logging_callback_manager.remove_callback_from_all_lists(always_on)

    @pytest.mark.asyncio
    async def test_a_caller_cannot_suppress_a_guardrail_by_naming_it_in_metadata(self):
        """Regression: request metadata is caller-controlled, so a bare guardrail name
        there must not switch off a PII, content-filter, or compression guardrail."""
        guardrail = _RecordingCompressionGuardrail(guardrail_name="always-on-compression")
        forged = {
            "model": "smart-router",
            "metadata": {AUTO_ROUTER_SUPPRESSED_COMPRESSION_GUARDRAILS_KEY: ["always-on-compression"]},
        }

        assert guardrail._suppressed_by_auto_router_compression(forged) is False

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
        await arm_pre_call(data=data, llm_router=router)
        assert data["metadata"]["guardrails"] == ["headroom-b"]

    @pytest.mark.asyncio
    async def test_snapshot_never_lands_in_persisted_metadata(self):
        """Regression: refresh_proxy_server_request_body_snapshot copies metadata into
        proxy_server_request.body, which deployments persist to spend logs. The
        pre-compression snapshot holds the prompt before any masking guardrail ran, so
        it must live outside anything that gets serialized."""
        router = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-a"})])
        original_messages = [{"role": "user", "content": "my ssn is 123-45-6789"}]
        data = {"model": "smart-router", "messages": original_messages}

        await arm_pre_call(data=data, llm_router=router)

        assert "123-45-6789" not in json.dumps(data["metadata"])
        assert [dict(m) for m in auto_router_compression._snapshot_messages()] == original_messages

    @pytest.mark.asyncio
    async def test_snapshot_is_a_copy_not_the_live_message_list(self):
        router = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-a"})])
        original_messages = [{"role": "user", "content": "hi"}]

        await arm_pre_call(data={"model": "smart-router", "messages": original_messages}, llm_router=router)
        original_messages[0]["content"] = "mutated after the snapshot"

        assert [dict(m) for m in auto_router_compression._snapshot_messages()] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_a_request_without_a_policy_clears_a_previous_snapshot(self):
        router_with = _FakeRouter([_marker({"auto_router_routing_compression": "headroom-a"})])
        await arm_pre_call(
            data={"model": "smart-router", "messages": [{"role": "user", "content": "first"}]}, llm_router=router_with
        )

        router_without = _FakeRouter([{"model_name": "plain", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
        await arm_pre_call(
            data={"model": "plain", "messages": [{"role": "user", "content": "second"}]}, llm_router=router_without
        )

        assert auto_router_compression._snapshot_messages() is None


class TestMessagesForRouting:
    @pytest.fixture(autouse=True)
    def _clear_snapshot(self):
        auto_router_compression._routing_messages_snapshot.set(None)
        yield
        auto_router_compression._routing_messages_snapshot.set(None)

    @pytest.mark.asyncio
    async def test_no_policy_returns_none(self):
        assert await messages_for_routing(policy=None, messages=[], request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_routing_none_with_no_model_compression_returns_none(self):
        """Nothing compressed either hop, so the caller's own messages are already right."""
        policy = AutoRouterCompressionPolicy(routing=None, model=None)
        assert await messages_for_routing(policy=policy, messages=[], request_kwargs={}) is None

    @pytest.mark.asyncio
    async def test_routing_none_with_model_compression_routes_on_the_snapshot(self):
        """Regression: with routing explicitly off and the model side compressed, the
        messages in hand are the model-side guardrail's output. Routing asked for no
        compression, so it must read the pre-compression snapshot instead."""
        original = [{"role": "user", "content": "the full original conversation"}]
        auto_router_compression._routing_messages_snapshot.set(tuple(dict(m) for m in original))
        policy = AutoRouterCompressionPolicy(routing=None, model="headroom-a")
        model_compressed = [{"role": "user", "content": "[COMPRESSED] the full original conversation"}]

        result = await messages_for_routing(policy=policy, messages=model_compressed, request_kwargs={})

        assert result == original

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
    async def test_uses_the_snapshot_when_present(self, registered_guardrail):
        policy = AutoRouterCompressionPolicy(routing="fake-compress", model="headroom-b")
        auto_router_compression._routing_messages_snapshot.set(({"role": "user", "content": "original"},))
        # `messages` here stands in for whatever the model-side guardrail already
        # rewrote `data["messages"]` to -- routing must ignore it and compress the
        # pristine snapshot instead.
        already_rewritten = [{"role": "user", "content": "rewritten by another guardrail"}]
        result = await messages_for_routing(policy=policy, messages=already_rewritten, request_kwargs={})
        assert result == [{"role": "user", "content": "[COMPRESSED] original"}]

    @pytest.mark.asyncio
    async def test_guardrail_receives_a_throwaway_request_data_not_the_real_request_kwargs(self, registered_guardrail):
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
