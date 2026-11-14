from collections import Counter

import pytest

from litellm.integrations.custom_logger_v2 import (
    HOOK_SPECS,
    Abort,
    Field,
    Hook,
    InvalidDeclarationError,
    Reroute,
    Respond,
    Subscription,
)

F = Field

_LIFECYCLE_HOOKS = [hook for hook in Hook if not HOOK_SPECS[hook].request_path]


def test_every_hook_has_exactly_one_table_row() -> None:
    assert set(HOOK_SPECS) == set(Hook)


def test_each_legacy_method_maps_onto_one_hook_point() -> None:
    counts = Counter(method for spec in HOOK_SPECS.values() for method in spec.legacy_methods)
    assert [method for method, count in counts.items() if count > 1] == []


def test_legacy_methods_are_real_custom_logger_methods() -> None:
    from litellm.integrations.custom_logger import CustomLogger

    missing = sorted(
        method
        for spec in HOOK_SPECS.values()
        for method in spec.legacy_methods
        if not callable(getattr(CustomLogger, method, None))
    )
    assert missing == []


def test_legacy_guardrail_events_are_real_guardrail_event_hooks() -> None:
    from litellm.types.guardrails import GuardrailEventHooks

    events = {event for spec in HOOK_SPECS.values() for event in spec.legacy_guardrail_events}
    assert events <= {member.value for member in GuardrailEventHooks}
    assert events == {"pre_call", "during_call", "post_call", "post_mcp_call"}


def test_outcomes_are_allowed_only_on_request_path_hooks() -> None:
    assert {hook for hook in Hook if HOOK_SPECS[hook].outcomes} <= {
        hook for hook in Hook if HOOK_SPECS[hook].request_path
    }


def test_reroute_is_allowed_only_before_the_provider_call_on_pre_call() -> None:
    assert {hook for hook in Hook if Reroute in HOOK_SPECS[hook].outcomes} == {Hook.pre_call}


def test_reroute_is_the_only_non_terminal_outcome() -> None:
    assert (Abort.terminal, Respond.terminal, Reroute.terminal) == (True, True, False)


@pytest.mark.parametrize("hook", _LIFECYCLE_HOOKS)
def test_abort_on_a_lifecycle_hook_is_rejected(hook: Hook) -> None:
    with pytest.raises(InvalidDeclarationError, match="allows no outcome"):
        Subscription(hook=hook, handler="h", may_abort=True)


def test_abort_on_response_headers_is_rejected_because_its_errors_are_swallowed() -> None:
    with pytest.raises(InvalidDeclarationError, match="allows no outcome"):
        Subscription(hook=Hook.response_headers, handler="h", writes=frozenset({F.response_headers}), may_abort=True)


@pytest.mark.parametrize(
    ("hook", "field"),
    [
        (Hook.during_call, F.messages),
        (Hook.post_call_success, F.messages),
        (Hook.pre_call, F.response),
        (Hook.pre_call, F.deprecated_kwargs),
        (Hook.attempt_ended, F.metadata),
        (Hook.provider_request_sent, F.provider_request_body),
        (Hook.log_prepare, F.messages),
    ],
)
def test_write_the_hook_cannot_apply_is_rejected(hook: Hook, field: Field) -> None:
    with pytest.raises(InvalidDeclarationError, match=f"writes {field.value}, which this hook point cannot apply"):
        Subscription(hook=hook, handler="h", writes=frozenset({field}))


def test_pre_call_may_write_tools_model_and_input() -> None:
    writes = frozenset({F.tools, F.model, F.input, F.messages, F.metadata, F.guardrail_info})
    assert Subscription(hook=Hook.pre_call, handler="h", writes=writes, may_abort=True).writes == writes


def test_log_prepare_may_rewrite_the_standard_logging_payload() -> None:
    writes = frozenset({F.standard_logging_payload})
    assert Subscription(hook=Hook.log_prepare, handler="h", writes=writes).writes == writes


def test_stream_chunk_cannot_be_subscribed_while_streams_are_out_of_scope() -> None:
    with pytest.raises(InvalidDeclarationError, match="no V2 subscription support"):
        Subscription(hook=Hook.stream_chunk, handler="h", reads=frozenset({F.stream_chunk}))


@pytest.mark.parametrize(
    ("writes", "may_abort"),
    [(frozenset({F.metadata}), False), (frozenset[Field](), True)],
)
def test_awaited_subscription_that_writes_or_aborts_is_rejected(writes: frozenset[Field], may_abort: bool) -> None:
    with pytest.raises(InvalidDeclarationError, match="awaited subscription must neither write nor abort"):
        Subscription(hook=Hook.pre_call, handler="h", writes=writes, may_abort=may_abort, awaited=True)


def test_awaited_read_only_subscription_is_accepted_on_a_lifecycle_hook() -> None:
    subscription = Subscription(
        hook=Hook.call_ended, handler="release", reads=frozenset({F.key_identity}), awaited=True
    )
    assert (subscription.awaited, subscription.writes, subscription.may_abort) == (True, frozenset(), False)


@pytest.mark.parametrize(
    ("writes", "may_abort"),
    [(frozenset[Field](), False), (frozenset({F.messages}), True)],
)
def test_parallel_is_only_for_abort_only_subscriptions(writes: frozenset[Field], may_abort: bool) -> None:
    with pytest.raises(InvalidDeclarationError, match="parallel is only for"):
        Subscription(hook=Hook.pre_call, handler="h", writes=writes, may_abort=may_abort, parallel=True)
