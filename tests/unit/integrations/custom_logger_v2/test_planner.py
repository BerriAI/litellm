from collections.abc import Set

from litellm.integrations.custom_logger_v2 import (
    CallbackManifest,
    CustomLoggerV2,
    Field,
    Hook,
    HookPlan,
    Selection,
    Subscription,
    plan_hook,
)
from litellm.integrations.custom_logger_v2.planner import Entry

F = Field


def _entry(
    position: int,
    name: str,
    *,
    reads: Set[Field] = frozenset(),
    writes: Set[Field] = frozenset(),
    may_abort: bool = False,
    parallel: bool = False,
    awaited: bool = False,
    hook: Hook = Hook.pre_call,
) -> Entry:
    subscription = Subscription(
        hook=hook,
        handler=name,
        reads=frozenset(reads),
        writes=frozenset(writes),
        may_abort=may_abort,
        parallel=parallel,
        awaited=awaited,
    )
    manifest = CallbackManifest(
        logger=CustomLoggerV2(),
        selection=Selection(name),
        subscriptions={hook: subscription},
        request_state=None,
    )
    return Entry(position, manifest, subscription)


def _names(plan: HookPlan) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(entry.subscription.handler for entry in stage) for stage in plan.stages)


def test_design_doc_worked_example() -> None:
    entries = (
        _entry(0, "pii_mask", reads={F.messages}, writes={F.messages}, may_abort=True),
        _entry(1, "tagger", writes={F.metadata}),
        _entry(2, "limiter", reads={F.key_identity}, may_abort=True),
        _entry(3, "secret_scan", reads={F.messages}, writes={F.messages}, may_abort=True),
        _entry(4, "audit", reads={F.messages}),
    )

    plan = plan_hook(entries)

    assert _names(plan) == (("pii_mask", "tagger"), ("limiter",), ("secret_scan",))
    assert plan.off_path == (entries[4],)
    assert plan.awaited == ()
    assert plan.reads == {F.messages, F.key_identity}


def test_rule_one_only_writers_and_abortables_block() -> None:
    reader = _entry(0, "reader", reads={F.messages})
    awaited = _entry(1, "awaited", reads={F.key_identity}, awaited=True)
    writer = _entry(2, "writer", writes={F.metadata})

    plan = plan_hook((reader, awaited, writer))

    assert _names(plan) == (("writer",),)
    assert plan.off_path == (reader,)
    assert plan.awaited == (awaited,)


def test_rule_two_every_abortable_waits_for_all_earlier_blocking_entries() -> None:
    plan = plan_hook(
        (
            _entry(0, "tagger", writes={F.metadata}),
            _entry(1, "first", reads={F.key_identity}, may_abort=True),
            _entry(2, "second", reads={F.key_identity}, may_abort=True),
        )
    )

    assert _names(plan) == (("tagger",), ("first",), ("second",))


def test_rule_three_writers_wait_only_on_conflicting_earlier_writers() -> None:
    plan = plan_hook(
        (
            _entry(0, "slow_messages", writes={F.messages}),
            _entry(1, "metadata", writes={F.metadata}),
            _entry(2, "reads_metadata", reads={F.metadata}, writes={F.guardrail_info}),
            _entry(3, "rewrites_messages", reads={F.messages}, writes={F.messages}),
        )
    )

    assert _names(plan) == (("slow_messages", "metadata"), ("reads_metadata", "rewrites_messages"))


def test_rule_three_writer_never_lands_before_an_earlier_reader_of_its_field() -> None:
    plan = plan_hook(
        (
            _entry(0, "tagger", writes={F.metadata}),
            _entry(1, "scanner", reads={F.messages}, may_abort=True),
            _entry(2, "rewriter", writes={F.messages}),
        )
    )

    assert _names(plan) == (("tagger",), ("scanner", "rewriter"))


def test_rule_three_writer_without_conflicts_joins_the_first_stage_past_abortables() -> None:
    plan = plan_hook(
        (
            _entry(0, "limiter", reads={F.key_identity}, may_abort=True),
            _entry(1, "scanner", reads={F.key_identity}, may_abort=True),
            _entry(2, "tagger", writes={F.metadata}),
        )
    )

    assert _names(plan) == (("limiter", "tagger"), ("scanner",))


def test_parallel_abort_only_entries_run_together_after_the_serial_chain() -> None:
    plan = plan_hook(
        (
            _entry(0, "fast_check", reads={F.messages}, may_abort=True, parallel=True),
            _entry(1, "masker", reads={F.messages}, writes={F.messages}, may_abort=True),
            _entry(2, "slow_check", reads={F.messages}, may_abort=True, parallel=True),
        )
    )

    assert _names(plan) == (("masker",), ("fast_check", "slow_check"))


def test_no_entries_plan_nothing() -> None:
    assert plan_hook(()) == HookPlan()
