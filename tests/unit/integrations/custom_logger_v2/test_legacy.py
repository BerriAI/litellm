import asyncio
import gc
from collections.abc import Awaitable, Generator
from contextvars import ContextVar
from typing import Final

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import (
    CallContext,
    ContractViolation,
    CustomLoggerV2,
    Field,
    Hook,
    InvalidPatchError,
    Patch,
    on,
)
from litellm.integrations.custom_logger_v2.host import reset_host_for_tests
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    adapter_for,
    clear_subscribers_cache_for_tests,
    hook_subscribers,
    kwargs_after,
    kwargs_patch,
    run_subscribers,
    run_subscribers_sync,
)
from litellm.types.guardrails import GuardrailEventHooks

F = Field


@pytest.fixture(autouse=True)
def fresh() -> Generator[None]:
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()
    yield
    reset_host_for_tests()
    clear_subscribers_cache_for_tests()


class PreCallOnly(CustomLogger):
    async def async_pre_call_hook(self, *args: object, **kwargs: object) -> None:
        return None


class InheritsPreCall(PreCallOnly):
    pass


class Moderator(CustomLogger):
    async def async_moderation_hook(self, *args: object, **kwargs: object) -> None:
        return None


class Exporter(CustomLogger):
    def log_success_event(self, *args: object, **kwargs: object) -> None:
        return None


class Masking(CustomLogger):
    def logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        return kwargs, result


class Writer(CustomLoggerV2):
    def __init__(self, journal: list[object], replacement: object) -> None:
        self.journal = journal
        self.replacement = replacement

    @on(Hook.pre_call, reads={F.messages}, writes={F.messages}, may_abort=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> Patch:
        self.journal.append(("v2", getattr(view, "messages")))
        return Patch().set(F.messages, self.replacement)


class Tagger(CustomLoggerV2):
    def __init__(self, journal: list[object]) -> None:
        self.journal = journal

    @on(Hook.pre_call, writes={F.model})
    async def pre_call(self, view: object, ctx: CallContext[object]) -> Patch:
        self.journal.append("tagger")
        return Patch().set(F.model, "tagged")


class Observer(CustomLoggerV2):
    def __init__(self, journal: list[object]) -> None:
        self.journal = journal

    @on(Hook.attempt_ended, reads={F.deprecated_kwargs})
    async def ended(self, view: object, ctx: CallContext[object]) -> None:
        self.journal.append("observer")


class BlockingAttemptEnded(CustomLoggerV2):
    @on(Hook.log_prepare, writes={F.standard_logging_payload})
    async def prepare(self, view: object, ctx: CallContext[object]) -> None:
        return None


def test_manifest_subscribes_only_the_hooks_the_class_overrides_through_its_mro() -> None:
    direct: Final = adapter_for(PreCallOnly()).manifest.subscriptions
    inherited: Final = adapter_for(InheritsPreCall()).manifest.subscriptions

    assert set(direct) == {Hook.pre_call}
    assert set(inherited) == {Hook.pre_call}
    assert direct[Hook.pre_call].reads == {F.deprecated_kwargs, F.deprecated_response_obj}
    assert direct[Hook.pre_call].writes == {F.deprecated_kwargs}
    assert (direct[Hook.pre_call].may_abort, direct[Hook.pre_call].parallel) == (True, False)


def test_during_call_and_parallel_guardrails_gather_and_write_nothing() -> None:
    moderation: Final = adapter_for(Moderator()).manifest.subscriptions[Hook.during_call]
    guardrail: Final = adapter_for(
        CustomGuardrail(guardrail_name="g", event_hook=GuardrailEventHooks.pre_call, run_in_parallel=True)
    ).manifest.subscriptions

    assert (moderation.writes, moderation.may_abort, moderation.parallel) == (frozenset(), True, True)
    assert (guardrail[Hook.pre_call].writes, guardrail[Hook.pre_call].parallel) == (frozenset(), True)
    assert guardrail[Hook.during_call].parallel
    assert guardrail[Hook.mcp_tool_result].writes == {F.deprecated_kwargs}


def test_log_hooks_neither_write_nor_abort_but_log_prepare_keeps_its_kwargs_rewrite() -> None:
    attempt: Final = adapter_for(Exporter()).manifest.subscriptions[Hook.attempt_ended]
    prepare: Final = adapter_for(Masking()).manifest.subscriptions[Hook.log_prepare]

    assert (attempt.writes, attempt.may_abort) == (frozenset(), False)
    assert (prepare.writes, prepare.may_abort) == ({F.deprecated_kwargs}, False)


def test_every_logger_admits_loggers_that_do_not_override_the_hook() -> None:
    exporter: Final = Exporter()

    assert hook_subscribers(Hook.pre_call, [exporter]).entries == ()
    assert len(hook_subscribers(Hook.pre_call, [exporter], admit=Admit.every_logger).entries) == 1


def test_adapters_are_cached_per_instance_and_dropped_with_it() -> None:
    logger = PreCallOnly()
    adapter = adapter_for(logger)

    assert adapter_for(logger) is adapter
    assert adapter.inner is logger
    del logger
    gc.collect()
    assert adapter.inner is None


def test_subscribers_are_cached_per_list_content_and_rebuilt_when_it_changes() -> None:
    first: Final = PreCallOnly()
    callbacks: Final[list[object]] = ["a-string-callback", first]
    subscribers: Final = hook_subscribers(Hook.pre_call, callbacks)

    assert hook_subscribers(Hook.pre_call, callbacks) is subscribers
    callbacks.append(PreCallOnly())
    grown: Final = hook_subscribers(Hook.pre_call, callbacks)
    assert grown is not subscribers
    assert len(grown.entries) == 2


def test_a_cached_subscribers_is_dropped_when_a_legacy_logger_in_it_dies() -> None:
    for _ in range(5):
        dying = PreCallOnly()
        _ = hook_subscribers(Hook.pre_call, [dying])
        del dying
        gc.collect()
        replacement = PreCallOnly()

        assert hook_subscribers(Hook.pre_call, [replacement]).legacy == (replacement,)


def test_legacy_kwargs_conflict_with_every_field_so_list_order_is_kept() -> None:
    journal: Final[list[object]] = []
    subscribers: Final = hook_subscribers(Hook.pre_call, [Tagger(journal), PreCallOnly(), Tagger(journal)])

    assert [len(stage) for stage in subscribers.plan.stages] == [1, 1, 1]
    assert [entry.position for stage in subscribers.plan.stages for entry in stage] == [0, 1, 2]


def _recording(journal: list[object], name: str, replacement: object = None):
    def start(logger: object, data: object) -> Awaitable[Patch | None]:
        async def run() -> Patch | None:
            journal.append((name, KwargsFields(data, PROXY_REQUEST_KEYS)[F.messages]))
            return None if replacement is None else kwargs_patch(data, replacement)

        return run()

    return start


async def test_a_v2_callback_between_two_legacy_entries_sees_the_first_edit_and_patches_what_the_second_sees() -> None:
    journal: Final[list[object]] = []
    first: Final = PreCallOnly()
    second: Final = PreCallOnly()
    subscribers: Final = hook_subscribers(Hook.pre_call, [first, Writer(journal, ["v2"]), second])
    replaced: Final[dict[str, object]] = {"messages": ["first"]}

    def start(logger: object, data: object) -> Awaitable[Patch | None]:
        async def run() -> Patch | None:
            journal.append(
                ("first" if logger is first else "second", KwargsFields(data, PROXY_REQUEST_KEYS)[F.messages])
            )
            return kwargs_patch(data, replaced) if logger is first else None

        return run()

    result: Final = await run_subscribers(
        subscribers, KwargsFields({"messages": ["original"]}, PROXY_REQUEST_KEYS), start
    )

    assert journal == [("first", ["original"]), ("v2", ["first"]), ("second", ["v2"])]
    assert kwargs_after(result) is replaced
    assert replaced["messages"] == ["v2"]


async def test_a_v2_callback_writer_without_abort_still_waits_for_the_legacy_entry_before_it() -> None:
    journal: Final[list[object]] = []
    callbacks: Final[list[object]] = [PreCallOnly(), Tagger(journal)]
    subscribers: Final = hook_subscribers(Hook.pre_call, callbacks)
    data: Final[dict[str, object]] = {"model": "m", "messages": []}

    def start(logger: object, kwargs: object) -> Awaitable[None]:
        async def run() -> None:
            journal.append("legacy")
            await asyncio.sleep(0)

        return run()

    await run_subscribers(subscribers, KwargsFields(data, PROXY_REQUEST_KEYS), start)

    assert journal == ["legacy", "tagger"]
    assert data["model"] == "tagged"


REQUEST_STASH: Final[ContextVar[object]] = ContextVar("request_stash", default=None)


class StashClaimer(CustomLoggerV2):
    def __init__(self, stash: object) -> None:
        self.stash = stash

    @on(Hook.pre_call, writes={F.metadata}, may_abort=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> None:
        _ = REQUEST_STASH.set(self.stash)


async def test_a_blocking_v2_callback_sets_context_a_later_legacy_callback_and_the_caller_see() -> None:
    stash: Final = object()
    seen: Final[list[object]] = []
    callbacks: Final[list[object]] = [StashClaimer(stash), PreCallOnly()]

    def start(logger: object, data: object) -> Awaitable[None]:
        async def run() -> None:
            seen.append(REQUEST_STASH.get())

        return run()

    await run_subscribers(
        hook_subscribers(Hook.pre_call, callbacks),
        KwargsFields({"messages": []}, PROXY_REQUEST_KEYS),
        start,
    )

    assert (seen, REQUEST_STASH.get()) == ([stash], stash)


async def test_legacy_only_and_dispatched_subscribers_give_the_same_result() -> None:
    first: Final = PreCallOnly()
    second: Final = PreCallOnly()
    replacement: Final[dict[str, object]] = {"messages": ["replaced"]}

    async def run(callbacks: list[object]) -> tuple[list[object], object]:
        journal: list[object] = []
        subscribers = hook_subscribers(Hook.pre_call, callbacks)
        result = await run_subscribers(
            subscribers,
            KwargsFields({"messages": ["original"]}, PROXY_REQUEST_KEYS),
            _recording(journal, "x", replacement),
        )
        return journal, kwargs_after(result)

    plain: Final = await run([first, second])
    dispatched: Final = await run([first, second, Observer([])])

    assert plain == dispatched
    assert plain[1] is replacement


async def test_a_legacy_exception_propagates_unchanged_through_both_paths() -> None:
    error: Final = RuntimeError("legacy said no")

    def start(logger: object, data: object) -> Awaitable[None]:
        async def run() -> None:
            raise error

        return run()

    for callbacks in ([PreCallOnly()], [PreCallOnly(), Writer([], [])]):
        with pytest.raises(RuntimeError) as raised:
            await run_subscribers(
                hook_subscribers(Hook.pre_call, callbacks), KwargsFields({"messages": []}, PROXY_REQUEST_KEYS), start
            )
        assert raised.value is error


def test_kwargs_fields_write_through_to_the_live_request() -> None:
    metadata: Final[dict[str, object]] = {"kept": 1}
    data: Final[dict[str, object]] = {"model": "m", "metadata": metadata}
    fields: Final = KwargsFields(data, {**PROXY_REQUEST_KEYS, F.metadata: "metadata"})

    patched: Final = fields.apply(Patch().set(F.model, "n").merge(F.metadata, {"added": 2}))

    assert patched[F.deprecated_kwargs] is data
    assert data == {"model": "n", "metadata": {"kept": 1, "added": 2}}
    assert data["metadata"] is metadata
    assert F.messages not in fields
    with pytest.raises(InvalidPatchError, match="no place"):
        fields.apply(Patch().set(F.request_headers, {}))


def test_sync_subscribers_run_legacy_inline_in_order_and_hand_v2_callbacks_to_the_off_path() -> None:
    journal: Final[list[object]] = []
    first: Final = Exporter()
    second: Final = Exporter()
    subscribers: Final = hook_subscribers(Hook.attempt_ended, [first, Observer(journal), second])

    def start(logger: object, data: object) -> None:
        journal.append("first" if logger is first else "second")

    run_subscribers_sync(subscribers, KwargsFields({}, {}), start)

    assert journal[:2] == ["first", "second"]


class SyncStamp(CustomLoggerV2):
    @on(Hook.log_prepare, reads={F.standard_logging_payload}, writes={F.standard_logging_payload})
    async def prepare(self, view: object, ctx: CallContext[object]) -> Patch:
        return Patch().merge(F.standard_logging_payload, {"stamped": True})


def test_sync_subscribers_run_a_blocking_v2_callback_inline_in_list_order() -> None:
    journal: Final[list[object]] = []
    details: Final[dict[str, object]] = {"standard_logging_object": {}}
    masking: Final = Masking()
    subscribers: Final = hook_subscribers(Hook.log_prepare, [SyncStamp(), masking, BlockingAttemptEnded()])

    def start(logger: object, data: object) -> None:
        journal.append((logger, data))

    run_subscribers_sync(
        subscribers, KwargsFields(details, {F.standard_logging_payload: "standard_logging_object"}), start
    )

    assert journal == [(masking, details)]
    assert details["standard_logging_object"] == {"stamped": True}


class AwaitedEnd(CustomLoggerV2):
    def __init__(self, journal: list[object]) -> None:
        self.journal = journal

    @on(Hook.attempt_ended, awaited=True)
    async def ended(self, view: object, ctx: CallContext[object]) -> None:
        self.journal.append("awaited")


@pytest.mark.parametrize("hook", [Hook.log_prepare, Hook.attempt_ended])
def test_a_sync_line_that_runs_v2_callbacks_inline_keeps_the_threads_event_loop(hook: Hook) -> None:
    journal: Final[list[object]] = []
    details: Final[dict[str, object]] = {"standard_logging_object": {}}
    callbacks: Final = [SyncStamp(), SyncStamp(), AwaitedEnd(journal), AwaitedEnd(journal)]
    loop: Final = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        run_subscribers_sync(
            hook_subscribers(hook, callbacks),
            KwargsFields(details, {F.standard_logging_payload: "standard_logging_object"}),
            lambda logger, data: None,
        )
        current = asyncio.get_event_loop()
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert current is loop
    assert (journal, details["standard_logging_object"]) == (
        (["awaited", "awaited"], {}) if hook is Hook.attempt_ended else ([], {"stamped": True})
    )


def test_every_entry_keeps_string_and_callable_callbacks_in_list_order() -> None:
    journal: Final[list[object]] = []

    def plain_callback() -> None:
        return None

    exporter: Final = Exporter()
    callbacks: Final[list[object]] = ["langfuse", plain_callback, exporter]

    def start(logger: object, data: object) -> None:
        journal.append(logger)

    run_subscribers_sync(
        hook_subscribers(Hook.attempt_ended, callbacks, admit=Admit.every_entry), KwargsFields({}, {}), start
    )

    assert journal == ["langfuse", plain_callback, exporter]
    assert hook_subscribers(Hook.attempt_ended, callbacks).legacy == (exporter,)
