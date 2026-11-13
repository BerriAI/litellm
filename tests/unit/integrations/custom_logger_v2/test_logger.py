from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest

from litellm.integrations.custom_logger_v2 import (
    Abort,
    Activation,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    InvalidDeclarationError,
    Patch,
    Selection,
    Subscription,
    build_manifest,
    on,
)

F = Field


class Prometheus(CustomLoggerV2):
    @on(Hook.attempt_ended, reads={F.outcome, F.deployment, F.key_identity, F.usage, F.cost, F.request_tags})
    async def attempt_ended(self, view: object, ctx: CallContext[None]) -> None: ...

    @on(Hook.call_ended, reads={F.outcome, F.deployment, F.key_identity, F.error})
    async def call_ended(self, view: object, ctx: CallContext[None]) -> None: ...


class PiiMask(CustomLoggerV2):
    @dataclass
    class request_state:
        tokens: dict[str, str] = field(default_factory=dict)

    @on(Hook.pre_call, reads={F.messages}, writes={F.messages}, may_abort=True)
    async def mask(self, view: object, ctx: CallContext[object]) -> Patch | None: ...

    @on(Hook.post_call_success, reads={F.response}, writes={F.response})
    async def unmask(self, view: object, ctx: CallContext[object]) -> Patch | None: ...


class ParallelRequestLimiter(CustomLoggerV2):
    selection = Selection("parallel_request_limiter", activation=Activation.required)

    @on(Hook.pre_call, reads={F.key_identity}, may_abort=True)
    async def acquire(self, view: object, ctx: CallContext[None]) -> Abort | None: ...

    @on(Hook.response_headers, reads={F.key_identity}, writes={F.response_headers})
    async def headers(self, view: object, ctx: CallContext[None]) -> Patch: ...

    @on(Hook.call_ended, reads={F.key_identity, F.usage, F.outcome})
    async def release(self, view: object, ctx: CallContext[None]) -> None: ...


def test_prometheus_example_declares_two_read_only_lifecycle_subscriptions() -> None:
    assert dict(Prometheus.subscriptions) == {
        Hook.attempt_ended: Subscription(
            hook=Hook.attempt_ended,
            handler="attempt_ended",
            reads=frozenset({F.outcome, F.deployment, F.key_identity, F.usage, F.cost, F.request_tags}),
        ),
        Hook.call_ended: Subscription(
            hook=Hook.call_ended,
            handler="call_ended",
            reads=frozenset({F.outcome, F.deployment, F.key_identity, F.error}),
        ),
    }


def test_pii_mask_example_declares_its_writes_and_request_state() -> None:
    assert dict(PiiMask.subscriptions) == {
        Hook.pre_call: Subscription(
            hook=Hook.pre_call,
            handler="mask",
            reads=frozenset({F.messages}),
            writes=frozenset({F.messages}),
            may_abort=True,
        ),
        Hook.post_call_success: Subscription(
            hook=Hook.post_call_success,
            handler="unmask",
            reads=frozenset({F.response}),
            writes=frozenset({F.response}),
        ),
    }
    assert build_manifest(PiiMask()).request_state is PiiMask.request_state


def test_parallel_request_limiter_example_is_required_and_aborts_without_writing() -> None:
    manifest = build_manifest(ParallelRequestLimiter())
    assert manifest.selection == Selection("parallel_request_limiter", activation=Activation.required)
    assert dict(manifest.subscriptions) == {
        Hook.pre_call: Subscription(
            hook=Hook.pre_call, handler="acquire", reads=frozenset({F.key_identity}), may_abort=True
        ),
        Hook.response_headers: Subscription(
            hook=Hook.response_headers,
            handler="headers",
            reads=frozenset({F.key_identity}),
            writes=frozenset({F.response_headers}),
        ),
        Hook.call_ended: Subscription(
            hook=Hook.call_ended, handler="release", reads=frozenset({F.key_identity, F.usage, F.outcome})
        ),
    }


def test_a_declaration_the_hook_rejects_fails_when_the_class_is_defined() -> None:
    with pytest.raises(InvalidDeclarationError, match="writes response, which this hook point cannot apply"):

        class Broken(CustomLoggerV2):
            @on(Hook.pre_call, writes={F.response})
            async def handle(self, view: object, ctx: CallContext[None]) -> None: ...


def test_during_call_abort_subscriptions_are_parallel_unless_they_opt_out() -> None:
    class Moderation(CustomLoggerV2):
        @on(Hook.during_call, reads={F.messages}, may_abort=True)
        async def moderate(self, view: object, ctx: CallContext[None]) -> Abort | None: ...

    class SerialModeration(CustomLoggerV2):
        @on(Hook.during_call, reads={F.messages}, may_abort=True, parallel=False)
        async def moderate(self, view: object, ctx: CallContext[None]) -> Abort | None: ...

    class Watcher(CustomLoggerV2):
        @on(Hook.during_call, reads={F.messages})
        async def watch(self, view: object, ctx: CallContext[None]) -> None: ...

    assert Moderation.subscriptions[Hook.during_call].parallel is True
    assert SerialModeration.subscriptions[Hook.during_call].parallel is False
    assert Watcher.subscriptions[Hook.during_call].parallel is False


def test_pre_call_abort_subscriptions_stay_serial_by_default() -> None:
    assert ParallelRequestLimiter.subscriptions[Hook.pre_call].parallel is False


def test_two_handlers_for_one_hook_in_one_class_are_rejected() -> None:
    with pytest.raises(InvalidDeclarationError, match="more than one handler for pre_call"):

        class Twice(CustomLoggerV2):
            @on(Hook.pre_call, reads={F.messages})
            async def first(self, view: object, ctx: CallContext[None]) -> None: ...

            @on(Hook.pre_call, reads={F.metadata})
            async def second(self, view: object, ctx: CallContext[None]) -> None: ...


@pytest.mark.parametrize(
    ("hook", "writes", "may_abort", "awaited"),
    [
        (Hook.pre_call, frozenset({F.metadata}), False, False),
        (Hook.pre_call, frozenset[Field](), True, False),
        (Hook.call_ended, frozenset[Field](), False, True),
    ],
)
def test_sync_handler_that_blocks_the_request_is_rejected(
    hook: Hook, writes: frozenset[Field], may_abort: bool, awaited: bool
) -> None:
    with pytest.raises(InvalidDeclarationError, match="handle must be async"):

        class Sync(CustomLoggerV2):
            @on(hook, writes=writes, may_abort=may_abort, awaited=awaited)
            def handle(self, view: object, ctx: CallContext[None]) -> None: ...


def test_sync_handler_is_allowed_for_off_path_reads() -> None:
    class SyncLogger(CustomLoggerV2):
        @on(Hook.attempt_ended, reads={F.usage})
        def log(self, view: object, ctx: CallContext[None]) -> None: ...

    assert SyncLogger.subscriptions[Hook.attempt_ended].handler == "log"


async def test_undecorated_override_keeps_the_parent_subscription_and_runs_the_override() -> None:
    override_patch = Patch().set(F.messages, ["masked"])

    class Masked(PiiMask):
        async def mask(self, view: object, ctx: CallContext[object]) -> Patch | None:
            return override_patch

    subscription = Masked.subscriptions[Hook.pre_call]
    handler: Callable[[object, CallContext[object]], Awaitable[Patch | None]] = getattr(Masked(), subscription.handler)
    assert subscription is PiiMask.subscriptions[Hook.pre_call]
    assert await handler(None, CallContext(hook=Hook.pre_call, call=None, state=None)) is override_patch


def test_redecorated_override_replaces_the_parent_subscription() -> None:
    class WiderMask(PiiMask):
        @on(Hook.pre_call, reads={F.messages, F.tools}, writes={F.messages, F.tools}, may_abort=True)
        async def mask(self, view: object, ctx: CallContext[object]) -> Patch | None: ...

    assert WiderMask.subscriptions[Hook.pre_call].writes == frozenset({F.messages, F.tools})
    assert WiderMask.subscriptions[Hook.post_call_success] is PiiMask.subscriptions[Hook.post_call_success]


def test_new_handler_for_an_inherited_hook_replaces_the_parent_handler() -> None:
    class Renamed(PiiMask):
        @on(Hook.pre_call, reads={F.messages})
        async def inspect_only(self, view: object, ctx: CallContext[object]) -> None: ...

    assert Renamed.subscriptions[Hook.pre_call] == Subscription(
        hook=Hook.pre_call, handler="inspect_only", reads=frozenset({F.messages})
    )


def test_redecorating_an_inherited_handler_for_another_hook_is_rejected() -> None:
    with pytest.raises(InvalidDeclarationError, match="mask is decorated for pre_route but also inherits"):

        class Moved(PiiMask):
            @on(Hook.pre_route, reads={F.messages})
            async def mask(self, view: object, ctx: CallContext[object]) -> Patch | None: ...


def test_replacing_an_inherited_handler_with_a_non_callable_is_rejected() -> None:
    with pytest.raises(InvalidDeclarationError, match="unmask handles post_call_success but is not callable"):

        class Removed(PiiMask):
            unmask = None  # pyright: ignore[reportAssignmentType]  # the test removes the handler on purpose


def test_multiple_inheritance_merges_hooks_and_the_nearest_class_wins() -> None:
    class Masking(CustomLoggerV2):
        @on(Hook.pre_call, reads={F.messages}, writes={F.messages})
        async def mask(self, view: object, ctx: CallContext[None]) -> Patch | None: ...

    class Auditing(CustomLoggerV2):
        @on(Hook.call_ended, reads={F.outcome})
        async def audit(self, view: object, ctx: CallContext[None]) -> None: ...

        @on(Hook.pre_call, reads={F.metadata})
        async def peek(self, view: object, ctx: CallContext[None]) -> None: ...

    class MaskingAudit(Masking, Auditing):
        pass

    assert MaskingAudit.subscriptions == {
        Hook.pre_call: Masking.subscriptions[Hook.pre_call],
        Hook.call_ended: Auditing.subscriptions[Hook.call_ended],
    }
