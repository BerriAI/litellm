import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter

from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import (
    BY_REFERENCE,
    DERIVED_ONLY,
    Activation,
    Field,
    FieldTier,
    Hook,
    OptOutBy,
)
from litellm.integrations.custom_logger_v2.hooks import HOOK_SPECS
from litellm.integrations.custom_logger_v2.logger import CustomLoggerV2, on
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.selection import Selection
from litellm.rust_bridge.execution import Execution, drive
from litellm.types.integrations.custom_logger_v2 import Strict


class RustHandle(Protocol):
    def manifest(self) -> str: ...

    def request_state(self) -> object: ...

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]: ...


@runtime_checkable
class HostedRust(Protocol):
    def start(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, host: object, /
    ) -> Execution: ...


def _streamless(execution: Execution, head: object) -> object:
    raise RuntimeError("a rust callback cannot open a stream")


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WireSubscription(_Wire):
    hook: Hook
    reads: frozenset[Field]
    writes: frozenset[Field]
    may_abort: bool
    parallel: bool
    awaited: bool
    activation: Activation | None


class WireSelection(_Wire):
    name: str
    aliases: frozenset[str]
    activation: Activation
    opt_out_by: frozenset[OptOutBy]
    honors_no_log: bool


class WireManifest(_Wire):
    selection: WireSelection
    subscriptions: tuple[WireSubscription, ...]
    has_request_state: bool


class _SetOp(_Wire):
    op: Literal["set"]
    field: Field
    value: object


class _MergeOp(_Wire):
    op: Literal["merge"]
    field: Field
    keys: Mapping[str, object]


class _WirePatch(_Wire):
    ops: tuple[Annotated[_SetOp | _MergeOp, Discriminator("op")], ...]


class WireAbortError(_Wire):
    kind: str
    fields: Mapping[str, object]


class WireAbort(_Wire):
    status: int
    message: str
    code: str | None
    headers: Mapping[str, str]
    error: WireAbortError | None = None


class _WireVerdict(_Wire):
    patch: _WirePatch | None = None
    abort: WireAbort | None = None


_VERDICT: Final[TypeAdapter[_WireVerdict | None]] = TypeAdapter(_WireVerdict | None)

AbortErrorFactory: TypeAlias = Callable[[WireAbort], Exception]

_LOGGER: Final = logging.getLogger("LiteLLM")

ABORT_ERRORS: Final[dict[str, AbortErrorFactory]] = {}  # mutable-ok: each port registers its error kind at import


def register_abort_error(kind: str, factory: AbortErrorFactory) -> None:
    ABORT_ERRORS[kind] = factory


def _patch(wire: _WirePatch) -> Patch:
    patch: Patch = Patch()  # rebind-ok: built one op at a time
    for op in wire.ops:
        patch = patch.set(op.field, op.value) if isinstance(op, _SetOp) else patch.merge(op.field, op.keys)
    return patch


def _exception(wire: WireAbort) -> Exception | None:
    if wire.error is None:
        return None
    factory: Final = ABORT_ERRORS.get(wire.error.kind)
    if factory is None:
        _LOGGER.error("no abort error is registered for kind %r, raising the generic abort error", wire.error.kind)
        return None
    return factory(wire)


def verdict(returned: object) -> Patch | Abort | None:
    wire: Final = _VERDICT.validate_python(returned)
    if wire is None:
        return None
    if wire.abort is not None:
        return Abort(
            status=wire.abort.status,
            message=wire.abort.message,
            code=wire.abort.code,
            headers=wire.abort.headers,
            exception=_exception(wire.abort),
        )
    return None if wire.patch is None else _patch(wire.patch)


def _plain(value: object) -> object:
    if isinstance(value, Strict):
        return value.model_dump(mode="python")
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return tuple(map(_plain, value))  # pyright: ignore[reportUnknownArgumentType]  # a typed field's tuple of records or scalars
    return value


_ABSENT: Final = object()


class RustState:
    __slots__ = ("handle",)
    rust: ClassVar[RustHandle]

    def __init__(self) -> None:
        self.handle: Final = type(self).rust.request_state()


Projection: TypeAlias = Callable[[object], object]

NO_PROJECTIONS: Final[Mapping[Field, Projection]] = MappingProxyType({})


class RustLogger(CustomLoggerV2):
    def __init__(
        self, rust: RustHandle, projections: Mapping[Field, Projection] = NO_PROJECTIONS, host: object = None
    ) -> None:
        self.rust: Final = rust
        self.projections: Final = projections
        self.host: Final = host

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        fields: Final = {
            field.value: self.projections.get(field, _plain)(value)
            for field in reads
            if (value := getattr(view, field.value, _ABSENT)) is not _ABSENT
        }
        call: Final = None if ctx.call is None else asdict(ctx.call)
        state: Final = ctx.state.handle if isinstance(ctx.state, RustState) else None
        return verdict(await self._handled(hook, fields, call, state))

    def _handled(
        self, hook: Hook, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object
    ) -> Awaitable[object]:
        if self.host is None:
            return self.rust.handle(hook.value, fields, call, state)
        if not isinstance(self.rust, HostedRust):
            raise TypeError(f"{type(self.rust).__name__} cannot serve host calls")
        return drive(self.rust.start(hook.value, fields, call, state, self.host), _streamless)


def _handler(
    hook: Hook, reads: frozenset[Field]
) -> Callable[[RustLogger, object, CallContext[object]], Awaitable[Patch | Abort | None]]:
    async def handle(self: RustLogger, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        return await self.run_rust(hook, reads, view, ctx)

    return handle


def _namespace(rust: RustHandle, manifest: WireManifest) -> dict[str, object]:  # mutable-ok: class namespace
    subscriptions: Final = manifest.subscriptions
    handlers: Final[dict[str, object]] = {  # mutable-ok: class namespace
        f"on_{subscription.hook.value}": on(
            subscription.hook,
            reads=subscription.reads,
            writes=subscription.writes,
            may_abort=subscription.may_abort,
            parallel=subscription.parallel,
            awaited=subscription.awaited,
        )(_handler(subscription.hook, subscription.reads))
        for subscription in subscriptions
    }
    selection: Final = manifest.selection
    return {
        **handlers,
        "selection": Selection(
            name=selection.name,
            aliases=selection.aliases,
            activation=selection.activation,
            opt_out_by=selection.opt_out_by,
            honors_no_log=selection.honors_no_log,
        ),
        "activation_overrides": MappingProxyType(
            {s.hook: s.activation for s in subscriptions if s.activation is not None}
        ),
        "request_state": (
            type(f"{selection.name}_state", (RustState,), {"rust": rust, "__slots__": ()})
            if manifest.has_request_state
            else None
        ),
    }


def rust_logger(
    rust: RustHandle, projections: Mapping[Field, Projection] = NO_PROJECTIONS, host: object = None
) -> RustLogger:
    manifest: Final = WireManifest.model_validate_json(rust.manifest())
    cls: Final = type(manifest.selection.name, (RustLogger,), _namespace(rust, manifest))
    return cls(rust, projections, host)


def contract_table() -> Mapping[str, object]:
    return {
        "hooks": [
            {
                "name": hook.value,
                "bit": bit,
                "request_path": HOOK_SPECS[hook].request_path,
                "subscribable": HOOK_SPECS[hook].subscribable,
                "writes": sorted(field.value for field in HOOK_SPECS[hook].writes),
                "outcomes": sorted(outcome.__name__.lower() for outcome in HOOK_SPECS[hook].outcomes),
            }
            for bit, hook in enumerate(Hook)
        ],
        "fields": [{"name": field.value, "bit": field.bit, "tier": field.tier.name} for field in Field],
        "tiers": [{"name": tier.name, "rank": tier.value} for tier in FieldTier],
        "derived_only": sorted(field.value for field in DERIVED_ONLY),
        "by_reference": sorted(field.value for field in BY_REFERENCE),
        "activations": [activation.value for activation in Activation],
        "opt_out_by": [scope.value for scope in OptOutBy],
    }


__all__ = [
    "ABORT_ERRORS",
    "HostedRust",
    "Projection",
    "RustHandle",
    "RustLogger",
    "RustState",
    "WireAbort",
    "contract_table",
    "register_abort_error",
    "rust_logger",
    "verdict",
]
