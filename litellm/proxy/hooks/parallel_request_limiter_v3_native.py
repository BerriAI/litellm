import builtins
import dataclasses
import functools
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import active_request_redis_batches
from litellm.constants import BATCH_TPD_DESCRIPTOR_SUFFIX
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.rust import (
    RustLogger,
    RustState,
    WireAbort,
    WireManifest,
    _namespace,  # pyright: ignore[reportPrivateUsage]  # the limiter's logger subclasses the contract's generated class
    _plain,  # pyright: ignore[reportPrivateUsage]  # the fields reach the bridge in the contract's wire shape
    register_abort_error,
    verdict,
)
from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
from litellm.litellm_core_utils.total_deadline import total_timeout_held
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError, map_v3_rate_limit_type
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BATCH_ENQUEUED_REFUND_STATUSES,
    BatchEnqueuedTokenOutcome,
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    BatchEnqueuedTokenStore,
    batch_response_view,
    canonical_provider_batch_id,
    project_has_model_batch_enqueued_token_limits,
    project_limits_every_model,
    resolve_batch_enqueued_token_scopes,
    resolve_project_model_batch_enqueued_token_scopes,
)
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.rust_bridge.execution import drive
from litellm.types.integrations.custom_logger_v2 import Record
from litellm.types.utils import MCP_GUARDRAIL_CALL_TYPES, CallTypes, Usage

_IO_KEYS: Final = frozenset({"model_per_project_itpm", "model_per_project_otpm"})
_AGENT_DESCRIPTOR_KEYS: Final = frozenset({"agent", "agent_session"})
PROJECT_OTPM_DESCRIPTOR_KEY: Final = "model_per_project_otpm"
PTU_TEAM_DESCRIPTOR_KEY: Final = "model_per_team_ptu"

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy.hooks.batch_rate_limiter import PROXY_BatchRateLimiter
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        ParallelRequestLimiterHook,
        RateLimitDescriptor,
        RateLimitResponse,
        RequestRateLimiterStash,
    )
    from litellm.proxy.utils import InternalUsageCache
    from litellm.redis.stores.rate_limits import RateLimitStore
    from litellm.rust_bridge._native import (
        NativeParallelRequestLimiter,
        NativeParallelRequestLimiterState,
        NativeRequestRedis,
    )

RATE_LIMITED_KIND: Final = "proxy_rate_limit"
UNVERIFIABLE_KIND: Final = "rate_limit_unverifiable"
PTU_SHARE_EXCEEDED_KIND: Final = "ptu_share_exceeded"
BATCH_RATE_LIMITED_KIND: Final = "batch_rate_limit"
PYTHON_ERROR_KIND: Final = "python_error"
_HEADERS: Final = TypeAdapter(list[tuple[str, str]])
_SCOPES: Final = TypeAdapter(tuple[BatchEnqueuedTokenScope, ...])
_RESERVATION: Final = TypeAdapter(BatchEnqueuedTokenReservation)
_OVER_LIMIT: Final = TypeAdapter(BatchEnqueuedTokenOverLimit)
_OUTCOME: Final = TypeAdapter(dict[str, object])
_PTU_FIELDS: Final = TypeAdapter(dict[str, str | int | float])
_READS: Final = frozenset(
    {
        Field.key_identity,
        Field.key_limits,
        Field.limiter_request,
        Field.rate_limit_statuses,
        Field.output_cap,
        Field.input,
        Field.usage,
        Field.standard_logging_payload,
    }
)
_TYPED_READS: Final = _READS - {Field.input}

_PRE_CALL_AUTH: Final[ContextVar[UserAPIKeyAuth | None]] = ContextVar(
    "native_parallel_request_limiter_pre_call_auth", default=None
)


class PreCallAuthScope:
    __slots__ = ("_auth", "_token")

    def __init__(self, user_api_key_dict: UserAPIKeyAuth | None) -> None:
        self._auth: Final = user_api_key_dict
        self._token: Token[UserAPIKeyAuth | None] | None = None

    def __enter__(self) -> None:
        self._token = _PRE_CALL_AUTH.set(self._auth)

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _PRE_CALL_AUTH.reset(self._token)


def _member(source: object, name: str) -> object:
    value: Final[object] = getattr(source, name, None)
    return value


def _rate_limited(wire: WireAbort) -> Exception:
    fields: Final[Mapping[str, object]] = wire.error.fields if wire.error is not None else MappingProxyType({})
    requested_model: Final = fields.get("requested_model")
    resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(
        requested_model if isinstance(requested_model, str) else None
    )
    headers: Final = fields.get("headers")
    rate_limit_type: Final = fields.get("rate_limit_type")
    descriptor_key: Final = fields.get("descriptor_key")
    return ProxyRateLimitError(
        detail=str(fields.get("detail", wire.message)),
        headers=dict(_HEADERS.validate_python(headers)) if headers is not None else dict(wire.headers),
        rate_limit_type=map_v3_rate_limit_type(rate_limit_type if isinstance(rate_limit_type, str) else None),
        model=resolved_model,
        llm_provider=llm_provider,
        descriptor_key=descriptor_key if isinstance(descriptor_key, str) else None,
    )


def _unverifiable(wire: WireAbort) -> Exception:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError

    return RateLimitUnverifiableError()


def _ptu_share_exceeded(wire: WireAbort) -> Exception:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PTUShareExceededError

    fields: Final = _PTU_FIELDS.validate_python(wire.error.fields if wire.error is not None else {})
    ceiling: Final = PTUTeamCeiling(
        model_group=str(fields["model_group"]),
        tpm_limit=int(fields["tpm_limit"]),
        output_to_input_ratio=float(fields["output_to_input_ratio"]),
        cached_input_ratio=float(fields["cached_input_ratio"]),
    )
    return PTUShareExceededError(ceiling, int(fields["needed_tokens"]))


def _batch_rate_limited(wire: WireAbort) -> Exception:
    from litellm.exceptions import RateLimitErrorCategory

    fields: Final[Mapping[str, object]] = wire.error.fields if wire.error is not None else MappingProxyType({})
    requested_model: Final = fields.get("requested_model")
    resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(
        requested_model if isinstance(requested_model, str) else None
    )
    rate_limit_type: Final = fields.get("rate_limit_type")
    return ProxyRateLimitError(
        detail=str(fields.get("detail", wire.message)),
        headers=dict(_HEADERS.validate_python(fields.get("headers") or ())),
        category=RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT,
        rate_limit_type=map_v3_rate_limit_type(rate_limit_type if isinstance(rate_limit_type, str) else None),
        model=resolved_model,
        llm_provider=llm_provider,
    )


def _python_error(wire: WireAbort) -> Exception:
    fields: Final[Mapping[str, object]] = wire.error.fields if wire.error is not None else MappingProxyType({})
    named: Final = fields.get("exception")
    raised: Final = getattr(builtins, named, None) if isinstance(named, str) else None
    message: Final = str(fields.get("message", wire.message))
    if isinstance(raised, type) and issubclass(raised, Exception):
        return raised(message)
    return RuntimeError(f"{named}: {message}")


register_abort_error(RATE_LIMITED_KIND, _rate_limited)
register_abort_error(PYTHON_ERROR_KIND, _python_error)
register_abort_error(BATCH_RATE_LIMITED_KIND, _batch_rate_limited)
register_abort_error(UNVERIFIABLE_KIND, _unverifiable)
register_abort_error(PTU_SHARE_EXCEEDED_KIND, _ptu_share_exceeded)


def raise_wire_abort(returned: object) -> None:
    outcome: Final = verdict(returned)
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception


def pre_call_request() -> "NativeRequestRedis | None":
    batches: Final = active_request_redis_batches()
    return None if batches is None else batches.native()


def post_call_request() -> "NativeRequestRedis | None":
    batches: Final = active_request_redis_batches()
    return None if batches is None else batches.native()


def _auth(data: Mapping[str, object]) -> UserAPIKeyAuth | None:
    scoped: Final = _PRE_CALL_AUTH.get()
    if scoped is not None:
        return scoped
    for key in ("metadata", "litellm_metadata"):
        metadata = data.get(key)
        if isinstance(metadata, dict):
            auth = metadata.get("user_api_key_auth")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # request metadata is untyped
            if isinstance(auth, UserAPIKeyAuth):
                return auth
    return None


def limiter_state(value: object) -> "NativeParallelRequestLimiterState | None":
    from litellm.rust_bridge._native import NativeParallelRequestLimiterState

    return value if isinstance(value, NativeParallelRequestLimiterState) else None


_PROBED_IDENTITY: Final = (
    "api_key",
    "user_id",
    "team_id",
    "org_id",
    "project_id",
    "end_user_id",
    "agent_id",
    "invoked_agent_id",
)
_PROBED_LIMITS: Final = (
    "rpm_limit",
    "tpm_limit",
    "max_parallel_requests",
    "budget_throttle_pct",
    "user_rpm_limit",
    "user_tpm_limit",
    "team_rpm_limit",
    "team_tpm_limit",
    "team_member_rpm_limit",
    "team_member_tpm_limit",
    "end_user_rpm_limit",
    "end_user_tpm_limit",
    "organization_rpm_limit",
    "organization_tpm_limit",
)
_PROBED_POLICY: Final = ("rpm_limit", "tpm_limit", "session_rpm_limit", "session_tpm_limit")


def _rejected(record: object) -> frozenset[str]:
    return frozenset(record.rejected or ()) if isinstance(record, Record) else frozenset()


def _raise_as_python(raw: object) -> None:
    _ = bool(raw)
    _ = str(raw)
    try:
        _ = hash(raw)
    except TypeError:
        return


def _raise_rejected(view: object, data: Mapping[str, object], auth: UserAPIKeyAuth | None) -> None:
    """A member the contract refused is one whose bool(), str() or hash() raises, and the Python limiter raised that
    same error the first time it tested or formatted the member. Re-raising it here keeps that error. A member the
    limiter never read stays tolerated, as it was."""
    limits: Final = _member(view, Field.key_limits.value)
    probed: Final = (
        (_member(view, Field.key_identity.value), auth, _PROBED_IDENTITY),
        (limits, auth, _PROBED_LIMITS),
        (_member(limits, "managed_agent_policy"), _member(auth, "managed_agent_policy"), _PROBED_POLICY),
        (_member(limits, "invoked_agent_policy"), _member(auth, "invoked_agent_policy"), _PROBED_POLICY),
    )
    for record, source, read in probed:
        rejected = _rejected(record)
        for name in read:
            if name in rejected:
                _raise_as_python(_member(source, name))
    if "resolved_agent_id" in _rejected(_member(view, Field.limiter_request.value)):
        _raise_as_python(
            _metadata_member(data.get("metadata"), "agent_id")
            or _metadata_member(data.get("litellm_metadata"), "agent_id")
        )


def _request_data(value: object) -> dict[str, object]:  # mutable-ok: the request body is written in place
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]  # the hook's request data is the untyped proxy body


class ParallelRequestLimiterNativeLogger(RustLogger):
    limiter: "ParallelRequestLimiterHook"

    def __init__(self, native: "NativeParallelRequestLimiter", limiter: "ParallelRequestLimiterHook") -> None:
        super().__init__(native)  # pyright: ignore[reportArgumentType]  # the bridge takes its own state type
        self.limiter = limiter
        self.bridge: Final = native

    def _state(self, ctx: CallContext[object]) -> "NativeParallelRequestLimiterState | None":
        from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash

        stash: Final = get_request_stash()
        stashed: Final = None if stash is None else limiter_state(stash.native_state)
        if stashed is not None:
            return stashed
        return limiter_state(ctx.state.handle) if isinstance(ctx.state, RustState) else None

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        data: Final = _request_data(_member(view, Field.deprecated_kwargs.value))
        match hook:
            case Hook.pre_call:
                return await self._pre_call(data, reads, view, ctx)
            case Hook.post_call_success:
                response: Final = _member(view, Field.deprecated_response_obj.value)
                return await self._call(
                    hook, reads, view, ctx, {"call_id": _call_id(data), "batch": _batch_response(response)}
                )
            case Hook.post_call_failure:
                return await self._call(
                    hook,
                    reads,
                    view,
                    ctx,
                    {
                        "combined_is_usage": isinstance(data.get("combined_usage_object"), Usage),
                        "token_rate_limit_type": _general_setting("token_rate_limit_type", "total"),
                    },
                )
            case Hook.log_prepare:
                return await self._call(hook, reads, view, ctx, {"call_id": _call_id(data)})
            case _:
                return None

    async def _call(
        self,
        hook: Hook,
        reads: frozenset[Field],
        view: object,
        ctx: CallContext[object],
        inputs: Mapping[str, object],
        state: "NativeParallelRequestLimiterState | None" = None,
        request: "NativeRequestRedis | None" = None,
    ) -> Patch | Abort | None:
        handle: Final = state if state is not None else self._state(ctx)
        if handle is None:
            return None
        call: Final = None if ctx.call is None else dataclasses.asdict(ctx.call)
        returned: Final = await self.bridge.handle(
            hook.value, _fields(view, ctx, reads), call, handle, request=request, inputs=inputs
        )
        return verdict(returned)

    @total_timeout_held()
    async def _pre_call(
        self,
        data: dict[str, object],  # mutable-ok: the output cap is written into the request body
        reads: frozenset[Field],
        view: object,
        ctx: CallContext[object],
    ) -> Patch | Abort | None:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            claim_request_stash_for_data,
            current_time,
            get_rate_limiter_for_call_type,
            get_request_stash,
            request_identity,
        )

        call_type: Final = ctx.call.kind.call_type if ctx.call is not None else ""
        if _member(view, Field.key_limits.value) is None:
            if call_type in MCP_GUARDRAIL_CALL_TYPES:
                return None  # the proxy runs an MCP call's pre_call hooks without auth, and no limit keys on it
            raise ValueError(
                f"the parallel request limiter's pre_call got no key auth for a {call_type or 'request'} call, "
                "so it has no key, user or team to limit"
            )
        auth: Final = _auth(data)
        _raise_rejected(view, data, auth)
        held: Final = get_request_stash()
        shared: Final = None if held is None else limiter_state(held.native_state)
        state: Final = (
            shared
            or (limiter_state(ctx.state.handle) if isinstance(ctx.state, RustState) else None)
            or self.bridge.request_state()
        )
        limiter: Final = self.limiter
        stash: Final = claim_request_stash_for_data(data)
        if (
            call_type == CallTypes.call_mcp_tool.value
            and shared is not None
            and self.bridge.holds_parallel_slot(shared)
        ):
            stash.parallel_slot_reused_at = current_time(limiter)
        common: Final = {
            "call_id": _call_id(data),
            # A nested call without an id keeps the outer request's identity, which the native holds already.
            "identity": request_identity(data) or (None if shared is not None else uuid.uuid4().hex),
            "nested": shared is not None,
            "policy": limiter_policy(limiter),
        }
        batch_limiter: Final = get_rate_limiter_for_call_type(limiter, call_type)
        if batch_limiter is not None:
            stash.call_type = call_type
            stash.native_state = state
            prepared: Final = None if auth is None else await _prepared_batch(batch_limiter, data, auth)  # pyright: ignore[reportArgumentType]  # only the batch limiter is keyed by call type
            if isinstance(prepared, Abort):
                return prepared
            batch_outcome: Final = await self._call(
                Hook.pre_call,
                reads,
                view,
                ctx,
                {**common, "phase": "first_pass", "batch_call": True, "batch": prepared},
                state=state,
                request=pre_call_request(),
            )
            mirror_rate_limit_response(self.bridge, state)
            return batch_outcome
        lookups, ceiling = await _lookups(limiter, view, data, ctx)
        stash.call_type = call_type
        stash.native_state = state
        try:
            return await self._hosted(
                reads,
                view,
                ctx,
                {
                    **common,
                    "metadata": _auth_metadata(auth),
                    "lookups": lookups,
                    "held_ptu_ceiling": None if stash.ptu_ceiling is None else dataclasses.asdict(stash.ptu_ceiling),
                },
                state,
                _PreCallHost(limiter, data, ctx, auth),
            )
        finally:
            self._stash_descriptors(stash, state, ceiling)
            mirror_rate_limit_response(self.bridge, state)

    async def _hosted(
        self,
        reads: frozenset[Field],
        view: object,
        ctx: CallContext[object],
        inputs: Mapping[str, object],
        state: "NativeParallelRequestLimiterState",
        host: "_PreCallHost",
    ) -> Patch | Abort | None:
        call: Final = None if ctx.call is None else dataclasses.asdict(ctx.call)
        execution: Final = self.bridge.start(
            Hook.pre_call.value,
            _fields(view, ctx, reads),
            call,
            state,
            host,
            request=pre_call_request(),
            inputs=inputs,
            body=host.data,
        )
        return verdict(await drive(execution, _streamless))

    def _stash_descriptors(
        self,
        stash: "RequestRateLimiterStash",
        state: "NativeParallelRequestLimiterState",
        ceiling: PTUTeamCeiling | None,
    ) -> None:
        descriptors: Final = _descriptors().validate_python(self.bridge.descriptors(state))
        stash.tpm_limited_tags = frozenset(  # rebind-ok: the Python settlement reads the native's tag scopes here
            descriptor["value"]
            for descriptor in descriptors
            if descriptor["key"] == "tag" and (descriptor["rate_limit"] or {}).get("tokens_per_unit") is not None
        )
        stash.socket_agent_windows = tuple(  # rebind-ok: the socket settlement reads the native's agent windows here
            descriptor for descriptor in descriptors if descriptor["key"] in _AGENT_DESCRIPTOR_KEYS
        )
        if ceiling is not None and any(descriptor["key"] == PTU_TEAM_DESCRIPTOR_KEY for descriptor in descriptors):
            stash.ptu_ceiling = ceiling  # rebind-ok: the next pre_call line and settlement read the held ceiling


def _fields(view: object, ctx: CallContext[object], reads: frozenset[Field]) -> Mapping[str, object]:
    projected: Final = reads & (_READS if ctx.call is None or not ctx.call.kind.call_type else _TYPED_READS)
    return {
        Field.deprecated_kwargs.value: None,
        **{field.value: _plain(value) for field in projected if (value := _member(view, field.value)) is not None},
    }


def _streamless(execution: object, head: object) -> object:
    raise RuntimeError("the native limiter's pre-call cannot open a stream")


class _PreCallHost:
    __slots__ = ("auth", "call_type", "data", "limiter", "model", "team_id")

    def __init__(
        self,
        limiter: "ParallelRequestLimiterHook",
        data: dict[str, object],  # mutable-ok: the output cap is written into the request body
        ctx: CallContext[object],
        auth: UserAPIKeyAuth | None,
    ) -> None:
        self.limiter: Final = limiter
        self.data: Final = data
        self.call_type: Final = ctx.call.kind.call_type if ctx.call is not None else ""
        self.model: Final = ctx.call.target.model if ctx.call is not None and ctx.call.target.model else None
        self.auth: Final = auth
        self.team_id: Final = None if auth is None else auth.team_id

    def configured_output_tokens(self, request: Mapping[str, object]) -> int | None:
        from litellm.proxy.auth.auth_utils import get_estimated_output_tokens

        return (
            None
            if self.auth is None
            else get_estimated_output_tokens(user_api_key_dict=self.auth, model_name=self.model)
        )

    def write_output_cap(self, request: Mapping[str, object]) -> bool:
        return _write_output_cap(self.data, self.call_type, _HOST_INT.validate_python(request["cap"]))

    async def media_estimate(self, request: Mapping[str, object]) -> list[int | bool]:
        """The input and output estimate of a request whose media (images, audio, video, files) only Python's
        estimator prices, or that holds a value Rust cannot read, and whether it counted embedding media no
        deployment prices as text. Text, chat and Responses requests the native estimator reads never come here."""
        from litellm.litellm_core_utils.token_counter import offload_token_count
        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            RememberedTokenEstimate,
            estimate_tokens_and_unpriced_media,
            get_request_stash,
        )

        include_media: Final = request["include_media"] is True
        estimate: Final = await offload_token_count(estimate_tokens_and_unpriced_media)(
            data=self.data,
            min_configured_tpm_limit=_HOST_OPTIONAL_INT.validate_python(request["min_configured_tpm_limit"]),
            call_type=self.call_type,
            configured_output_tokens=_HOST_OPTIONAL_INT.validate_python(request["configured_output_tokens"]),
            include_media=include_media,
            team_id=self.team_id,
            media_model_resolver=self.limiter.media_model_resolver,
        )
        stash: Final = get_request_stash()
        model: Final = self.data.get("model")
        if stash is not None and include_media:
            stash.token_estimate = RememberedTokenEstimate(model if isinstance(model, str) else None, estimate)
        return [estimate.input_tokens, estimate.output_tokens, estimate.unpriced_media]

    def precise_input_tokens(self, request: Mapping[str, object]) -> Awaitable[int]:
        from litellm.litellm_core_utils.token_counter import offload_token_count
        from litellm.proxy.hooks.parallel_request_limiter_v3 import estimate_precise_input_tokens

        return offload_token_count(estimate_precise_input_tokens)(
            self.limiter, data=self.data, model=self.model, call_type=self.call_type, team_id=self.team_id
        )


_HOST_INT: Final = TypeAdapter(int)
_HOST_OPTIONAL_INT: Final[TypeAdapter[int | None]] = TypeAdapter(int | None)


@functools.cache
def _descriptors() -> "TypeAdapter[tuple[RateLimitDescriptor, ...]]":
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitDescriptor

    return TypeAdapter(tuple[RateLimitDescriptor, ...])


@functools.cache
def _rate_limit_response() -> "TypeAdapter[RateLimitResponse]":
    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitResponse

    return TypeAdapter(RateLimitResponse)


def _scalar_setting(raw: object) -> object:
    if raw is None or isinstance(raw, (bool, int, str)):
        return raw
    return raw if isinstance(raw, float) and math.isfinite(raw) else str(raw)


def _general_setting(name: str, default: object = None) -> object:
    from litellm.proxy.proxy_server import general_settings_view

    return _scalar_setting(general_settings_view().get(name, default))


def _forced_grouping() -> object:
    return _scalar_setting(litellm.force_redis_hash_tag_grouping)


def _entries(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]  # auth and request metadata are untyped


def _metadata_member(metadata: object, name: str) -> object:
    entries: Final = _entries(metadata)
    return entries.get(name) if entries else None


def _auth_metadata(auth: UserAPIKeyAuth | None) -> Mapping[str, object]:
    if auth is None:
        return MappingProxyType({})
    budgets: Final = _entries(_member(auth, "model_max_budget"))
    return {
        "key_model_rpm_limit": _metadata_member(_member(auth, "metadata"), "model_rpm_limit"),
        "key_model_tpm_limit": _metadata_member(_member(auth, "metadata"), "model_tpm_limit"),
        "key_tag_rpm_limit": _metadata_member(_member(auth, "metadata"), "tag_rpm_limit"),
        "model_max_budget": {model: budget for model, budget in budgets.items() if _entries(budget) is not None}
        if budgets
        else None,
        "team_model_rpm_limit": _metadata_member(_member(auth, "team_metadata"), "model_rpm_limit"),
        "team_model_tpm_limit": _metadata_member(_member(auth, "team_metadata"), "model_tpm_limit"),
        "organization_model_rpm_limit": _metadata_member(_member(auth, "organization_metadata"), "model_rpm_limit"),
        "organization_model_tpm_limit": _metadata_member(_member(auth, "organization_metadata"), "model_tpm_limit"),
        "project_model_rpm_limit": _metadata_member(_member(auth, "project_metadata"), "model_rpm_limit"),
        "project_model_tpm_limit": _metadata_member(_member(auth, "project_metadata"), "model_tpm_limit"),
        "project_model_itpm_limit": _metadata_member(_member(auth, "project_metadata"), "model_itpm_limit"),
        "project_model_otpm_limit": _metadata_member(_member(auth, "project_metadata"), "model_otpm_limit"),
        "end_user_model_rpm_limit": _member(auth, "end_user_model_rpm_limit"),
        "end_user_model_tpm_limit": _member(auth, "end_user_model_tpm_limit"),
    }


def _agent_limits(policy: object) -> Mapping[str, object] | None:
    if policy is None:
        return None
    return {
        name: _member(policy, name) for name in ("rpm_limit", "tpm_limit", "session_rpm_limit", "session_tpm_limit")
    }


async def _lookups(
    limiter: "ParallelRequestLimiterHook",
    view: object,
    data: Mapping[str, object],
    ctx: CallContext[object],
) -> tuple[Mapping[str, object], PTUTeamCeiling | None]:
    from litellm.litellm_core_utils.model_latest_aliases import expand_model_latest_alias
    from litellm.proxy.auth.auth_utils import (
        _get_deployment_default_limit,  # pyright: ignore[reportPrivateUsage]  # the key's per-model default comes from the router's deployments
    )
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        dynamic_rate_limiter_covers,
        model_has_recent_failures,
        registry_agent,
    )

    identity: Final = _member(view, Field.key_identity.value)
    limits: Final = _member(view, Field.key_limits.value)
    auth: Final = _auth(data)
    team_id: Final = None if auth is None else auth.team_id
    model: Final = ctx.call.target.model if ctx.call is not None else ""
    candidates: Final = expand_model_latest_alias(model)
    named: Final = tuple(candidate for candidate in candidates if candidate)
    groups: Final = {candidate: limiter.model_group_resolver(candidate) for candidate in named}
    resolved: Final = tuple(dict.fromkeys(group or candidate for candidate, group in groups.items()))
    settings: Final = litellm.priority_reservation_settings
    borrowers: Final[frozenset[str]] = frozenset() if settings is None else settings.model_limit_borrowing_entities
    dynamic: Final = "dynamic" in (_member(limits, "rpm_limit_type"), _member(limits, "tpm_limit_type"))
    requested_group: Final = (groups.get(model) or model) if model else None
    ceiling: Final = (
        limiter.ptu_team_ceiling_resolver(team_id, requested_group) if team_id and requested_group else None
    )
    tags: Final = tuple(dict.fromkeys(get_tags_from_request_body(data)))
    resolve_tags: Final = limiter.tag_rate_limit_resolver
    tag_limits: Final[Mapping[str, object]] = await resolve_tags(tags) if tags else MappingProxyType({})
    agent_ids: Final = tuple(
        dict.fromkeys(
            agent_id
            for agent_id in (
                _member(identity, "agent_id"),
                _member(identity, "invoked_agent_id"),
                _metadata_member(data.get("metadata"), "agent_id"),
                _metadata_member(data.get("litellm_metadata"), "agent_id"),
            )
            if isinstance(agent_id, str) and agent_id
        )
    )
    return {
        "candidates": list(candidates),
        "groups": groups,
        "deployment_limits": {
            group: {
                "rpm": _get_deployment_default_limit(group, "default_api_key_rpm_limit"),
                "tpm": _get_deployment_default_limit(group, "default_api_key_tpm_limit"),
            }
            for group in resolved
        },
        "borrows": "team" in borrowers,
        "borrows_key": "key" in borrowers,
        "dynamic_covered": [group for group in resolved if borrowers and dynamic_rate_limiter_covers(group)],
        "model_has_failures": bool(model) and dynamic and await model_has_recent_failures(model=model),
        "ptu_ceiling": None if ceiling is None else dataclasses.asdict(ceiling),
        "tag_limits": [
            [tag, _member(limit, "rpm_limit"), _member(limit, "tpm_limit")]
            for tag in tags
            if (limit := tag_limits.get(tag)) is not None
        ],
        "registry_agents": [[agent_id, _agent_limits(registry_agent(agent_id))] for agent_id in agent_ids],
    }, ceiling


async def scoped_inputs(
    limiter: "ParallelRequestLimiterHook",
    auth: UserAPIKeyAuth | None,
    data: Mapping[str, object],
    call_type: str | None = None,
    *,
    lookups: bool = True,
) -> dict[str, object]:  # mutable-ok: the caller adds its selection and action before handing it to the bridge
    """The typed key records and lookups an entry point outside the V2 pre-call hands the native
    builder, derived the way the pre-call derives them for ``auth`` making a request with ``data``."""
    from litellm.integrations.custom_logger_v2.derivations import key_identity, key_limits
    from litellm.integrations.custom_logger_v2.field_sources import HookArgs
    from litellm.proxy.v2_proxy_fields import limiter_request, proxy_call_facts

    args: Final = HookArgs(auth=auth, request=data, call_type=call_type)
    identity: Final = key_identity(args)
    limits: Final = key_limits(args)
    request: Final = limiter_request(args)
    view: Final = SimpleNamespace(
        **{Field.key_identity.value: identity, Field.key_limits.value: limits, Field.limiter_request.value: request}
    )
    _raise_rejected(view, data, auth)
    ctx: Final[CallContext[object]] = CallContext(
        hook=Hook.pre_call, call=proxy_call_facts(data, call_type, auth), state=None
    )
    model: Final = data.get("model")
    found: Mapping[str, object] = MappingProxyType({})  # rebind-ok: filled only when the selection needs lookups
    if lookups:
        with PreCallAuthScope(auth):
            found, _ = await _lookups(limiter, view, data, ctx)
    return {
        Field.key_identity.value: _plain(identity),
        Field.key_limits.value: _plain(limits),
        Field.limiter_request.value: _plain(request),
        "requested_model": model if isinstance(model, str) else None,
        "metadata": dict(_auth_metadata(auth)),
        "lookups": dict(found),
        "policy": dict(limiter_policy(limiter)),
    }


def _write_output_cap(
    data: dict[str, object],  # mutable-ok: the cap is written into the routed request
    call_type: str,
    cap: int | None,
) -> bool:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import write_implicit_output_cap

    return cap is not None and write_implicit_output_cap(data, call_type, cap)


async def _prepared_batch(
    batch: "PROXY_BatchRateLimiter",
    data: dict[str, object],  # mutable-ok: the batch counts are written into the request body
    auth: UserAPIKeyAuth,
) -> Mapping[str, object] | Abort | None:
    from starlette.exceptions import HTTPException

    from litellm.proxy.hooks.batch_rate_limiter import (
        _rows_were_rewritten_on_upload,  # pyright: ignore[reportPrivateUsage]  # the batch limiter's own upload check
    )

    try:
        input_file_id: Final = data.get("input_file_id")
        if not input_file_id:
            return None
        key_and_team_scopes: Final = resolve_batch_enqueued_token_scopes(auth)
        should_skip, descriptors = await batch._should_skip_batch_input_file_processing(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the batch limiter's own skip decision over the untyped body
            data=data,
            user_api_key_dict=auth,
            has_enqueued_scopes=bool(key_and_team_scopes) or project_has_model_batch_enqueued_token_limits(auth),
        )
        if should_skip:
            return None
        provider: Final = data.get("custom_llm_provider", "openai")
        usage: Final = await batch.count_input_file_usage(  # pyright: ignore[reportUnknownMemberType]  # the batch limiter reads the untyped body
            file_id=input_file_id,  # pyright: ignore[reportArgumentType]  # passed through as the Python limiter does
            custom_llm_provider=provider,  # pyright: ignore[reportArgumentType]  # passed through as the Python limiter does
            user_api_key_dict=auth,
            data=data,
            descriptors=descriptors,
        )
        data["_batch_token_count"] = usage.total_tokens  # rebind-ok: the batch limiter reads the counts here
        data["_batch_request_count"] = usage.request_count  # rebind-ok: the batch limiter reads the counts here
        routing_model: Final = batch._get_batch_routing_model(data)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the batch limiter's own routing model over the untyped body
        row_tokens_by_model: Final[Mapping[str, int]] = MappingProxyType(
            {
                row_model: model_usage.get("total_tokens", 0)
                for (*_, row_model), model_usage in usage.per_model_usage.items()
            }
            if not (isinstance(input_file_id, str) and _rows_were_rewritten_on_upload(input_file_id))
            else {}
        )
        enqueued_scopes: Final = (
            *key_and_team_scopes,
            *resolve_project_model_batch_enqueued_token_scopes(
                auth,
                row_tokens_by_model=row_tokens_by_model,
                routing_model=routing_model,
                batch_tokens=usage.total_tokens,
                model_group_of=batch._model_group_of,  # pyright: ignore[reportPrivateUsage]  # the batch limiter's own model grouping
            ),
        )
        project_covers_batch: Final = (
            bool(enqueued_scopes)
            and not key_and_team_scopes
            and routing_model is not None
            and project_limits_every_model(
                auth,
                (routing_model, *row_tokens_by_model),
                batch._model_group_of,  # pyright: ignore[reportPrivateUsage]  # the batch limiter's own model grouping
            )
        )
        uncharged: Final = bool(enqueued_scopes and key_and_team_scopes)
        built: Final = (
            descriptors
            if descriptors is not None or uncharged
            else await batch._create_batch_rate_limit_descriptors(user_api_key_dict=auth, data=data)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the batch limiter's own descriptors over the untyped body
        )
        charged: Final = (
            ()
            if uncharged
            else tuple(
                descriptor
                for descriptor in built or ()
                if not project_covers_batch or descriptor["key"].endswith(BATCH_TPD_DESCRIPTOR_SUFFIX)
            )
        )
        io_descriptors, io_increments = (
            await batch._create_project_io_descriptors_for_models(  # pyright: ignore[reportPrivateUsage]  # the batch limiter's own descriptors
                user_api_key_dict=auth, per_model_usage=usage.per_model_usage
            )
            if not (enqueued_scopes and key_and_team_scopes) and not project_covers_batch
            else ([], [])
        )
    except HTTPException as error:
        return Abort(status=error.status_code, message=str(error.detail), exception=error)
    except Exception:  # noqa: BLE001  # the Python batch limiter lets the request through when its preparation fails
        return None
    return {
        "descriptors": [dict(descriptor) for descriptor in (*charged, *io_descriptors)],
        "increments": [
            *((usage.request_count, usage.total_tokens) for _ in charged),
            *((increment["requests"], increment["tokens"]) for increment in io_increments),
        ],
        "request_count": usage.request_count,
        "total_tokens": usage.total_tokens,
        "output_tokens": usage.output_tokens,
        "per_model_usage": [
            {"models": list(models), "usage": dict(model_usage)}
            for models, model_usage in usage.per_model_usage.items()
        ],
        "enqueued_scopes": [dataclasses.asdict(scope) for scope in enqueued_scopes],
        "counters_after_reservation": bool(enqueued_scopes)
        and not key_and_team_scopes
        and (bool(charged) or not project_covers_batch),
    }


def _batch_response(response: object) -> Mapping[str, object] | None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash

    stash: Final = get_request_stash()
    if stash is None or stash.native_state is None:
        return None
    try:
        view: Final = batch_response_view(response)
        if view is None:
            return None
        return {
            "id": canonical_provider_batch_id(view.id),
            "terminal": view.status.lower() in BATCH_ENQUEUED_REFUND_STATUSES,
        }
    except Exception as e:  # noqa: BLE001  # post-call batch accounting must never fail the response
        verbose_proxy_logger.exception("Error in batch enqueued-token post-call hook: %s", e)
        return None


class NativeBatchEnqueuedTokenStore(BatchEnqueuedTokenStore):
    def __init__(self, internal_usage_cache: "InternalUsageCache", native: "NativeParallelRequestLimiter") -> None:
        super().__init__(internal_usage_cache=internal_usage_cache)
        self.native: Final = native

    async def reserve(
        self,
        tokens: int,
        scopes: tuple[BatchEnqueuedTokenScope, ...],
        litellm_parent_otel_span: "Span | None" = None,
    ) -> BatchEnqueuedTokenOutcome:
        outcome: Final = _OUTCOME.validate_python(
            await self.native.enqueued_reserve(tokens, _SCOPES.dump_json(scopes).decode())
        )
        if "over_limit" in outcome:
            return _OVER_LIMIT.validate_python(outcome["over_limit"])
        return _RESERVATION.validate_python(outcome["reservation"])

    async def refund(
        self, reservation: BatchEnqueuedTokenReservation, litellm_parent_otel_span: "Span | None" = None
    ) -> None:
        await self.native.enqueued_refund(_RESERVATION.dump_json(reservation).decode())

    async def save_reservation(
        self,
        batch_id: str,
        reservation: BatchEnqueuedTokenReservation,
        litellm_parent_otel_span: "Span | None" = None,
    ) -> None:
        await self.native.enqueued_save(batch_id, _RESERVATION.dump_json(reservation).decode())

    async def pop_reservation(
        self, batch_id: str, litellm_parent_otel_span: "Span | None" = None
    ) -> BatchEnqueuedTokenReservation | None:
        popped: Final = await self.native.enqueued_pop(batch_id)
        return None if popped is None else _RESERVATION.validate_python(popped)


def mirror_rate_limit_response(
    bridge: "NativeParallelRequestLimiter", state: "NativeParallelRequestLimiterState"
) -> None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash

    stash: Final = get_request_stash()
    response: Final = bridge.rate_limit_response(state)
    if stash is None or response is None:
        return
    stash.rate_limit_response = _rate_limit_response().validate_python(response)


def policy(
    fail_closed_resolver: Callable[[], bool] | None = None,
    force_grouping_resolver: Callable[[], bool] | None = None,
) -> Mapping[str, object]:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING

    return {
        "fail_closed": _general_setting(FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING)
        if fail_closed_resolver is None
        else fail_closed_resolver(),
        "force_grouping": _forced_grouping() if force_grouping_resolver is None else force_grouping_resolver(),
    }


def limiter_policy(limiter: "ParallelRequestLimiterHook") -> Mapping[str, object]:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _fail_closed_rate_limit_enforcement_from_general_settings,  # pyright: ignore[reportPrivateUsage]  # the default resolver reads the raw setting
        _force_redis_hash_tag_grouping_from_litellm_settings,  # pyright: ignore[reportPrivateUsage]  # the default resolver reads the raw setting
    )

    fail_closed: Final = limiter.fail_closed_resolver
    force_grouping: Final = limiter.force_hash_tag_grouping_resolver
    return policy(
        None if fail_closed is _fail_closed_rate_limit_enforcement_from_general_settings else fail_closed,
        None if force_grouping is _force_redis_hash_tag_grouping_from_litellm_settings else force_grouping,
    )


def _call_id(data: Mapping[str, object]) -> str | None:
    call_id: Final = data.get("litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def native_engine(
    rate_limits: "RateLimitStore",
    window_size: int,
    tpm_reservation_enabled: bool,
    clock: Callable[[], datetime] = datetime.now,
) -> "NativeParallelRequestLimiter":
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        DEFAULT_MAX_TOKENS_ESTIMATE,
        NO_MAX_TOKENS_TPM_RESERVATION_FRACTION,
        RESPONSES_API_MIN_OUTPUT_TOKENS,
    )
    from litellm.rust_bridge._native import NativeParallelRequestLimiter

    return NativeParallelRequestLimiter(
        rate_limits._handle,  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # both limiters share one store and memory tier
        window_size,
        tpm_reservation_enabled,
        None if clock is datetime.now else (lambda: clock().timestamp()),
        time.monotonic,
        output_cap=(
            NO_MAX_TOKENS_TPM_RESERVATION_FRACTION,
            DEFAULT_MAX_TOKENS_ESTIMATE,
            RESPONSES_API_MIN_OUTPUT_TOKENS,
        ),
    )


def limiter_engine(limiter: "ParallelRequestLimiterHook") -> "NativeParallelRequestLimiter":
    return native_engine(
        limiter.rate_limits,
        limiter.window_size,
        limiter.tpm_reservation_enabled,
        limiter.time_provider,
    )


def native_parallel_request_limiter(
    limiter: "ParallelRequestLimiterHook",
) -> ParallelRequestLimiterNativeLogger:
    bridge: Final = limiter.native_companion
    manifest: Final = WireManifest.model_validate_json(bridge.manifest())
    cls: Final = type(manifest.selection.name, (ParallelRequestLimiterNativeLogger,), _namespace(bridge, manifest))  # pyright: ignore[reportArgumentType]  # the bridge takes its own state type
    logger: Final = cls(bridge, limiter)
    store: Final = limiter.batch_enqueued_token_store
    if not isinstance(store, NativeBatchEnqueuedTokenStore):
        native_store: Final = NativeBatchEnqueuedTokenStore(store.internal_usage_cache, bridge)
        limiter.batch_enqueued_token_store = native_store  # rebind-ok: the native owns the enqueued-token state
    return logger


__all__ = [
    "NativeBatchEnqueuedTokenStore",
    "ParallelRequestLimiterNativeLogger",
    "PreCallAuthScope",
    "limiter_engine",
    "limiter_policy",
    "limiter_state",
    "mirror_rate_limit_response",
    "native_engine",
    "native_parallel_request_limiter",
    "policy",
    "post_call_request",
    "pre_call_request",
    "raise_wire_abort",
    "scoped_inputs",
]
