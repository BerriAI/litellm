import builtins
import dataclasses
import functools
import math
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import active_request_redis_batches
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
    resolve_batch_enqueued_token_scopes,
)
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.rust_bridge.execution import drive
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth
from litellm.types.utils import CallTypes, Usage

_IO_KEYS: Final = frozenset({"model_per_project_itpm", "model_per_project_otpm"})
_AGENT_DESCRIPTOR_KEYS: Final = frozenset({"agent", "agent_session"})
PROJECT_OTPM_DESCRIPTOR_KEY: Final = "model_per_project_otpm"
PTU_TEAM_DESCRIPTOR_KEY: Final = "model_per_team_ptu"

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy.hooks.batch_rate_limiter import PROXY_BatchRateLimiter
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        PROXY_MaxParallelRequestsHandler_v3,
        RateLimitDescriptor,
        RateLimitResponse,
        RequestRateLimiterStash,
    )
    from litellm.proxy.utils import InternalUsageCache
    from litellm.rust_bridge._native import (
        NativeParallelRequestLimiter,
        NativeParallelRequestLimiterState,
        NativeRequestRedis,
    )

NATIVE_PARALLEL_REQUEST_LIMITER_ENV: Final = "LITELLM_NATIVE_PARALLEL_REQUEST_LIMITER"
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


def native_parallel_request_limiter_enabled() -> bool:
    return os.getenv(NATIVE_PARALLEL_REQUEST_LIMITER_ENV, "false").strip().lower() == "true"


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
    return None if batches is None else batches.native_post_call()


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


def _holds_rejected(*records: object) -> bool:
    return any(_member(record, "rejected") for record in records)


def _request_data(value: object) -> dict[str, object]:  # mutable-ok: the request body is written in place
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]  # the hook's request data is the untyped proxy body


class ParallelRequestLimiterNativeLogger(RustLogger):
    limiter: "PROXY_MaxParallelRequestsHandler_v3"

    def __init__(self, native: "NativeParallelRequestLimiter", limiter: "PROXY_MaxParallelRequestsHandler_v3") -> None:
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
                return await self._pre_call(data, view, ctx)
            case Hook.post_call_success:
                response: Final = _member(view, Field.deprecated_response_obj.value)
                return await self._call(
                    hook, view, ctx, {"call_id": _call_id(data), "batch": _batch_response(response)}
                )
            case Hook.post_call_failure:
                return await self._call(
                    hook,
                    view,
                    ctx,
                    {
                        "combined_is_usage": isinstance(data.get("combined_usage_object"), Usage),
                        "token_rate_limit_type": _general_setting("token_rate_limit_type", "total"),
                    },
                )
            case Hook.log_prepare:
                return await self._call(hook, view, ctx, {"call_id": _call_id(data)})
            case _:
                return None

    async def _call(
        self,
        hook: Hook,
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
            hook.value, _fields(view, ctx), call, handle, request=request, inputs=inputs
        )
        return verdict(returned)

    def _declines(self, data: Mapping[str, object], view: object) -> bool:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash, request_identity

        stash: Final = get_request_stash()
        limits: Final = _member(view, Field.key_limits.value)
        return (
            limits is None
            or (stash is not None and stash.native_state is None and stash.call_type is not None)
            or request_identity(data) is None
            or _holds_rejected(
                _member(view, Field.key_identity.value),
                limits,
                _member(limits, "managed_agent_policy"),
                _member(limits, "invoked_agent_policy"),
                _member(view, Field.limiter_request.value),
            )
        )

    async def _pre_call(
        self,
        data: dict[str, object],  # mutable-ok: the output cap is written into the request body
        view: object,
        ctx: CallContext[object],
    ) -> Patch | Abort | None:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import (
            claim_request_stash_for_data,
            get_request_stash,
            request_identity,
        )

        call_type: Final = ctx.call.kind.call_type if ctx.call is not None else ""
        if self._declines(data, view):
            return None
        auth: Final = _auth(data)
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
            stash.parallel_slot_reused_at = limiter._get_current_time()  # pyright: ignore[reportPrivateUsage]  # the reuse time is read by the Python settlement
        common: Final = {
            "call_id": _call_id(data),
            "identity": request_identity(data),
            "nested": shared is not None,
            "policy": policy(limiter),
        }
        batch_limiter: Final = limiter.get_rate_limiter_for_call_type(call_type)
        if batch_limiter is not None:
            stash.call_type = call_type
            stash.native_state = state
            prepared: Final = None if auth is None else await _prepared_batch(batch_limiter, data, auth)  # pyright: ignore[reportArgumentType]  # only the batch limiter is keyed by call type
            if isinstance(prepared, Abort):
                return prepared
            batch_outcome: Final = await self._call(
                Hook.pre_call,
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
        view: object,
        ctx: CallContext[object],
        inputs: Mapping[str, object],
        state: "NativeParallelRequestLimiterState",
        host: "_PreCallHost",
    ) -> Patch | Abort | None:
        call: Final = None if ctx.call is None else dataclasses.asdict(ctx.call)
        execution: Final = self.bridge.start(
            Hook.pre_call.value,
            _fields(view, ctx),
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


def _fields(view: object, ctx: CallContext[object]) -> Mapping[str, object]:
    reads: Final = _READS if ctx.call is None or not ctx.call.kind.call_type else _TYPED_READS
    return {
        Field.deprecated_kwargs.value: None,
        **{field.value: _plain(value) for field in reads if (value := _member(view, field.value)) is not None},
    }


def _streamless(execution: object, head: object) -> object:
    raise RuntimeError("the native limiter's pre-call cannot open a stream")


class _PreCallHost:
    __slots__ = ("auth", "call_type", "data", "limiter", "model", "team_id")

    def __init__(
        self,
        limiter: "PROXY_MaxParallelRequestsHandler_v3",
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

    def estimate_tokens(self, request: Mapping[str, object]) -> Awaitable[int]:
        from litellm.litellm_core_utils.token_counter import offload_token_count

        return offload_token_count(self.limiter._estimate_tokens_for_request)(  # pyright: ignore[reportPrivateUsage]  # the estimate Rust hands back to the Python limiter
            data=self.data,
            model=self.model,
            min_configured_tpm_limit=_HOST_INT.validate_python(request["min_configured_tpm_limit"]),
            call_type=self.call_type,
            configured_output_tokens=_HOST_OPTIONAL_INT.validate_python(request["configured_output_tokens"]),
            team_id=self.team_id,
        )

    async def estimate_ptu_tokens(self, request: Mapping[str, object]) -> list[int]:
        from litellm.litellm_core_utils.token_counter import offload_token_count

        limiter: Final = self.limiter
        estimated: Final = await offload_token_count(limiter._estimate_input_and_output_tokens)(  # pyright: ignore[reportPrivateUsage]  # the estimate Rust hands back to the Python limiter
            data=self.data,
            min_configured_tpm_limit=_HOST_INT.validate_python(request["min_configured_tpm_limit"]),
            call_type=self.call_type,
            configured_output_tokens=_HOST_OPTIONAL_INT.validate_python(request["configured_output_tokens"]),
            team_id=self.team_id,
            media_model_resolver=limiter._media_model_resolver,  # pyright: ignore[reportPrivateUsage]  # the estimate Rust hands back to the Python limiter
        )
        return list(estimated)

    def estimate_io_output_tokens(self, request: Mapping[str, object]) -> int:
        _, output = self.limiter._estimate_input_and_output_tokens(  # pyright: ignore[reportPrivateUsage]  # the estimate Rust hands back to the Python limiter
            data=self.data,
            min_configured_tpm_limit=_HOST_OPTIONAL_INT.validate_python(request["min_configured_tpm_limit"]),
            call_type=self.call_type,
            include_media=False,
        )
        return output

    def precise_input_tokens(self, request: Mapping[str, object]) -> Awaitable[int]:
        from litellm.litellm_core_utils.token_counter import offload_token_count

        return offload_token_count(self.limiter._estimate_precise_input_tokens)(  # pyright: ignore[reportPrivateUsage]  # only the Python tokenizer counts the precise input
            data=self.data, model=self.model, call_type=self.call_type, team_id=self.team_id
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
    from litellm.proxy.proxy_server import general_settings

    return _setting_entry(general_settings, name, default)


def _setting_entry(settings: Mapping[str, object], name: str, default: object) -> object:
    return _scalar_setting(settings.get(name, default))


def _setting(resolver: Callable[[], bool], default: Callable[[], bool], raw: Callable[[], object]) -> object:
    return raw() if resolver is default else resolver()


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
        "customer_scopes": _customer_scopes(auth.customer_hierarchy),
    }


def _customer_scopes(hierarchy: CustomerHierarchyAuth | None) -> list[Mapping[str, object]] | None:
    if hierarchy is None:
        return None
    return [
        scope.model_dump(include={"key", "entity_id", "rpm_limit", "tpm_limit", "model_rpm_limit", "model_tpm_limit"})
        for scope in hierarchy.scopes
    ]


def _agent_limits(policy: object) -> Mapping[str, object] | None:
    if policy is None:
        return None
    return {
        name: _member(policy, name) for name in ("rpm_limit", "tpm_limit", "session_rpm_limit", "session_tpm_limit")
    }


async def _lookups(
    limiter: "PROXY_MaxParallelRequestsHandler_v3",
    view: object,
    data: Mapping[str, object],
    ctx: CallContext[object],
) -> tuple[Mapping[str, object], PTUTeamCeiling | None]:
    from litellm.litellm_core_utils.model_latest_aliases import expand_model_latest_alias
    from litellm.proxy.auth.auth_utils import (
        _get_deployment_default_limit,  # pyright: ignore[reportPrivateUsage]  # the key's per-model default comes from the router's deployments
    )
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
    from litellm.proxy.hooks.parallel_request_limiter_v3 import dynamic_rate_limiter_covers

    identity: Final = _member(view, Field.key_identity.value)
    limits: Final = _member(view, Field.key_limits.value)
    auth: Final = _auth(data)
    team_id: Final = None if auth is None else auth.team_id
    model: Final = ctx.call.target.model if ctx.call is not None else ""
    candidates: Final = expand_model_latest_alias(model)
    named: Final = tuple(candidate for candidate in candidates if candidate)
    groups: Final = {candidate: limiter._model_group_resolver(candidate) for candidate in named}  # pyright: ignore[reportPrivateUsage]  # the router's alias lookup
    resolved: Final = tuple(dict.fromkeys(group or candidate for candidate, group in groups.items()))
    settings: Final = litellm.priority_reservation_settings
    borrows: Final = settings is not None and settings.team_model_limit_borrows_idle_capacity_from_other_teams
    dynamic: Final = "dynamic" in (_member(limits, "rpm_limit_type"), _member(limits, "tpm_limit_type"))
    requested_group: Final = (groups.get(model) or model) if model else None
    ceiling: Final = (
        limiter._ptu_team_ceiling_resolver(team_id, requested_group)  # pyright: ignore[reportPrivateUsage]  # the router's PTU share lookup
        if team_id and requested_group
        else None
    )
    tags: Final = tuple(dict.fromkeys(get_tags_from_request_body(data)))
    resolve_tags: Final = limiter._tag_rate_limit_resolver  # pyright: ignore[reportPrivateUsage]  # the database's tag budgets
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
        "borrows": borrows,
        "dynamic_covered": [group for group in resolved if borrows and dynamic_rate_limiter_covers(group)],
        "model_has_failures": bool(model) and dynamic and await limiter._check_model_has_recent_failures(model=model),  # pyright: ignore[reportPrivateUsage]  # the router's failure counts
        "ptu_ceiling": None if ceiling is None else dataclasses.asdict(ceiling),
        "tag_limits": [
            [tag, _member(limit, "rpm_limit"), _member(limit, "tpm_limit")]
            for tag in tags
            if (limit := tag_limits.get(tag)) is not None
        ],
        "registry_agents": [
            [agent_id, _agent_limits(limiter._get_agent_from_registry(agent_id))]  # pyright: ignore[reportPrivateUsage]  # the agent registry
            for agent_id in agent_ids
        ],
    }, ceiling


def _write_output_cap(
    data: dict[str, object],  # mutable-ok: the cap is written into the routed request
    call_type: str,
    cap: int | None,
) -> bool:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3

    return cap is not None and PROXY_MaxParallelRequestsHandler_v3.write_implicit_output_cap(data, call_type, cap)


async def _prepared_batch(
    batch: "PROXY_BatchRateLimiter",
    data: dict[str, object],  # mutable-ok: the batch counts are written into the request body
    auth: UserAPIKeyAuth,
) -> Mapping[str, object] | Abort | None:
    from starlette.exceptions import HTTPException

    try:
        input_file_id: Final = data.get("input_file_id")
        if not input_file_id:
            return None
        enqueued_scopes: Final = resolve_batch_enqueued_token_scopes(auth)
        should_skip, descriptors = batch._should_skip_batch_input_file_processing(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the batch limiter's own skip decision over the untyped body
            data=data, user_api_key_dict=auth, has_enqueued_scopes=bool(enqueued_scopes)
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
        if enqueued_scopes:
            return {
                "descriptors": [],
                "increments": [],
                "request_count": usage.request_count,
                "total_tokens": usage.total_tokens,
                "enqueued_scopes": [dataclasses.asdict(scope) for scope in enqueued_scopes],
            }
        charged: Final = (
            descriptors
            if descriptors is not None
            else batch._create_batch_rate_limit_descriptors(user_api_key_dict=auth, data=data)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the batch limiter's own descriptors over the untyped body
        )
        io_descriptors, io_increments = batch._create_project_io_descriptors_for_models(  # pyright: ignore[reportPrivateUsage]  # the batch limiter's own descriptors
            user_api_key_dict=auth, per_model_usage=usage.per_model_usage
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
        "per_model_usage": usage.per_model_usage,
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


def usage_facts(usage: object) -> Mapping[str, int] | None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _ReconciledUsage,  # pyright: ignore[reportPrivateUsage]  # the limiter's usage shape
    )

    if not isinstance(usage, _ReconciledUsage):
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": usage.cached_tokens,
    }


def failure_facts(limiter: "PROXY_MaxParallelRequestsHandler_v3", source: Mapping[str, object]) -> Mapping[str, object]:
    recovered: Final = limiter._recovered_partial_usage(source)  # pyright: ignore[reportPrivateUsage]  # the native settles with the limiter's usage recovery
    return {
        "call_id": _call_id(source),
        "tpm_actual": limiter._recovered_partial_usage_tokens(source)[0],  # pyright: ignore[reportPrivateUsage]  # the native settles with the limiter's usage recovery
        "partial_usage": None if recovered is None else usage_facts(limiter._resolve_reconciled_usage(recovered)),  # pyright: ignore[reportPrivateUsage]  # the native settles with the limiter's usage recovery
    }


def policy(limiter: "PROXY_MaxParallelRequestsHandler_v3") -> Mapping[str, object]:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING,
        _fail_closed_rate_limit_enforcement_from_general_settings,  # pyright: ignore[reportPrivateUsage]  # the default resolver reads the raw setting
        _force_redis_hash_tag_grouping_from_litellm_settings,  # pyright: ignore[reportPrivateUsage]  # the default resolver reads the raw setting
    )

    return {
        "fail_closed": _setting(
            limiter._fail_closed_resolver,  # pyright: ignore[reportPrivateUsage]  # read live per call like the Python limiter
            _fail_closed_rate_limit_enforcement_from_general_settings,
            lambda: _general_setting(FAIL_CLOSED_RATE_LIMIT_ENFORCEMENT_SETTING),
        ),
        "force_grouping": _setting(
            limiter._force_hash_tag_grouping_resolver,  # pyright: ignore[reportPrivateUsage]  # read live per call like the Python limiter
            _force_redis_hash_tag_grouping_from_litellm_settings,
            _forced_grouping,
        ),
    }


def _call_id(data: Mapping[str, object]) -> str | None:
    call_id: Final = data.get("litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def native_engine(limiter: "PROXY_MaxParallelRequestsHandler_v3") -> "NativeParallelRequestLimiter":
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        DEFAULT_MAX_TOKENS_ESTIMATE,
        NO_MAX_TOKENS_TPM_RESERVATION_FRACTION,
        RESPONSES_API_MIN_OUTPUT_TOKENS,
    )
    from litellm.rust_bridge._native import NativeParallelRequestLimiter

    clock: Final = limiter._time_provider  # pyright: ignore[reportPrivateUsage]  # the native reads the limiter's clock
    return NativeParallelRequestLimiter(
        limiter.rate_limits._handle,  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # both limiters share one store and memory tier
        limiter.window_size,
        limiter.tpm_reservation_enabled,
        None if clock is datetime.now else (lambda: clock().timestamp()),
        time.monotonic,
        output_cap=(
            NO_MAX_TOKENS_TPM_RESERVATION_FRACTION,
            DEFAULT_MAX_TOKENS_ESTIMATE,
            RESPONSES_API_MIN_OUTPUT_TOKENS,
        ),
    )


def native_parallel_request_limiter(
    limiter: "PROXY_MaxParallelRequestsHandler_v3",
) -> ParallelRequestLimiterNativeLogger:
    bridge: Final = limiter.native_companion or native_engine(limiter)
    manifest: Final = WireManifest.model_validate_json(bridge.manifest())
    cls: Final = type(manifest.selection.name, (ParallelRequestLimiterNativeLogger,), _namespace(bridge, manifest))  # pyright: ignore[reportArgumentType]  # the bridge takes its own state type
    logger: Final = cls(bridge, limiter)
    limiter.native_companion = bridge  # rebind-ok: the Python limiter settles through the native's bridge
    store: Final = limiter.batch_enqueued_token_store
    if not isinstance(store, NativeBatchEnqueuedTokenStore):
        native_store: Final = NativeBatchEnqueuedTokenStore(store.internal_usage_cache, bridge)
        limiter.batch_enqueued_token_store = native_store  # rebind-ok: the native owns the enqueued-token state
    return logger


__all__ = [
    "NATIVE_PARALLEL_REQUEST_LIMITER_ENV",
    "NativeBatchEnqueuedTokenStore",
    "ParallelRequestLimiterNativeLogger",
    "PreCallAuthScope",
    "failure_facts",
    "limiter_state",
    "mirror_rate_limit_response",
    "native_engine",
    "native_parallel_request_limiter",
    "native_parallel_request_limiter_enabled",
    "policy",
    "post_call_request",
    "pre_call_request",
    "raise_wire_abort",
    "usage_facts",
]
