import asyncio
import json
from collections.abc import Awaitable, Coroutine, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm._internal_context import service_target
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import Admit, hook_subscribers
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.rust import RustHandle, RustLogger, rust_logger
from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
from litellm.litellm_core_utils.safe_json_dumps import safe_json_structure
from litellm.litellm_core_utils.secret_redaction import (
    _token_redactor,  # pyright: ignore[reportPrivateUsage]  # the callback shares the process's compiled token redactor
)
from litellm.proxy._types import SpendLogsPayload, UserAPIKeyAuth
from litellm.proxy.db.db_spend_update_writer import (
    _turn_chains_through_the_db,  # pyright: ignore[reportPrivateUsage]  # the writer's own Responses chaining check
    debitable_model_access_groups,
    get_llm_router,
)
from litellm.proxy.db.rust_spend_log_writer import rust_spend_log_writer
from litellm.proxy.hooks.proxy_track_cost_callback import (
    FAILURE_SPEND_COUNTER_WAIT_SECONDS,
    ProxyDBLogger,
    _batch_accounting_for,  # pyright: ignore[reportPrivateUsage]  # the cost callback's batch costing outcome
)
from litellm.proxy.hooks.search_free_tier import record_freed_cost
from litellm.proxy.spend_tracking.customer_cost_margin import apply_callback_customer_cost_margin
from litellm.proxy.spend_tracking.rust_spend_counters import (
    _CounterHost,  # pyright: ignore[reportPrivateUsage]  # the counter step's host, which the callback's host delegates to
    charge_counter_steps,
)
from litellm.proxy.spend_tracking.rust_spend_log_inputs import SpendLogCall, spend_log_document
from litellm.proxy.spend_tracking.spend_tracking_utils import get_request_model_access_groups
from litellm.redis.stores._route import post_call_request
from litellm.responses.streaming_iterator import spawn_logging_task
from litellm.rust_bridge._native import NativeSpendCounters, RustSpendLog, RustSpendLogWriter
from litellm.rust_bridge.execution import Execution
from litellm.types.proxy.auth.customer_hierarchy import (
    customer_budget_scopes_from_request,
    customer_hierarchy_ids_from_request,
)

_FIELDS: Final = TypeAdapter(dict[str, object])
_STEPS: Final = TypeAdapter(list[dict[str, object]])
_KEY_IDENTITY: Final = (
    ("key_alias", "user_api_key_alias"),
    ("user_id", "user_api_key_user_id"),
    ("team_id", "user_api_key_team_id"),
    ("org_id", "user_api_key_org_id"),
    ("project_id", "user_api_key_project_id"),
    ("team_alias", "user_api_key_team_alias"),
)
_ROW_TIMES: Final = ("completionStartTime",)
_SPEND_LOG_CALL: Final[ContextVar[SpendLogCall | None]] = ContextVar("rust_spend_log_call", default=None)
_FAILURE_COUNTER_STEPS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: holds each running step until it ends


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}  # pyright: ignore[reportUnknownVariableType]  # host requests are JSON objects


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _cost(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _texts(value: object) -> Sequence[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []  # pyright: ignore[reportUnknownVariableType]  # a JSON array


def _row_kwargs(position: object) -> Mapping[str, object]:
    call: Final = _SPEND_LOG_CALL.get()
    if call is None or not isinstance(position, int) or not 0 <= position < len(call.rows):
        return {}
    return call.rows[position]


def _started_at(kwargs: Mapping[str, object]) -> datetime | None:
    started: Final = kwargs.get("start_time")
    if isinstance(started, datetime):
        return started
    logged: Final = getattr(kwargs.get("litellm_logging_obj"), "start_time", None)
    return logged if isinstance(logged, datetime) else None


def _response() -> object:
    call: Final = _SPEND_LOG_CALL.get()
    return None if call is None else call.response


def _spawn(coroutine: Coroutine[object, object, object], task_name: str) -> None:
    _ = spawn_logging_task(asyncio.get_running_loop(), coroutine, task_name=task_name)


def _payload(row: Mapping[str, object]) -> SpendLogsPayload:
    return {  # pyright: ignore[reportReturnType]  # the Rust row's columns, the payload's keys
        key: datetime.fromisoformat(value) if key in _ROW_TIMES and isinstance(value, str) else value
        for key, value in row.items()
    }


def _written_row(request: Mapping[str, object]) -> Mapping[str, object]:
    row: Final = _mapping(request.get("row"))
    accounting_payload: Final = request.get("accounting_payload")
    return row if accounting_payload is None else {**row, "accounting_payload": accounting_payload}


def _savings(row: Mapping[str, object]) -> Mapping[str, object]:
    from litellm.accounting.snapshot import (
        _savings_or_zero,  # pyright: ignore[reportPrivateUsage]  # savings pricing stays in Python and feeds the Rust row
    )

    metadata: Final = _FIELDS.validate_json(_text(row.get("metadata")) or "{}")
    return _savings_or_zero(row, metadata, get_llm_router)._asdict()  # pyright: ignore[reportArgumentType]  # the built row's savings columns


async def _committed_tracked(request: Mapping[str, object]) -> Sequence[str]:
    from litellm.proxy.proxy_server import spend_counter_store
    from litellm.proxy.spend_tracking.committed_spend import committed_spend_tracked, warm_committed_counter_keys
    from litellm.types.proxy.carried_budget_state import EntitySpendCounterKeys

    if request.get("warm_committed") is not True or not committed_spend_tracked():
        return []
    warmed: Final = await warm_committed_counter_keys(
        spend_counter_store,
        EntitySpendCounterKeys.of(
            token=_text(request.get("token")),
            team_id=_text(request.get("team_id")),
            user_id=_text(request.get("user_id")),
            org_id=_text(request.get("org_id")),
            end_user_id=_text(request.get("end_user_id")),
        ),
    )
    return list(warmed.tracked())


def _log_failed_counter_steps(increment: "asyncio.Task[None]") -> None:
    error: Final = None if increment.cancelled() else increment.exception()
    if error is not None:
        verbose_proxy_logger.error("Failed to raise spend counters for a failure spend row: %s", error, exc_info=error)


@dataclass(frozen=True, slots=True)
class SpendLogHost:
    """The callback's host calls, with the counter step's own calls answered by its host."""

    counters: _CounterHost = field(default_factory=_CounterHost)

    async def budget_windows(self, request: object) -> object:
        return await self.counters.budget_windows(request)

    async def stored_spend(self, request: object) -> tuple[float | None, ...]:
        return await self.counters.stored_spend(request)

    async def cached_spend(self, request: object) -> tuple[float, ...]:
        return await self.counters.cached_spend(request)

    async def charge_failure_counters(self, request: Mapping[str, object]) -> None:
        """``_increment_failure_spend_counters``: the failure rows' counter steps, sent before the client sees the
        error. A counter store slower than ``FAILURE_SPEND_COUNTER_WAIT_SECONDS`` finishes them in the background, in a
        task of this proxy's own, so their host calls are still answered after the hook returned.
        """
        increment: Final = asyncio.create_task(charge_counter_steps(_STEPS.validate_python(request.get("steps", []))))
        _FAILURE_COUNTER_STEPS.add(increment)
        increment.add_done_callback(_FAILURE_COUNTER_STEPS.discard)
        increment.add_done_callback(_log_failed_counter_steps)
        finished, _ = await asyncio.wait((increment,), timeout=FAILURE_SPEND_COUNTER_WAIT_SECONDS)
        if not finished:
            verbose_proxy_logger.warning(
                "Spend counters for failure spend rows took over %ss, finishing them in the background",
                FAILURE_SPEND_COUNTER_WAIT_SECONDS,
            )

    async def enrich_key_identity(self, request: Mapping[str, object]) -> Mapping[str, object]:
        metadata: Final[dict[str, object]] = {  # mutable-ok: the enrichment fills it in place
            "user_api_key": request.get("api_key"),
            **{logged: request.get(member) for member, logged in _KEY_IDENTITY},
        }
        _ = await ProxyDBLogger._enrich_failure_metadata_with_key_info(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]  # the cost callback's own enrichment fills the dict in place
            metadata=metadata, resolve_missing_key_identity=request.get("resolve_missing_key_identity") is True
        )
        return {member: _text(metadata.get(logged)) for member, logged in _KEY_IDENTITY}

    async def accounting_context(self, request: Mapping[str, object]) -> Mapping[str, object]:
        from litellm.proxy.proxy_server import accounting_window_keys, customer_accounting_counters

        kwargs: Final = _row_kwargs(request.get("position"))
        built: Final = request.get("built")
        row: Final = _mapping(built) if built is not None else None
        token: Final = _text(request.get("token"))
        customer: Final = customer_hierarchy_ids_from_request(kwargs)
        charges: Final = request.get("charges") is True
        customer_counters: Final = (
            await customer_accounting_counters(
                customer, customer_budget_scopes_from_request(kwargs), started_at=_started_at(kwargs)
            )
            if charges
            else ()
        )
        windows: Final = (
            await accounting_window_keys(
                token, _text(request.get("team_id")), customer_counters, user_id=_text(request.get("user_id"))
            )
            if charges and row is not None
            else ()
        )
        router: Final = get_llm_router()
        return {
            "windows": json.loads(json.dumps(safe_json_structure(list(windows)))),
            "customer_entities": [] if customer is None else list(customer.entities().items()),
            "customer_windows": [
                {
                    "counter_key": counter.counter_key,
                    "entity_type": counter.entity_type,
                    "entity_id": counter.entity_id,
                    "window_duration": counter.window_duration,
                    "window_start": None if counter.window_start is None else counter.window_start.isoformat(),
                }
                for counter in customer_counters
            ],
            "committed_tracked": await _committed_tracked(request),
            "savings": None if row is None else _savings(row),
            "model_access_groups": []
            if row is None
            else list(
                debitable_model_access_groups(
                    get_request_model_access_groups(kwargs), _text(row.get("model_id")), router
                )
            ),
            "chains_through_db": row is not None and _turn_chains_through_the_db(row, router),
        }

    def after_write(self, request: Mapping[str, object]) -> None:
        from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj

        _spawn(
            proxy_logging_obj.db_spend_update_writer.record_after_ledger_row(
                payload=_payload(_written_row(request)),
                prisma_client=prisma_client,
                kwargs=_row_kwargs(request.get("position")),
                completion_response=_response(),
                hashed_token=_text(request.get("token")),
                team_id=_text(request.get("team_id")),
                billable_cost=_cost(request.get("billable_cost")),
                spend_logs_kept=request.get("spend_logs_kept") is True,
                claimed=request.get("claimed") is True,
            ),
            "spend log post-write",
        )

    def after_charge(self, request: Mapping[str, object]) -> None:
        from litellm.proxy.proxy_server import proxy_logging_obj, update_cache

        token: Final = _text(request.get("token"))
        end_user_id: Final = _text(request.get("end_user_id"))
        response_cost: Final = _cost(request.get("response_cost"))
        tags: Final = request.get("tags")
        max_budget: Final = request.get("end_user_max_budget")
        _spawn(
            update_cache(
                token=token,
                user_id=_text(request.get("user_id")),
                end_user_id=end_user_id,
                response_cost=response_cost,
                team_id=_text(request.get("team_id")),
                parent_otel_span=get_parent_otel_span_from_kwargs(dict(_row_kwargs(0))),
                tags=None if tags is None else list(_texts(tags)),
            ),
            "update cache",
        )
        _spawn(
            proxy_logging_obj.slack_alerting_instance.customer_spend_alert(
                token=token,
                key_alias=_text(request.get("key_alias")),
                end_user_id=end_user_id,
                response_cost=response_cost,
                max_budget=float(max_budget) if isinstance(max_budget, int | float) else None,
            ),
            "customer spend alert",
        )

    def failed_tracking_alert(self, request: Mapping[str, object]) -> None:
        from litellm.proxy.proxy_server import proxy_logging_obj

        _spawn(
            proxy_logging_obj.failed_tracking_alert(
                error_message=_text(request.get("error_message")) or "",
                failing_model=_text(request.get("failing_model")) or "",
            ),
            "failed tracking alert",
        )

    def batch_accounting(self, request: Mapping[str, object]) -> None:
        outcome: Final = _batch_accounting_for(_row_kwargs(0).get("litellm_call_id"))
        if outcome is None:
            return
        outcome.write_started = outcome.write_started or request.get("write_started") is True
        outcome.failed = outcome.failed or request.get("failed") is True

    def logged_cost(self, request: Mapping[str, object]) -> None:
        kwargs: Final = _row_kwargs(0)
        payload: Final = kwargs.get("standard_logging_object")
        if not isinstance(payload, dict):
            return
        cost: Final = _cost(request.get("cost"))
        discount: Final = request.get("discount")
        margin: Final = request.get("customer_margin")
        _ = apply_callback_customer_cost_margin(
            payload,  # pyright: ignore[reportUnknownArgumentType, reportArgumentType]  # the live standard logging payload
            cost
            if discount is None
            else record_freed_cost(
                dict(kwargs),
                payload,  # pyright: ignore[reportUnknownArgumentType, reportArgumentType]  # the live standard logging payload
                cost,
                _cost(discount),
            ),
            float(margin) if isinstance(margin, int | float) else None,
        )

    async def increment_search_free_tier(self, request: Mapping[str, object]) -> float | None:
        from litellm.proxy.proxy_server import proxy_logging_obj

        counter: Final = proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache
        key: Final = _text(request.get("key"))
        units: Final = request.get("units")
        ttl: Final = request.get("ttl")
        if counter is None or key is None:
            return None
        if not isinstance(units, int) or not isinstance(ttl, int):
            return None
        with service_target("search_free_tier"):
            return await counter.async_increment(key, units, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # RedisCache leaves its key parameter untyped


def _on_async_line(hook: Hook) -> bool:
    from litellm.litellm_core_utils.litellm_logging import V2_ASYNC_LOGGING_LINE

    return hook is not Hook.attempt_ended or V2_ASYNC_LOGGING_LINE.get()


class _SpendLogRustLogger(RustLogger):
    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        if not _on_async_line(hook):
            return None
        call: Final = getattr(view, Field.spend_log_input.value, None)
        token: Final = _SPEND_LOG_CALL.set(call if isinstance(call, SpendLogCall) else None)
        try:
            return await super().run_rust(hook, reads, view, ctx)
        finally:
            _SPEND_LOG_CALL.reset(token)


@dataclass(frozen=True, slots=True)
class _OnRequestBatch:
    """Starts each hook with the request's post-call Redis batch, so the counter increments ride it and a later
    reader in the request, such as the Prometheus budget gauges, can tell whether its own charge has landed."""

    spend_log: RustSpendLog
    callback: RustHandle

    def manifest(self) -> str:
        return self.callback.manifest()

    def request_state(self) -> object:
        return self.callback.request_state()

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, /
    ) -> Awaitable[object]:
        return self.callback.handle(hook, fields, call, state)

    def start(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, state: object, host: object, /
    ) -> Execution:
        from litellm.proxy.proxy_server import spend_counter_store

        return self.spend_log.start(
            hook, fields, call, host, post_call_request() if spend_counter_store.has_redis else None
        )


def spend_logger(writer: RustSpendLogWriter, spend_counters: NativeSpendCounters) -> RustLogger:
    spend_log: Final = RustSpendLog(writer, spend_counters, _token_redactor())
    return rust_logger(
        _OnRequestBatch(spend_log, spend_log.callback()),
        projections=MappingProxyType({Field.spend_log_input: spend_log_document}),
        host=SpendLogHost(),
        base=_SpendLogRustLogger,
    )


def rust_spend_loggers(prisma_client: object) -> RustLogger | None:
    from litellm.proxy.proxy_server import spend_counter_store

    writer: Final = rust_spend_log_writer(prisma_client)
    handle: Final = spend_counter_store.handle
    if writer is None or not isinstance(handle, NativeSpendCounters):
        verbose_proxy_logger.debug("Spend tracking - no Rust spend log writer, keeping the Python cost callback")
        return None
    return spend_logger(writer, handle)


def _registered_rust_spend_logger() -> RustLogger | None:
    import litellm

    for callback in litellm.callbacks:
        if isinstance(callback, _SpendLogRustLogger):
            return callback
    return None


def _no_legacy_binding(callback: object, request: object) -> None:
    return None


async def bill_failure_in_rust(
    user_api_key_dict: UserAPIKeyAuth,
    exception: Exception,
    request_data: dict[str, object],  # mutable-ok: the legacy request dict the failure hooks fill in place
) -> None:
    from litellm.proxy.utils import HookRun, hook_fields, run_hook_subscribers
    from litellm.proxy.v2_proxy_fields import DERIVED_POST_CALL_FAILURE_PROXY_FIELDS

    logger: Final = _registered_rust_spend_logger()
    if logger is None:
        return
    await run_hook_subscribers(
        hook_subscribers(Hook.post_call_failure, [logger], admit=Admit.every_entry),
        with_sources(
            hook_fields(request_data, HookRun(response=exception), Field.error),
            DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
            HookArgs(auth=user_api_key_dict, request=request_data, result=exception),
        ),
        _no_legacy_binding,
    )


__all__ = ["SpendLogHost", "bill_failure_in_rust", "rust_spend_loggers", "spend_logger"]
