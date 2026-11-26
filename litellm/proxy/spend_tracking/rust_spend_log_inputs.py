import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import litellm
from litellm._internal_context import reporting_nested_tool_call_failure
from litellm._uuid import uuid
from litellm.constants import BACKGROUND_INTERACTION_COST_POLLING_ENABLED
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.litellm_core_utils.batch_utils import defers_batch_cost
from litellm.litellm_core_utils.core_helpers import (
    budget_reservation_from_metadata,
    finite_cost,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper's dict parameter is untyped upstream
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
from litellm.proxy.db.db_spend_update_writer import debitable_model_access_groups, get_llm_router
from litellm.proxy.hooks.proxy_track_cost_callback import (
    _batch_accounting_for,  # pyright: ignore[reportPrivateUsage]  # the cost callback's batch costing outcome
    _committed_spend_scope,  # pyright: ignore[reportPrivateUsage]  # the failure writer's committed twins
    _is_unbilled_in_progress_interaction,  # pyright: ignore[reportPrivateUsage]  # the cost callback's interaction checks
    _is_unbilled_interaction_response,  # pyright: ignore[reportPrivateUsage]  # the cost callback's interaction checks
    _tracked_app_id,  # pyright: ignore[reportPrivateUsage]  # the cost callback's app attribution
    failure_spend_rows,
    failure_spend_start_time,
    tracks_failure_spend,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.spend_tracking.customer_cost_margin import (
    failed_request_customer_attribution,
    request_customer_cost_margin,
)
from litellm.proxy.spend_tracking.rust_spend_log_rows import spend_log_row_config, spend_log_row_input
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    configured_spend_logs_metadata_fields,
    get_request_model_access_groups,
)
from litellm.proxy.utils import ProxyUpdateSpend
from litellm.types.integrations.custom_logger_v2 import Outcome
from litellm.types.proxy.auth.user_api_key_auth import committed_budget_spend_of_row
from litellm.utils import (
    get_end_user_id_for_cost_tracking,  # pyright: ignore[reportUnknownVariableType]  # the helper's dict parameter is untyped upstream
)

_UNCHARGED_OUTCOMES: Final = frozenset({Outcome.failure, Outcome.fallback_failure})


@dataclass(frozen=True, slots=True)
class SpendLogCall:
    document: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    response: object


def _local_now() -> datetime:
    return datetime.now(timezone.utc).astimezone().replace(tzinfo=None)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}  # pyright: ignore[reportUnknownVariableType]  # logging dicts are str-keyed


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _settings() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import (
        disable_spend_logs,  # pyright: ignore[reportUnknownVariableType]  # the legacy untyped global
        general_settings_view,
        litellm_proxy_budget_name,
    )

    general: Final = general_settings_view()
    retention: Final = configured_spend_logs_metadata_fields()
    return {
        "disable_spend_updates": ProxyUpdateSpend.disable_spend_updates() is True,
        "disable_spend_logs": disable_spend_logs is True,
        "disable_error_logs": general.get("disable_error_logs") is True,
        "disable_responses_id_security": general.get("disable_responses_id_security") is True,
        "row_config": spend_log_row_config()._asdict(),
        "proxy_budget_name": litellm_proxy_budget_name if litellm.max_budget > 0 else None,
        "apply_search_free_tier": litellm.apply_search_free_tier is True,
        "background_interaction_cost_polling": bool(BACKGROUND_INTERACTION_COST_POLLING_ENABLED),
        "spend_logs_metadata_fields": None if retention is None else retention.model_dump(mode="json"),
    }


def _window_resets(reservation: object) -> Sequence[str | None]:
    """Each reservation entry's window reset, as the counter step reads it."""
    from litellm.proxy.spend_tracking.native_budgets import reservation_inputs

    return list(reservation_inputs(_mapping(reservation) or None)[1])


def _metadata(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(get_litellm_metadata_from_kwargs(dict(kwargs)))  # pyright: ignore[reportUnknownArgumentType]  # the metadata helper returns an untyped dict


def _carried_committed(kwargs: Mapping[str, object]) -> Sequence[str] | None:
    carried: Final = committed_budget_spend_of_row(_metadata(kwargs), _mapping(kwargs.get("litellm_params")))
    return None if carried is None else list(carried.counter_keys.tracked())


def _customer_hierarchy(kwargs: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    from litellm.proxy.spend_tracking.rust_spend_counters import (
        _customer_scopes,  # pyright: ignore[reportPrivateUsage]  # the counter step's own hierarchy reader
    )

    return list(_customer_scopes(_metadata(kwargs).get("user_api_key_customer_hierarchy")))


def _row_input(kwargs: Mapping[str, object], response: object, start_time: datetime, end_time: datetime) -> object:
    try:
        built: Final = spend_log_row_input(
            dict(kwargs),
            response,
            start_time,
            end_time,
            get_llm_router(),
            time.time(),
            str(uuid.uuid4()),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # litellm's uuid shim is untyped
        )
    except Exception as error:  # noqa: BLE001  # update_database logs a row it cannot build and still charges it
        return {"unbuildable": str(error)}
    return {key: value for key, value in built.items() if key != "request_id"}


def _row(
    kwargs: Mapping[str, object],
    response: object,
    start_time: datetime,
    end_time: datetime,
    committed: Sequence[str] | None,
    served_model_id: object,
    customer_margin: float | None,
) -> Mapping[str, object]:
    return {
        "row": _row_input(kwargs, response, start_time, end_time),
        "model_access_groups": list(
            debitable_model_access_groups(
                attributed=get_request_model_access_groups(kwargs),
                served_model_id=served_model_id if isinstance(served_model_id, str) else None,
                router=get_llm_router(),
            )
        ),
        "committed_tracked": committed,
        "customer_hierarchy": _customer_hierarchy(kwargs),
        "customer_margin": customer_margin,
        "cost_finalized": bool(_mapping(kwargs.get("standard_logging_object")).get("cost_finalized", False)),
    }


def spend_log_success_input(args: HookArgs) -> SpendLogCall | None:
    """A failed attempt is left to the fallback, so it gets no input and no row input is built."""
    if args.outcome in _UNCHARGED_OUTCOMES:
        return None
    kwargs: Final = _mapping(args.request)
    if not kwargs or not isinstance(args.start_time, datetime) or not isinstance(args.end_time, datetime):
        return None
    response: Final = (
        kwargs.get("async_complete_streaming_response")
        if getattr(args.logging_obj, "stream", None) is True
        else args.result
    )
    metadata: Final = _metadata(kwargs)
    payload: Final = kwargs.get("standard_logging_object")
    logged: Final = _mapping(payload)
    end_user_max_budget: Final = metadata.get("user_api_end_user_max_budget")
    unbilled_interaction: Final = _is_unbilled_interaction_response(response)
    document: Final = {
        "kind": "success",
        "settings": _settings(),
        "identity": {
            "token": _text(metadata.get("user_api_key")),
            "user_id": _text(metadata.get("user_api_key_user_id")),
            "team_id": _text(metadata.get("user_api_key_team_id")),
            "org_id": _text(metadata.get("user_api_key_org_id")),
            "project_id": _text(metadata.get("user_api_key_project_id")),
            "end_user_id": get_end_user_id_for_cost_tracking(dict(_mapping(kwargs.get("litellm_params")))),
            "key_alias": _text(metadata.get("user_api_key_alias")),
            "agent_id": _text(metadata.get("billing_agent_id") or metadata.get("agent_id")),
            "app_id": _tracked_app_id(metadata),
        },
        "call_type": _text(kwargs.get("call_type")),
        "response": {
            "defers_batch_cost": defers_batch_cost(kwargs, response),
            "unbilled_interaction": unbilled_interaction,
            "interaction_in_progress": unbilled_interaction and _is_unbilled_in_progress_interaction(response),
        },
        "cache_hit": bool(logged.get("cache_hit")),
        "kwargs_response_cost": finite_cost(kwargs.get("response_cost")) if payload is None else None,
        "stream": kwargs.get("stream") is True,
        "stream_complete": "complete_streaming_response" in kwargs,
        "end_user_max_budget": float(end_user_max_budget) if isinstance(end_user_max_budget, int | float) else None,
        "batch_accounting": _batch_accounting_for(kwargs.get("litellm_call_id")) is not None,
        "window_resets": _window_resets(budget_reservation_from_metadata(dict(metadata))),
        "row": _row(
            kwargs,
            response,
            args.start_time,
            args.end_time,
            _carried_committed(kwargs),
            logged.get("model_id"),
            request_customer_cost_margin(metadata),
        ),
    }
    return SpendLogCall(document=document, rows=(kwargs,), response=response)


def _failure_served_model_id(row: Mapping[str, object]) -> object:
    metadata: Final = _mapping(_mapping(row.get("litellm_params")).get("metadata"))
    return _mapping(metadata.get("model_info")).get("id") or _mapping(row.get("standard_logging_object")).get(
        "model_id"
    )


def spend_log_failure_input(args: HookArgs) -> SpendLogCall | None:
    request: Final = _mapping(args.request)
    auth: Final = args.auth
    exception: Final = args.result
    if not isinstance(auth, UserAPIKeyAuth) or not isinstance(exception, Exception):
        return None
    completed_cost: Final = _mapping(request.get("standard_logging_object")).get("cost_finalized") is True
    track: Final = tracks_failure_spend(request, exception, auth, completed_cost)
    rows: Final = failure_spend_rows(request, exception, auth, args.traceback_str) if track else ()
    start_time: Final = failure_spend_start_time(request)
    end_time: Final = datetime.now(timezone.utc) if completed_cost else _local_now()
    committed: Final = list(_committed_spend_scope(auth).tracked())
    document: Final = {
        "kind": "failure",
        "settings": _settings(),
        "identity": {
            "token": LiteLLMProxyRequestSetup.get_logged_api_key(auth),
            "user_id": auth.user_id,
            "team_id": auth.team_id,
            "org_id": auth.org_id,
            "project_id": auth.project_id,
            "end_user_id": None if litellm.disable_end_user_cost_tracking else auth.end_user_id,
            "key_alias": auth.key_alias,
        },
        "enrich": not isinstance(exception, DBLookupDeadlineExceeded),
        "track": track,
        "nested_tool_call": reporting_nested_tool_call_failure(),
        "window_resets": _window_resets(auth.budget_reservation),
        "rows": [
            _row(
                row,
                exception,
                start_time,
                end_time,
                committed,
                _failure_served_model_id(row),
                request_customer_cost_margin(failed_request_customer_attribution(row)),
            )
            for row in rows
        ],
    }
    return SpendLogCall(document=document, rows=rows, response=exception)


def spend_log_document(call: object) -> object:
    return call.document if isinstance(call, SpendLogCall) else call


__all__ = ["SpendLogCall", "spend_log_document", "spend_log_failure_input", "spend_log_success_input"]
