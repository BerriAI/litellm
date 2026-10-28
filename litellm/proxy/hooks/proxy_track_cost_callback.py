import asyncio
import traceback
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, cast

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.batch_utils import batch_cost_is_final
from litellm.constants import BACKGROUND_INTERACTION_COST_POLLING_ENABLED, BILLED_RETRY_ATTEMPT_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    budget_reservation_from_metadata,
    get_litellm_metadata_from_kwargs,
    get_metadata_variable_name_from_kwargs,
    get_parent_otel_span_from_kwargs,
)
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import guardrail_information_cost
from litellm.litellm_core_utils.redact_messages import MESSAGE_REDACTION_ENABLE_HEADERS
from litellm.llms.azure.ptu_capacity import is_str_keyed_mapping
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    get_key_object,
    get_team_object,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
from litellm.proxy.db.db_spend_update_writer import (
    DBSpendUpdateWriter,
    debitable_model_access_groups,
    get_llm_router,
)
from litellm.proxy.hooks.search_free_tier import apply_search_free_tier
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup, metadata_variable_name_for_route
from litellm.proxy.route_llm_request import ProxyModelNotFoundError
from litellm.proxy.spend_tracking.spend_counter_batch import post_call_counter_keys, spend_counter_batch_scope
from litellm.proxy.spend_tracking.spend_event import (
    ObjectMapping,
    SpendEventBuildError,
    SpendEventDecodeError,
    build_spend_event,
    decode_spend_event,
    is_offloadable_success,
    spend_event_callback_args,
)
from litellm.proxy.spend_tracking.spend_event_producer import SpendEventProducer
from litellm.proxy.spend_tracking.spend_log_error_logger import (
    should_suppress_spend_log_tracebacks,
    spend_log_error,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import (  # noqa: F401  # legacy module exports
    _sanitize_error_information_for_spend_logs,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    deployment_params_provider,
    get_request_model_access_groups,
    sanitize_error_information_for_spend_logs,
    should_store_prompts_and_responses_in_spend_logs,
)
from litellm.proxy.utils import (
    REJECTED_BEFORE_REQUEST_SETUP,
    REJECTED_BEFORE_REQUEST_SETUP_KEY,
    BilledFailedAttempt,
    ProxyUpdateSpend,
)
from litellm.types.router import Deployment
from litellm.types.utils import (
    CallTypes,
    LiteLLMBatch,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
)
from litellm.utils import get_end_user_id_for_cost_tracking

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging


class BatchAccountingOutcome:
    def __init__(self, call_id: str) -> None:
        self.call_id: Final = call_id
        self.write_started = False
        self.failed = False

    @property
    def retryable_failure(self) -> bool:
        return self.failed and not self.write_started


_BATCH_ACCOUNTING: Final[ContextVar[BatchAccountingOutcome | None]] = ContextVar("batch_accounting", default=None)
_BATCH_CALLBACK_FIELDS: Final = TypeAdapter(dict[str, object])


@contextmanager
def track_batch_accounting(call_id: str) -> Generator[BatchAccountingOutcome, None, None]:
    outcome: Final = BatchAccountingOutcome(call_id)
    token: Final = _BATCH_ACCOUNTING.set(outcome)
    try:
        yield outcome
    finally:
        _BATCH_ACCOUNTING.reset(token)


def _batch_accounting_for(call_id: object) -> BatchAccountingOutcome | None:
    outcome: Final = _BATCH_ACCOUNTING.get()
    return outcome if outcome is not None and call_id == outcome.call_id else None


_UNATTRIBUTED_TRACKABLE_CALL_TYPES: Final[frozenset[str]] = frozenset(
    {
        CallTypes.pass_through.value,
        CallTypes.llm_passthrough_route.value,
        CallTypes.allm_passthrough_route.value,
        CallTypes.call_mcp_tool.value,
        # CheckBatchCost's synthetic logging_obj for a completed managed batch carries
        # whatever LiteLLM_ManagedObjectTable stored at create time, and all of it is
        # None for a batch created before those columns were persisted, or by the master
        # key. The batch already incurred real provider cost, so track it regardless.
        CallTypes.aretrieve_batch.value,
    }
)

# Both spellings, because call_type reaches the callback as str(...) of either the
# enum member or its value.
_CAPTURED_IDENTITY_CALL_TYPES: Final[frozenset[str]] = frozenset(
    (
        CallTypes.aretrieve_batch.value,
        str(CallTypes.aretrieve_batch),
    )
)


_OBJECT_MAPPING: Final = TypeAdapter(ObjectMapping)


def _proxy_stamped_used_client_oauth_token(
    request_data: Mapping[str, object], request_route: str | None
) -> bool | None:
    proxy_bucket: Final = (
        get_metadata_variable_name_from_kwargs(request_data)
        if request_route is None
        else metadata_variable_name_for_route(request_route)
    )
    proxy_metadata: Final = request_data.get(proxy_bucket)
    stamped: Final = proxy_metadata.get("used_client_oauth_token") if isinstance(proxy_metadata, dict) else None
    return stamped if isinstance(stamped, bool) else None


def _str_keyed_fields(value: object) -> Mapping[str, object]:
    return value if is_str_keyed_mapping(value) else {}


def _is_model_group(value: object) -> bool:
    return isinstance(value, str) and bool(value)


_TAG_LIST: Final = TypeAdapter(list[str])


def _is_tag_list(value: object) -> bool:
    try:
        return bool(_TAG_LIST.validate_python(value, strict=True))
    except ValidationError:
        return False


_ROUTED_METADATA_FIELD_CHECKS: Final = MappingProxyType({"model_group": _is_model_group, "tags": _is_tag_list})


def _routed_metadata_fallback(request_data: Mapping[str, object]) -> Mapping[str, object]:
    caller_metadata: Final = _str_keyed_fields(request_data.get("metadata"))
    routed_metadata: Final = _str_keyed_fields(request_data.get("litellm_metadata"))
    return {
        field: routed_metadata[field]
        for field, is_valid in _ROUTED_METADATA_FIELD_CHECKS.items()
        if not caller_metadata.get(field) and is_valid(routed_metadata.get(field))
    }


def _failure_snapshot_with_guardrails_recorded_after_it(
    failure_snapshot: object, recorded_guardrail_information: object
) -> Mapping[str, object] | None:
    if (
        not recorded_guardrail_information
        or not isinstance(failure_snapshot, dict)
        or failure_snapshot.get("guardrail_information")
    ):
        return None
    return {**failure_snapshot, "guardrail_information": recorded_guardrail_information}


def _guardrail_info_lifted_from_litellm_metadata(
    litellm_metadata: ObjectMapping, *, guardrail_info_already_set: bool
) -> ObjectMapping:
    guardrail_info: Final = litellm_metadata.get("standard_logging_guardrail_information")
    if guardrail_info is None or guardrail_info_already_set:
        return {}
    return {"standard_logging_guardrail_information": guardrail_info}


def _redaction_opt_in_lifted_from_litellm_metadata(
    litellm_metadata: ObjectMapping, *, caller_set_headers: bool
) -> ObjectMapping:
    request_headers: Final = litellm_metadata.get("headers")
    if caller_set_headers or not isinstance(request_headers, dict):
        return {}
    headers: Final = _OBJECT_MAPPING.validate_python(request_headers)
    opt_in_headers: Final = {name: headers[name] for name in MESSAGE_REDACTION_ENABLE_HEADERS if name in headers}
    return {"headers": opt_in_headers} if opt_in_headers else {}


def _proxy_spend_writer() -> DBSpendUpdateWriter:
    from litellm.proxy.proxy_server import proxy_logging_obj

    return proxy_logging_obj.db_spend_update_writer


DeploymentLookup: TypeAlias = Callable[[str], Deployment | None]


def _proxy_deployment(model_id: str) -> Deployment | None:
    router: Final = get_llm_router()
    return None if router is None else router.get_deployment(model_id=model_id)


def _billed_failed_attempts(request_data: Mapping[str, object]) -> tuple[BilledFailedAttempt, ...]:
    """The earlier billed attempts ``_failure_fields_to_lift`` stashed on the request, if any."""
    attempts: Final = request_data.get("billed_failed_attempts")
    if not isinstance(attempts, tuple):
        return ()
    lifted: Final = cast("tuple[object, ...]", attempts)  # cast-ok: lifted by _failure_fields_to_lift, checked below
    return tuple(attempt for attempt in lifted if isinstance(attempt, BilledFailedAttempt))


def _failure_spend_rows(
    request_data: Mapping[str, object],
    final_cost: float,
    late_guardrails_snapshot: object,
    deployment_lookup: DeploymentLookup,
) -> tuple[tuple[float, Mapping[str, object]], ...]:
    """One ``(response_cost, kwargs)`` per spend row a failure writes: the final attempt under the
    request id, then every earlier attempt the provider also billed under its own row."""
    final_row: Final[Mapping[str, object]] = (
        request_data
        if late_guardrails_snapshot is None
        else {**request_data, "standard_logging_object": late_guardrails_snapshot}
    )
    return (
        (final_cost, final_row),
        *(
            (
                attempt.response_cost,
                _attempt_spend_row_kwargs(
                    request_data, attempt, position, _attempt_deployment(attempt, deployment_lookup)
                ),
            )
            for position, attempt in enumerate(_billed_failed_attempts(request_data))
        ),
    )


def _attempt_deployment(attempt: BilledFailedAttempt, deployment_lookup: DeploymentLookup) -> Deployment | None:
    return None if attempt.deployment_id is None else deployment_lookup(attempt.deployment_id)


def _deployment_attribution(deployment: Deployment) -> Mapping[str, object]:
    """The spend-log attribution fields of the deployment that served an attempt: the model string it
    was called with, its provider, its configured ``api_base`` and its model group."""
    return MappingProxyType(
        {
            "model": deployment.litellm_params.model,
            "custom_llm_provider": deployment_params_provider(deployment.litellm_params),
            "api_base": deployment.litellm_params.api_base,
            "model_group": deployment.model_name,
        }
    )


_NO_ATTRIBUTION: Final[Mapping[str, object]] = MappingProxyType({})


def _attempt_spend_row_kwargs(
    request_data: Mapping[str, object], attempt: BilledFailedAttempt, position: int, deployment: Deployment | None
) -> Mapping[str, object]:
    """The final attempt's row kwargs re-keyed for one earlier billed attempt: its own
    ``request_id`` (``<id>-attempt-<position>``, since ``request_id`` is the spend-log primary key
    and the retry count restarts after a fallback hop), its estimated usage and cost, its retry
    count under ``attempted_retries``, the deployment that billed it under ``model_info.id`` and,
    when the router still knows that deployment, its model, provider, ``api_base`` and model group.
    The row is stamped ``billed_retry_attempt`` so the daily spend tables keep its spend, tokens
    and model usage without counting it as a request the caller sent."""
    suffix: Final = f"-attempt-{position}"
    attribution: Final = _NO_ATTRIBUTION if deployment is None else _deployment_attribution(deployment)
    raw_params: Final = cast("Mapping[str, object]", request_data.get("litellm_params") or {})  # cast-ok: request dict
    litellm_params: Final = {
        **raw_params,
        **({} if deployment is None else {"api_base": attribution["api_base"]}),
    }
    metadata: Final = {
        **cast("Mapping[str, object]", litellm_params.get("metadata") or {}),  # cast-ok: request dict
        "attempted_retries": attempt.attempted_retries,
        BILLED_RETRY_ATTEMPT_METADATA_KEY: True,
        **({} if deployment is None else {"model_group": attribution["model_group"]}),
    }
    metadata.pop("previous_models", None)
    if attempt.deployment_id is not None:
        metadata["model_info"] = {
            **cast("Mapping[str, object]", metadata.get("model_info") or {}),  # cast-ok: router dict
            "id": attempt.deployment_id,
        }
    litellm_params["metadata"] = metadata
    row: Final = {
        **request_data,
        **({} if deployment is None else {k: attribution[k] for k in ("model", "custom_llm_provider")}),
        "litellm_params": litellm_params,
        "combined_usage_object": attempt.usage,
        "response_cost": attempt.response_cost,
    }
    row.pop("billed_failed_attempts", None)
    call_id: Final = request_data.get("litellm_call_id") or litellm_params.get("litellm_call_id")
    if isinstance(call_id, str):
        row["litellm_call_id"] = f"{call_id}{suffix}"
    standard_logging_object: Final = request_data.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        payload: Final = cast("Mapping[str, object]", standard_logging_object)  # cast-ok: StandardLoggingPayload dict
        if isinstance(payload.get("id"), str):
            row["standard_logging_object"] = {
                **payload,
                "id": f"{payload['id']}{suffix}",
                **({} if deployment is None else {**attribution, "model_id": attempt.deployment_id}),
            }
    return row


class _ProxyDBLogger(CustomLogger):
    def __init__(
        self,
        spend_event_producer: SpendEventProducer | None = None,
        *,
        turn_off_message_logging: bool = False,
        message_logging: bool = True,
        spend_writer: Callable[[], DBSpendUpdateWriter] = _proxy_spend_writer,
        deployment_lookup: DeploymentLookup = _proxy_deployment,
    ) -> None:
        super().__init__(turn_off_message_logging=turn_off_message_logging, message_logging=message_logging)
        self.spend_event_producer = spend_event_producer
        self._spend_writer: Final = spend_writer
        self._deployment_lookup: Final = deployment_lookup

    async def async_log_success_event(
        self, kwargs: ObjectMapping, response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if self.spend_event_producer is None or not is_offloadable_success(response_obj):
            await self._PROXY_track_cost_callback(kwargs, response_obj, start_time, end_time)
            return
        event: Final = build_spend_event(
            kwargs,
            response_obj,
            start_time,
            end_time,
            store_bodies=should_store_prompts_and_responses_in_spend_logs(),
        )
        if isinstance(event, SpendEventBuildError):
            verbose_proxy_logger.warning("collector: tracking cost in-process, event not buildable: %s", event.reason)
            await self._PROXY_track_cost_callback(kwargs, response_obj, start_time, end_time)
            return
        await self.spend_event_producer.publish(event)

    async def run_spend_event(self, line: bytes) -> None:
        """Run the unchanged cost pipeline on a serialized spend event (sidecar consumer and in-process fallback)."""
        event: Final = decode_spend_event(line)
        if isinstance(event, SpendEventDecodeError):
            verbose_proxy_logger.error("collector: discarding undecodable spend event: %s", event.reason)
            return
        args: Final = spend_event_callback_args(event)
        await self._PROXY_track_cost_callback(args.kwargs, args.response_obj, args.start_time, args.end_time)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        try:
            await _release_budget_reservation(budget_reservation=user_api_key_dict.budget_reservation)
        except Exception:
            verbose_proxy_logger.exception("Failed to release budget reservation during failure handling")
            try:
                await _invalidate_budget_reservation_counters(budget_reservation=user_api_key_dict.budget_reservation)
                if user_api_key_dict.budget_reservation is not None:
                    user_api_key_dict.budget_reservation["finalized"] = True
            except Exception:
                verbose_proxy_logger.exception(
                    "Failed to invalidate budget reservation counters after failure release failed"
                )

        request_route: Final = user_api_key_dict.request_route
        if (
            ProxyDBLogger._should_track_errors_in_db() is False
            or ProxyDBLogger._skips_pre_routing_rejections()
            and is_pre_routing_rejection(original_exception, _REQUEST_DATA.validate_python(request_data))
            or request_route is not None
            and not (
                RouteChecks.is_llm_api_route(route=request_route) or RouteChecks.is_info_route(route=request_route)
            )
            or request_route is not None
            and RouteChecks.is_dashboard_session_info_route(request_route, user_api_key_dict.team_id)
        ):
            return

        _metadata = dict(
            LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(user_api_key_dict=user_api_key_dict)
        )
        _metadata["user_api_key"] = LiteLLMProxyRequestSetup.get_logged_api_key(user_api_key_dict)
        _metadata["status"] = "failure"
        _error_information = StandardLoggingPayloadSetup.get_error_information(
            original_exception=original_exception,
            traceback_str=traceback_str,
        )
        if should_suppress_spend_log_tracebacks():
            # Drop the traceback key entirely so the per-row Metadata pane in
            # the UI (which renders the JSON blob verbatim) doesn't show a
            # noisy ``"traceback": ""`` line. Downstream consumers all use
            # ``.get("traceback")`` / truthy checks, and the TypedDict marks
            # the field as optional, so omitting is type-safe.
            _error_information.pop("traceback", None)
        # Strip echoed request input + apply DB-size cap before storing in
        # the spend-log metadata column (LIT-2992). Result is never None
        # here because the input above is constructed non-None.
        _error_information = cast(
            StandardLoggingPayloadErrorInformation,
            sanitize_error_information_for_spend_logs(_error_information, original_exception=original_exception),
        )
        _metadata["error_information"] = _error_information

        _metadata = await ProxyDBLogger._enrich_failure_metadata_unless_db_stalled(  # rebind-ok: pre-existing rebinding on a rename-only line
            metadata=_metadata, original_exception=original_exception
        )

        request_fields: Final = _str_keyed_fields(request_data)
        _metadata["used_client_oauth_token"] = _proxy_stamped_used_client_oauth_token(request_fields, request_route)

        existing_metadata: Final[dict] = request_data.get("metadata", None) or {}
        existing_metadata.update({**_routed_metadata_fallback(request_fields), **_metadata})

        litellm_metadata_bucket: Final = request_data.get("litellm_metadata")
        litellm_metadata: Final[ObjectMapping] = (
            _OBJECT_MAPPING.validate_python(litellm_metadata_bucket)
            if isinstance(litellm_metadata_bucket, dict)
            else {}
        )
        existing_metadata.update(
            _guardrail_info_lifted_from_litellm_metadata(
                litellm_metadata,
                guardrail_info_already_set="standard_logging_guardrail_information" in existing_metadata,
            )
        )

        if "litellm_params" not in request_data:
            request_data["litellm_params"] = {}

        existing_litellm_params: Final = request_data.get("litellm_params", {})
        existing_litellm_metadata: Final = existing_litellm_params.get("metadata", {}) or {}

        # Preserve tags from existing metadata
        if existing_litellm_metadata.get("tags"):
            existing_metadata["tags"] = existing_litellm_metadata.get("tags")

        request_data["litellm_params"]["proxy_server_request"] = (
            request_data.get("proxy_server_request") or existing_litellm_params.get("proxy_server_request") or {}
        )
        request_data["litellm_params"]["metadata"] = {
            **existing_metadata,
            **_redaction_opt_in_lifted_from_litellm_metadata(
                litellm_metadata, caller_set_headers="headers" in existing_metadata
            ),
        }

        # Preserve model name and custom_llm_provider
        if "model" not in request_data:
            request_data["model"] = existing_litellm_params.get("model") or request_data.get("model", "")
        if "custom_llm_provider" not in request_data:
            request_data["custom_llm_provider"] = existing_litellm_params.get(
                "custom_llm_provider"
            ) or request_data.get("custom_llm_provider", "")

        # Propagate standard_logging_object and litellm_trace_id from the Logging
        # instance so the failure row carries the same trace_id Langfuse received.
        _litellm_logging_obj: Final = request_data.get("litellm_logging_obj")
        if _litellm_logging_obj is not None:
            if not request_data.get("standard_logging_object"):
                request_data["standard_logging_object"] = getattr(_litellm_logging_obj, "model_call_details", {}).get(
                    "standard_logging_object"
                )
            if request_data.get("litellm_trace_id") is None:
                request_data["litellm_trace_id"] = getattr(_litellm_logging_obj, "litellm_trace_id", None)

        lifted_start_time: Final = request_data.get("start_time")
        actual_start_time: Final = (
            lifted_start_time
            if isinstance(lifted_start_time, datetime)
            else getattr(_litellm_logging_obj, "start_time", None) or datetime.now()
        )

        # A stream that broke mid-flight still billed the provider for the
        # chunks already delivered. ``post_call_failure_hook`` lifts that
        # recovered cost onto request_data (the usage rides along in
        # ``combined_usage_object`` for the token columns), so attribute the
        # real partial spend to this failure row instead of zero.
        recovered_stream_cost: Final = (
            max(float(request_data.get("response_cost") or 0.0), 0.0)
            if isinstance(request_data.get("combined_usage_object"), litellm.Usage)
            else 0.0
        )
        recovered_response_cost: Final = recovered_stream_cost + guardrail_information_cost(
            existing_metadata.get("standard_logging_guardrail_information")
        )

        failure_snapshot_with_late_guardrails: Final = _failure_snapshot_with_guardrails_recorded_after_it(
            request_data.get("standard_logging_object"),
            existing_metadata.get("standard_logging_guardrail_information"),
        )

        await self._write_failure_spend_rows(
            user_api_key_dict=user_api_key_dict,
            original_exception=original_exception,
            start_time=actual_start_time,
            spend_rows=_failure_spend_rows(
                cast("Mapping[str, object]", request_data),  # cast-ok: request dict, copied per row below
                recovered_response_cost,
                failure_snapshot_with_late_guardrails,
                self._deployment_lookup,
            ),
        )

    async def _write_failure_spend_rows(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        original_exception: Exception,
        start_time: datetime,
        spend_rows: Sequence[tuple[float, Mapping[str, object]]],
    ) -> None:
        """One spend row (and one key/team/user spend deduction) per ``(response_cost, kwargs)`` pair."""
        for row_cost, row_kwargs in spend_rows:
            await self._spend_writer().update_database(
                token=LiteLLMProxyRequestSetup.get_logged_api_key(user_api_key_dict),
                response_cost=row_cost,
                user_id=user_api_key_dict.user_id,
                end_user_id=None if litellm.disable_end_user_cost_tracking else user_api_key_dict.end_user_id,
                team_id=user_api_key_dict.team_id,
                kwargs=dict(row_kwargs),
                completion_response=original_exception,
                start_time=start_time,
                end_time=datetime.now(),
                org_id=user_api_key_dict.org_id,
                project_id=user_api_key_dict.project_id,
            )

    async def _PROXY_track_cost_callback(
        self,
        kwargs,  # kwargs to completion
        completion_response: litellm.ModelResponse | object | None,  # response from completion
        start_time=None,
        end_time=None,  # start/end time for completion
    ):
        from litellm.proxy.proxy_server import (
            increment_spend_counters,
            proxy_logging_obj,
            update_cache,
        )

        accounting: Final = (
            _batch_accounting_for(_BATCH_CALLBACK_FIELDS.validate_python(kwargs).get("litellm_call_id"))
            if _BATCH_ACCOUNTING.get() is not None
            else None
        )
        verbose_proxy_logger.debug("INSIDE _PROXY_track_cost_callback")
        try:
            verbose_proxy_logger.debug(
                "kwargs stream: %s + complete streaming response: %s",
                kwargs.get("stream", None),
                kwargs.get("complete_streaming_response", None),
            )
            parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs=kwargs)
            litellm_params: Final = kwargs.get("litellm_params", {}) or {}
            end_user_id: Final = get_end_user_id_for_cost_tracking(litellm_params)
            metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
            # Only fetch key details when user_id wasn't already populated (e.g. direct MCP REST calls).
            # Avoids a cache/DB lookup on every normal LLM request.
            if metadata.get("user_api_key") and not metadata.get("user_api_key_user_id"):
                metadata = await ProxyDBLogger._enrich_failure_metadata_with_key_info(  # rebind-ok: enriched metadata replaces the original
                    metadata=metadata,
                    resolve_missing_key_identity=str(kwargs.get("call_type")) not in _CAPTURED_IDENTITY_CALL_TYPES,
                )
                _write_spend_metadata_to_kwargs(kwargs=kwargs, metadata=metadata)
            budget_reservation: Final = _get_budget_reservation_from_metadata(metadata=metadata)
            if accounting is not None and (budget_reservation is not None or litellm.apply_search_free_tier):
                accounting.write_started = True
            if (
                isinstance(completion_response, LiteLLMBatch)
                and kwargs.get("call_type") == CallTypes.aretrieve_batch.value
                and not batch_cost_is_final(completion_response)
            ):
                verbose_proxy_logger.debug(
                    "Cost tracking deferred for batch %s still in status %s",
                    completion_response.id,
                    completion_response.status,
                )
                await _release_budget_reservation(budget_reservation=budget_reservation)
                return
            user_id: Final = cast(str | None, metadata.get("user_api_key_user_id", None))
            team_id: Final = cast(str | None, metadata.get("user_api_key_team_id", None))
            org_id: Final = cast(str | None, metadata.get("user_api_key_org_id", None))
            project_id: Final = (
                project_id_value
                if isinstance(project_id_value := metadata.get("user_api_key_project_id"), str)
                else None
            )
            key_alias: Final = cast(str | None, metadata.get("user_api_key_alias", None))
            end_user_max_budget: Final = metadata.get("user_api_end_user_max_budget", None)
            sl_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)
            raw_cost: Final = _str_keyed_fields(
                sl_object if sl_object is not None else kwargs  # pyright: ignore[reportUnknownArgumentType]  # untyped success-callback payload
            ).get("response_cost")
            priced_cost: Final = raw_cost if isinstance(raw_cost, (int, float)) else None
            response_cost: Final = (
                await apply_search_free_tier(
                    kwargs=kwargs,  # pyright: ignore[reportUnknownArgumentType]  # untyped success-callback payload
                    sl_object=sl_object,  # pyright: ignore[reportUnknownArgumentType]  # payload carries untyped lists
                    response_cost=priced_cost,
                    completion_response=completion_response,
                    counter=proxy_logging_obj.internal_usage_cache.dual_cache.redis_cache,
                    now=datetime.now(timezone.utc),
                )
                if litellm.apply_search_free_tier and sl_object is not None and priced_cost is not None
                else priced_cost
            )
            tags: Final = _get_request_tags_for_cost_tracking(
                sl_object=sl_object,
                metadata=metadata,
            )
            model_access_groups: Final = debitable_model_access_groups(
                attributed=get_request_model_access_groups(kwargs),
                served_model_id=sl_object.get("model_id") if sl_object is not None else None,
                router=get_llm_router(),
            )

            if response_cost is not None:
                user_api_key: Final = metadata.get("user_api_key", None)
                verbose_proxy_logger.debug(
                    "user_api_key %s, user_id %s, team_id %s, end_user_id %s",
                    user_api_key,
                    user_id,
                    team_id,
                    end_user_id,
                )
                call_type: str | None = kwargs.get("call_type")
                if _should_track_cost_callback(
                    user_api_key=user_api_key,
                    user_id=user_id,
                    team_id=team_id,
                    end_user_id=end_user_id,
                    call_type=call_type,
                    agent_id=metadata.get("billing_agent_id") or metadata.get("agent_id"),
                ):
                    ## UPDATE DATABASE
                    charged: Final = await _update_database_and_spend_counters(
                        proxy_logging_obj=proxy_logging_obj,
                        increment_spend_counters=increment_spend_counters,
                        user_api_key=user_api_key,
                        user_id=user_id,
                        end_user_id=end_user_id,
                        team_id=team_id,
                        org_id=org_id,
                        kwargs=kwargs,
                        completion_response=completion_response,
                        start_time=start_time,
                        end_time=end_time,
                        response_cost=response_cost,
                        budget_reservation=budget_reservation,
                        request_tags=tags,
                        model_access_groups=model_access_groups,
                        project_id=project_id,
                    )
                    if not charged:
                        return

                    # update cache (fire-and-forget for backward compat:
                    # cached object fields, soft budget alerts, etc.)
                    asyncio.create_task(
                        update_cache(
                            token=user_api_key,
                            user_id=user_id,
                            end_user_id=end_user_id,
                            response_cost=response_cost,
                            team_id=team_id,
                            parent_otel_span=parent_otel_span,
                            tags=tags,
                        )
                    )

                    await proxy_logging_obj.slack_alerting_instance.customer_spend_alert(
                        token=user_api_key,
                        key_alias=key_alias,
                        end_user_id=end_user_id,
                        response_cost=response_cost,
                        max_budget=end_user_max_budget,
                    )
                elif budget_reservation is not None:
                    await _release_budget_reservation(budget_reservation=budget_reservation)
            else:
                if _is_unbilled_interaction_response(completion_response):
                    if BACKGROUND_INTERACTION_COST_POLLING_ENABLED and _is_unbilled_in_progress_interaction(
                        completion_response
                    ):
                        verbose_proxy_logger.debug(
                            "Cost tracking deferred for in-progress background interaction; "
                            "the budget reservation stays open until the poll task logs the final usage"
                        )
                        return
                    await _release_budget_reservation(budget_reservation=budget_reservation)
                    verbose_proxy_logger.debug(
                        "Released the budget reservation for an interaction create with no usage "
                        "that no poll task will settle"
                    )
                    return
                await _release_budget_reservation(budget_reservation=budget_reservation)
                # Non-model call types (health checks, afile_delete) have no model or standard_logging_object.
                # Use .get() for "stream" to avoid KeyError on health checks.
                if sl_object is None and (not kwargs.get("model") or kwargs.get("call_type") == "_arealtime"):
                    verbose_proxy_logger.warning(
                        "Cost tracking - skipping, no standard_logging_object for call_type=%s",
                        kwargs.get("call_type", "unknown"),
                    )
                    return
                if kwargs.get("stream") is not True or (
                    kwargs.get("stream") is True and "complete_streaming_response" in kwargs
                ):
                    if sl_object is not None:
                        cost_tracking_failure_debug_info: dict | str = (
                            sl_object["response_cost_failure_debug_info"]
                            or "response_cost_failure_debug_info is None in standard_logging_object"
                        )
                    else:
                        cost_tracking_failure_debug_info = "standard_logging_object not found"
                    model = kwargs.get("model")
                    raise Exception(
                        f"Cost tracking failed for model={model}.\nDebug info - {cost_tracking_failure_debug_info}\nAdd custom pricing - https://docs.litellm.ai/docs/proxy/custom_pricing"
                    )
        except Exception as e:
            if accounting is not None:
                accounting.failed = True
            failing_model: Final = kwargs.get("model", "")
            failing_call_type: Final = kwargs.get("call_type", "")
            error_msg: Final = (
                f"Error in tracking cost callback - {e}\n Traceback:{traceback.format_exc()}\n"
                f" Args to _PROXY_track_cost_callback\n model: {failing_model}\n call_type: {failing_call_type}\n"
            )
            failing_litellm_params: Final = kwargs.get("litellm_params") or {}
            verbose_proxy_logger.debug(
                "Cost tracking callback failed for model=%s call_type=%s;"
                " chosen_metadata keys=%s litellm_metadata keys=%s old_metadata keys=%s",
                failing_model,
                failing_call_type,
                _metadata_keys(get_litellm_metadata_from_kwargs(kwargs=kwargs)),
                _metadata_keys(failing_litellm_params.get("litellm_metadata")),
                _metadata_keys(failing_litellm_params.get("metadata")),
            )
            asyncio.create_task(
                proxy_logging_obj.failed_tracking_alert(
                    error_message=error_msg,
                    failing_model=failing_model,
                )
            )

            spend_log_error("Error in tracking cost callback - %s", str(e), exc=e)

    @staticmethod
    async def _enrich_failure_metadata_unless_db_stalled(
        metadata: dict[str, object], original_exception: Exception
    ) -> dict[str, object]:
        if isinstance(original_exception, DBLookupDeadlineExceeded):
            return metadata
        return await ProxyDBLogger._enrich_failure_metadata_with_key_info(metadata=metadata)

    @staticmethod
    async def _enrich_failure_metadata_with_key_info(metadata: dict, resolve_missing_key_identity: bool = True) -> dict:
        """
        Enriches failure spend log metadata by looking up the key object (and team object)
        from cache/DB when key fields are missing.

        This handles two scenarios:
        1. Auth errors (401): UserAPIKeyAuth is created with only api_key set, all other
           fields are null. We look up the full key object to fill in alias, user_id,
           team_id, etc.
        2. Post-auth failures (provider errors, rate limits): key fields are populated
           but team_alias is missing because LiteLLM_VerificationTokenView SQL view
           doesn't include it. We look up the team object to fill in team_alias.

        Scenario 1 reads the key's identity as it stands right now, so it is only correct
        for a log emitted within the request it describes. Callers that log after a delay,
        against an identity captured earlier, pass resolve_missing_key_identity=False and
        keep their own user_id, team_id and org_id.
        """
        api_key_hash: Final = metadata.get("user_api_key")
        if not api_key_hash:
            return metadata

        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        # Step 1: If key fields are missing, look up the full key object
        if resolve_missing_key_identity and metadata.get("user_api_key_alias") is None:
            try:
                key_obj: Final = await get_key_object(
                    hashed_token=api_key_hash,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if metadata.get("user_api_key_alias") is None:
                    metadata["user_api_key_alias"] = key_obj.key_alias
                if metadata.get("user_api_key_user_id") is None:
                    metadata["user_api_key_user_id"] = key_obj.user_id
                if metadata.get("user_api_key_team_id") is None:
                    metadata["user_api_key_team_id"] = key_obj.team_id
                if metadata.get("user_api_key_org_id") is None:
                    metadata["user_api_key_org_id"] = key_obj.org_id
                if metadata.get("user_api_key_project_id") is None:
                    metadata["user_api_key_project_id"] = key_obj.project_id
            except Exception:
                verbose_proxy_logger.debug(
                    "Failed to enrich failure metadata with key info for api_key=%s",
                    api_key_hash,
                )

        # Step 2: If team_id is known but team_alias is missing, look up the team object
        team_id: Final = metadata.get("user_api_key_team_id")
        if team_id and metadata.get("user_api_key_team_alias") is None:
            try:
                team_obj: Final = await get_team_object(
                    team_id=team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if team_obj.team_alias is not None:
                    metadata["user_api_key_team_alias"] = team_obj.team_alias
            except Exception:
                verbose_proxy_logger.debug(
                    "Failed to enrich failure metadata with team_alias for team_id=%s",
                    team_id,
                )
        return metadata

    @staticmethod
    def _should_track_errors_in_db():
        """
        Returns True if errors should be tracked in the database

        By default, errors are tracked in the database

        If users want to disable error tracking, they can set the disable_error_logs flag in the general_settings
        """
        if _general_setting_enabled("disable_error_logs"):
            return False
        return

    @staticmethod
    def _skips_pre_routing_rejections() -> bool:
        """``general_settings.disable_pre_routing_error_logs`` keeps requests the proxy rejected before
        selecting a deployment out of the spend logs, so a client retrying a bad key or an exhausted
        budget cannot flood ``LiteLLM_SpendLogs`` with zero-spend rows."""
        return _general_setting_enabled("disable_pre_routing_error_logs")


def _general_setting_enabled(name: str) -> bool:
    from litellm.proxy.proxy_server import general_settings

    return name in general_settings and general_settings[name] is True


ProxyDBLogger: Final = _ProxyDBLogger


_REQUEST_DATA: Final = TypeAdapter(dict[str, object])


def is_pre_routing_rejection(exception: Exception, request_data: Mapping[str, object]) -> bool:
    """The pipeline sets up the logging object only after auth, body parsing and model resolution succeed, and
    before guardrails or routing run. ``ProxyLogging._handle_logging_proxy_only_error`` marks a failure that
    arrives without one with the ``REJECTED_BEFORE_REQUEST_SETUP`` sentinel, so that request never had a provider
    considered. An unknown model is the one later rejection that still never reached a provider."""
    return request_data.get(REJECTED_BEFORE_REQUEST_SETUP_KEY) is REJECTED_BEFORE_REQUEST_SETUP or isinstance(
        exception, ProxyModelNotFoundError
    )


def _write_spend_metadata_to_kwargs(kwargs: dict, metadata: dict) -> None:
    patch = {k: v for k, v in metadata.items() if (k.startswith("user_api_key") or k == "tags") and v is not None}
    if not patch:
        return

    litellm_params: Final = kwargs.setdefault("litellm_params", {})
    for bucket_name in ("litellm_metadata", "metadata"):
        bucket = litellm_params.get(bucket_name)
        if isinstance(bucket, dict):
            for key, value in patch.items():
                if bucket.get(key) is None:
                    bucket[key] = value


async def run_spend_event(line: bytes) -> None:
    await ProxyDBLogger().run_spend_event(line)


def _is_unbilled_interaction_response(completion_response: object) -> bool:
    from litellm.interactions.background_cost_polling import missing_usage_is_expected
    from litellm.types.interactions import InteractionsAPIResponse

    if not isinstance(completion_response, InteractionsAPIResponse):
        return False
    return completion_response.usage is None and missing_usage_is_expected(completion_response)


def _is_unbilled_in_progress_interaction(completion_response: object) -> bool:
    from litellm.interactions.background_cost_polling import is_pollable_background_interaction
    from litellm.types.interactions import InteractionsAPIResponse

    if not isinstance(completion_response, InteractionsAPIResponse):
        return False
    return completion_response.usage is None and is_pollable_background_interaction(completion_response)


def _should_track_cost_callback(
    user_api_key: str | None,
    user_id: str | None,
    team_id: str | None,
    end_user_id: str | None,
    call_type: str | None = None,
    agent_id: str | None = None,
) -> bool:
    """
    Determine if the cost callback should be tracked based on the kwargs

    Pass-through endpoints can be configured with ``auth=false``, which leaves
    the request with no key/user/team/end-user to attribute spend to. Those
    requests still forward real provider traffic that operators expect to see
    in request/usage logs, so they are tracked even when unauthenticated.
    The same reasoning applies to a completed managed batch's cost event
    (see _UNATTRIBUTED_TRACKABLE_CALL_TYPES).
    """

    # don't run track cost callback if user opted into disabling spend
    if ProxyUpdateSpend.disable_spend_updates() is True:
        return False

    if (
        agent_id is not None
        or user_api_key is not None
        or user_id is not None
        or team_id is not None
        or end_user_id is not None
    ):
        return True
    return call_type in _UNATTRIBUTED_TRACKABLE_CALL_TYPES


def _metadata_keys(metadata: object) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    return tuple(sorted(str(key) for key in metadata))


def _get_budget_reservation_from_metadata(metadata: dict) -> dict | None:
    return budget_reservation_from_metadata(metadata)


def _get_request_tags_for_cost_tracking(
    sl_object: StandardLoggingPayload | None,
    metadata: dict,
) -> list[str] | None:
    if sl_object is not None:
        request_tags: Final = sl_object.get("request_tags", None)
        if isinstance(request_tags, list):
            return request_tags

    metadata_tags: Final = metadata.get("tags", None)
    if isinstance(metadata_tags, list):
        return metadata_tags

    return None


class _IncrementSpendCounters(Protocol):
    """The ``increment_spend_counters`` coroutine :func:`_update_database_and_spend_counters` awaits."""

    async def __call__(
        self,
        token: str | None,
        team_id: str | None,
        user_id: str | None,
        response_cost: float | None,
        org_id: str | None = None,
        budget_reservation: dict[str, object] | None = None,
        end_user_id: str | None = None,
        tags: list[str] | None = None,
        request_started_at: datetime | None = None,
        model_access_groups: Sequence[str] | None = None,
    ) -> None: ...


async def _update_database_and_spend_counters(
    proxy_logging_obj: "ProxyLogging",
    increment_spend_counters: _IncrementSpendCounters,
    user_api_key: str | None,
    user_id: str | None,
    end_user_id: str | None,
    team_id: str | None,
    org_id: str | None,
    kwargs: dict,
    completion_response: object,
    start_time: datetime | None,
    end_time: datetime | None,
    response_cost: float,
    budget_reservation: dict | None,
    request_tags: list[str] | None = None,
    model_access_groups: Sequence[str] | None = None,
    project_id: str | None = None,
) -> bool:
    """The reservation is reconciled before the spend is persisted, from its own read. One spend counter batch then
    spans the database write and the counter update, so the post-call counters are read with a single MGET after the
    write and their increments leave in a single pipeline."""
    from litellm.proxy.proxy_server import spend_counter_cache
    from litellm.proxy.spend_tracking.budget_reservation import get_reserved_counter_keys

    if budget_reservation is not None:
        await _reconcile_budget_reservation_before_db_update(
            budget_reservation=budget_reservation, response_cost=response_cost
        )
    counter_keys: Final = frozenset(
        get_reserved_counter_keys(budget_reservation=budget_reservation)
    ) | post_call_counter_keys(
        token=user_api_key,
        team_id=team_id,
        user_id=user_id,
        org_id=org_id,
        end_user_id=end_user_id,
        tags=request_tags,
        model_access_groups=model_access_groups,
        project_id=project_id,
    )
    with spend_counter_batch_scope(spend_counter_cache.redis_cache, counter_keys=counter_keys):
        return await _update_database_and_spend_counters_in_batch(
            proxy_logging_obj=proxy_logging_obj,
            increment_spend_counters=increment_spend_counters,
            user_api_key=user_api_key,
            user_id=user_id,
            end_user_id=end_user_id,
            team_id=team_id,
            org_id=org_id,
            kwargs=kwargs,
            completion_response=completion_response,
            start_time=start_time,
            end_time=end_time,
            response_cost=response_cost,
            budget_reservation=budget_reservation,
            request_tags=request_tags,
            model_access_groups=model_access_groups,
            project_id=project_id,
        )


async def _update_database_and_spend_counters_in_batch(
    proxy_logging_obj: "ProxyLogging",
    increment_spend_counters: _IncrementSpendCounters,
    user_api_key: str | None,
    user_id: str | None,
    end_user_id: str | None,
    team_id: str | None,
    org_id: str | None,
    kwargs: dict,
    completion_response: object,
    start_time: datetime | None,
    end_time: datetime | None,
    response_cost: float,
    budget_reservation: dict | None,
    request_tags: list[str] | None,
    model_access_groups: Sequence[str] | None,
    project_id: str | None,
) -> bool:
    try:
        writer: Final = proxy_logging_obj.db_spend_update_writer
        accounting: Final = (
            _batch_accounting_for(_BATCH_CALLBACK_FIELDS.validate_python(kwargs).get("litellm_call_id"))
            if _BATCH_ACCOUNTING.get() is not None
            else None
        )
        if accounting is not None:
            accounting.write_started = True
        charged: Final = await writer.update_database(
            token=user_api_key,
            response_cost=response_cost,
            user_id=user_id,
            end_user_id=end_user_id,
            team_id=team_id,
            kwargs=kwargs,
            completion_response=completion_response,
            start_time=start_time,
            end_time=end_time,
            org_id=org_id,
            project_id=project_id,
        )
    except Exception:
        if budget_reservation is not None:
            try:
                await _release_budget_reservation(budget_reservation=budget_reservation)
            except Exception:
                verbose_proxy_logger.exception("Failed to release budget reservation after database update failed")
                try:
                    await _invalidate_budget_reservation_counters(budget_reservation=budget_reservation)
                except Exception:
                    verbose_proxy_logger.exception(
                        "Failed to invalidate budget reservation counters after release failed"
                    )
        raise
    if not charged:
        await _release_budget_reservation(budget_reservation=budget_reservation)
        return False

    try:
        await increment_spend_counters(
            token=user_api_key,
            team_id=team_id,
            user_id=user_id,
            response_cost=response_cost,
            org_id=org_id,
            budget_reservation=budget_reservation,
            end_user_id=end_user_id,
            tags=request_tags,
            request_started_at=start_time,
            model_access_groups=model_access_groups,
            project_id=project_id,
        )
    except Exception:
        if budget_reservation is not None:
            try:
                await _invalidate_budget_reservation_counters(budget_reservation=budget_reservation)
            except Exception:
                verbose_proxy_logger.exception(
                    "Failed to invalidate budget reservation counters after spend counter update failed"
                )
            finally:
                budget_reservation["finalized"] = True
        raise
    return True


async def _reconcile_budget_reservation_before_db_update(
    budget_reservation: dict,  # mutable-ok: reconcile_budget_reservation stamps applied_adjustment on the caller's shared reservation dict
    response_cost: float,
) -> None:
    """Reseeds the reserved counters that were flushed since reservation; the adjustments themselves are written by ``increment_spend_counters`` in the same pipeline as its increments, or by
    the release / invalidation that runs when the spend write fails."""
    from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation

    try:
        _ = await reconcile_budget_reservation(
            budget_reservation=budget_reservation, actual_cost=response_cost, finalize=False, apply_consistent=False
        )
    except Exception:  # noqa: BLE001  # a failed reconcile must not block the spend write; the counters are dropped instead
        verbose_proxy_logger.warning(
            "Failed to reconcile budget reservation before persisting spend; invalidating reserved counters"
        )
        try:
            await _invalidate_budget_reservation_counters(budget_reservation=budget_reservation)
        except Exception:  # noqa: BLE001  # nothing left to try; the finalized stamp below keeps it from being reprocessed
            verbose_proxy_logger.exception(
                "Failed to invalidate budget reservation counters after pre-persist reconcile failed"
            )
        finally:
            budget_reservation["finalized"] = True  # rebind-ok: stamps the caller's shared dict for the counter update


async def _release_budget_reservation(budget_reservation: dict | None) -> None:
    if budget_reservation is None:
        return

    from litellm.proxy.spend_tracking.budget_reservation import (
        release_budget_reservation,
    )

    await release_budget_reservation(
        budget_reservation=budget_reservation,
    )


async def _invalidate_budget_reservation_counters(
    budget_reservation: dict | None,
) -> None:
    if budget_reservation is None:
        return

    from litellm.proxy.spend_tracking.budget_reservation import (
        invalidate_budget_reservation_counters,
    )

    await invalidate_budget_reservation_counters(
        budget_reservation=budget_reservation,
    )
