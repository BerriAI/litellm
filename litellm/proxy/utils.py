import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import os
import smtplib
import ssl
import sys
import threading
import time
import traceback
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Optional, Protocol, TypeVar, Union, cast, overload

from typing_extensions import ReadOnly, TypedDict

from litellm import _custom_logger_compatible_callbacks_literal
from litellm.constants import (
    DEFAULT_MODEL_CREATED_AT_TIME,
    LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL,
    MAX_TEAM_LIST_LIMIT,
    SPEND_LOG_QUEUE_MAX_BYTES,
    SPEND_LOG_WRITE_BATCH_MAX_BYTES,
    SPEND_LOG_WRITE_BATCH_MAX_ROWS,
)
from litellm.proxy._types import (
    CommonProxyErrors,
    ProxyErrorTypes,
    ProxyException,
    SpendLogsMetadata,
    SpendLogsPayload,
)
from litellm.proxy.spend_tracking.spend_log_error_logger import spend_log_error
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.model_listing import ModelInfoResponse
from litellm.types.utils import CallTypes, CallTypesLiteral, ModelInfo, Usage

try:
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import (
        BaseEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
        ResendEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email import (
        SendGridEmailLogger,
    )
    from litellm_enterprise.enterprise_callbacks.send_emails.smtp_email import (
        SMTPEmailLogger,
    )
except ImportError:
    BaseEmailLogger = None
    SendGridEmailLogger = None
    SMTPEmailLogger = None
    ResendEmailLogger = None

try:
    import backoff
except ImportError:
    raise ImportError("backoff is not installed. Please install it via 'pip install backoff'")

from fastapi import HTTPException, status

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
    Router,
)
from litellm._logging import _redact_string, verbose_proxy_logger
from litellm._service_logger import ServiceLogging, ServiceTypes
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.dual_cache import LimitedSizeOrderedDict
from litellm.exceptions import RejectedRequestError, SensitiveDataRouteException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.utils import _add_langfuse_trace_id_to_alert
from litellm.litellm_core_utils.core_helpers import coerce_token_limit, is_expected_client_error
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms import load_guardrail_translation_mappings
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    LiteLLM_VerificationTokenView,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.config_sync_pubsub import publish_config_param_change
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.create_views import (
    create_missing_views,
    create_view_tolerating_race,
    should_create_missing_views,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.exception_handler import (
    PrismaDBExceptionHandler,
    call_with_db_reconnect_retry,
)
from litellm.proxy.db.log_db_metrics import log_db_metrics
from litellm.proxy.db.prisma_client import (
    PrismaWrapper,
    parse_iam_endpoint_from_url,
)
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.db.spend_log_batching import (
    spend_log_queue_within_budget,
    spend_log_row_bytes,
    spend_log_write_batches,
)
from litellm.proxy.db.token_auth import (
    DatabaseTokenAuth,
    mint_database_token,
    resolve_database_token_auth,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.proxy.hooks import PROXY_HOOKS, get_proxy_hook
from litellm.proxy.hooks.cache_control_check import _PROXY_CacheControlCheck
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.hooks.sensitive_data_routing import (
    _PROXY_SensitiveDataRoutingHandler,
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.management_helpers.key_settings_audit import with_settings_updated_at
from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.table_repositories import (
    EndUserRepository,
    HealthCheckRepository,
    SpendLogsRepository,
    UserNotificationsRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.slack_alerting import DEFAULT_ALERT_TYPES
from litellm.types.mcp import (
    MCPDuringCallResponseObject,
    MCPPreCallRequestObject,
    MCPPreCallResponseObject,
)
from litellm.types.proxy.policy_engine.pipeline_types import PipelineExecutionResult
from litellm.types.utils import LLMResponseTypes, LoggedLiteLLMParams

if TYPE_CHECKING:
    from mcp.types import CallToolResult
    from opentelemetry.trace import Span as _Span
    from prisma import models as prisma_models
    from prisma.actions import LiteLLM_DeprecatedVerificationTokenActions
    from prisma.client import TransactionManager
    from prisma.models import LiteLLM_DeprecatedVerificationToken
    from prisma.types import HttpConfig

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.models.team import LiteLLM_TeamTableCachedObj
    from litellm.proxy.db.autorouter_session_rollup import AutoRouterTurnTransaction
    from litellm.proxy.db.spend_log_tool_index import ToolUsageTransaction
    from litellm.repositories.prisma_protocols import TableActions
    from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline

    Span = _Span | object
else:
    Span = Any

_T: Final = TypeVar("_T")


class _ViewCountRow(TypedDict):
    view_count: ReadOnly[int]
    view_names: ReadOnly[Sequence[str] | None]


class _RelTuplesRow(TypedDict):
    reltuples: ReadOnly[int]


class _EndUserBatchTable(Protocol):
    def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class _EndUserSpendBatch(Protocol):
    @property
    def litellm_endusertable(self) -> _EndUserBatchTable: ...


unified_guardrail: Final = UnifiedLLMGuardrails()

NON_OPENAI_STREAM_GUARDRAIL_TRANSLATION_CALL_TYPES: "frozenset[CallTypes]" = frozenset({CallTypes.anthropic_messages})


def print_verbose(print_statement: object):
    """
    Prints the given `print_statement` to the console if `litellm.set_verbose` is True.
    Also logs the `print_statement` at the debug level using `verbose_proxy_logger`.

    :param print_statement: The statement to be printed and logged.
    :type print_statement: Any
    """
    import traceback

    verbose_proxy_logger.debug("%s\n%s", print_statement, traceback.format_exc())
    if litellm.set_verbose:
        print(f"LiteLLM Proxy: {_redact_string(str(print_statement))}")  # noqa: T201


def _get_email_logger_class():
    """
    Determine which email logger class to use based on environment variables.
    Priority: SendGrid > Resend > SMTP > BaseEmailLogger (fallback)

    Returns:
        The email logger class to use, or None if BaseEmailLogger is not available
    """
    if BaseEmailLogger is None:
        return None

    # Check for SendGrid API key
    if SendGridEmailLogger is not None and os.getenv("SENDGRID_API_KEY"):
        return SendGridEmailLogger

    # Check for Resend API key
    if ResendEmailLogger is not None and os.getenv("RESEND_API_KEY"):
        return ResendEmailLogger

    # Check for SMTP configuration
    if SMTPEmailLogger is not None and os.getenv("SMTP_HOST"):
        return SMTPEmailLogger

    # Fallback to BaseEmailLogger (though it won't actually send emails)
    return BaseEmailLogger


class InternalUsageCache:
    def __init__(self, dual_cache: DualCache):
        self.dual_cache: DualCache = dual_cache

    async def async_get_cache(
        self,
        key: str,
        litellm_parent_otel_span: Span | None,
        local_only: bool = False,
        **kwargs: object,
    ) -> Any:
        return await self.dual_cache.async_get_cache(
            key=key,
            local_only=local_only,
            parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_set_cache(
        self,
        key: str,
        value: object,
        litellm_parent_otel_span: Span | None,
        local_only: bool = False,
        **kwargs: object,
    ) -> None:
        return await self.dual_cache.async_set_cache(
            key=key,
            value=value,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_batch_set_cache(
        self,
        cache_list: list[tuple[str, object]],
        litellm_parent_otel_span: Span | None,
        local_only: bool = False,
        **kwargs: object,
    ) -> None:
        return await self.dual_cache.async_set_cache_pipeline(
            cache_list=cache_list,
            local_only=local_only,
            litellm_parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    async def async_batch_get_cache(
        self,
        keys: Sequence[str | None],
        parent_otel_span: Span | None = None,
        local_only: bool = False,
    ):
        return await self.dual_cache.async_batch_get_cache(
            keys=list(keys),
            parent_otel_span=parent_otel_span,
            local_only=local_only,
        )

    async def async_increment_cache(
        self,
        key: str,
        value: float,
        litellm_parent_otel_span: Span | None,
        local_only: bool = False,
        **kwargs,
    ):
        return await self.dual_cache.async_increment_cache(
            key=key,
            value=value,
            local_only=local_only,
            parent_otel_span=litellm_parent_otel_span,
            **kwargs,
        )

    def set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,
    ) -> None:
        return self.dual_cache.set_cache(
            key=key,
            value=value,
            local_only=local_only,
            **kwargs,
        )

    def get_cache(
        self,
        key: str,
        local_only: bool = False,
        **kwargs: object,
    ) -> Any:
        return self.dual_cache.get_cache(
            key=key,
            local_only=local_only,
            **kwargs,
        )


### LOGGING ###

# Cache for inspect.signature checks — avoids repeated introspection per request
_CALLBACK_ACCEPTS_CALL_INFO: Final[dict[int, bool]] = {}


def _accepts_litellm_call_info(cb: CustomLogger) -> bool:
    key: Final = id(type(cb))
    if key not in _CALLBACK_ACCEPTS_CALL_INFO:
        sig: Final = inspect.signature(cb.async_post_call_response_headers_hook)
        _CALLBACK_ACCEPTS_CALL_INFO[key] = "litellm_call_info" in sig.parameters
    return _CALLBACK_ACCEPTS_CALL_INFO[key]


def _enrich_http_exception_with_guardrail_context(exc: BaseException, callback: object) -> None:
    """
    If `exc` is an HTTPException with a dict `detail`, mutate it in place to
    add `guardrail_name` and `guardrail_mode` taken from the callback instance.

    Uses setdefault so guardrails that already populate these fields explicitly
    win over the inferred defaults. No-op for non-HTTPException, non-dict-detail,
    or callbacks without `guardrail_name`. Never raises.
    """
    if not isinstance(exc, HTTPException):
        return
    detail: Final = getattr(exc, "detail", None)
    if not isinstance(detail, dict):
        return
    guardrail_name: Final[object] = getattr(callback, "guardrail_name", None)
    if guardrail_name:
        detail.setdefault("guardrail_name", guardrail_name)
    event_hook: Final[object] = getattr(callback, "event_hook", None)
    if event_hook:
        detail.setdefault("guardrail_mode", event_hook)


def _exception_changes_request_flow(exc: BaseException) -> bool:
    """
    True for guardrail exceptions the proxy turns into an alternate request flow
    (a reroute or a passthrough response) rather than a block. A pipeline step
    configured to block must honor that block, so these are surfaced as the
    generic pipeline block instead of being re-raised verbatim.
    """
    return isinstance(exc, (SensitiveDataRouteException, ModifyResponseException))


def _policy_state_metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    """
    Return the metadata bucket the policy engine wrote its pipeline state into.

    The route decides the bucket (``litellm_metadata`` for ``/v1/messages``,
    responses, batches, files and bedrock, ``metadata`` everywhere else), and both
    buckets can be present at once because callers send their own provider-facing
    ``metadata`` (Claude Code sends ``metadata.user_id``) or their own
    ``litellm_metadata``. Pipeline slots are stripped from caller input before the
    policy engine runs, so whichever bucket carries them is the proxy's own write.
    """
    return next(
        (
            bucket
            for bucket in (data.get("metadata"), data.get("litellm_metadata"))
            if isinstance(bucket, dict)
            and ("_guardrail_pipelines" in bucket or "_pipeline_managed_guardrails" in bucket)
        ),
        {},
    )


def _policy_pipelines(data: Mapping[str, object]) -> tuple[tuple[str, "GuardrailPipeline"], ...]:
    pipelines: Final = _policy_state_metadata(data).get("_guardrail_pipelines")
    return (
        tuple(cast("Sequence[tuple[str, GuardrailPipeline]]", pipelines))  # cast-ok: the policy engine wrote the slot
        if pipelines
        else ()
    )


def _pipeline_managed_guardrail_names(data: Mapping[str, object]) -> frozenset[str]:
    managed: Final = _policy_state_metadata(data).get("_pipeline_managed_guardrails")
    return (
        frozenset(cast("Collection[str]", managed))  # cast-ok: the policy engine wrote these guardrail names
        if managed
        else frozenset()
    )


def _prompt_block_text(block: object) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    block_text: Final = block.get("text")
    return block_text if isinstance(block_text, str) else ""


def _system_prompt_text(system_input: object) -> str:
    if isinstance(system_input, str):
        return system_input
    if not isinstance(system_input, list):
        return ""
    return "".join(_prompt_block_text(block) for block in system_input)


def _count_request_input_tokens(model: str, request_input: object, system_input: object) -> int:
    system_text: Final = _system_prompt_text(system_input)
    system_tokens: Final = litellm.token_counter(model=model, text=system_text) if system_text else 0
    if isinstance(request_input, str):
        return system_tokens + litellm.token_counter(model=model, text=request_input)
    if not isinstance(request_input, list) or not request_input:
        return system_tokens
    text_entries: Final = tuple(entry for entry in request_input if isinstance(entry, str))
    if len(text_entries) == len(request_input):
        return system_tokens + litellm.token_counter(model=model, text="".join(text_entries))
    return system_tokens + litellm.token_counter(
        model=model, messages=request_input, use_default_image_token_count=True
    )


def _estimate_dispatched_failure_usage(model: str, request_input: object, system_input: object) -> Usage | None:
    """A request that failed after dispatch consumed provider-billed input
    tokens, but no provider usage ever came back. Estimate the input side with
    the same tokenizer fallback interrupted streams use, so the spend log's
    failure row records what was sent instead of zero."""
    try:
        input_tokens: Final = _count_request_input_tokens(
            model=model, request_input=request_input, system_input=system_input
        )
    except Exception:
        return None
    if input_tokens <= 0:
        return None
    return Usage(prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens)


_INPUT_ESTIMABLE_CALL_TYPES: Final = frozenset(
    call_type.value
    for call_type in (
        CallTypes.completion,
        CallTypes.acompletion,
        CallTypes.text_completion,
        CallTypes.atext_completion,
        CallTypes.anthropic_messages,
        CallTypes.aanthropic_messages,
        CallTypes.responses,
        CallTypes.aresponses,
        CallTypes.embedding,
        CallTypes.aembedding,
        CallTypes.moderation,
        CallTypes.amoderation,
        CallTypes.image_generation,
        CallTypes.aimage_generation,
        CallTypes.speech,
        CallTypes.aspeech,
        CallTypes.rerank,
        CallTypes.arerank,
        CallTypes.generate_content,
        CallTypes.agenerate_content,
        CallTypes.generate_content_stream,
        CallTypes.agenerate_content_stream,
    )
)


def _failure_usage_to_lift(
    model_call_details: Mapping[str, object],
    request_body: Mapping[str, object],
    dispatched: bool,
) -> tuple[object, object] | None:
    """A stream that broke mid-flight still billed the provider for the chunks
    already delivered; the streaming handler stashes that recovered usage and
    cost in model_call_details, so prefer it. Otherwise a request that was
    dispatched to a provider and failed without upstream usage gets an
    estimated input-side Usage with zero cost. The raw request body backfills
    the system prompt when the SDK bridges an endpoint (e.g. /v1/messages on a
    chat-completions provider) without filling optional_params. Returns the
    (combined_usage_object, response_cost) pair to lift, or None."""
    recovered_usage: Final = model_call_details.get("combined_usage_object")
    if recovered_usage is not None:
        return recovered_usage, model_call_details.get("response_cost")
    if not dispatched or model_call_details.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL):
        return None
    if str(model_call_details.get("call_type")) not in _INPUT_ESTIMABLE_CALL_TYPES:
        return None
    optional_params: Final = model_call_details.get("optional_params")
    dispatched_system: Final = (
        (optional_params.get("system") or optional_params.get("instructions"))
        if isinstance(optional_params, dict)
        else None
    )
    system_input: Final = dispatched_system or request_body.get("system") or request_body.get("instructions")
    estimated_usage: Final = _estimate_dispatched_failure_usage(
        model=str(model_call_details.get("model") or ""),
        request_input=model_call_details.get("messages"),
        system_input=system_input,
    )
    if estimated_usage is None:
        return None
    return estimated_usage, 0.0


_EMPTY_LIFT: Final = MappingProxyType({})


def _failure_fields_to_lift(request_data: Mapping[str, object]) -> Mapping[str, object]:
    """Failure-path callbacks run after ``litellm_logging_obj`` is popped from
    request_data (it is not serialisable), so the caller merges these fields
    onto request_data first: the first-handoff instant for preprocessing
    latency, recovered or estimated usage for token counts, and the standard
    logging object for deployment attribution on failed-request spend logs."""
    _logging_obj: Final = request_data.get("litellm_logging_obj")
    if _logging_obj is None:
        return _EMPTY_LIFT
    _model_call_details: Final = getattr(_logging_obj, "model_call_details", {})
    _first_handoff: Final = _model_call_details.get("first_api_call_start_time")
    _usage_to_lift: Final = _failure_usage_to_lift(
        model_call_details=_model_call_details,
        request_body=request_data,
        dispatched=_first_handoff is not None,
    )
    _entries: Final = (
        ("first_api_call_start_time", _first_handoff),
        ("combined_usage_object", None if _usage_to_lift is None else _usage_to_lift[0]),
        ("response_cost", None if _usage_to_lift is None else (_usage_to_lift[1] or 0.0)),
        ("standard_logging_object", _model_call_details.get("standard_logging_object")),
    )
    return MappingProxyType({key: value for key, value in _entries if value is not None})


@dataclass(frozen=True)
class _CallbackCapabilities:
    """Cached per-hook capability flags derived from ``litellm.callbacks``.

    Recomputing this per request walked the callback list and resolved every
    string entry via ``get_custom_logger_compatible_class`` — a measurable
    chunk of overhead on streaming and non-streaming chat completions.
    """

    has_post_call_response_headers: bool = False
    has_iterator_override: bool = False
    has_streaming_chunk_override: bool = False
    has_guardrail: bool = False
    has_pre_call_override: bool = False
    has_content_enforcer: bool = False
    # Tuple[(resolved_callback, "override" | "apply_guardrail"), ...]
    # Ordered the same as ``litellm.callbacks``; used to build the streaming
    # iterator chain without re-scanning per request.
    iterator_overrides: tuple[tuple[Any, str], ...] = field(default_factory=tuple)
    # Resolved CustomLogger callbacks in original order. Pre-resolving once
    # avoids the per-request ``get_custom_logger_compatible_class`` walk for
    # every string entry in ``litellm.callbacks``.
    resolved_callbacks: tuple[object, ...] = field(default_factory=tuple)


class ProxyLogging:
    """
    Logging/Custom Handlers for proxy.

    Implemented mainly to:
    - log successful/failed db read/writes
    - support the max parallel request integration
    """

    def __init__(
        self,
        user_api_key_cache: UserApiKeyCache,
        premium_user: bool = False,
    ):
        ## INITIALIZE  LITELLM CALLBACKS ##
        self.call_details: dict = {}
        self.call_details["user_api_key_cache"] = user_api_key_cache
        self.internal_usage_cache: InternalUsageCache = InternalUsageCache(
            dual_cache=DualCache(default_in_memory_ttl=1)  # ping redis cache every 1s
        )
        self.max_parallel_request_limiter = _PROXY_MaxParallelRequestsHandler(self.internal_usage_cache)
        self.max_budget_limiter = _PROXY_MaxBudgetLimiter()
        self.cache_control_check = _PROXY_CacheControlCheck()
        self.alerting: list | None = None
        self.alerting_threshold: float = 300  # default to 5 min. threshold
        self.alert_types: list[AlertType] = DEFAULT_ALERT_TYPES
        self.alert_to_webhook_url: dict | None = None
        self.slack_alerting_instance: SlackAlerting = SlackAlerting(
            alerting_threshold=self.alerting_threshold,
            alerting=self.alerting,
            internal_usage_cache=self.internal_usage_cache.dual_cache,
        )
        self.email_logging_instance: Any | None = None
        if BaseEmailLogger is not None:
            email_logger_class: Final = _get_email_logger_class()
            if email_logger_class is not None:
                # All email logger classes now accept internal_usage_cache
                self.email_logging_instance = email_logger_class(
                    internal_usage_cache=self.internal_usage_cache.dual_cache,
                )
        self.premium_user = premium_user
        self.service_logging_obj = ServiceLogging()
        self.db_spend_update_writer = DBSpendUpdateWriter()
        self.proxy_hook_mapping: dict[str, CustomLogger] = {}

        # Guard flags to prevent duplicate background tasks
        self.daily_report_started: bool = False
        self.hanging_requests_check_started: bool = False
        self.deprecation_check_started: bool = False

    def startup_event(
        self,
        llm_router: Router | None,
        redis_usage_cache: RedisCache | None,
    ):
        """Initialize logging and alerting on proxy startup"""
        ## UPDATE SLACK ALERTING ##
        self.slack_alerting_instance.update_values(llm_router=llm_router)

        ## UPDATE INTERNAL USAGE CACHE ##
        self.update_values(
            redis_cache=redis_usage_cache
        )  # used by parallel request limiter for rate limiting keys across instances

        self._init_litellm_callbacks(
            llm_router=llm_router
        )  # INITIALIZE LITELLM CALLBACKS ON SERVER STARTUP <- do this to catch any logging errors on startup, not when calls are being made

        if (
            self.slack_alerting_instance is not None
            and "daily_reports" in self.slack_alerting_instance.alert_types
            and not self.daily_report_started
        ):
            asyncio.create_task(
                self.slack_alerting_instance._run_scheduled_daily_report(
                    llm_router=llm_router,
                    pod_lock_manager=self.db_spend_update_writer.pod_lock_manager,
                )
            )  # RUN DAILY REPORT (if scheduled)
            self.daily_report_started = True

        if (
            self.slack_alerting_instance is not None
            and AlertType.llm_requests_hanging in self.slack_alerting_instance.alert_types
            and not self.hanging_requests_check_started
        ):
            asyncio.create_task(
                self.slack_alerting_instance.hanging_request_check.check_for_hanging_requests()
            )  # RUN HANGING REQUEST CHECK (if user wants to alert on hanging requests)
            self.hanging_requests_check_started = True

        self._ensure_deprecation_check_scheduled()

    def _ensure_deprecation_check_scheduled(self) -> None:
        """Alerting can be configured at startup or by a later config reload, so schedule from either path"""
        if self.alerting is None or self.deprecation_check_started:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        asyncio.create_task(
            self.slack_alerting_instance.run_scheduled_deprecation_check(
                pod_lock_manager=self.db_spend_update_writer.pod_lock_manager
            )
        )
        self.deprecation_check_started = True

    def update_values(
        self,
        alerting: list | None = None,
        alerting_threshold: float | None = None,
        redis_cache: RedisCache | None = None,
        alert_types: list[AlertType] | None = None,
        alerting_args: dict | None = None,
        alert_to_webhook_url: dict | None = None,
        alert_type_config: dict | None = None,
    ):
        updated_slack_alerting: bool = False
        if alerting is not None:
            self.alerting = alerting
            updated_slack_alerting = True
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold
            updated_slack_alerting = True
        if alert_types is not None:
            self.alert_types = alert_types
            updated_slack_alerting = True
        if alert_to_webhook_url is not None:
            self.alert_to_webhook_url = alert_to_webhook_url
            updated_slack_alerting = True
        if alert_type_config is not None:
            updated_slack_alerting = True

        if updated_slack_alerting is True:
            self._ensure_deprecation_check_scheduled()
            self.slack_alerting_instance.update_values(
                alerting=self.alerting,
                alerting_threshold=self.alerting_threshold,
                alert_types=self.alert_types,
                alerting_args=alerting_args,
                alert_to_webhook_url=self.alert_to_webhook_url,
                alert_type_config=alert_type_config,
            )

            if self.alerting is not None and "slack" in self.alerting:
                # NOTE: ENSURE we only add callbacks when alerting is on
                # We should NOT add callbacks when alerting is off
                if (
                    "daily_reports" in self.alert_types
                    or "outage_alerts" in self.alert_types
                    or "region_outage_alerts" in self.alert_types
                ):
                    litellm.logging_callback_manager.add_litellm_callback(self.slack_alerting_instance)
                litellm.logging_callback_manager.add_litellm_success_callback(
                    self.slack_alerting_instance.response_taking_too_long_callback
                )

        if redis_cache is not None:
            self.internal_usage_cache.dual_cache.redis_cache = redis_cache
            self.db_spend_update_writer.redis_update_buffer.redis_cache = redis_cache
            self.db_spend_update_writer.pod_lock_manager.redis_cache = redis_cache

    def _add_proxy_hooks(self, llm_router: Router | None = None):
        """
        Add proxy hooks to litellm.callbacks
        """
        from litellm.proxy.proxy_server import prisma_client

        for hook in PROXY_HOOKS:
            proxy_hook = get_proxy_hook(hook)
            expected_args = inspect.getfullargspec(proxy_hook).args
            if "prisma_client" in expected_args and prisma_client is None:
                verbose_proxy_logger.debug(
                    "Skipping proxy hook %s: it requires a database and no prisma client is configured", hook
                )
                continue
            passed_in_args: dict[str, Any] = {}
            if "internal_usage_cache" in expected_args:
                passed_in_args["internal_usage_cache"] = self.internal_usage_cache
            if "prisma_client" in expected_args:
                passed_in_args["prisma_client"] = prisma_client
            proxy_hook_obj = cast(CustomLogger, proxy_hook(**passed_in_args))
            litellm.logging_callback_manager.add_litellm_callback(proxy_hook_obj)

            self.proxy_hook_mapping[hook] = proxy_hook_obj

    def get_proxy_hook(self, hook: str) -> CustomLogger | None:
        """
        Get a proxy hook from the proxy_hook_mapping
        """
        return self.proxy_hook_mapping.get(hook)

    def _init_litellm_callbacks(self, llm_router: Router | None = None):
        self._add_proxy_hooks(llm_router)
        litellm.logging_callback_manager.add_litellm_callback(self.service_logging_obj)

        # Track string callbacks and their initialized instances so we can
        # replace them in-place, preventing duplicates (string + instance) in
        # litellm.callbacks which caused double-counting of metrics.
        string_callbacks_to_replace: Final[dict[int, CustomLogger]] = {}

        for idx, callback in enumerate(litellm.callbacks):
            if isinstance(callback, str):
                initialized_callback = litellm.litellm_core_utils.litellm_logging._init_custom_logger_compatible_class(
                    cast(_custom_logger_compatible_callbacks_literal, callback),
                    internal_usage_cache=self.internal_usage_cache.dual_cache,
                    llm_router=llm_router,
                )

                if initialized_callback is not None:
                    string_callbacks_to_replace[idx] = initialized_callback

        # Replace string entries in litellm.callbacks with initialized instances
        for idx, initialized_callback in string_callbacks_to_replace.items():
            litellm.callbacks[idx] = initialized_callback

        # Fan ``litellm.callbacks`` (the "all events" registry) out into the
        # success/failure event lists eagerly, at startup. ``completion()`` does
        # this lazily in ``function_setup`` on the first call, but request paths
        # that build their own logging object and never run ``function_setup`` —
        # notably pass-through endpoints — read ``litellm._async_success_callback``
        # directly. Without this, a config-registered logger (e.g. ``otel``) is
        # invisible to pass-through traffic until some other request warms the
        # global lists. The manager dedupes, so this is idempotent with
        # ``function_setup``.
        for callback in litellm.callbacks:
            if isinstance(callback, CustomLogger):
                litellm.logging_callback_manager.add_litellm_success_callback(callback)
                litellm.logging_callback_manager.add_litellm_failure_callback(callback)
                litellm.logging_callback_manager.add_litellm_async_success_callback(callback)
                litellm.logging_callback_manager.add_litellm_async_failure_callback(callback)

    async def update_request_status(self, litellm_call_id: str, status: Literal["success", "fail"]):
        # only use this if slack alerting is being used
        if self.alerting is None:
            return

        # current alerting threshold
        alerting_threshold: float = self.alerting_threshold

        # add a 100 second buffer to the alerting threshold
        # ensures we don't send errant hanging request slack alerts
        alerting_threshold += 100

        await self.internal_usage_cache.async_set_cache(
            key=f"request_status:{litellm_call_id}",
            value=status,
            local_only=True,
            ttl=alerting_threshold,
            litellm_parent_otel_span=None,
        )

    def _convert_user_api_key_auth_to_dict(self, user_api_key_auth_obj):
        """
        Helper function to convert UserAPIKeyAuth object to dictionary.
        Handles both Pydantic models and regular objects.
        """
        if user_api_key_auth_obj is not None:
            if hasattr(user_api_key_auth_obj, "model_dump"):
                # If it's a Pydantic model, convert to dict
                return user_api_key_auth_obj.model_dump()
            elif hasattr(user_api_key_auth_obj, "__dict__"):
                # If it's a regular object, convert to dict
                return user_api_key_auth_obj.__dict__
        return {}

    def _convert_mcp_to_llm_format(self, request_obj, kwargs: dict) -> dict:
        """
        Convert MCP tool call to LLM message format for existing guardrail validation.
        """
        from litellm.types.llms.openai import ChatCompletionUserMessage

        # Create a synthetic message that represents the tool call
        tool_call_content: Final = f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"

        synthetic_message: Final = ChatCompletionUserMessage(role="user", content=tool_call_content)

        # Create synthetic LLM data that guardrails can process
        synthetic_data: Final = {
            "messages": [synthetic_message],
            "model": kwargs.get("model", "mcp-tool-call"),
            "user_api_key_user_id": kwargs.get("user_api_key_user_id"),
            "user_api_key_team_id": kwargs.get("user_api_key_team_id"),
            "user_api_key_end_user_id": kwargs.get("user_api_key_end_user_id"),
            "user_api_key_hash": kwargs.get("user_api_key_hash"),
            "user_api_key_request_route": kwargs.get("user_api_key_request_route"),
            "mcp_tool_name": request_obj.tool_name,  # Keep original for reference
            "mcp_arguments": request_obj.arguments,  # Keep original for reference
            # Surface the per-MCP-server rate-limit identity so the
            # ParallelRequestLimiterV3 hook can apply mcp_rpm_limit on the
            # synthetic call_mcp_tool payload (otherwise a key with
            # mcp_rpm_limit could exceed it via the MCP path).
            "mcp_server_name": kwargs.get("mcp_rate_limit_server_name"),
            # Raw Bearer token from the original HTTP request — allows guardrails
            # (e.g. MCPJWTSigner) to independently verify the caller's identity
            # before re-signing an outbound token (FR-5 verify+re-sign).
            "incoming_bearer_token": kwargs.get("incoming_bearer_token"),
            "metadata": {"headers": kwargs.get("headers") or {}},
        }

        return synthetic_data

    def _convert_llm_result_to_mcp_response(self, llm_result, request_obj) -> MCPPreCallResponseObject | None:
        """
        Convert LLM guardrail result back to MCP response format.
        """
        from litellm.types.mcp import MCPPreCallResponseObject

        # If result is an exception, it means the guardrail blocked the request
        if isinstance(llm_result, Exception):
            return MCPPreCallResponseObject(
                should_proceed=False,
                error_message=str(llm_result),
                modified_arguments=None,
            )

        # If result is a dict with modified messages, check for content filtering
        if isinstance(llm_result, dict):
            modified_messages: Final = llm_result.get("messages")
            if modified_messages:
                # Check if content was blocked/modified
                original_content: Final = f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"
                new_content: Final = modified_messages[0].get("content", "") if modified_messages else ""

                if new_content != original_content:
                    # Content was modified - could be masking, redaction, or blocking
                    if not new_content or "blocked" in new_content.lower() or "violation" in new_content.lower():
                        # Content was blocked completely
                        return MCPPreCallResponseObject(
                            should_proceed=False,
                            error_message="Content blocked by guardrail",
                            modified_arguments=None,
                        )
                    else:
                        # Content was masked/redacted - extract the modified arguments
                        try:
                            # Try to parse the modified arguments from the masked content
                            modified_args = self._extract_modified_arguments_from_content(new_content, request_obj)
                            if modified_args is not None:
                                # Return the masked/redacted arguments for the MCP call to use
                                return MCPPreCallResponseObject(
                                    should_proceed=True,
                                    error_message=None,
                                    modified_arguments=modified_args,
                                )
                            else:
                                # Could not parse modified arguments, allow original call but warn
                                verbose_proxy_logger.warning(
                                    "Could not parse modified arguments from guardrail response: %s", new_content
                                )
                                return None
                        except Exception as e:
                            verbose_proxy_logger.error("Error parsing modified arguments: %s", e)
                            # Fallback: allow original call
                            return None

        # If result is a string, it's likely an error message
        if isinstance(llm_result, str):
            return MCPPreCallResponseObject(should_proceed=False, error_message=llm_result, modified_arguments=None)

        return None

    def _extract_modified_arguments_from_content(self, masked_content: str, request_obj) -> dict | None:
        """
        Extract modified/masked arguments from the guardrail response content.
        """
        import json

        verbose_proxy_logger.debug("Extracting modified args from content: %s", masked_content)

        try:
            # The format should be: "Tool: <tool_name>\nArguments: <json_arguments>"
            # Parse the arguments section
            lines: Final = masked_content.strip().split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Arguments:"):
                    # Get the arguments part - everything after "Arguments: "
                    args_text = line[len("Arguments:") :].strip()

                    verbose_proxy_logger.debug("Found arguments text: %s", args_text)

                    # Try to parse as JSON first
                    try:
                        modified_args = json.loads(args_text)
                        verbose_proxy_logger.debug("Successfully parsed JSON args: %s", modified_args)
                        return modified_args
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, try to extract key-value pairs manually
                        verbose_proxy_logger.debug("Failed to parse JSON arguments: %s, error: %s", args_text, e)
                        return self._parse_arguments_manually(args_text, request_obj.arguments)

            # If we can't find the Arguments: line, return None
            verbose_proxy_logger.warning("Could not find 'Arguments:' line in masked content")
            return None

        except Exception as e:
            verbose_proxy_logger.error("Error extracting modified arguments: %s", e)
            return None

    def _parse_arguments_manually(self, args_text: str, original_args: dict) -> dict | None:
        """
        Try to manually parse arguments when JSON parsing fails.
        This is a fallback for cases where the guardrail modifies the format.
        """
        import re

        try:
            # Start with original arguments and try to apply modifications
            modified_args: Final = original_args.copy()

            # Look for simple key-value patterns
            # This is a basic implementation - can be enhanced based on specific guardrail formats
            for key, original_value in original_args.items():
                if isinstance(original_value, str):
                    # Look for the key in the masked content and try to extract its value
                    pattern = rf"['\"]?{re.escape(key)}['\"]?\s*:\s*['\"]?([^,'\"]*)['\"]?"
                    match = re.search(pattern, args_text, re.IGNORECASE)
                    if match:
                        new_value = match.group(1).strip()
                        if new_value:
                            modified_args[key] = new_value

            return modified_args

        except Exception as e:
            verbose_proxy_logger.error("Error in manual argument parsing: %s", e)
            return None

    def _convert_llm_result_to_mcp_during_response(self, llm_result, request_obj) -> MCPDuringCallResponseObject | None:
        """
        Convert LLM guardrail result back to MCP during call response format.
        """
        # If result is an exception, it means the guardrail wants to stop execution
        if isinstance(llm_result, Exception):
            return MCPDuringCallResponseObject(should_continue=False, error_message=str(llm_result))

        # If result is a dict with modified messages, check for content filtering
        if isinstance(llm_result, dict):
            modified_messages: Final = llm_result.get("messages")
            if modified_messages:
                # Check if content was blocked/modified
                original_content: Final = f"Tool: {request_obj.tool_name}\nArguments: {request_obj.arguments}"
                new_content: Final = modified_messages[0].get("content", "") if modified_messages else ""

                if new_content != original_content:
                    # Content was modified, could be masking or blocking
                    if not new_content or "blocked" in new_content.lower():
                        # Content was blocked
                        return MCPDuringCallResponseObject(
                            should_continue=False,
                            error_message="Content blocked by guardrail during execution",
                        )
                    else:
                        # Content was masked/modified - for now, stop execution
                        return MCPDuringCallResponseObject(
                            should_continue=False,
                            error_message="Content modified by guardrail during execution",
                        )

        # If result is a string, it's likely an error message
        if isinstance(llm_result, str):
            return MCPDuringCallResponseObject(should_continue=False, error_message=llm_result)

        return None

    def get_combined_callback_list(self, dynamic_success_callbacks: list | None, global_callbacks: list) -> list:
        if dynamic_success_callbacks is None:
            return list(global_callbacks)
        return list(dict.fromkeys(dynamic_success_callbacks + global_callbacks))

    def _parse_pre_mcp_call_hook_response(
        self,
        response: MCPPreCallResponseObject,
        original_request: MCPPreCallRequestObject,
    ) -> Mapping[str, object]:
        """
        Parse the response from the pre_mcp_tool_call_hook

        1. Check if the call should proceed
        2. Apply any argument modifications
        3. Handle validation errors
        """
        result: Final = {
            "should_proceed": response.should_proceed,
            "modified_arguments": response.modified_arguments or original_request.arguments,
            "error_message": response.error_message,
            "hidden_params": response.hidden_params,
        }
        return result

    def _create_mcp_request_object_from_kwargs(self, kwargs: dict) -> "MCPPreCallRequestObject":
        """
        Helper function to create MCPPreCallRequestObject from kwargs for standard pre_call_hook.
        """
        from litellm.types.llms.base import HiddenParams
        from litellm.types.mcp import MCPPreCallRequestObject

        user_api_key_auth_dict: Final = self._convert_user_api_key_auth_to_dict(kwargs.get("user_api_key_auth"))

        return MCPPreCallRequestObject(
            tool_name=kwargs.get("name", ""),
            arguments=kwargs.get("arguments", {}),
            server_name=kwargs.get("server_name"),
            user_api_key_auth=user_api_key_auth_dict,
            hidden_params=HiddenParams(),
        )

    def _convert_mcp_hook_response_to_kwargs(self, response_data: dict | None, original_kwargs: dict) -> dict:
        """
        Helper function to convert pre_call_hook response back to kwargs for MCP usage.

        Supports:
        - modified_arguments: Override tool call arguments
        - extra_headers: Inject custom headers into the outbound MCP request
        """
        if not response_data:
            return original_kwargs

        modified_kwargs: Final = original_kwargs.copy()

        if response_data.get("modified_arguments"):
            modified_kwargs["arguments"] = response_data["modified_arguments"]

        if response_data.get("extra_headers"):
            # Merge rather than replace — a prior guardrail in the chain may have
            # already injected headers (e.g. tracing IDs).  Later guardrails win on
            # key collisions so that the most-specific guardrail (e.g. JWT signer)
            # takes precedence over earlier ones.
            existing: Final = modified_kwargs.get("extra_headers") or {}
            modified_kwargs["extra_headers"] = {
                **existing,
                **response_data["extra_headers"],
            }

        return modified_kwargs

    async def process_pre_call_hook_response(self, response, data, call_type):
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            if call_type in ["completion", "text_completion"]:
                raise RejectedRequestError(
                    message=response,
                    model=data.get("model", ""),
                    llm_provider="",
                    request_data=data,
                )
            else:
                raise HTTPException(status_code=400, detail={"error": response})
        return data

    def _should_use_guardrail_load_balancing(
        self,
        guardrail_name: str,
    ) -> bool:
        """
        Check if load balancing should be used for this guardrail.

        Returns True if the router has multiple deployments for this guardrail name.
        """
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None or not hasattr(llm_router, "guardrail_list"):
            return False

        matching: Final = [g for g in llm_router.guardrail_list if g.get("guardrail_name") == guardrail_name]
        return len(matching) > 1

    async def _execute_guardrail_hook(
        self,
        callback: "CustomGuardrail",
        hook_type: str,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth | None,
        call_type: CallTypesLiteral,
        response: LLMResponseTypes | None = None,
    ) -> object:
        """
        Execute a single guardrail's hook.

        Args:
            callback: The guardrail callback to execute
            hook_type: One of "pre_call", "during_call", "post_call"
            data: Request data
            user_api_key_dict: User API key auth
            call_type: Type of call
            response: Response object (for post_call hooks)

        Returns:
            Result from the guardrail execution
        """
        # Use unified_guardrail if callback has apply_guardrail method
        has_apply_guardrail: Final = "apply_guardrail" in type(callback).__dict__ and not getattr(
            callback, "use_native_lifecycle_hooks", False
        )
        use_unified: Final = has_apply_guardrail and not (
            hook_type == "during_call" and getattr(callback, "use_native_during_call_hook", False)
        )
        if use_unified:
            data["guardrail_to_apply"] = callback

        target: Final = unified_guardrail if use_unified else callback

        if hook_type == "pre_call":
            return await target.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=self.call_details["user_api_key_cache"],
                data=data,
                call_type=call_type,
            )
        elif hook_type == "during_call":
            return await target.async_moderation_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
            )
        elif hook_type == "post_call":
            return await target.async_post_call_success_hook(
                user_api_key_dict=user_api_key_dict,
                data=data,
                response=response,
            )
        else:
            raise ValueError(f"Unknown hook_type: {hook_type}")

    async def _execute_guardrail_with_load_balancing(
        self,
        guardrail_name: str,
        hook_type: str,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth | None,
        call_type: CallTypesLiteral,
        response: LLMResponseTypes | None = None,
    ) -> object:
        """
        Execute a guardrail using the router's load balancing.

        Args:
            guardrail_name: Name of the guardrail
            hook_type: One of "pre_call", "during_call", "post_call"
            data: Request data
            user_api_key_dict: User API key auth
            call_type: Type of call
            response: Response object (for post_call hooks)

        Returns:
            Result from the guardrail execution
        """
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            raise ValueError("Router not initialized")

        # Select guardrail using router's load balancing
        selected_guardrail: Final = llm_router.get_available_guardrail(guardrail_name=guardrail_name)

        callback: Final[CustomGuardrail | None] = selected_guardrail.get("callback")
        if callback is None:
            raise ValueError(f"No callback found for guardrail: {guardrail_name}")

        return await self._execute_guardrail_hook(
            callback=callback,
            hook_type=hook_type,
            data=data,
            user_api_key_dict=user_api_key_dict,
            call_type=call_type,
            response=response,
        )

    async def _process_guardrail_callback(
        self,
        callback: CustomGuardrail,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth | None,
        call_type: CallTypesLiteral,
        event_type: GuardrailEventHooks,
    ) -> dict | None:
        """
        Process a guardrail callback during pre-call hook.

        Supports load balancing when multiple guardrail deployments exist.

        Args:
            callback: The CustomGuardrail callback to process
            data: The request data dictionary
            user_api_key_dict: User API key authentication details
            call_type: The type of API call being made

        Returns:
            Updated data dictionary if guardrail passes, None if guardrail should be skipped
        """
        from litellm.types.guardrails import GuardrailEventHooks

        # Determine the event type based on call type
        if event_type is GuardrailEventHooks.pre_call and call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.pre_mcp_call

        # Check if the guardrail should run for this request
        if callback.should_run_guardrail(data=data, event_type=event_type) is not True:
            return None

        guardrail_name: Final = callback.guardrail_name

        # Track timing and errors for prometheus metrics
        # Use time.perf_counter() for more accurate duration measurements
        guardrail_start_time: Final = time.perf_counter()
        status = "success"
        error_type = None

        try:
            # Check if load balancing should be used
            if guardrail_name and self._should_use_guardrail_load_balancing(guardrail_name):
                response = await self._execute_guardrail_with_load_balancing(
                    guardrail_name=guardrail_name,
                    hook_type="pre_call",
                    data=data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=call_type,
                )
            else:
                # Single guardrail - execute directly
                response = await self._execute_guardrail_hook(
                    callback=callback,
                    hook_type="pre_call",
                    data=data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=call_type,
                )

            # Process the response if one was returned
            if response is not None:
                data = await self.process_pre_call_hook_response(response=response, data=data, call_type=call_type)

            callback.mark_pre_call_hook_ran(data)

        except SensitiveDataRouteException:
            status = "intervened"
            raise
        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            _enrich_http_exception_with_guardrail_context(e, callback)
            # Re-raise the exception to maintain existing behavior
            raise
        finally:
            # Record prometheus metrics
            guardrail_end_time: Final = time.perf_counter()
            latency_seconds: Final = guardrail_end_time - guardrail_start_time

            # Get guardrail name for metrics (fallback if not set)
            metrics_guardrail_name: Final = (
                guardrail_name or getattr(callback, "guardrail_name", callback.__class__.__name__) or "unknown"
            )

            self._emit_guardrail_metrics(
                guardrail_name=metrics_guardrail_name,
                latency_seconds=latency_seconds,
                status=status,
                error_type=error_type,
                hook_type="pre_call",
            )

        return data

    async def _process_prompt_template(
        self,
        data: dict,
        litellm_logging_obj: Any,
        prompt_id: str,
        prompt_version: int | None,
        call_type: CallTypesLiteral,
    ) -> None:
        """Process prompt template if applicable."""

        from litellm.proxy.prompts.prompt_endpoints import (
            construct_versioned_prompt_id,
            get_latest_version_prompt_id,
        )
        from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
        from litellm.responses.utils import ResponsesAPIRequestUtils
        from litellm.utils import get_non_default_completion_params

        if prompt_version is None:
            lookup_prompt_id = get_latest_version_prompt_id(
                prompt_id=prompt_id,
                all_prompt_ids=IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS,
            )
        else:
            lookup_prompt_id = construct_versioned_prompt_id(prompt_id=prompt_id, version=prompt_version)

        custom_logger: Final = IN_MEMORY_PROMPT_REGISTRY.get_prompt_callback_by_id(lookup_prompt_id)
        prompt_spec: Final = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(lookup_prompt_id)
        litellm_prompt_id: str | None = None
        if prompt_spec is not None:
            litellm_prompt_id = prompt_spec.litellm_params.prompt_id
            data.pop("prompt_id", None)

        if custom_logger and prompt_spec is not None:
            is_responses_call: Final = call_type == "aresponses"
            original_responses_input: Final = data.get("input", "") if is_responses_call else ""
            client_messages: Final = (
                ResponsesAPIRequestUtils.responses_input_to_chat_messages(original_responses_input)
                if is_responses_call
                else data.get("messages", [])
            )
            (
                model,
                messages,
                optional_params,
            ) = await litellm_logging_obj.async_get_chat_completion_prompt(
                model=data.get("model", ""),
                messages=client_messages,
                non_default_params=get_non_default_completion_params(kwargs=data) or {},
                prompt_id=litellm_prompt_id,
                prompt_spec=prompt_spec,
                prompt_management_logger=custom_logger,
                prompt_variables=data.pop("prompt_variables", None) or {},
                prompt_label=data.pop("prompt_label", None) or {},
                prompt_version=data.pop("prompt_version", None) or {},
            )

            data.update(optional_params)
            data["model"] = model
            if is_responses_call:
                data["input"] = ResponsesAPIRequestUtils.merge_prompt_management_input(
                    original_input=original_responses_input,
                    client_input=client_messages,
                    merged_input=messages,
                )
            else:
                data["messages"] = messages
            # prevent re-processing the prompt template
            data.pop("prompt_id", None)
            data.pop("prompt_variables", None)
            data.pop("prompt_label", None)
            data.pop("prompt_version", None)

    def _process_guardrail_metadata(self, data: dict) -> None:
        """Process guardrails from metadata and add to applied_guardrails."""
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        metadata_standard: Final = data.get("metadata") or {}
        metadata_litellm: Final = data.get("litellm_metadata") or {}

        guardrails_in_metadata = []
        if isinstance(metadata_standard, dict) and "guardrails" in metadata_standard:
            guardrails_in_metadata = metadata_standard.get("guardrails", [])
        elif isinstance(metadata_litellm, dict) and "guardrails" in metadata_litellm:
            guardrails_in_metadata = metadata_litellm.get("guardrails", [])

        if guardrails_in_metadata and isinstance(guardrails_in_metadata, list):
            applied_guardrails = []
            if isinstance(metadata_standard, dict) and "applied_guardrails" in metadata_standard:
                applied_guardrails = metadata_standard.get("applied_guardrails", [])
            elif isinstance(metadata_litellm, dict) and "applied_guardrails" in metadata_litellm:
                applied_guardrails = metadata_litellm.get("applied_guardrails", [])

            if not isinstance(applied_guardrails, list):
                applied_guardrails = []

            for guardrail_name in guardrails_in_metadata:
                if isinstance(guardrail_name, str) and guardrail_name not in applied_guardrails:
                    add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=guardrail_name)

    async def _maybe_execute_pipelines(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
        event_hook: str,
    ) -> dict:
        """
        Execute guardrail pipelines if any are configured for this request.

        Checks metadata for pipelines resolved by the policy engine
        and executes them. Handles the result (allow/block/modify_response).

        Returns the (possibly modified) data dict.
        """
        pipelines: Final = _policy_pipelines(data)
        if not pipelines:
            return data

        for policy_name, pipeline in pipelines:
            if pipeline.mode != event_hook:
                continue

            result: PipelineExecutionResult = await PipelineExecutor.execute_steps(
                steps=pipeline.steps,
                mode=pipeline.mode,
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
                policy_name=policy_name,
            )

            data = self._handle_pipeline_result(
                result=result,
                data=data,
                policy_name=policy_name,
            )

        return data

    @staticmethod
    def _handle_pipeline_result(
        result: PipelineExecutionResult,
        data: dict,
        policy_name: str,
    ) -> dict:
        """
        Handle a PipelineExecutionResult — allow, block, or modify_response.

        Returns data dict if allowed, raises on block/modify_response.
        """
        if result.terminal_action == "allow":
            if result.modified_data is not None:
                data.update(result.modified_data)
            return data

        if result.terminal_action == "block":
            original_exception: Final = result.original_exception
            if original_exception is not None and not _exception_changes_request_flow(original_exception):
                blocking_step: Final = result.step_results[-1] if result.step_results else None
                if blocking_step is not None:
                    callback: Final = PipelineExecutor.find_guardrail_callback(blocking_step.guardrail_name)
                    if callback is not None:
                        _enrich_http_exception_with_guardrail_context(original_exception, callback)
                raise original_exception

            step_results_serializable: Final = [
                {
                    "guardrail": sr.guardrail_name,
                    "outcome": sr.outcome,
                    "action": sr.action_taken,
                }
                for sr in result.step_results
            ]
            error_detail: Final = {
                "error": {
                    "message": f"Content blocked by guardrail pipeline '{policy_name}'",
                    "type": "guardrail_pipeline_error",
                    "pipeline_context": {
                        "policy": policy_name,
                        "step_results": step_results_serializable,
                    },
                }
            }
            raise HTTPException(status_code=400, detail=error_detail)

        if result.terminal_action == "modify_response":
            raise ModifyResponseException(
                message=result.modify_response_message or "Response modified by pipeline",
                model=data.get("model", "unknown"),
                request_data=data,
                guardrail_name=f"pipeline:{policy_name}",
                detection_info=None,
            )

        return data

    def has_pre_call_guardrails(self, request_metadata: Mapping[str, object]) -> bool:
        """
        Whether anything configured would inspect the content of a request carrying this metadata.

        Evaluated with the same predicate the pre-call loop uses, so a proxy configured only with
        post-call guardrails answers False. Callers that must pay a real cost to build the hook's
        input, such as streaming a batch input file off disk, use this to skip that work.

        A content-enforcing ``CustomLogger`` counts too. It is not a guardrail and has no event
        hook to consult, but it judges the payload the same way, so a proxy configured only with
        one of those still has something to say about every record.
        """
        if request_metadata.get("_guardrail_pipelines"):
            return True
        caps: Final = ProxyLogging._callback_capabilities()
        if caps.has_content_enforcer:
            return True
        probe: Final = {"metadata": dict(request_metadata)}  # mutable-ok: should_run_guardrail takes a dict
        return any(
            isinstance(callback, CustomGuardrail)
            and callback.should_run_guardrail(data=probe, event_type=GuardrailEventHooks.pre_call)
            for callback in caps.resolved_callbacks
        )

    # The actual implementation of the function
    @overload
    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: None,
        call_type: CallTypesLiteral,
        guardrails_only: bool = False,
    ) -> None:
        pass

    @overload
    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        call_type: CallTypesLiteral,
        guardrails_only: bool = False,
    ) -> dict:
        pass

    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict | None,
        call_type: CallTypesLiteral,
        guardrails_only: bool = False,
    ) -> dict | None:
        """
        Allows users to modify/reject the incoming request to the proxy, without having to deal with parsing Request body.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation

        With ``guardrails_only`` the walk is limited to guardrails and guardrail pipelines: rate
        limiting, budget accounting, prompt templates and hanging-request alerting are skipped.
        Use it to scan a payload that is not itself a request, such as one record of a batch file.
        """
        verbose_proxy_logger.debug("Inside Proxy Logging Pre-call hook!")

        if not guardrails_only:
            self._init_response_taking_too_long_task(data=data)

        if data is None:
            return None

        litellm_logging_obj: Final = cast(Optional["LiteLLMLoggingObj"], data.get("litellm_logging_obj", None))
        prompt_id: Final[str | None] = data.get("prompt_id", None)
        prompt_version: Final[int | None] = data.get("prompt_version", None)

        ## PROMPT TEMPLATE CHECK ##

        if (
            not guardrails_only
            and litellm_logging_obj is not None
            and prompt_id is not None
            and (call_type == "completion" or call_type == "acompletion" or call_type == "aresponses")
        ):
            await self._process_prompt_template(
                data=data,
                litellm_logging_obj=litellm_logging_obj,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                call_type=call_type,
            )

        try:
            # Execute guardrail pipelines before the normal callback loop
            data = await self._maybe_execute_pipelines(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
                event_hook="pre_call",
            )

            # Get pipeline-managed guardrails to skip in normal loop
            pipeline_managed: Final = _pipeline_managed_guardrail_names(data)

            caps: Final = ProxyLogging._callback_capabilities()
            # Skip the per-request callback walk entirely when nothing in
            # ``litellm.callbacks`` overrides ``async_pre_call_hook`` and no
            # CustomGuardrail is configured. Saves the loop overhead +
            # ``time.time()`` x2 per registered callback for the common
            # "callbacks=[]" case on small / dev deployments.
            if (
                not caps.has_guardrail
                and not caps.has_content_enforcer
                and (guardrails_only or not caps.has_pre_call_override)
            ):
                if data is not None:
                    self._process_guardrail_metadata(data)
                return data

            parallel_guardrails: Final[tuple[CustomGuardrail, ...]] = tuple(
                cb
                for cb in caps.resolved_callbacks
                if isinstance(cb, CustomGuardrail)
                and getattr(cb, "run_in_parallel", False)
                and not (cb.guardrail_name and cb.guardrail_name in pipeline_managed)
            )

            deferred_route_exc: SensitiveDataRouteException | None = None
            for _callback in caps.resolved_callbacks:
                start_time = time.time()
                try:
                    if isinstance(_callback, CustomGuardrail) and data is not None:
                        # Skip guardrails managed by a pipeline
                        if _callback.guardrail_name and _callback.guardrail_name in pipeline_managed:
                            continue

                        if getattr(_callback, "run_in_parallel", False):
                            continue

                        result = await self._process_guardrail_callback(
                            callback=_callback,
                            data=data,
                            user_api_key_dict=user_api_key_dict,
                            call_type=call_type,
                            event_type=GuardrailEventHooks.pre_call,
                        )
                        if result is None:
                            continue
                        data = result

                    elif (
                        _callback is not None
                        and isinstance(_callback, CustomLogger)
                        and (not guardrails_only or _callback.enforces_request_content)
                        and "async_pre_call_hook" in vars(_callback.__class__)
                        and _callback.__class__.async_pre_call_hook != CustomLogger.async_pre_call_hook
                    ):
                        if call_type == "call_mcp_tool" and user_api_key_dict is None:
                            continue

                        response: Exception | str | Mapping[str, object] | None = await _callback.async_pre_call_hook(
                            user_api_key_dict=user_api_key_dict,
                            cache=self.call_details["user_api_key_cache"],
                            data=data,
                            call_type=call_type,
                        )
                        if response is not None:
                            data = await self.process_pre_call_hook_response(
                                response=response, data=data, call_type=call_type
                            )
                except SensitiveDataRouteException as e:
                    # Defer the reroute until remaining guardrails have run so later
                    # security checks are not skipped; the first reroute wins and a
                    # later guardrail that blocks still propagates. Fall through to the
                    # service-span recording below so the triggering guardrail is still
                    # timed like every other callback.
                    if deferred_route_exc is None:
                        deferred_route_exc = e

                end_time = time.time()
                duration = end_time - start_time
                if (
                    hasattr(self, "service_logging_obj") and duration > 0.01
                ):  # only if duration is non-negligible - don't spam the logs
                    await self.service_logging_obj.async_service_success_hook(
                        service=ServiceTypes.PROXY_PRE_CALL,
                        duration=duration,
                        call_type=f"{_callback.__class__.__name__}",
                        parent_otel_span=user_api_key_dict.parent_otel_span,
                        start_time=start_time,
                        end_time=end_time,
                    )

            if deferred_route_exc is not None and data is not None:
                data = await self._handle_sensitive_data_route_exception(deferred_route_exc, data, user_api_key_dict)

            if parallel_guardrails and data is not None:
                await self._run_parallel_pre_call_guardrails(
                    guardrails=parallel_guardrails,
                    data=data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=call_type,
                )

            if data is not None:
                self._process_guardrail_metadata(data)

            return data
        except SensitiveDataRouteException as e:
            data = await self._handle_sensitive_data_route_exception(e, data, user_api_key_dict)
            if data is not None:
                self._process_guardrail_metadata(data)
            return data
        except Exception as e:
            raise e

    async def _run_parallel_pre_call_guardrails(
        self,
        guardrails: tuple[CustomGuardrail, ...],
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> None:
        """
        Run opted-in pre_call guardrails concurrently against one shared payload
        snapshot. These guardrails are declared block-only, so any modified data
        they return is discarded; they run for their blocking side effect (raising
        to reject the request before it reaches the LLM). Every guardrail is
        awaited to completion (``return_exceptions=True``) so a raise by one never
        leaves the others running as unobserved background tasks. A guardrail that
        blocks (any exception other than a reroute or passthrough) takes precedence
        over one that only changes the request flow, so a fast reroute can never
        let a slower block be bypassed; the request is rejected before it reaches
        the LLM, preserving the pre-call barrier that ``during_call`` guardrails
        cannot provide. Per-guardrail latency is recorded by
        ``_process_guardrail_callback``'s own metrics.
        """
        results: Final = await asyncio.gather(
            *(
                self._process_guardrail_callback(
                    callback=callback,
                    data=data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=call_type,
                    event_type=GuardrailEventHooks.pre_call,
                )
                for callback in guardrails
            ),
            return_exceptions=True,
        )
        raised: Final = tuple(result for result in results if isinstance(result, BaseException))
        blocking: Final = next((exc for exc in raised if not _exception_changes_request_flow(exc)), None)
        if blocking is not None:
            raise blocking
        if raised:
            raise raised[0]

    async def _handle_sensitive_data_route_exception(
        self,
        exc: SensitiveDataRouteException,
        data: dict | None,
        user_api_key_dict: UserAPIKeyAuth | None,
    ) -> dict | None:
        """
        Handle SensitiveDataRouteException by rerouting the current request to
        the target model and, when sticky_session_routing is enabled, persisting
        the session override so subsequent requests reuse the same model.
        """
        if data is None:
            return None

        verbose_proxy_logger.info(
            "SensitiveDataRouteException caught: session_id=%s route_to_model=%s guardrail=%s sticky=%s",
            exc.session_id,
            exc.route_to_model,
            exc.guardrail_name,
            exc.sticky_session_routing,
        )

        if exc.sticky_session_routing:
            sensitive_routing_hook: Final = self.get_proxy_hook("sensitive_data_routing")
            if isinstance(sensitive_routing_hook, _PROXY_SensitiveDataRoutingHandler):
                await sensitive_routing_hook.set_session_routing(
                    session_id=exc.session_id,
                    model=exc.route_to_model,
                    user_api_key_dict=user_api_key_dict,
                    guardrail_name=exc.guardrail_name,
                )
            else:
                verbose_proxy_logger.warning(
                    "SensitiveDataRouteException requested sticky routing for session_id=%s "
                    "but the 'sensitive_data_routing' hook is not registered. Only this request "
                    "will be rerouted; subsequent requests will not be sticky.",
                    exc.session_id,
                )

        original_model: Final = data.get("model")
        data["model"] = exc.route_to_model

        metadata: Final = data.get("metadata") or {}
        metadata["sensitive_data_routing_applied"] = True
        metadata["sensitive_data_routing_original_model"] = original_model
        metadata["sensitive_data_routing_guardrail"] = exc.guardrail_name
        metadata["sensitive_data_routing_detection_info"] = exc.detection_info
        data["metadata"] = metadata

        return data

    @staticmethod
    def _emit_guardrail_metrics(
        guardrail_name: str,
        latency_seconds: float,
        status: str,
        error_type: str | None,
        hook_type: str,
    ) -> None:
        for prom_callback in litellm.callbacks:
            if isinstance(prom_callback, PrometheusLogger):
                prom_callback._record_guardrail_metrics(
                    guardrail_name=guardrail_name,
                    latency_seconds=latency_seconds,
                    status=status,
                    error_type=error_type,
                    hook_type=hook_type,
                )
                break

    @staticmethod
    async def _run_guardrail_with_metrics(callback: object, coro: Awaitable[_T], hook_type: str) -> _T:
        """
        Await `coro`, recording its latency and status to the
        `litellm_guardrail_latency_seconds` metric under `hook_type`, and
        enriching any raised HTTPException with the originating callback's
        `guardrail_name`/`guardrail_mode` before re-raising.
        """
        guardrail_name: Final = getattr(callback, "guardrail_name", None) or type(callback).__name__
        start_time: Final = time.perf_counter()
        status = "success"
        error_type: str | None = None
        try:
            return await coro
        except SensitiveDataRouteException:
            status = "intervened"
            raise
        except Exception as e:
            status = "error"
            error_type = type(e).__name__
            _enrich_http_exception_with_guardrail_context(e, callback)
            raise
        finally:
            ProxyLogging._emit_guardrail_metrics(
                guardrail_name=guardrail_name,
                latency_seconds=time.perf_counter() - start_time,
                status=status,
                error_type=error_type,
                hook_type=hook_type,
            )

    @staticmethod
    async def _wrap_streaming_iterator_with_enrichment(
        callback: object, gen: AsyncGenerator[_T, None]
    ) -> AsyncGenerator[_T, None]:
        """
        Yield from `gen`; if iteration raises an HTTPException with dict detail,
        enrich the detail with the originating callback's `guardrail_name` and
        `guardrail_mode` before re-raising. Used to wrap each layer of the
        async_post_call_streaming_iterator_hook chain so the enrichment is
        attributed to the callback that produced the chunk pipeline at that
        point in the chain.
        """
        try:
            async for chunk in gen:
                yield chunk
        except Exception as e:
            _enrich_http_exception_with_guardrail_context(e, callback)
            raise

    # Cache for callback-capability detection. Keyed on a signature of
    # litellm.callbacks (length + each item's id) so we recompute when the
    # callback list mutates (add/remove) without iterating every request.
    _callback_capabilities_cache: ClassVar[dict[tuple[int, tuple[int, ...]], "_CallbackCapabilities"]] = {}

    @staticmethod
    def _callback_capabilities() -> "_CallbackCapabilities":
        """
        Inspect ``litellm.callbacks`` once and answer the per-hook capability
        questions used to short-circuit no-op work on the chat-completions hot
        path. Per-request callers iterated ``litellm.callbacks`` and called
        ``get_custom_logger_compatible_class`` for every string entry — that
        scanning cost dominated the proxy overhead on low-config deployments.

        Cache invalidates whenever the list length or member identities change.
        """
        callbacks: Final = litellm.callbacks
        sig: Final = (len(callbacks), tuple(id(c) for c in callbacks))
        cache: Final = ProxyLogging._callback_capabilities_cache
        cached: Final = cache.get(sig)
        if cached is not None:
            return cached

        has_post_call_response_headers = False
        has_iterator_override = False
        has_streaming_chunk_override = False
        has_guardrail = False
        has_pre_call_override = False
        has_content_enforcer = False
        iterator_overrides: Final[list[tuple[Any, str]]] = []  # (callback, kind)
        resolved_callbacks: Final[list[CustomLogger]] = []

        for callback in callbacks:
            if isinstance(callback, str):
                resolved = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                    cast(_custom_logger_compatible_callbacks_literal, callback)
                )
            else:
                resolved = callback
            if resolved is None or not isinstance(resolved, CustomLogger):
                continue
            resolved_callbacks.append(resolved)
            cls = type(resolved)
            if cls is CustomLogger:
                continue
            if isinstance(resolved, CustomGuardrail):
                has_guardrail = True
            # Use the same leaf-class ``__dict__`` check as the other hook
            # capabilities: only callbacks that actually override the hook
            # contribute to the flag. Setting this for every ``CustomLogger``
            # instance (the prior behaviour) forced the full
            # ``post_call_response_headers_hook`` body to run on every request
            # even when no registered callback customized response headers.
            cls_attrs = cls.__dict__
            if "async_post_call_response_headers_hook" in cls_attrs:
                has_post_call_response_headers = True
            if "async_post_call_streaming_iterator_hook" in cls_attrs:
                has_iterator_override = True
                iterator_overrides.append((resolved, "override"))
            elif "apply_guardrail" in cls_attrs and not getattr(resolved, "use_native_lifecycle_hooks", False):
                iterator_overrides.append((resolved, "apply_guardrail"))
            # Walk the MRO for ``async_post_call_streaming_hook`` rather than
            # using the leaf-class ``__dict__`` check used by the other flags:
            # before this PR the hook was unconditionally invoked, so a
            # callback that inherits an override from an intermediate parent
            # (e.g. a vendor base class providing the override, with the
            # registered class adding nothing else) MUST still be detected.
            # A leaf-class miss here would silently drop the inherited hook.
            base_streaming_hook = CustomLogger.async_post_call_streaming_hook
            cls_streaming_hook = getattr(
                cls,
                "async_post_call_streaming_hook",
                base_streaming_hook,
            )
            if getattr(cls_streaming_hook, "__func__", cls_streaming_hook) is not getattr(
                base_streaming_hook, "__func__", base_streaming_hook
            ):
                has_streaming_chunk_override = True
            if "async_pre_call_hook" in cls_attrs:
                has_pre_call_override = True
                if resolved.enforces_request_content is True:
                    has_content_enforcer = True

        caps: Final = _CallbackCapabilities(
            has_post_call_response_headers=has_post_call_response_headers,
            has_iterator_override=has_iterator_override
            or any(kind == "apply_guardrail" for _, kind in iterator_overrides),
            has_streaming_chunk_override=has_streaming_chunk_override,
            has_guardrail=has_guardrail,
            has_pre_call_override=has_pre_call_override,
            has_content_enforcer=has_content_enforcer,
            iterator_overrides=tuple(iterator_overrides),
            resolved_callbacks=tuple(resolved_callbacks),
        )
        # Limit cache to handle test churn without leaking; production
        # callback lists are stable so this rarely grows past 1 entry.
        if len(cache) >= 32:
            cache.clear()
        cache[sig] = caps
        return caps

    @staticmethod
    def _stream_requires_guardrail_translation(user_api_key_dict: UserAPIKeyAuth) -> bool:
        from litellm.litellm_core_utils.api_route_to_call_types import (
            get_call_types_for_route,
        )

        route: Final = user_api_key_dict.request_route
        if not route:
            return False
        call_types: Final = get_call_types_for_route(route)
        if not call_types:
            return False
        return call_types[0] in NON_OPENAI_STREAM_GUARDRAIL_TRANSLATION_CALL_TYPES

    @staticmethod
    def has_post_call_response_headers_callbacks() -> bool:
        return ProxyLogging._callback_capabilities().has_post_call_response_headers

    @staticmethod
    def has_streaming_callbacks() -> bool:
        caps: Final = ProxyLogging._callback_capabilities()
        return caps.has_iterator_override or caps.has_streaming_chunk_override or caps.has_guardrail

    @staticmethod
    def has_streaming_chunk_hook_overrides() -> bool:
        """True iff any callback overrides ``async_post_call_streaming_hook``
        (the per-chunk hook, distinct from the iterator wrapper)."""
        caps: Final = ProxyLogging._callback_capabilities()
        return caps.has_streaming_chunk_override or caps.has_guardrail

    def needs_iterator_wrap(self) -> bool:
        """Whether ``async_data_generator`` needs to wrap the upstream stream
        through ``async_post_call_streaming_iterator_hook``. Instance method
        so tests can override the gate via ``MagicMock(spec=ProxyLogging)``.
        """
        return ProxyLogging._callback_capabilities().has_iterator_override

    def needs_per_chunk_streaming_hook(self) -> bool:
        """Whether ``async_data_generator`` needs to call the per-chunk
        ``_apply_streaming_chunk_hooks`` for every emitted chunk. Instance
        method for the same reason as :py:meth:`needs_iterator_wrap`.
        """
        caps: Final = ProxyLogging._callback_capabilities()
        return caps.has_streaming_chunk_override or caps.has_guardrail

    @staticmethod
    def has_during_call_guardrails() -> bool:
        return ProxyLogging._callback_capabilities().has_guardrail

    async def during_call_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth | None,
        call_type: CallTypesLiteral,
    ):
        """
        Runs the CustomGuardrail's async_moderation_hook() in parallel
        """
        # Fast path: skip the entire guardrail scan when no CustomGuardrail
        # callbacks are registered. Saves per-request iteration over
        # ``litellm.callbacks`` plus an ``asyncio.gather([])`` round trip on
        # deployments with no guardrails configured.
        if not ProxyLogging._callback_capabilities().has_guardrail:
            return data
        # Step 1: Collect all guardrail tasks to run in parallel
        guardrail_tasks: Final = []

        for callback in litellm.callbacks:
            if isinstance(callback, CustomGuardrail):
                ################################################################
                # Check if guardrail should be run for GuardrailEventHooks.during_call hook
                ################################################################

                # V1 implementation - backwards compatibility
                if callback.event_hook is None and hasattr(callback, "moderation_check"):
                    if callback.moderation_check == "pre_call":
                        return
                else:
                    # Main - V2 Guardrails implementation
                    from litellm.types.guardrails import GuardrailEventHooks

                    event_type = GuardrailEventHooks.during_call
                    if call_type == CallTypes.call_mcp_tool.value:
                        event_type = GuardrailEventHooks.during_mcp_call

                    if callback.should_run_guardrail(data=data, event_type=event_type) is not True:
                        continue
                # Convert user_api_key_dict to proper format for async_moderation_hook
                if call_type == CallTypes.call_mcp_tool.value:
                    user_api_key_auth_dict = self._convert_user_api_key_auth_to_dict(user_api_key_dict)
                else:
                    user_api_key_auth_dict = user_api_key_dict
                # Add task to list for parallel execution
                if (
                    "apply_guardrail" in type(callback).__dict__
                    and not callback.use_native_lifecycle_hooks
                    and user_api_key_dict is not None
                    and not getattr(callback, "use_native_during_call_hook", False)
                ):
                    data["guardrail_to_apply"] = callback
                    guardrail_task = self._run_guardrail_with_metrics(
                        callback,
                        unified_guardrail.async_moderation_hook(
                            user_api_key_dict=user_api_key_dict,
                            data=data,
                            call_type=call_type,
                        ),
                        "during_call",
                    )
                else:
                    guardrail_task = self._run_guardrail_with_metrics(
                        callback,
                        callback.async_moderation_hook(
                            data=data,
                            user_api_key_dict=user_api_key_auth_dict,
                            call_type=call_type,
                        ),
                        "during_call",
                    )
                guardrail_tasks.append(guardrail_task)

        # Step 2: Run all guardrail tasks in parallel
        if guardrail_tasks:
            try:
                await asyncio.gather(*guardrail_tasks)
            except Exception as e:
                # If any guardrail raises an exception, it will propagate here
                raise e

        return data

    async def failed_tracking_alert(
        self,
        error_message: str,
        failing_model: str,
    ):
        if self.alerting is None:
            return

        if self.slack_alerting_instance:
            await self.slack_alerting_instance.failed_tracking_alert(
                error_message=error_message,
                failing_model=failing_model,
            )

    async def budget_alerts(
        self,
        type: Literal[
            "token_budget",
            "user_budget",
            "soft_budget",
            "max_budget_alert",
            "team_budget",
            "organization_budget",
            "proxy_budget",
            "projected_limit_exceeded",
            "project_budget",
        ],
        user_info: CallInfo,
    ):
        # For soft_budget alerts with alert_emails set, allow email sending even if alerting is None
        # This enables team-specific soft budget email alerts via metadata.soft_budget_alerting_emails
        # Note: user_info is a CallInfo that can represent user/team/org level info. For team budgets,
        # alert_emails is populated from team_object.metadata.soft_budget_alerting_emails (see auth_checks.py)
        is_soft_budget_with_alert_emails: Final = (
            type == "soft_budget" and user_info.alert_emails is not None and len(user_info.alert_emails) > 0
        )

        if self.alerting is None and not is_soft_budget_with_alert_emails:
            # do nothing if alerting is not switched on (unless it's a soft_budget alert with team-specific emails)
            return

        if self.alerting is not None and "slack" in self.alerting:
            if self.slack_alerting_instance is not None:
                await self.slack_alerting_instance.budget_alerts(
                    type=type,
                    user_info=user_info,
                )

        # Call email_logging_instance if:
        # 1. "email" is in alerting config, OR
        # 2. It's a soft_budget alert with team-specific alert_emails (bypasses global alerting config)
        should_send_email = (self.alerting is not None and "email" in self.alerting) or is_soft_budget_with_alert_emails

        if should_send_email and self.email_logging_instance is not None:
            await self.email_logging_instance.budget_alerts(
                type=type,
                user_info=user_info,
            )

    async def alerting_handler(
        self,
        message: str,
        level: Literal["Low", "Medium", "High"],
        alert_type: AlertType,
        request_data: dict | None = None,
    ):
        """
        Alerting based on thresholds: - https://github.com/BerriAI/litellm/issues/1298

        - Responses taking too long
        - Requests are hanging
        - Calls are failing
        - DB Read/Writes are failing
        - Proxy Close to max budget
        - Key Close to max budget

        Parameters:
            level: str - Low|Medium|High - if calls might fail (Medium) or are failing (High); Currently, no alerts would be 'Low'.
            message: str - what is the alert about
        """
        if self.alerting is None:
            return

        from datetime import datetime

        # Get the current timestamp
        current_time: Final = datetime.now().strftime("%H:%M:%S")
        _proxy_base_url: Final = os.getenv("PROXY_BASE_URL", None)
        formatted_message = f"Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
        if _proxy_base_url is not None:
            formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

        extra_kwargs: Final = {}
        alerting_metadata = {}
        if request_data is not None:
            _url: Final = await _add_langfuse_trace_id_to_alert(request_data=request_data)

            if _url is not None:
                extra_kwargs["🪢 Langfuse Trace"] = _url
                formatted_message += f"\n\n🪢 Langfuse Trace: {_url}"
            if (
                "metadata" in request_data
                and request_data["metadata"].get("alerting_metadata", None) is not None
                and isinstance(request_data["metadata"]["alerting_metadata"], dict)
            ):
                alerting_metadata = request_data["metadata"]["alerting_metadata"]
        for client in self.alerting:
            if client == "slack":
                await self.slack_alerting_instance.send_alert(
                    message=message,
                    level=level,
                    alert_type=alert_type,
                    user_info=None,
                    alerting_metadata=alerting_metadata,
                    **extra_kwargs,
                )
            elif client == "sentry":
                if litellm.utils.sentry_sdk_instance is not None:
                    litellm.utils.sentry_sdk_instance.capture_message(formatted_message)
                else:
                    raise Exception("Missing SENTRY_DSN from environment")

    async def failure_handler(self, original_exception, duration: float, call_type: str, traceback_str=""):
        """
        Log failed db read/writes

        Currently only logs exceptions to sentry
        """
        ### ALERTING ###
        if AlertType.db_exceptions not in self.alert_types:
            return
        if isinstance(original_exception, HTTPException):
            if isinstance(original_exception.detail, str):
                error_message = original_exception.detail
            elif isinstance(original_exception.detail, dict):
                error_message = json.dumps(original_exception.detail)
            else:
                error_message = str(original_exception)
        else:
            error_message = str(original_exception)
        if isinstance(traceback_str, str):
            error_message += traceback_str[:1000]
        error_message = _redact_string(error_message)
        asyncio.create_task(
            self.alerting_handler(
                message=f"DB read/write call failed: {error_message}",
                level="High",
                alert_type=AlertType.db_exceptions,
                request_data={},
            )
        )

        if hasattr(self, "service_logging_obj"):
            await self.service_logging_obj.async_service_failure_hook(
                service=ServiceTypes.DB,
                duration=duration,
                error=error_message,
                call_type=call_type,
            )

        if litellm.utils.capture_exception:
            litellm.utils.capture_exception(error=original_exception)

    async def post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        error_type: ProxyErrorTypes | None = None,
        route: str | None = None,
        traceback_str: str | None = None,
    ) -> HTTPException | None:
        """
        Allows users to raise custom exceptions/log when a call fails, without having to deal with parsing Request body.
        Callbacks can return or raise HTTPException to transform error responses sent to clients.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation

        Args:
            - request_data: dict - The request data.
            - original_exception: Exception - The original exception.
            - user_api_key_dict: UserAPIKeyAuth - The user api key dict.
            - error_type: Optional[ProxyErrorTypes] - The error type.
            - route: Optional[str] - The route.
            - traceback_str: Optional[str] - The traceback string, sometimes upstream endpoints might need to send the upstream traceback. In which case we use this

        Returns:
            - Optional[HTTPException]: If any callback returns or raises an HTTPException, the first one found is returned.
                                      Otherwise, returns None and the original exception is used.
        """

        ### ALERTING ###
        await self.update_request_status(litellm_call_id=request_data.get("litellm_call_id", ""), status="fail")
        if AlertType.llm_exceptions in self.alert_types and not isinstance(
            original_exception, (HTTPException, ProxyException)
        ):
            """
            Just alert on LLM API exceptions. Do not alert on user errors

            Related issue - https://github.com/BerriAI/litellm/issues/3395
            """
            litellm_debug_info: Final[str | None] = getattr(original_exception, "litellm_debug_info", None)
            exception_str = str(original_exception)
            if litellm_debug_info is not None:
                exception_str += litellm_debug_info

            asyncio.create_task(
                self.alerting_handler(
                    message=_redact_string(f"LLM API call failed: `{exception_str}`"),
                    level="High",
                    alert_type=AlertType.llm_exceptions,
                    request_data=request_data,
                )
            )

        # Auth and pass-through failure bodies are unstripped client input, and
        # the logging handler below flattens body keys into model_call_details,
        # so drop the key before it can masquerade as the built payload.
        request_data.pop("standard_logging_object", None)

        ### LOGGING ###
        if self._is_proxy_only_llm_api_error(
            original_exception=original_exception,
            error_type=error_type,
            route=user_api_key_dict.request_route,
        ):
            await self._handle_logging_proxy_only_error(
                request_data=request_data,
                user_api_key_dict=user_api_key_dict,
                route=route,
                original_exception=original_exception,
            )

        request_data.update(_failure_fields_to_lift(request_data))

        # Remove before callbacks iterate — not serialisable
        request_data.pop("litellm_logging_obj", None)

        # Track the first HTTPException returned or raised by any callback
        transformed_exception: HTTPException | None = None

        for callback in litellm.callbacks:
            try:
                _callback: CustomLogger | None = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback
                if _callback is not None and isinstance(_callback, CustomLogger):
                    try:
                        hook_result = await _callback.async_post_call_failure_hook(
                            request_data=request_data,
                            user_api_key_dict=user_api_key_dict,
                            original_exception=original_exception,
                            traceback_str=traceback_str,
                        )
                        # If callback returned an HTTPException, use it (first one wins)
                        if isinstance(hook_result, HTTPException) and transformed_exception is None:
                            transformed_exception = hook_result
                    except HTTPException as e:
                        # If callback raised an HTTPException, use it (first one wins)
                        if transformed_exception is None:
                            transformed_exception = e
                    except Exception as e:
                        # Log non-HTTPException errors from callbacks but don't break the flow
                        verbose_proxy_logger.exception(
                            "[Non-Blocking] Error in async_post_call_failure_hook callback: %s", e
                        )
            except Exception as e:
                verbose_proxy_logger.exception("[Non-Blocking] Error setting up post_call_failure_hook callback: %s", e)

        return transformed_exception

    def _is_proxy_only_llm_api_error(
        self,
        original_exception: Exception,
        error_type: ProxyErrorTypes | None = None,
        route: str | None = None,
    ) -> bool:
        """
        Return True if the error is a Proxy Only LLM API Error

        Prevents double logging of LLM API exceptions

        e.g should only return True for:
            - Authentication Errors from user_api_key_auth
            - HTTP HTTPException (rate limit errors)
            - ProxyException (guardrail blocks, budget / rate-limit errors)
        """

        #########################################################
        # Only log LLM API and info route errors for proxy level hooks
        # eg. Authentication errors, rate limit errors, etc.
        # Note: This fixes a security issue where we
        #       would log temporary keys/auth info
        #       from management endpoints
        #########################################################
        if route is None:
            return False
        if not (RouteChecks.is_llm_api_route(route) or RouteChecks.is_info_route(route)):
            return False

        return isinstance(original_exception, (HTTPException, ProxyException)) or (
            error_type == ProxyErrorTypes.auth_error
        )

    async def _handle_logging_proxy_only_error(
        self,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        route: str | None = None,
        original_exception: Exception | None = None,
    ):
        """
        Handle logging for proxy only errors by calling `litellm_logging_obj.async_failure_handler`

        Is triggered when self._is_proxy_only_error() returns True
        """
        litellm_logging_obj: Logging | None = request_data.get("litellm_logging_obj", None)
        if litellm_logging_obj is None:
            from litellm._uuid import uuid

            request_data["litellm_call_id"] = str(uuid.uuid4())
            user_api_key_logged_metadata: Final = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
                user_api_key_dict=user_api_key_dict
            )

            litellm_logging_obj, data = litellm.utils.function_setup(
                original_function=route or "IGNORE_THIS",
                rules_obj=litellm.utils.Rules(),
                start_time=datetime.now(),
                **request_data,
            )
            if "metadata" not in request_data:
                request_data["metadata"] = {}
            request_data["metadata"].update(user_api_key_logged_metadata)

        if litellm_logging_obj is not None:
            ## UPDATE LOGGING INPUT
            _optional_params: Final = {}
            _litellm_params: Final = {}

            litellm_param_keys: Final = LoggedLiteLLMParams.__annotations__.keys()
            for k, v in request_data.items():
                if k in litellm_param_keys:
                    _litellm_params[k] = v
                elif k not in ("model", "user", "litellm_logging_obj"):
                    _optional_params[k] = v

            litellm_logging_obj.update_environment_variables(
                model=request_data.get("model", ""),
                user=request_data.get("user", ""),
                optional_params=_optional_params,
                litellm_params=_litellm_params,
            )

            input: list | str | dict = ""
            normalized_call_type: str | None = None
            if "messages" in request_data and isinstance(request_data["messages"], list):
                input = request_data["messages"]
                litellm_logging_obj.model_call_details["messages"] = input
                if litellm_logging_obj.call_type != CallTypes.pass_through.value:
                    normalized_call_type = CallTypes.acompletion.value
            elif "prompt" in request_data and isinstance(request_data["prompt"], str):
                input = request_data["prompt"]
                litellm_logging_obj.model_call_details["prompt"] = input
                if litellm_logging_obj.call_type != CallTypes.pass_through.value:
                    normalized_call_type = CallTypes.atext_completion.value
            elif "input" in request_data and isinstance(request_data["input"], list):
                input = request_data["input"]
                litellm_logging_obj.model_call_details["input"] = input
                if litellm_logging_obj.call_type != CallTypes.pass_through.value:
                    normalized_call_type = CallTypes.aembedding.value
            if normalized_call_type is not None:
                litellm_logging_obj.call_type = normalized_call_type
                litellm_logging_obj.model_call_details["call_type"] = normalized_call_type
            # Pass-through endpoints are logged via the callback loop's
            # async_post_call_failure_hook — skip pre_call and failure handlers.
            if litellm_logging_obj.call_type == CallTypes.pass_through.value:
                return
            # This is a proxy-gate error (auth/rate-limit) for a request that never
            # reached a provider. ``pre_call`` below still fires every callback's
            # input hook so the failure is logged — but tracing callbacks must not
            # fabricate an LLM-call span for a call that did not happen (and, since
            # this runs inside the live ``auth`` phase span, would otherwise nest it
            # under auth). The marker tells them to skip span creation.
            litellm_logging_obj.model_call_details[LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL] = True
            litellm_logging_obj.pre_call(
                input=input,
                api_key="",
            )

            await self._dispatch_proxy_only_failure_handlers(
                litellm_logging_obj=litellm_logging_obj,
                original_exception=original_exception,
            )

    @staticmethod
    async def _dispatch_proxy_only_failure_handlers(
        litellm_logging_obj: Logging,
        original_exception: Exception | None,
    ) -> None:
        """Runs the async failure handler plus the threaded sync handler. Expected
        client (4xx) errors skip traceback formatting unless
        litellm.log_client_error_tracebacks is set."""
        include_traceback: Final = litellm.log_client_error_tracebacks or not is_expected_client_error(
            original_exception
        )
        traceback_str: Final = traceback.format_exc() if include_traceback else ""
        await litellm_logging_obj.async_failure_handler(
            exception=original_exception,
            traceback_exception=traceback_str,
        )

        threading.Thread(
            target=litellm_logging_obj.failure_handler,
            args=(
                original_exception,
                traceback_str,
            ),
            daemon=True,
        ).start()

    async def post_call_success_hook(
        self,
        data: dict,
        response: LLMResponseTypes,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        Allow user to modify outgoing data

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation
        4. /files
        """

        from litellm.proxy.proxy_server import llm_router
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail_callbacks: Final[list[CustomGuardrail]] = []
        other_callbacks: Final[list[CustomLogger]] = []
        try:
            for callback in litellm.callbacks:
                _callback: CustomLogger | None = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback

                if _callback is not None:
                    if isinstance(_callback, CustomGuardrail):
                        guardrail_callbacks.append(_callback)
                    else:
                        other_callbacks.append(_callback)
                    ############## Handle Guardrails ########################################
                    #############################################################################

            # Merge model-level guardrails before checking which guardrails to run
            guardrail_data: Final = _check_and_merge_model_level_guardrails(data=data, llm_router=llm_router)

            parallel_guardrails: Final[tuple[CustomGuardrail, ...]] = tuple(
                callback for callback in guardrail_callbacks if getattr(callback, "run_in_parallel", False)
            )

            for callback in guardrail_callbacks:
                # Main - V2 Guardrails implementation

                if getattr(callback, "run_in_parallel", False):
                    continue

                if (
                    callback.should_run_guardrail(
                        data=guardrail_data,
                        event_type=GuardrailEventHooks.post_call,
                    )
                    is not True
                ):
                    continue

                guardrail_response: Any | None = None

                if "apply_guardrail" in type(callback).__dict__ and not callback.use_native_lifecycle_hooks:
                    data["guardrail_to_apply"] = callback
                    guardrail_response = await self._run_guardrail_with_metrics(
                        callback,
                        unified_guardrail.async_post_call_success_hook(
                            user_api_key_dict=user_api_key_dict,
                            data=data,
                            response=response,
                        ),
                        "post_call",
                    )
                else:
                    guardrail_response = await self._run_guardrail_with_metrics(
                        callback,
                        callback.async_post_call_success_hook(
                            user_api_key_dict=user_api_key_dict,
                            data=data,
                            response=response,
                        ),
                        "post_call",
                    )

                if guardrail_response is not None:
                    response = guardrail_response

            if parallel_guardrails:
                await self._run_parallel_post_call_guardrails(
                    guardrails=parallel_guardrails,
                    data=data,
                    guardrail_data=guardrail_data,
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                )

            ############ Handle CustomLogger ###############################
            #################################################################

            for callback in other_callbacks:
                callback_response: LLMResponseTypes | None = await callback.async_post_call_success_hook(
                    user_api_key_dict=user_api_key_dict, data=data, response=response
                )
                if callback_response is not None:
                    response = callback_response
        except Exception as e:
            raise e
        return response

    async def _run_parallel_post_call_guardrails(
        self,
        guardrails: tuple[CustomGuardrail, ...],
        data: dict,
        guardrail_data: dict,
        response: LLMResponseTypes,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """
        Run opted-in post_call guardrails concurrently against the response
        produced by the sequential guardrails. These guardrails are declared
        block-only, so any modified response they return is discarded; they run
        for their blocking side effect (raising to reject the response before it
        reaches the client). Every guardrail is awaited to completion
        (``return_exceptions=True``) so a raise by one never leaves the others
        running as unobserved background tasks. A guardrail that blocks (any
        exception other than a passthrough) takes precedence over one that only
        changes the response flow, so a fast passthrough can never let a slower
        block be bypassed. Each per-guardrail coroutine sets ``guardrail_to_apply``
        immediately before awaiting, and the unified hook pops it before its first
        suspension point, so concurrent guardrails never race on that key.
        """

        async def _run_one(callback: CustomGuardrail) -> None:
            if callback.should_run_guardrail(data=guardrail_data, event_type=GuardrailEventHooks.post_call) is not True:
                return
            if "apply_guardrail" in type(callback).__dict__ and not callback.use_native_lifecycle_hooks:
                data["guardrail_to_apply"] = callback
                await self._run_guardrail_with_metrics(
                    callback,
                    unified_guardrail.async_post_call_success_hook(
                        user_api_key_dict=user_api_key_dict,
                        data=data,
                        response=response,
                    ),
                    "post_call",
                )
            else:
                await self._run_guardrail_with_metrics(
                    callback,
                    callback.async_post_call_success_hook(
                        user_api_key_dict=user_api_key_dict,
                        data=data,
                        response=response,
                    ),
                    "post_call",
                )

        results: Final = await asyncio.gather(
            *(_run_one(callback) for callback in guardrails),
            return_exceptions=True,
        )
        raised: Final = tuple(result for result in results if isinstance(result, BaseException))
        blocking: Final = next((exc for exc in raised if not _exception_changes_request_flow(exc)), None)
        if blocking is not None:
            raise blocking
        if raised:
            raise raised[0]

    async def post_mcp_call_hook(
        self,
        response: "CallToolResult",
        request_data: Mapping[str, Any],
        user_api_key_dict: UserAPIKeyAuth | None = None,
    ) -> "CallToolResult":
        """
        Run guardrails configured for ``post_mcp_call`` against an MCP tool result.

        The MCP counterpart of ``post_call_success_hook``: guardrails that
        implement ``apply_guardrail`` see the tool result's text through the
        unified guardrail seam (``MCPGuardrailTranslationHandler``), so a text
        guardrail can mask sensitive values in the result without any MCP-specific
        code of its own. Guardrails that instead implement
        ``async_post_mcp_tool_call_hook`` are dispatched by
        ``Logging.async_post_mcp_tool_call_hook`` and are not run here.

        A guardrail that rejects the result raises, and the exception propagates
        (matching the inbound ``pre_mcp_call`` behavior) rather than being
        swallowed into an unguarded result.
        """
        caps: Final = ProxyLogging._callback_capabilities()
        if not caps.has_guardrail:
            return response

        handler_cls: Final = load_guardrail_translation_mappings().get(CallTypes.call_mcp_tool)
        if handler_cls is None:
            verbose_proxy_logger.debug("MCP guardrail translation handler unavailable; skipping post_mcp_call hook")
            return response

        for callback in caps.resolved_callbacks:
            if not isinstance(callback, CustomGuardrail):
                continue
            if "apply_guardrail" not in type(callback).__dict__ or callback.use_native_lifecycle_hooks:
                continue
            if (
                callback.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_mcp_call)
                is not True
            ):
                continue
            response = await self._run_guardrail_with_metrics(
                callback,
                handler_cls().process_output_response(
                    response=response,
                    guardrail_to_apply=callback,
                    litellm_logging_obj=request_data.get("litellm_logging_obj"),
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                ),
                "post_mcp_call",
            )
        return response

    async def post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Calls async_post_call_response_headers_hook on all CustomLogger callbacks.
        Merges all returned header dicts (later callbacks override earlier ones).

        Returns:
            Dict[str, str]: Merged headers from all callbacks.
        """
        merged_headers: Final[dict[str, str]] = {}
        # Outer call sites in common_request_processing.py already gate this
        # call with ``has_post_call_response_headers_callbacks()``. The
        # cached detection makes the redundant interior guard cheap, but the
        # guard would still iterate every code path through this function so
        # keep it cheap and rely on the cached capability lookup.
        if not ProxyLogging._callback_capabilities().has_post_call_response_headers:
            return merged_headers

        try:
            # Build litellm_call_info — normalized routing metadata for callbacks
            litellm_call_info: Final = self._build_litellm_call_info(data=data, response=response)

            for callback in litellm.callbacks:
                _callback: CustomLogger | None = None
                if isinstance(callback, str):
                    _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                        cast(_custom_logger_compatible_callbacks_literal, callback)
                    )
                else:
                    _callback = callback

                if _callback is not None and isinstance(_callback, CustomLogger):
                    if _accepts_litellm_call_info(_callback):
                        result = await _callback.async_post_call_response_headers_hook(
                            data=data,
                            user_api_key_dict=user_api_key_dict,
                            response=response,
                            request_headers=request_headers,
                            litellm_call_info=litellm_call_info,
                        )
                    else:
                        # Backwards compat: callback doesn't accept litellm_call_info
                        result = await _callback.async_post_call_response_headers_hook(
                            data=data,
                            user_api_key_dict=user_api_key_dict,
                            response=response,
                            request_headers=request_headers,
                        )
                    if result is not None:
                        merged_headers.update(result)
        except Exception as e:
            verbose_proxy_logger.exception("Error in post_call_response_headers_hook: %s", str(e))
        return merged_headers

    @staticmethod
    def _build_litellm_call_info(data: dict, response: object) -> dict[str, object]:
        """
        Build a normalized dict of routing metadata from response._hidden_params
        and data, abstracting away the metadata vs litellm_metadata split.
        """
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}

        # model_info: check both metadata keys (chat uses "metadata", responses uses "litellm_metadata")
        model_info: Final = (
            (data.get("metadata") or {}).get("model_info")
            or (data.get("litellm_metadata") or {}).get("model_info")
            or {}
        )

        return {
            "custom_llm_provider": hidden_params.get("custom_llm_provider")
            or getattr(response, "custom_llm_provider", None),
            "model_info": model_info,
            "api_base": hidden_params.get("api_base"),
            "model_id": hidden_params.get("model_id"),
        }

    def is_a2a_streaming_response(self, response: dict) -> bool:
        expected_keys: Final = ["jsonrpc", "id", "result"]
        return all(key in response for key in expected_keys)

    async def async_post_call_streaming_hook(
        self,
        data: dict,
        response: ModelResponse | EmbeddingResponse | ImageResponse | ModelResponseStream,
        user_api_key_dict: UserAPIKeyAuth,
        str_so_far: str | None = None,
    ):
        """
        Allow user to modify outgoing streaming data -> per chunk

        Covers:
        1. /chat/completions
        """
        # Per-chunk fast path: skip the response-string materialization and
        # callback scan when no configured callback overrides
        # ``async_post_call_streaming_hook`` AND no CustomGuardrail is
        # active. ``get_response_string`` walks every choice/delta on the
        # chunk so paying it per chunk for no-op callbacks dominated stream
        # CPU time even after the iterator-chain fix.
        caps: Final = ProxyLogging._callback_capabilities()
        if not caps.has_streaming_chunk_override and not caps.has_guardrail:
            return response

        from litellm.proxy.proxy_server import llm_router

        response_str: str | None = None
        if isinstance(response, (ModelResponse, ModelResponseStream)):
            response_str = litellm.get_response_string(response_obj=response)
        elif isinstance(response, dict) and self.is_a2a_streaming_response(response):
            from litellm.llms.a2a.common_utils import extract_text_from_a2a_response

            response_str = extract_text_from_a2a_response(response)
        if response_str is not None:
            # Cache model-level guardrails check per-request to avoid repeated
            # dict lookups + llm_router.get_deployment() per callback per chunk.
            _cached_guardrail_data: dict | None = None
            _guardrail_data_computed = False

            for callback in litellm.callbacks:
                try:
                    _callback: CustomLogger | None = None
                    if isinstance(callback, CustomGuardrail):
                        # Main - V2 Guardrails implementation
                        from litellm.types.guardrails import GuardrailEventHooks

                        ## CHECK FOR MODEL-LEVEL GUARDRAILS (cached per-request)
                        if not _guardrail_data_computed:
                            _cached_guardrail_data = _check_and_merge_model_level_guardrails(
                                data=data, llm_router=llm_router
                            )
                            _guardrail_data_computed = True

                        if (
                            callback.should_run_guardrail(
                                data=_cached_guardrail_data,
                                event_type=GuardrailEventHooks.post_call,
                            )
                            is not True
                        ):
                            continue
                    if isinstance(callback, str):
                        _callback = litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class(
                            cast(_custom_logger_compatible_callbacks_literal, callback)
                        )
                    else:
                        _callback = callback
                    if _callback is not None and isinstance(_callback, CustomLogger):
                        if str_so_far is not None:
                            complete_response = str_so_far + response_str
                        else:
                            complete_response = response_str
                        callback_response: (
                            ModelResponse | EmbeddingResponse | ImageResponse | ModelResponseStream | None
                        )
                        callback_response = await _callback.async_post_call_streaming_hook(
                            user_api_key_dict=user_api_key_dict,
                            response=complete_response,
                        )
                        if callback_response is not None:
                            response = callback_response
                except Exception as e:
                    raise e
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        response,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,
    ):
        """
        Allow user to modify outgoing streaming data -> Given a whole response iterator.
        This hook is best used when you need to modify multiple chunks of the response at once.

        Covers:
        1. /chat/completions
        """
        caps: Final = ProxyLogging._callback_capabilities()
        # Fast path: no real overrides. Internal proxy CustomLogger callbacks
        # (e.g. _PROXY_MaxBudgetLimiter, ManagedFiles) inherit the default
        # ``async for chunk: yield chunk`` body, so wrapping the iterator
        # through each of them adds N pass-through trampolines per chunk for
        # zero behavior change. Skip the chain entirely and stream through.
        if not caps.iterator_overrides:
            async for chunk in response:
                yield chunk
            ProxyLogging._fire_deferred_stream_logging(request_data)
            return

        from litellm.proxy.proxy_server import llm_router

        # Merge model-level guardrails before checking which guardrails to run
        request_data = _check_and_merge_model_level_guardrails(data=request_data, llm_router=llm_router)

        current_response = response
        stream_needs_translation: Final = ProxyLogging._stream_requires_guardrail_translation(user_api_key_dict)

        for resolved_callback, kind in caps.iterator_overrides:
            if isinstance(resolved_callback, CustomGuardrail):
                if (
                    resolved_callback.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call)
                    is not True
                ):
                    continue
            effective_kind = (
                "apply_guardrail"
                if (
                    kind == "override"
                    and stream_needs_translation
                    and isinstance(resolved_callback, CustomGuardrail)
                    and resolved_callback.uses_apply_guardrail_interface()
                    and getattr(resolved_callback, "use_native_lifecycle_hooks", False) is not True
                    and not resolved_callback.mask_response_content
                )
                else kind
            )
            if effective_kind == "override":
                current_response = self._wrap_streaming_iterator_with_enrichment(
                    resolved_callback,
                    resolved_callback.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user_api_key_dict,
                        response=current_response,
                        request_data=request_data,
                    ),
                )
            else:
                # kind == "apply_guardrail": route through unified_guardrail
                current_response = self._wrap_streaming_iterator_with_enrichment(
                    resolved_callback,
                    unified_guardrail.async_post_call_streaming_iterator_hook(
                        user_api_key_dict=user_api_key_dict,
                        request_data=request_data,
                        response=current_response,
                        guardrail_to_apply=resolved_callback,
                        buffer_until_moderated_default=(kind == "override"),
                    ),
                )

        # Actually iterate through the chained async generator and yield chunks
        async for chunk in current_response:
            yield chunk

        # Fire deferred logging AFTER all guardrail end-of-stream blocks
        # completed.  unified_guardrail writes guardrail_information during
        # its end-of-stream block (inside current_response), so by the time
        # we reach this point the metadata is fully populated.
        ProxyLogging._fire_deferred_stream_logging(request_data)

    @staticmethod
    def _fire_deferred_stream_logging(request_data: dict) -> None:
        """
        Fire the deferred streaming logging callback after the full streaming
        pipeline (including guardrail end-of-stream blocks) has completed.

        CSW.__anext__ stores the callback and args on logging_obj instead of
        scheduling via create_task (which would race with unified_guardrail's
        end-of-stream block).  This method retrieves and fires them.
        """
        logging_obj: Final = request_data.get("litellm_logging_obj")
        if logging_obj is None:
            return
        _deferred_cb: Final[Callable[..., Coroutine[object, object, object]] | None] = getattr(
            logging_obj, "_on_deferred_stream_complete", None
        )
        _args: Final[tuple[object, ...] | None] = getattr(logging_obj, "_deferred_stream_complete_args", None)
        if _deferred_cb is not None and _args is not None:
            logging_obj._on_deferred_stream_complete = None
            logging_obj._deferred_stream_complete_args = None
            asyncio.create_task(_deferred_cb(*_args))

    async def _arelease_max_parallel_requests_on_disconnect(
        self,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """
        Release the api-key max_parallel_requests slot when a streaming
        response is cancelled mid-flight (client disconnect) and no logging
        callback fired for it. Neither the success nor failure callback runs on
        the resulting CancelledError / GeneratorExit, so the pre-call +1 would
        otherwise leak.

        Awaited from the shielded streaming cleanup rather than scheduled
        fire-and-forget, so the caller can make it the single owner of the
        release: when a disconnect-time success event does fire (partial-spend
        billing or a deferred-guardrail flush), that event's own limiter
        callback releases the slot and this is not called at all. Two
        concurrent releases of the same acquisition would otherwise race and
        double-decrement under the limiter's in-memory fallback.
        """
        limiter: Final = self.get_proxy_hook("parallel_request_limiter")
        if not isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
            return
        await limiter.async_release_max_parallel_requests_on_disconnect(user_api_key_dict)

    def _init_response_taking_too_long_task(self, data: dict | None = None):
        """
        Initialize the response taking too long task if user is using slack alerting

        Only run task if user is using slack alerting

        This handles checking for if a request is hanging for too long
        """
        ## ALERTING ###
        if self.slack_alerting_instance and self.slack_alerting_instance.alerting is not None:
            asyncio.create_task(self.slack_alerting_instance.response_taking_too_long(request_data=data))


### DB CONNECTOR ###
# Define the retry decorator with backoff strategy
# Function to be called whenever a retry is about to happen
def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    print_verbose(f"Backing off... this was attempt #{details['tries']}")


def jsonify_object(data: dict) -> dict:
    db_data: Final = copy.deepcopy(data)

    for k, v in db_data.items():
        if isinstance(v, dict):
            try:
                db_data[k] = json.dumps(v)
            except Exception:
                # This avoids Prisma retrying this 5 times, and making 5 clients
                db_data[k] = "failed-to-serialize-json"
    return db_data


# In-memory cache for deprecated key lookups:
# maps old_token_hash -> (active_token_id, cache_expires_at_ts, revoke_at_ts).
# Avoids a DB query on every auth request for non-deprecated keys.
# Bounded to prevent memory leaks from accumulated rotations.
_deprecated_key_cache: Final[LimitedSizeOrderedDict] = LimitedSizeOrderedDict(max_size=1000)
_DEPRECATED_KEY_CACHE_TTL_SECONDS: Final = 60


async def _lookup_deprecated_key(
    db: PrismaWrapper | RoutingPrismaWrapper,
    hashed_token: str,
) -> str | None:
    """
    Check if a token exists in the deprecated keys table and is still within its grace period.

    Returns the active_token_id if found and valid, otherwise None.
    Uses an in-memory cache to avoid DB queries on every auth request.
    """
    now: Final = datetime.now(timezone.utc)
    now_ts: Final = now.timestamp()

    # Check cache first
    cached: Final = _deprecated_key_cache.get(hashed_token)
    if cached is not None:
        active_token_id, cache_expires_at_ts, revoke_at_ts = cached
        if now_ts < cache_expires_at_ts and now_ts < revoke_at_ts:
            return active_token_id
        _deprecated_key_cache.pop(hashed_token, None)

    try:
        deprecated_keys_table: Final[
            LiteLLM_DeprecatedVerificationTokenActions[LiteLLM_DeprecatedVerificationToken]
        ] = db.litellm_deprecatedverificationtoken
        deprecated_row: Final = await deprecated_keys_table.find_first(
            where={
                "token": hashed_token,
                "revoke_at": {"gt": now},
            }
        )
        if deprecated_row and deprecated_row.active_token_id:
            revoke_at: Final = deprecated_row.revoke_at
            _deprecated_key_cache[hashed_token] = (
                deprecated_row.active_token_id,
                now_ts + _DEPRECATED_KEY_CACHE_TTL_SECONDS,
                revoke_at.timestamp(),
            )
            return deprecated_row.active_token_id
        # Only cache positive results; negative lookups are fast on indexed columns
        # and caching them risks evicting real deprecated key entries.
    except Exception as e:
        verbose_proxy_logger.debug("Deprecated key lookup skipped: %s", e)

    return None


# DualCache for LiteLLM_Config param_name reads.
# Redis layer is attached in proxy_server._init_cache.
LITELLM_CONFIG_CACHE_TTL_SECONDS: Final[int] = int(os.environ.get("LITELLM_CONFIG_PARAM_CACHE_TTL_SECONDS", "60"))
_CONFIG_CACHE_MISS: Final[str] = "__litellm_config_param_miss__"

litellm_config_cache: Final[DualCache] = DualCache(
    default_in_memory_ttl=LITELLM_CONFIG_CACHE_TTL_SECONDS,
    default_redis_ttl=LITELLM_CONFIG_CACHE_TTL_SECONDS,
)


class _ConfigRow:
    """Mimics the Prisma litellm_config row shape for cached entries."""

    __slots__ = ("param_name", "param_value")

    def __init__(self, param_name: str, param_value: Any) -> None:
        self.param_name = param_name
        self.param_value = param_value


def _config_cache_key(param_name: str) -> str:
    return f"litellm_config:param:{param_name}"


def _pack_config_row(row: Any) -> dict[str, object]:
    return {"param_name": row.param_name, "param_value": row.param_value}


def _unpack_config_row(cached: Any) -> _ConfigRow | None:
    if cached is None or cached == _CONFIG_CACHE_MISS:
        return None
    if isinstance(cached, dict):
        return _ConfigRow(cached["param_name"], cached["param_value"])
    return None


async def get_config_param(prisma_client: "PrismaClient", param_name: str) -> Any | None:
    """Cached read of a LiteLLM_Config row; returns row, _ConfigRow shim, or None."""
    cache_key: Final = _config_cache_key(param_name)
    cached: Final = await litellm_config_cache.async_get_cache(cache_key)
    if cached is not None:
        return _unpack_config_row(cached)

    row: Final = await prisma_client.get_generic_data(key="param_name", value=param_name, table_name="config")
    cache_value: Final[Mapping[str, object] | str] = _pack_config_row(row) if row is not None else _CONFIG_CACHE_MISS
    await litellm_config_cache.async_set_cache(cache_key, cache_value, ttl=LITELLM_CONFIG_CACHE_TTL_SECONDS)
    return row


async def evict_config_param(param_name: str) -> None:
    await litellm_config_cache.async_delete_cache(_config_cache_key(param_name))


async def invalidate_config_param(param_name: str) -> None:
    """Evict from both cache layers; call after every LiteLLM_Config write."""
    await evict_config_param(param_name)
    await publish_config_param_change(param_name)


async def prefetch_config_params(prisma_client: "PrismaClient | None", param_names: list[str]) -> None:
    """Batch-load LiteLLM_Config rows into the cache with one find_many."""
    if not param_names:
        return
    try:
        config_table: Final = cast(  # cast-ok: ConfigRepository.table is prisma's litellm_config actions object
            "TableActions[prisma_models.LiteLLM_Config]", ConfigRepository(prisma_client).table
        )
        rows: Final = await config_table.find_many(where={"param_name": {"in": param_names}})
    except Exception as e:
        verbose_proxy_logger.debug(
            "prefetch_config_params failed, falling through to per-param queries: %s",
            e,
        )
        return
    by_name: Final = {row.param_name: row for row in rows}
    for name in param_names:
        row = by_name.get(name)
        cache_value: Mapping[str, object] | str = _pack_config_row(row) if row is not None else _CONFIG_CACHE_MISS
        await litellm_config_cache.async_set_cache(
            _config_cache_key(name), cache_value, ttl=LITELLM_CONFIG_CACHE_TTL_SECONDS
        )


class _ForcedRecreateDeclined(Exception):
    """A forced recreate was declined by the engine-generation guard.

    Distinct from a reconnect *failure*: the machinery worked, it just found
    that another path had already replaced the writer, so it left the engines
    alone. The caller's engine may still be poisoned, so the cycle must not
    report success, but it must not count as a failure either, or the record
    of what could not be repaired would gate the retry that recovers.
    """


@dataclass(frozen=True, slots=True)
class _StaleReadEngine:
    """The read engine a query observed, identified rather than only counted.

    `PrismaClient.read_db` resolves to the reader while it is available and to
    the writer once it is not, and the two carry independent generation
    counters that both start at zero and advance on the same reconnect
    cadence. A bare generation compared across that switch would silently pit
    one engine's counter against another's, so the wrapper is carried with the
    number and a switch counts as the engine having moved.

    Holding the wrapper itself rather than its `id()` is load-bearing, not
    incidental: the strong reference keeps the wrapper alive, so its address
    cannot be recycled under a stored observation and match an unrelated
    engine later. It is only free because writer and reader both live as long
    as the client does; a replaceable reader would make this a retention leak.
    """

    wrapper: PrismaWrapper
    generation: int

    @classmethod
    def observe(cls, wrapper: PrismaWrapper) -> "_StaleReadEngine":
        return cls(wrapper=wrapper, generation=wrapper.engine_generation)

    def is_still_live(self, current: PrismaWrapper) -> bool:
        """Whether this exact engine is still serving reads, unreplaced.

        A True answer must never be the only thing standing between a poisoned
        engine and its repair. The generation moves only after a replacement
        connects, and a recreate whose connect raises leaves it unmoved until
        some later recreate succeeds, so this can report an engine as live
        after it has stopped working. What bounds that is the failed-repair
        record in `_cooldown_applies`, written by a repair attempt that fails
        rather than by whatever broke the engine: the two need not be the same
        recreate, since the synchronous token-refresh fallback in
        `PrismaWrapper.__getattr__` recreates outside the reconnect machinery
        and records nothing. The record is written only for callers that named
        an engine, and it collapses the rest of the burst for up to one
        cooldown window rather than guaranteeing a repair, since the cooldown
        conjunct underneath it still expires and lets a later caller retry.
        """
        return self.wrapper is current and self.generation == current.engine_generation


class PrismaClient:
    spend_log_transactions: list = []
    _spend_log_transactions_lock = asyncio.Lock()
    spend_log_flush_requested: ClassVar[asyncio.Event] = asyncio.Event()
    spend_log_queue_bytes: ClassVar[int] = 0
    spend_logs_queue_monitor_task: "asyncio.Task[None] | None" = None
    tool_usage_transactions: list["ToolUsageTransaction"] = []
    _tool_usage_transactions_lock = asyncio.Lock()
    autorouter_turn_transactions: ClassVar[
        list["AutoRouterTurnTransaction"]
    ] = []  # mutable-ok: drained queue, mirrors tool_usage_transactions
    _autorouter_turn_transactions_lock = asyncio.Lock()

    # How long a health probe failure waits for an in-flight planned engine
    # replacement to settle before deciding whether to report itself. Generous
    # against a replacement that takes well under a second, and far short of the
    # reconnect budget an outage-hung `connect()` runs under, so a real outage
    # is never waited out.
    PLANNED_ENGINE_REPLACEMENT_SETTLE_SECONDS: ClassVar[float] = 5.0

    def __init__(
        self,
        database_url: str,
        proxy_logging_obj: ProxyLogging,
        http_client: "HttpConfig | None" = None,
    ):
        ## init logging object
        self.proxy_logging_obj = proxy_logging_obj
        self.token_auth: DatabaseTokenAuth | None = resolve_database_token_auth()
        verbose_proxy_logger.debug("Creating Prisma Client..")
        try:
            from prisma import Prisma
        except Exception as e:
            verbose_proxy_logger.error("Failed to import Prisma client: %s", e)
            verbose_proxy_logger.error("This usually means 'prisma generate' hasn't been run yet.")
            verbose_proxy_logger.error("Please run 'prisma generate' to generate the Prisma client.")
            raise Exception("Unable to find Prisma binaries. Please run 'prisma generate' first.")
        token_auth: Final = self.token_auth
        # When read-replica routing is on, tag log lines with [writer]/[reader]
        # so the two wrappers' interleaved token refresh logs can be told apart.
        # Single-DB deployments get an empty prefix (logs unchanged).
        read_replica_url = os.getenv("DATABASE_URL_READ_REPLICA")
        writer_log_prefix: Final = "[writer]" if read_replica_url else ""
        if http_client is not None:
            writer_wrapper = PrismaWrapper(
                original_prisma=Prisma(http=http_client),
                token_auth=token_auth,
                log_prefix=writer_log_prefix,
            )
        else:
            writer_wrapper = PrismaWrapper(
                original_prisma=Prisma(),
                token_auth=token_auth,
                log_prefix=writer_log_prefix,
            )

        # Optional read-replica routing. When DATABASE_URL_READ_REPLICA is set,
        # reads (find_*, count, group_by, query_raw/_first) are routed to the
        # reader endpoint and writes stay on the writer. Falls back to the
        # writer-only wrapper when the env var is unset, preserving existing
        # single-DB deployments.
        self.db: PrismaWrapper | RoutingPrismaWrapper
        if read_replica_url:
            try:
                # If token auth is enabled, the reader refreshes its own token on
                # the same cadence as the writer. We parse the static endpoint
                # pieces (host/port/user/db) once from the reader URL — only
                # the token rotates after that.
                reader_iam_endpoint: Final = (
                    parse_iam_endpoint_from_url(read_replica_url) if token_auth is not None else None
                )
                # Mint a fresh token for the reader BEFORE constructing the
                # Prisma client. Mirrors what `proxy_cli.py` already does for
                # the writer — without this, the reader Prisma is built with
                # whatever placeholder URL the user supplied (no real token),
                # and the first query falls through to the synchronous fallback
                # path in `PrismaWrapper.__getattr__`, which deadlocks the event
                # loop and times out after 30s.
                if token_auth is not None and reader_iam_endpoint is not None:
                    reader_token: Final = mint_database_token(token_auth, reader_iam_endpoint)
                    read_replica_url = reader_iam_endpoint.build_url(reader_token)
                    os.environ["DATABASE_URL_READ_REPLICA"] = read_replica_url
                reader_kwargs: Final[dict[str, Any]] = {"datasource": {"url": read_replica_url}}
                if http_client is not None:
                    reader_prisma = Prisma(http=http_client, **reader_kwargs)
                else:
                    reader_prisma = Prisma(**reader_kwargs)
                reader_wrapper: Final = PrismaWrapper(
                    original_prisma=reader_prisma,
                    token_auth=token_auth,
                    db_url_env_var="DATABASE_URL_READ_REPLICA",
                    iam_endpoint=reader_iam_endpoint,
                    recreate_uses_datasource=True,
                    log_prefix="[reader]",
                )
                self.db = RoutingPrismaWrapper(writer=writer_wrapper, reader=reader_wrapper)
                verbose_proxy_logger.info(
                    "PrismaClient: read-replica routing enabled via DATABASE_URL_READ_REPLICA"
                    + (f" (with {token_auth.label} auto-refresh)" if token_auth is not None else "")
                )
            except Exception as e:
                # Reader is opt-in; never let its construction fail proxy
                # startup. Mirrors the runtime contract from
                # `RoutingPrismaWrapper.connect`: reader-side failures are
                # logged and we keep serving traffic via the writer alone.
                # This recovers from transient credential-provider hiccups
                # during the reader token mint, malformed DATABASE_URL_READ_REPLICA,
                # and Prisma construction errors. Operator restart is required
                # to retry read-routing once the underlying issue is resolved.
                verbose_proxy_logger.warning(
                    "Failed to initialize read replica Prisma client: %s. "
                    "Falling back to writer-only mode (no read routing) until proxy restart.",
                    e,
                )
                self.db = writer_wrapper
        else:
            self.db = writer_wrapper  # Client to connect to Prisma db
        self._db_reconnect_lock = asyncio.Lock()
        self._db_health_watchdog_task: asyncio.Task | None = None
        self._db_last_reconnect_attempt_ts: float = 0.0
        self._db_reconnect_cooldown_seconds: int = max(1, int(os.getenv("PRISMA_RECONNECT_COOLDOWN_SECONDS", "15")))
        self._db_health_watchdog_interval_seconds: int = max(
            5, int(os.getenv("PRISMA_HEALTH_WATCHDOG_INTERVAL_SECONDS", "30"))
        )
        self._db_health_watchdog_enabled: bool = (
            str_to_bool(os.getenv("PRISMA_HEALTH_WATCHDOG_ENABLED", "true")) is True
        )
        self._db_health_watchdog_probe_timeout_seconds: float = max(
            0.5,
            float(os.getenv("PRISMA_HEALTH_WATCHDOG_PROBE_TIMEOUT_SECONDS", "5.0")),
        )
        self._db_watchdog_reconnect_timeout_seconds: float = max(
            1.0, float(os.getenv("PRISMA_WATCHDOG_RECONNECT_TIMEOUT_SECONDS", "30.0"))
        )
        self._db_auth_reconnect_timeout_seconds: float = max(
            0.5, float(os.getenv("PRISMA_AUTH_RECONNECT_TIMEOUT_SECONDS", "2.0"))
        )
        self._db_auth_reconnect_lock_timeout_seconds: float = max(
            0.0,
            float(os.getenv("PRISMA_AUTH_RECONNECT_LOCK_TIMEOUT_SECONDS", "0.1")),
        )
        self._consecutive_reconnect_failures: int = 0
        # Last generation of each read engine whose repair was attempted and
        # failed. Scoped to the engine rather than counted globally so an
        # unrelated reconnect failure cannot suppress a stale reader's
        # recovery, and keyed per wrapper rather than held in one slot so a
        # writer failure cannot evict the reader's record and hand the waiver
        # back to a caller whose engine is still unrepaired. Bounded at two
        # entries: a client has one writer and at most one reader.
        self._failed_recreate_generations: Mapping[PrismaWrapper, int] = MappingProxyType({})
        self._reconnect_escalation_threshold: int = max(1, int(os.getenv("PRISMA_RECONNECT_ESCALATION_THRESHOLD", "3")))
        self._engine_pidfd: int = -1
        self._engine_pid: int = 0
        self._watching_engine: bool = False
        self._engine_confirmed_dead: bool = False
        self._engine_wait_thread: threading.Thread | None = None
        verbose_proxy_logger.debug("Success - Created Prisma Client")

    @property
    def writer_db(self) -> PrismaWrapper:
        """Underlying writer Prisma wrapper, regardless of read-replica routing."""
        if isinstance(self.db, RoutingPrismaWrapper):
            return self.db.writer
        return self.db

    @property
    def read_db(self) -> PrismaWrapper:
        """Underlying wrapper that top-level reads are dispatched to.

        Identical to `writer_db` without a read replica. With one configured
        it is the reader, which is the engine `query_first` actually runs on,
        so anything reasoning about the state of the connection that served a
        read has to consult this rather than the writer.
        """
        if isinstance(self.db, RoutingPrismaWrapper):
            return self.db.read_target
        return self.db

    def tx(self) -> "TransactionManager":
        """Open an interactive transaction on the writer.

        Callers go through this instead of reaching into ``self.db`` so writer
        selection and read-replica routing stay encapsulated in the wrapper.
        """
        return cast("TransactionManager", self.db.tx())  # cast-ok: wrappers delegate tx via __getattr__ (untyped)

    def get_request_status(self, payload: dict | SpendLogsPayload) -> Literal["success", "failure"]:
        """
        Determine if a request was successful or failed based on payload metadata.

        Args:
            payload (Union[dict, SpendLogsPayload]): Request payload containing metadata

        Returns:
            Literal["success", "failure"]: Request status
        """
        try:
            # Get metadata and convert to dict if it's a JSON string
            payload_metadata: Final[dict | SpendLogsMetadata | str] = payload.get("metadata", {})
            if isinstance(payload_metadata, str):
                payload_metadata_json: dict | SpendLogsMetadata = cast(dict, json.loads(payload_metadata))
            else:
                payload_metadata_json = payload_metadata

            # Check status in metadata dict
            return "failure" if payload_metadata_json.get("status") == "failure" else "success"

        except (json.JSONDecodeError, AttributeError):
            # Default to success if metadata parsing fails
            return "success"

    def hash_token(self, token: str):
        # Hash the string using SHA-256
        hashed_token: Final = hashlib.sha256(token.encode()).hexdigest()

        return hashed_token

    def jsonify_object(self, data: Mapping[str, object]) -> dict[str, object]:
        db_data: Final[dict[str, object]] = copy.deepcopy(dict(data))

        for k, v in db_data.items():
            if isinstance(v, dict):
                try:
                    db_data[k] = json.dumps(v)
                except Exception:
                    # This avoids Prisma retrying this 5 times, and making 5 clients
                    db_data[k] = "failed-to-serialize-json"
        return db_data

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def check_view_exists(self):
        """
        Checks if the LiteLLM_VerificationTokenView and MonthlyGlobalSpend exists in the user's db.

        LiteLLM_VerificationTokenView: This view is used for getting the token + team data in user_api_key_auth

        MonthlyGlobalSpend: This view is used for the admin view to see global spend for this month

        If the view doesn't exist, one will be created.
        """

        # Check to see if all of the necessary views exist and if they do, simply return
        # This is more efficient because it lets us check for all views in one
        # query instead of multiple queries.
        try:
            expected_views: Final = [
                "LiteLLM_VerificationTokenView",
                "MonthlyGlobalSpend",
                "Last30dKeysBySpend",
                "Last30dModelsBySpend",
                "MonthlyGlobalSpendPerKey",
                "MonthlyGlobalSpendPerUserPerKey",
                "Last30dTopEndUsersSpend",
                "DailyTagSpend",
            ]
            required_view: Final = "LiteLLM_VerificationTokenView"
            expected_views_str: Final = ", ".join(f"'{view}'" for view in expected_views)
            pg_schema: Final = os.getenv("DATABASE_SCHEMA", "public")
            ret: Final[Sequence[_ViewCountRow]] = await self.db.query_raw(f"""
                WITH existing_views AS (
                    SELECT viewname
                    FROM pg_views
                    WHERE schemaname = '{pg_schema}' AND viewname IN (
                        {expected_views_str}
                    )
                )
                SELECT
                    (SELECT COUNT(*) FROM existing_views) AS view_count,
                    ARRAY_AGG(viewname) AS view_names
                FROM existing_views
                """)
            expected_total_views: Final = len(expected_views)
            if ret[0]["view_count"] == expected_total_views:
                verbose_proxy_logger.info("All necessary views exist!")
                return
            else:
                ## check if required view exists ##
                if ret[0]["view_names"] and required_view not in ret[0]["view_names"]:
                    await self.health_check()  # make sure we can connect to db
                    await create_view_tolerating_race(
                        self.db,
                        "LiteLLM_VerificationTokenView",
                        """
                            CREATE VIEW "LiteLLM_VerificationTokenView" AS
                            SELECT
                            v.*,
                            t.spend AS team_spend,
                            t.max_budget AS team_max_budget,
                            t.tpm_limit AS team_tpm_limit,
                            t.rpm_limit AS team_rpm_limit
                            FROM "LiteLLM_VerificationToken" v
                            LEFT JOIN "LiteLLM_TeamTable" t ON v.team_id = t.team_id;
                        """,
                    )
                else:
                    should_create_views: Final = await should_create_missing_views(db=self.db)
                    if should_create_views:
                        await create_missing_views(db=self.db)
                    else:
                        # don't block execution if these views are missing
                        # Convert lists to sets for efficient difference calculation
                        ret_view_names_set: Final = set(ret[0]["view_names"]) if ret[0]["view_names"] else set()
                        expected_views_set: Final = set(expected_views)
                        # Find missing views
                        missing_views: Final = expected_views_set - ret_view_names_set

                        verbose_proxy_logger.warning(
                            "\n\n\x1b[93mNot all views exist in db, needed for UI 'Usage' tab. Missing=%s.\nRun 'create_views.py' from https://github.com/BerriAI/litellm/tree/main/db_scripts to create missing views.\x1b[0m\n",
                            missing_views,
                        )

        except Exception:
            raise
        return

    @log_db_metrics
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=1,  # maximum number of retries
        max_time=2,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def get_generic_data(
        self,
        key: str,
        value: object,
        table_name: Literal["users", "keys", "config", "spend"],
    ):
        """
        Generic implementation of get data.

        Self-heals across a single transient transport blip via
        `call_with_db_reconnect_retry`: on `httpx.ReadError` /
        `ClientNotConnectedError` / similar, attempt one DB reconnect and
        retry once before surfacing the failure. Restores the 1.82.6 behavior
        that was lost in 1.83.x — see issue #25143.
        """
        start_time: Final = time.time()

        async def _do_query():
            if table_name == "users":
                return await UserRepository(self).table.find_first(where={key: value})
            elif table_name == "keys":
                return await VerificationTokenRepository(self).table.find_first(where={key: value})
            elif table_name == "config":
                config_table: Final = cast(  # cast-ok: ConfigRepository.table is prisma's litellm_config actions object
                    "TableActions[prisma_models.LiteLLM_Config]", ConfigRepository(self).table
                )
                return await config_table.find_first(where={key: value})
            elif table_name == "spend":
                return await self.db.l.find_first(where={key: value})
            return None

        try:
            return await call_with_db_reconnect_retry(
                self,
                _do_query,
                reason=f"prisma_get_generic_data_{table_name}_lookup_failure",
            )
        except Exception as e:
            error_msg = f"LiteLLM Prisma Client Exception get_generic_data: {e}"
            verbose_proxy_logger.error(error_msg)
            error_msg = error_msg + f"\nException Type: {type(e)}"
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    traceback_str=error_traceback,
                    call_type="get_generic_data",
                )
            )

            raise e

    async def _query_first_with_cached_plan_fallback(self, sql_query: str, *args) -> dict | None:
        """
        Execute a query, recovering once from PostgreSQL's "cached plan must not
        change result type" error.

        That error surfaces during rolling deployments when a schema change
        invalidates the prepared-statement plans that pooled connections still
        hold. Clearing only the server-side plans with DEALLOCATE ALL makes
        things worse: Prisma's query engine keeps a per-connection client-side
        cache of prepared-statement names, so once the server drops a plan the
        engine re-sends a name PostgreSQL no longer recognizes and the
        connection breaks with `prepared statement "sN" does not exist`. With a
        small pool that connection stays poisoned and every auth lookup fails.

        Recreating the Prisma client kills the engine subprocess and drops the
        server-side plans and the engine's client-side name cache together, so
        the retried query is prepared fresh. We reconnect through
        `attempt_db_reconnect`, which is singleflight: when a schema change
        poisons every pooled connection at once, the first cached-plan error
        recreates the client and the concurrent waiters reuse that single
        recreate instead of racing to kill each other's fresh engine. We pass
        `force_recreate` so the reconnect skips its `SELECT 1` liveness probe:
        the connection is healthy here, it is the prepared statements on it
        that are stale, so a passing probe would otherwise skip the recreate
        and leave the retry to hit the same error. We then retry the identical
        query exactly once.

        The retry reuses the original query byte-for-byte. Mutating the SQL
        (e.g. injecting a unique comment) would defeat PostgreSQL's plan cache,
        forcing a fresh plan on every request and pegging the database CPU.

        The reconnect cooldown must not gate the engine this query itself saw
        as stale, or a migration landing within the cooldown of an earlier
        reconnect leaves auth failing until it elapses. The engine observed
        before the query names it, so the reconnect bypasses the cooldown only
        while that same engine is still the live one.

        It is observed from `read_db`, not `writer_db`: `query_first` is a
        top-level read, so with a read replica configured it runs on the reader
        and it is the reader's prepared statements that went stale. Naming the
        writer here would let an unrelated writer reconnect re-arm the cooldown
        while the reader stayed poisoned.
        """
        stale_read_engine: Final = _StaleReadEngine.observe(self.read_db)
        try:
            return await self.db.query_first(sql_query, *args)
        except Exception as e:
            if "cached plan must not change result type" not in str(e):
                raise
            verbose_proxy_logger.warning(
                "PostgreSQL cached plan error detected for token lookup; "
                "recreating the database connection and retrying with the same "
                "query. This may occur during rolling deployments when schema "
                "changes are applied."
            )
            await self.attempt_db_reconnect(
                reason="postgres_cached_plan_error",
                force_recreate=True,
                stale_read_engine=stale_read_engine,
            )
            return await self.db.query_first(sql_query, *args)

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    @log_db_metrics
    async def get_data(
        self,
        token: str | list | None = None,
        user_id: str | None = None,
        user_id_list: Sequence[str] | None = None,
        team_id: str | None = None,
        team_id_list: Sequence[str] | None = None,
        key_val: dict | None = None,
        table_name: Literal[
            "user", "key", "config", "spend", "enduser", "budget", "team", "user_notification", "combined_view"
        ]
        | None = None,
        query_type: Literal["find_unique", "find_all"] = "find_unique",
        expires: datetime | None = None,
        reset_at: datetime | None = None,
        offset: int | None = None,  # pagination, what row number to start from
        limit: int | None = None,  # pagination, number of rows to getch when find_all==True
        parent_otel_span: Span | None = None,
        proxy_logging_obj: ProxyLogging | None = None,
        budget_id_list: list[str] | None = None,
        check_deprecated: bool = True,
    ):
        args_passed_in: Final = locals()
        start_time: Final = time.time()
        hashed_token: str | None = None
        try:
            response: Any = None
            if (token is not None and table_name is None) or (table_name is not None and table_name == "key"):
                # check if plain text or hash
                if token is not None:
                    if isinstance(token, str):
                        hashed_token = _hash_token_if_needed(token=token)
                        verbose_proxy_logger.debug("PrismaClient: find_unique for token: %s", hashed_token)
                if query_type == "find_unique" and hashed_token is not None:
                    if token is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"No token passed in. Token={token}"},
                        )
                    response = await VerificationTokenRepository(self).table.find_unique(
                        where={"token": hashed_token},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None:
                        # for prisma we need to cast the expires time to str
                        if response.expires is not None and isinstance(response.expires, datetime):
                            response.expires = response.expires.isoformat()
                    else:
                        # Token does not exist.
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Authentication Error: invalid user key - user key does not exist in db. User Key={token}",
                        )
                elif query_type == "find_all" and user_id is not None:
                    response = await VerificationTokenRepository(self).table.find_many(
                        where={"user_id": user_id},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all" and team_id is not None:
                    response = await VerificationTokenRepository(self).table.find_many(
                        take=limit,
                        where={"team_id": team_id},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all" and expires is not None and reset_at is not None:
                    response = await VerificationTokenRepository(self).table.find_many(
                        take=limit,
                        where={
                            "OR": [
                                {"expires": None},
                                {"expires": {"gt": expires}},
                            ],
                            "budget_reset_at": {"lt": reset_at},
                            "NOT": {"budget_duration": None},
                        },
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all":
                    where_filter: Final[dict[str, dict[str, Sequence[str]]]] = {}
                    if token is not None:
                        where_filter["token"] = {}
                        if isinstance(token, str):
                            token = _hash_token_if_needed(token=token)
                            where_filter["token"]["in"] = [token]
                        elif isinstance(token, list):
                            hashed_tokens: Final[list[str]] = []
                            for t in token:
                                assert isinstance(t, str)
                                if t.startswith("sk-"):
                                    new_token = self.hash_token(token=t)
                                    hashed_tokens.append(new_token)
                                else:
                                    hashed_tokens.append(t)
                            where_filter["token"]["in"] = hashed_tokens
                    response = await VerificationTokenRepository(self).table.find_many(
                        order={"spend": "desc"},
                        where=where_filter,
                        include={"litellm_budget_table": True},
                    )
                if response is not None:
                    return response
                else:
                    # Token does not exist.
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication Error: invalid user key - token does not exist",
                    )
            elif (user_id is not None and table_name is None) or (table_name is not None and table_name == "user"):
                if query_type == "find_unique":
                    if key_val is None:
                        key_val = {"user_id": user_id}

                    response = await UserRepository(self).table.find_unique(
                        where=key_val,
                        include={"organization_memberships": True},
                    )

                elif query_type == "find_all" and key_val is not None:
                    response = await UserRepository(self).table.find_many(where=key_val)
                elif query_type == "find_all" and reset_at is not None:
                    response = await UserRepository(self).table.find_many(
                        take=limit,
                        where={
                            # A user seeded from default_internal_user_params
                            # (or created via /user/new without an explicit
                            # budget_reset_at) has budget_duration set but
                            # budget_reset_at = NULL. `{"lt": reset_at}` never
                            # matches NULL, so such users would never be reset
                            # and their spend would accumulate for the lifetime
                            # of the row, silently exceeding max_budget. Treat a
                            # NULL budget_reset_at with a non-NULL budget_duration
                            # as due, matching the budget-table query below.
                            "NOT": {"budget_duration": None},
                            "OR": [
                                {"budget_reset_at": None},
                                {"budget_reset_at": {"lt": reset_at}},
                            ],
                        },
                    )
                elif query_type == "find_all" and user_id_list is not None:
                    response = await UserRepository(self).table.find_many(where={"user_id": {"in": user_id_list}})
                elif query_type == "find_all":
                    if expires is not None:
                        response = await UserRepository(self).table.find_many(
                            order={"spend": "desc"},
                            where={
                                "OR": [
                                    {"expires": None},
                                    {"expires": {"gt": expires}},
                                ],
                            },
                        )
                    else:
                        # return all users in the table, get their key aliases ordered by spend
                        sql_query = """
                        SELECT
                            u.*,
                            json_agg(v.key_alias) AS key_aliases
                        FROM
                            "LiteLLM_UserTable" u
                        LEFT JOIN "LiteLLM_VerificationToken" v ON u.user_id = v.user_id
                        GROUP BY
                            u.user_id
                        ORDER BY u.spend DESC
                        LIMIT $1
                        OFFSET $2
                        """
                        response = await self.db.query_raw(sql_query, limit, offset)
                return response
            elif table_name == "spend":
                verbose_proxy_logger.debug("PrismaClient: get_data: table_name == 'spend'")
                if key_val is not None:
                    if query_type == "find_unique":
                        response = await SpendLogsRepository(self).table.find_unique(
                            where={
                                key_val["key"]: key_val["value"],
                            }
                        )
                    elif query_type == "find_all":
                        response = await SpendLogsRepository(self).table.find_many(
                            where={
                                key_val["key"]: key_val["value"],
                            }
                        )
                    return response
                else:
                    response = await SpendLogsRepository(self).table.find_many(
                        order={"startTime": "desc"},
                    )
                    return response
            elif table_name == "budget" and reset_at is not None:
                if query_type == "find_all":
                    response = await BudgetRepository(self).table.find_many(
                        take=limit,
                        where={
                            "NOT": {"budget_duration": None},
                            "OR": [
                                {"budget_reset_at": None},
                                {"budget_reset_at": {"lt": reset_at}},
                            ],
                        },
                    )
                    return response

            elif table_name == "enduser" and budget_id_list is not None:
                if query_type == "find_all":
                    response = await EndUserRepository(self).table.find_many(
                        where={"budget_id": {"in": budget_id_list}}
                    )
                    return response
            elif table_name == "team":
                if query_type == "find_unique":
                    response = await TeamRepository(self).table.find_unique(
                        where={"team_id": team_id},
                        include={"litellm_model_table": True},
                    )
                elif query_type == "find_all" and reset_at is not None:
                    response = await TeamRepository(self).table.find_many(
                        take=limit,
                        where={
                            # Same NULL budget_reset_at gap as the user query
                            # above: a team with a budget_duration but no
                            # initialized budget_reset_at would never be reset.
                            "NOT": {"budget_duration": None},
                            "OR": [
                                {"budget_reset_at": None},
                                {"budget_reset_at": {"lt": reset_at}},
                            ],
                        },
                    )
                elif query_type == "find_all" and user_id is not None:
                    response = await TeamRepository(self).table.find_many(
                        where={
                            "members": {"has": user_id},
                        },
                        include={"litellm_budget_table": True},
                    )
                elif query_type == "find_all" and team_id_list is not None:
                    response = await TeamRepository(self).table.find_many(where={"team_id": {"in": team_id_list}})
                elif query_type == "find_all" and team_id_list is None:
                    response = await TeamRepository(self).table.find_many(take=MAX_TEAM_LIST_LIMIT)
                return response
            elif table_name == "user_notification":
                if query_type == "find_unique":
                    response = await UserNotificationsRepository(self).table.find_unique(where={"user_id": user_id})
                elif query_type == "find_all":
                    response = await UserNotificationsRepository(self).table.find_many()
                return response
            elif table_name == "combined_view":
                # check if plain text or hash
                if token is not None:
                    if isinstance(token, str):
                        hashed_token = _hash_token_if_needed(token=token)
                        verbose_proxy_logger.debug("PrismaClient: find_unique for token: %s", hashed_token)
                if query_type == "find_unique":
                    if token is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"No token passed in. Token={token}"},
                        )

                    sql_query = """
                        SELECT 
                            v.*,
                            t.spend AS team_spend, 
                            t.max_budget AS team_max_budget,
                            t.soft_budget AS team_soft_budget,
                            t.tpm_limit AS team_tpm_limit,
                            t.rpm_limit AS team_rpm_limit,
                            t.models AS team_models,
                            t.metadata AS team_metadata,
                            t.blocked AS team_blocked,
                            t.team_alias AS team_alias,
                            t.metadata AS team_metadata,
                            t.members_with_roles AS team_members_with_roles,
                            t.object_permission_id AS team_object_permission_id,
                            t.organization_id as org_id,
                            p.project_alias AS project_alias,
                            tm.spend AS team_member_spend,
                            b_tm.tpm_limit AS team_member_tpm_limit,
                            b_tm.rpm_limit AS team_member_rpm_limit,
                            m.aliases AS team_model_aliases,
                            -- Added comma to separate b.* columns
                            b.max_budget AS litellm_budget_table_max_budget,
                            b.tpm_limit AS litellm_budget_table_tpm_limit,
                            b.rpm_limit AS litellm_budget_table_rpm_limit,
                            b.model_max_budget as litellm_budget_table_model_max_budget,
                            b.soft_budget as litellm_budget_table_soft_budget,
                            o.metadata as organization_metadata,
                            o.organization_alias as organization_alias,
                            b2.max_budget as organization_max_budget,
                            b2.tpm_limit as organization_tpm_limit,
                            b2.rpm_limit as organization_rpm_limit
                        FROM "LiteLLM_VerificationToken" AS v
                        LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
                        LEFT JOIN "LiteLLM_TeamMembership" AS tm ON v.team_id = tm.team_id AND tm.user_id = v.user_id
                        LEFT JOIN "LiteLLM_BudgetTable" AS b_tm ON tm.budget_id = b_tm.budget_id
                        LEFT JOIN "LiteLLM_ModelTable" m ON t.model_id = m.id
                        LEFT JOIN "LiteLLM_BudgetTable" AS b ON v.budget_id = b.budget_id
                        LEFT JOIN "LiteLLM_ProjectTable" AS p ON v.project_id = p.project_id
                        LEFT JOIN "LiteLLM_OrganizationTable" AS o ON v.organization_id = o.organization_id
                        LEFT JOIN "LiteLLM_BudgetTable" AS b2 ON o.budget_id = b2.budget_id
                        WHERE v.token = $1
                    """

                    response = await self._query_first_with_cached_plan_fallback(sql_query, hashed_token)

                    # If not found in main table, check deprecated keys (grace period)
                    # check_deprecated=False on the recursive call prevents unbounded chaining
                    if response is None and hashed_token is not None and check_deprecated:
                        active_token_id: Final = await _lookup_deprecated_key(db=self.db, hashed_token=hashed_token)
                        if active_token_id:
                            # The recursive call returns a finished
                            # LiteLLM_VerificationTokenView; the dict
                            # normalization below would crash subscripting it.
                            deprecated_response: Final = await self.get_data(
                                token=active_token_id,
                                table_name="combined_view",
                                query_type="find_unique",
                                parent_otel_span=parent_otel_span,
                                proxy_logging_obj=proxy_logging_obj,
                                check_deprecated=False,
                            )
                            if deprecated_response is not None:
                                verbose_proxy_logger.debug("Deprecated key used during grace period")
                            return deprecated_response

                    if response is not None:
                        if response["team_models"] is None:
                            response["team_models"] = []
                        if response["team_blocked"] is None:
                            response["team_blocked"] = False

                        team_member: Member | None = None
                        if response["team_members_with_roles"] is not None and response["user_id"] is not None:
                            ## find the team member corresponding to user id
                            """
                            [
                                {
                                    "role": "admin",
                                    "user_id": "default_user_id",
                                    "user_email": null
                                },
                                {
                                    "role": "user",
                                    "user_id": null,
                                    "user_email": "test@email.com"
                                }
                            ]
                            """
                            for tm in response["team_members_with_roles"]:
                                if tm.get("user_id") is not None and response["user_id"] == tm.get("user_id"):
                                    team_member = Member(**tm)
                        response["team_member"] = team_member
                        response = LiteLLM_VerificationTokenView(**response, last_refreshed_at=time.time())
                        # for prisma we need to cast the expires time to str
                        if response.expires is not None and isinstance(response.expires, datetime):
                            response.expires = response.expires.isoformat()
                    return response
        except Exception as e:
            import traceback

            prisma_query_info: Final = (
                f"LiteLLM Prisma Client Exception: Error with `get_data`. Args passed in: {args_passed_in}"
            )
            error_msg: Final = prisma_query_info + str(e)
            print_verbose(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            verbose_proxy_logger.debug(error_traceback)
            end_time: Final = time.time()
            _duration: Final = end_time - start_time

            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="get_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    def jsonify_team_object(self, db_data: Mapping[str, object]) -> dict[str, object]:
        db_data = self.jsonify_object(data=db_data)
        if db_data.get("members_with_roles", None) is not None and isinstance(db_data["members_with_roles"], list):
            db_data["members_with_roles"] = json.dumps(db_data["members_with_roles"])
        if db_data.get("budget_limits", None) is not None and isinstance(db_data["budget_limits"], list):
            db_data["budget_limits"] = json.dumps(db_data["budget_limits"])
        return db_data

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def insert_data(
        self,
        data: Mapping[str, object],
        table_name: Literal["user", "key", "config", "spend", "team", "user_notification"],
    ):
        """
        Add a key to the database. If it already exists, do nothing.
        """
        start_time: Final = time.time()
        try:
            verbose_proxy_logger.debug(
                "PrismaClient: insert_data: %s",
                {**data, "token": self.hash_token(token=cast("str", data["token"]))}  # cast-ok: a key token is a str
                if data.get("token") is not None
                else data,
            )
            if table_name == "key":
                token: Final = cast("str", data["token"])  # cast-ok: the key table's token column is a str
                hashed_token: Final = self.hash_token(token=token)
                db_data = self.jsonify_object(data=data)
                db_data["token"] = hashed_token
                # Prisma rejects nullable JSON fields set to None (no default).
                # Strip them so the DB stores NULL via the column's nullable constraint.
                if db_data.get("budget_limits") is None:
                    db_data.pop("budget_limits", None)
                print_verbose("PrismaClient: Before upsert into litellm_verificationtoken")
                new_verification_token: Final = await VerificationTokenRepository(self).table.upsert(
                    where={
                        "token": hashed_token,
                    },
                    data={
                        "create": {**db_data},
                        "update": {},  # don't do anything if it already exists
                    },
                    include={"litellm_budget_table": True},
                )
                verbose_proxy_logger.info("Data Inserted into Keys Table")
                return new_verification_token
            elif table_name == "user":
                db_data = self.jsonify_object(data=data)
                try:
                    new_user_row: Final = await UserRepository(self).table.upsert(
                        where={"user_id": data["user_id"]},
                        data={
                            "create": {**db_data},
                            "update": {},  # don't do anything if it already exists
                        },
                    )
                except Exception as e:
                    if (
                        "Foreign key constraint failed on the field: `LiteLLM_UserTable_organization_id_fkey (index)`"
                        in str(e)
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": f"Foreign Key Constraint failed. Organization ID={db_data['organization_id']} does not exist in LiteLLM_OrganizationTable. Create via `/organization/new`."
                            },
                        )
                    raise e
                verbose_proxy_logger.info("Data Inserted into User Table")
                return new_user_row
            elif table_name == "team":
                db_data = self.jsonify_team_object(db_data=data)
                new_team_row: Final = await TeamRepository(self).table.upsert(
                    where={"team_id": data["team_id"]},
                    data={
                        "create": {**db_data},
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Team Table")
                return new_team_row
            elif table_name == "config":
                """
                For each param,
                get the existing table values

                Add the new values

                Update DB
                """
                tasks: Final = []
                for k, v in data.items():
                    updated_data = v
                    updated_data = json.dumps(updated_data)
                    updated_table_row = ConfigRepository(self).table.upsert(
                        where={"param_name": k},
                        data={
                            "create": {"param_name": k, "param_value": updated_data},
                            "update": {"param_value": updated_data},
                        },
                    )

                    tasks.append(updated_table_row)
                await asyncio.gather(*tasks)
                # invalidate cache so other pods see writes from save_config
                for k in data:
                    await invalidate_config_param(k)
                verbose_proxy_logger.info("Data Inserted into Config Table")
            elif table_name == "spend":
                db_data = self.jsonify_object(data=data)
                new_spend_row: Final = await SpendLogsRepository(self).table.upsert(
                    where={"request_id": data["request_id"]},
                    data={
                        "create": {**db_data},
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Spend Table")
                return new_spend_row
            elif table_name == "user_notification":
                db_data = self.jsonify_object(data=data)
                new_user_notification_row: Final = await UserNotificationsRepository(self).table.upsert(
                    where={"request_id": data["request_id"]},
                    data={
                        "create": {**db_data},
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Model Request Table")
                return new_user_notification_row

        except Exception as e:
            import traceback

            error_msg: Final = f"LiteLLM Prisma Client Exception in insert_data: {e}"
            print_verbose(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="insert_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def update_data(
        self,
        token: str | None = None,
        data: Mapping[str, object] = {},
        data_list: list | None = None,
        user_id: str | None = None,
        team_id: str | None = None,
        query_type: Literal["update", "update_many"] = "update",
        table_name: Literal["user", "key", "config", "spend", "team", "enduser", "budget"] | None = None,
        update_key_values: dict[str, object] | None = None,
        update_key_values_custom_query: dict[str, object] | None = None,
    ):
        """
        Update existing data
        """
        verbose_proxy_logger.debug("PrismaClient: update_data, table_name: %s", table_name)
        start_time: Final = time.time()
        try:
            db_data: Final = self.jsonify_object(data=data)
            if update_key_values is not None:
                update_key_values = self.jsonify_object(data=update_key_values)
            if token is not None:
                print_verbose(f"token: [set={token is not None}]")
                # check if plain text or hash
                token = _hash_token_if_needed(token=token)
                db_data["token"] = token
                response: Final = await VerificationTokenRepository(self).table.update(
                    where={"token": token},
                    data=with_settings_updated_at(db_data),
                )
                verbose_proxy_logger.debug("\033[91m" + f"DB Token Table update succeeded {response}" + "\033[0m")
                _data: dict = {}
                if response is not None:
                    try:
                        _data = response.model_dump()
                    except Exception:
                        _data = response.dict()  # pyright: ignore[reportDeprecated]  # pydantic-v1 row fallback
                return {"token": token, "data": _data}
            elif user_id is not None or (table_name is not None and table_name == "user") and query_type == "update":
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                if user_id is None:
                    user_id = cast("str", db_data["user_id"])  # cast-ok: the user table's user_id column is a str
                if update_key_values is None:
                    if update_key_values_custom_query is not None:
                        update_key_values = update_key_values_custom_query
                    else:
                        update_key_values = db_data
                update_user_row: Final = await UserRepository(self).table.upsert(
                    where={"user_id": user_id},
                    data={
                        "create": {**db_data},
                        "update": {**update_key_values},  # just update user-specified values, if it already exists
                    },
                )
                verbose_proxy_logger.info(
                    "\033[91m" + f"DB User Table - update succeeded {update_user_row}" + "\033[0m"
                )
                return {"user_id": user_id, "data": update_user_row}
            elif team_id is not None or (table_name is not None and table_name == "team") and query_type == "update":
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                if team_id is None:
                    team_id = cast("str | None", db_data["team_id"])  # cast-ok: team_id column is a nullable str
                if update_key_values is None:
                    update_key_values = db_data
                if "team_id" not in db_data and team_id is not None:
                    db_data["team_id"] = team_id
                if "members_with_roles" in db_data and isinstance(db_data["members_with_roles"], list):
                    db_data["members_with_roles"] = json.dumps(db_data["members_with_roles"])
                if "members_with_roles" in update_key_values and isinstance(
                    update_key_values["members_with_roles"], list
                ):
                    update_key_values["members_with_roles"] = json.dumps(update_key_values["members_with_roles"])
                update_team_row: Final = await TeamRepository(self).table.upsert(
                    where={"team_id": team_id},
                    data={
                        "create": {**db_data},
                        "update": {**update_key_values},  # just update user-specified values, if it already exists
                    },
                )
                verbose_proxy_logger.info(
                    "\033[91m" + f"DB Team Table - update succeeded {update_team_row}" + "\033[0m"
                )
                return {"team_id": team_id, "data": update_team_row}
            elif (
                table_name is not None
                and table_name == "key"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for idx, t in enumerate(data_list):
                    # check if plain text or hash
                    if t.token.startswith("sk-"):
                        t.token = self.hash_token(token=t.token)
                    try:
                        data_json = self.jsonify_object(data=t.model_dump(exclude_none=True))
                    except Exception:
                        data_json = self.jsonify_object(data=t.dict(exclude_none=True))
                    batcher.litellm_verificationtoken.update(
                        where={"token": t.token},
                        data={**data_json},
                    )
                await batcher.commit()
                print_verbose("\033[91m" + "DB Token Table update succeeded" + "\033[0m")
            elif (
                table_name is not None
                and table_name == "user"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for idx, user in enumerate(data_list):
                    try:
                        data_json = self.jsonify_object(data=user.model_dump(exclude_none=True))
                    except Exception:
                        data_json = self.jsonify_object(data=user.dict())
                    batcher.litellm_usertable.upsert(
                        where={"user_id": user.user_id},
                        data={
                            "create": {**data_json},
                            "update": {**data_json},  # just update user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info("\033[91m" + "DB User Table Batch update succeeded" + "\033[0m")
            elif (
                table_name is not None
                and table_name == "enduser"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for enduser in data_list:
                    try:
                        data_json = self.jsonify_object(data=enduser.model_dump(exclude_none=True))
                    except Exception:
                        data_json = self.jsonify_object(data=enduser.dict())
                    batcher.litellm_endusertable.upsert(
                        where={"user_id": enduser.user_id},
                        data={
                            "create": {**data_json},
                            "update": {**data_json},  # just update end-user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info("\033[91m" + "DB End User Table Batch update succeeded" + "\033[0m")
            elif (
                table_name is not None
                and table_name == "budget"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for budget in data_list:
                    try:
                        data_json = self.jsonify_object(data=budget.model_dump(exclude_none=True))
                    except Exception:
                        data_json = self.jsonify_object(data=budget.dict())
                    batcher.litellm_budgettable.upsert(
                        where={"budget_id": budget.budget_id},
                        data={
                            "create": {**data_json},
                            "update": {**data_json},  # just update end-user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info("\033[91m" + "DB Budget Table Batch update succeeded" + "\033[0m")
            elif (
                table_name is not None
                and table_name == "team"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                # Batch write update queries
                batcher = self.db.batch_()
                for idx, team in enumerate(data_list):
                    try:
                        data_json = self.jsonify_team_object(db_data=team.model_dump(exclude_none=True))
                    except Exception:
                        data_json = self.jsonify_object(data=team.dict(exclude_none=True))
                    batcher.litellm_teamtable.upsert(
                        where={"team_id": team.team_id},
                        data={
                            "create": {**data_json},
                            "update": {**data_json},  # just update user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info("\033[91m" + "DB Team Table Batch update succeeded" + "\033[0m")

        except Exception as e:
            import traceback

            error_msg: Final = f"LiteLLM Prisma Client Exception - update_data: {e}"
            print_verbose(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="update_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def delete_data(
        self,
        tokens: Sequence[str | None] | None = None,
        team_id_list: Sequence[str] | None = None,
        table_name: Literal["user", "key", "config", "spend", "team"] | None = None,
        user_id: str | None = None,
    ):
        """
        Allow user to delete a key(s)

        Ensure user owns that key, unless admin.
        """
        start_time: Final = time.time()
        try:
            if tokens is not None and isinstance(tokens, list):
                hashed_tokens: Final[list[str | None]] = []
                for token in tokens:
                    if isinstance(token, str) and token.startswith("sk-"):
                        hashed_token = self.hash_token(token=token)
                    else:
                        hashed_token = token
                    hashed_tokens.append(hashed_token)
                filter_query: dict[str, object] = {}
                if user_id is not None:
                    filter_query = {"AND": [{"token": {"in": hashed_tokens}}, {"user_id": user_id}]}
                else:
                    filter_query = {"token": {"in": hashed_tokens}}

                deleted_tokens: Final[int] = await VerificationTokenRepository(self).table.delete_many(
                    where=filter_query
                )
                verbose_proxy_logger.debug("deleted_tokens: %s", deleted_tokens)
                return {"deleted_keys": deleted_tokens}
            elif table_name == "team" and team_id_list is not None and isinstance(team_id_list, list):
                # admin only endpoint -> `/team/delete`
                await TeamRepository(self).table.delete_many(where={"team_id": {"in": team_id_list}})
                return {"deleted_teams": team_id_list}
            elif table_name == "key" and team_id_list is not None and isinstance(team_id_list, list):
                # admin only endpoint -> `/team/delete`
                await VerificationTokenRepository(self).table.delete_many(where={"team_id": {"in": team_id_list}})
        except Exception as e:
            import traceback

            error_msg: Final = f"LiteLLM Prisma Client Exception - delete_data: {e}"
            print_verbose(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="delete_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def connect(self):
        start_time: Final = time.time()
        try:
            verbose_proxy_logger.debug("PrismaClient: connect() called Attempting to Connect to DB")
            if self.db.is_connected() is False:
                verbose_proxy_logger.debug("PrismaClient: DB not connected, Attempting to Connect to DB")
                await self.db.connect()
        except Exception as e:
            import traceback

            error_msg: Final = f"LiteLLM Prisma Client Exception connect(): {e}"
            verbose_proxy_logger.warning(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="connect",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def disconnect(self):
        start_time: Final = time.time()
        try:
            await self.db.disconnect()
        except Exception as e:
            import traceback

            error_msg: Final = f"LiteLLM Prisma Client Exception disconnect(): {e}"
            print_verbose(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="disconnect",
                    traceback_str=error_traceback,
                )
            )
            raise e

    def _get_engine_pid(self) -> int:
        """Get the PID of the writer's engine subprocess, or 0 if unavailable.

        Must never raise: prisma's ``_engine`` property raises
        ``ClientNotConnectedError`` on a disconnected client, and an exception
        escaping from the reconnect path would leave it unable to recover.
        """
        try:
            prisma_obj: Final = self.writer_db._original_prisma
            if prisma_obj.is_connected() is not True:
                return 0
            engine: Final = prisma_obj._engine
            process: Final = getattr(engine, "process", None) if engine is not None else None
            if process is not None:
                pid: Final[object] = process.pid
                if isinstance(pid, int):
                    return pid
        except (AttributeError, TypeError):
            pass
        return 0

    def _is_engine_alive(self) -> bool:
        if self._engine_pid <= 0:
            return True
        try:
            os.kill(self._engine_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True

    @staticmethod
    def _reap_all_zombies() -> set:
        """Reap ALL zombie child processes via waitpid(-1, WNOHANG).

        Returns a set of reaped PIDs.  As PID 1 in Docker (or any
        process that spawns children), we must reap ALL terminated
        children to prevent zombie accumulation.

        No-op on Windows: os.waitpid and os.WNOHANG are Unix-only.
        """
        if sys.platform == "win32":
            return set()
        reaped: Final[set] = set()
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                reaped.add(pid)
            except ChildProcessError:
                break
        return reaped

    def _try_waitpid_watch(self, pid: int) -> bool:
        """Watch engine PID via os.waitpid() in a dedicated thread.

        The thread blocks on os.waitpid(pid, 0) which is a kernel-level
        wait and with zero CPU overhead, instant detection when the process exits.
        When the process dies, the thread notifies the asyncio event loop
        via call_soon_threadsafe.

        Returns True if the thread was started, False on failure.
        On Windows, returns False immediately (os.waitpid/WNOHANG are Unix-only);
        caller falls back to os.kill polling.
        """
        if sys.platform == "win32":
            return False
        try:
            probe_pid, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            verbose_proxy_logger.debug(
                "PID %s is not a child process; skipping waitpid watch.",
                pid,
            )
            return False

        if probe_pid == pid:
            verbose_proxy_logger.warning(
                "prisma-query-engine PID %s already dead at watch start.",
                pid,
            )
            if self._consume_expected_death(pid):
                verbose_proxy_logger.info(
                    "PID %s death was planned (engine already replaced); not reconnecting.",
                    pid,
                )
                self._cleanup_engine_watcher()
                return True
            self._engine_confirmed_dead = True
            self._reap_all_zombies()
            self._cleanup_engine_watcher()
            asyncio.create_task(
                self.attempt_db_reconnect(
                    reason="engine_process_death",
                    force=True,
                )
            )
            return True

        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return False

        thread: Final = threading.Thread(
            target=self._waitpid_thread_func,
            args=(pid, loop),
            daemon=True,
            name=f"prisma-engine-waitpid-{pid}",
        )
        thread.start()
        self._engine_wait_thread = thread
        return True

    def _waitpid_thread_func(self, pid: int, loop: asyncio.AbstractEventLoop) -> None:
        """Thread function: block until engine PID exits, then notify event loop.

        Note: uvloop/libuv may reap the child first via waitpid(-1, WNOHANG)
        in its SIGCHLD handler. In that case our waitpid raises ChildProcessError.
        we still notify the event loop because the engine is dead either way.
        """
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        except OSError:
            pass
        try:
            loop.call_soon_threadsafe(self._on_engine_death_from_thread, pid)
        except RuntimeError:
            pass

    def _consume_expected_death(self, pid: int) -> bool:
        """True iff ``pid`` was killed on purpose by a planned recreate.

        `PrismaWrapper.recreate_prisma_client` records the old engine PID in
        `_expected_engine_deaths` before SIGTERM-ing it (IAM token refresh,
        guarded reconnect). When the watcher then sees that PID die, this lets
        it recognize the death as planned and skip its own reconnect, which
        would otherwise kill the engine the recreate just spawned (#29176).

        Consumes (removes) the PID so a later real crash of a reused PID is
        still handled. Tolerant of `self.db` stand-ins (tests / older clients)
        that don't expose a real set.
        """
        expected: Final = getattr(self.db, "_expected_engine_deaths", None)
        if isinstance(expected, set) and pid in expected:
            expected.discard(pid)
            return True
        return False

    def _on_engine_death_from_thread(self, dead_pid: int) -> None:
        """Called on the event loop thread when the waitpid thread detects engine death."""
        if self._engine_confirmed_dead:
            return
        if dead_pid != self._engine_pid:
            return
        if self._consume_expected_death(dead_pid):
            verbose_proxy_logger.info(
                "prisma-query-engine PID %s exited as part of a planned restart; "
                "not reconnecting (engine already replaced).",
                dead_pid,
            )
            self._cleanup_engine_watcher()
            return
        verbose_proxy_logger.error(
            "prisma-query-engine PID %s exited (waitpid thread); triggering reconnect.",
            dead_pid,
        )
        self._engine_confirmed_dead = True
        self._reap_all_zombies()
        self._cleanup_engine_watcher()
        asyncio.create_task(
            self.attempt_db_reconnect(
                reason="engine_process_death",
                force=True,
            )
        )

    def _try_pidfd_watch(self, pid: int) -> bool:
        """
        Watch engine PID via pidfd_open + asyncio event loop reader.

        Returns True if pidfd watch was set up, False if unavailable or failed.
        Broad OSError catch handles both ENOSYS and SECCOMP-blocked syscalls.
        """
        if not hasattr(os, "pidfd_open"):
            return False
        fd = -1
        try:
            fd = os.pidfd_open(pid, 0)
            asyncio.get_running_loop().add_reader(fd, self._on_pidfd_readable)
            self._engine_pidfd = fd
            return True
        except OSError:
            if fd >= 0:
                os.close(fd)
            return False

    def _on_pidfd_readable(self) -> None:
        """pidfd became readable: engine process exited or became zombie.

        Sets _engine_confirmed_dead BEFORE cleanup so _run_reconnect_cycle
        takes the heavy path (recreate Prisma client + re-arm watcher).
        """
        if self._engine_confirmed_dead:
            # Already handled -- just clean up pidfd resources.
            if self._engine_pidfd >= 0:
                try:
                    asyncio.get_running_loop().remove_reader(self._engine_pidfd)
                except Exception:
                    pass
                try:
                    os.close(self._engine_pidfd)
                except OSError:
                    pass
                self._engine_pidfd = -1
            return
        dead_pid: Final = self._engine_pid
        if self._consume_expected_death(dead_pid):
            verbose_proxy_logger.info(
                "prisma-query-engine PID %s exited (pidfd event) as part of a "
                "planned restart; not reconnecting (engine already replaced).",
                dead_pid,
            )
            self._cleanup_engine_watcher()
            return
        verbose_proxy_logger.error(
            "prisma-query-engine PID %s exited (pidfd event); triggering reconnect.",
            dead_pid,
        )
        self._engine_confirmed_dead = True
        self._reap_all_zombies()
        self._cleanup_engine_watcher()
        asyncio.create_task(
            self.attempt_db_reconnect(
                reason="engine_process_death",
                force=True,
            )
        )

    async def _poll_engine_proc(self) -> None:
        """poll via os.kill(pid, 0) every 1s.
        Only used when BOTH waitpid thread and pidfd are unavailable
        (e.g., PID is not our child process and pidfd_open fails)
        """
        while self._watching_engine and self._engine_pid > 0:
            try:
                os.kill(self._engine_pid, 0)
            except ProcessLookupError:
                dead_pid = self._engine_pid
                if self._consume_expected_death(dead_pid):
                    verbose_proxy_logger.info(
                        "prisma-query-engine PID %s gone as part of a planned "
                        "restart; not reconnecting (engine already replaced).",
                        dead_pid,
                    )
                    self._cleanup_engine_watcher()
                    return
                verbose_proxy_logger.error(
                    "prisma-query-engine PID %s gone; triggering reconnect.",
                    dead_pid,
                )
                self._engine_confirmed_dead = True
                self._reap_all_zombies()
                self._cleanup_engine_watcher()
                await self.attempt_db_reconnect(
                    reason="engine_process_death",
                    force=True,
                )
                return
            except (PermissionError, OSError):
                verbose_proxy_logger.debug(
                    "Cannot signal PID %s; stopping engine poll.",
                    self._engine_pid,
                )
                self._cleanup_engine_watcher()
                return
            await asyncio.sleep(1)

    def _cleanup_engine_watcher(self) -> None:
        """Clean up pidfd reader, waitpid thread ref, or stop polling and reset state."""
        self._watching_engine = False
        if self._engine_pidfd >= 0:
            try:
                asyncio.get_running_loop().remove_reader(self._engine_pidfd)
            except Exception:
                pass
            try:
                os.close(self._engine_pidfd)
            except OSError:
                pass
            self._engine_pidfd = -1
        self._engine_wait_thread = None
        self._engine_pid = 0

    async def _start_engine_watcher(self) -> None:
        """
        Start watching the Prisma query engine process for death.

        Detection priority:
        1. os.waitpid() in a dedicated thread, works with all event loops.
        2. pidfd_open kernel fd registered with asyncio.
        3. os.kill(pid, 0) polling (1s), last-resort fallback when neither
           waitpid thread nor pidfd are available.

        """
        if self._watching_engine or self._engine_pidfd >= 0 or self._engine_wait_thread is not None:
            return
        pid: Final = self._get_engine_pid()
        if pid == 0:
            verbose_proxy_logger.debug("Could not find prisma-query-engine PID; engine death detection unavailable.")
            return
        self._engine_pid = pid
        self._engine_confirmed_dead = False
        verbose_proxy_logger.info("Found prisma-query-engine at PID %s.", pid)
        waitpid_ok: Final = self._try_waitpid_watch(pid)
        pidfd_ok: Final = False if waitpid_ok else self._try_pidfd_watch(pid)
        if waitpid_ok:
            verbose_proxy_logger.info(
                "Watching engine PID %s via waitpid thread.",
                pid,
            )
        elif pidfd_ok:
            verbose_proxy_logger.info(
                "Watching engine PID %s via pidfd.",
                pid,
            )
        else:
            verbose_proxy_logger.info(
                "Watching engine PID %s via os.kill polling.",
                pid,
            )
            self._watching_engine = True
            asyncio.create_task(self._poll_engine_proc())

    def _stop_engine_watcher(self) -> None:
        """Stop watching the engine process and clean up all resources."""
        self._cleanup_engine_watcher()
        self._engine_confirmed_dead = False
        verbose_proxy_logger.debug("Stopped engine process watcher.")

    def _handle_writer_engine_replaced(self) -> None:
        """Re-arm the engine watcher after a planned writer-engine restart.

        Wired as `PrismaWrapper.on_engine_replaced` and invoked from inside
        `recreate_prisma_client` once the new engine is connected (IAM token
        refresh, guarded reconnect). The old watcher was tracking the engine
        we just intentionally killed, so we tear it down and re-arm on the new
        PID. Scheduling `_start_engine_watcher` as a task (rather than awaiting)
        keeps us from blocking the recreate while it still holds the wrapper's
        reconnection lock. Without this re-arm, a planned restart would leave
        the proxy with no engine-death detection until the next reconnect.
        """
        self._engine_confirmed_dead = False
        self._cleanup_engine_watcher()
        asyncio.create_task(self._start_engine_watcher())

    async def _run_reconnect_cycle(
        self,
        timeout_seconds: float | None = None,
        force_recreate: bool = False,
    ) -> None:
        """
        Run a reconnect cycle with a single overall timeout budget.

        Uses the _engine_confirmed_dead flag (set by waitpid thread / pidfd / poll
        handlers) to choose between heavy reconnect (engine dead -- recreate
        Prisma client, re-arm watcher) and direct reconnect (network blip --
        recreate Prisma client, re-arm watcher, SELECT 1). Both paths recreate
        the client via the non-blocking kill-then-construct flow rather than
        calling disconnect(), which blocks the event loop on the synchronous
        subprocess.Popen.wait() inside prisma-client-py (see issue #26191).

        `force_recreate` skips the direct path's liveness probe, for callers
        whose failure lives in the session state rather than the connection
        (stale prepared statements after a schema change): a reachable writer
        proves nothing about those, so the probe must not veto the recreate.
        """
        effective_timeout: Final = (
            timeout_seconds if timeout_seconds is not None else self._db_watchdog_reconnect_timeout_seconds
        )

        # Snapshot the writer's engine generation BEFORE any await. Both
        # reconnect branches forward it to recreate_prisma_client as an
        # optimistic-lock token: if a concurrent IAM token refresh replaces the
        # engine after this point, the generation moves and the recreate becomes
        # a no-op instead of killing the engine the refresh just spawned
        # (#29176). Captured here — atomically with the dead-engine decision
        # below — rather than inside the reconnect closures, because those run
        # after an `asyncio.wait_for(...)` yield during which a refresh could
        # otherwise slip in and bump the very generation the closure then reads.
        expected_generation: Final = getattr(self.writer_db, "_engine_generation", None)

        engine_is_dead: Final = self._engine_confirmed_dead or (self._engine_pid > 0 and not self._is_engine_alive())

        if engine_is_dead:
            dead_pid: Final = self._engine_pid
            verbose_proxy_logger.warning(
                "prisma-query-engine PID %s is dead; reconnecting.",
                dead_pid,
            )
            self._reap_all_zombies()
            self._cleanup_engine_watcher()

            async def _do_heavy_reconnect() -> None:
                db_url: Final = os.getenv("DATABASE_URL", "")
                if not db_url:
                    verbose_proxy_logger.error("DATABASE_URL not set; cannot recreate Prisma client.")
                    raise RuntimeError("DATABASE_URL not set")
                # Forward the entry-snapshot generation. The engine was
                # confirmed dead, but a concurrent IAM refresh may have already
                # respawned it; the guard makes this recreate a no-op in that
                # case rather than killing the fresh engine (#29176). Unlike the
                # direct path there is no SELECT 1 probe here, so the generation
                # guard is the only thing standing between a crash-reconnect and
                # a refresh that raced it.
                recreated: Final = await self.db.recreate_prisma_client(db_url, expected_generation=expected_generation)
                await self._start_engine_watcher()
                # Same contract as the direct path below: a forced caller asked
                # for its engine to be replaced, so a decline is not a success.
                # Reachable here because the escalation threshold flips
                # `_engine_confirmed_dead`, which routes the next cycle, forced
                # callers included, down this branch.
                if force_recreate is True and recreated is False:
                    # Clear the dead-engine flag first, restoring the policy the
                    # non-forced path already has: a decline does not raise for
                    # it, so it falls through to the clear below. Only the
                    # forced branch would strand the flag, and stranding it
                    # routes the next cycle back down this probe-free branch,
                    # where the refreshed generation now matches and the
                    # recreate kills the healthy engine a refresh just spawned
                    # (#29176). This has to stay AFTER `_start_engine_watcher`
                    # above: clearing the flag while the watcher is still torn
                    # down would be worse than either alone.
                    self._engine_confirmed_dead = False
                    raise _ForcedRecreateDeclined(
                        "Forced Prisma recreate declined by the generation guard; "
                        "the engine that failed was not replaced"
                    )

            await asyncio.wait_for(_do_heavy_reconnect(), timeout=effective_timeout)
            # Only clear the "dead engine" flag after the heavy reconnect
            # actually completed. If `_do_heavy_reconnect()` raises (timeout,
            # missing DATABASE_URL, recreate failure), the flag stays True so
            # the next attempt re-enters the heavy branch instead of silently
            # demoting to the lightweight path.
            self._engine_confirmed_dead = False
        else:
            verbose_proxy_logger.debug("Performing Prisma DB reconnect (engine alive or unknown).")

            async def _do_direct_reconnect() -> None:
                db_url: Final = os.getenv("DATABASE_URL", "")
                if not db_url:
                    verbose_proxy_logger.error("DATABASE_URL not set; cannot reconnect Prisma client.")
                    raise RuntimeError("DATABASE_URL not set")
                # Probe the writer BEFORE recreating. A concurrent IAM token
                # refresh may have just replaced the engine (issue #29176); if
                # the writer answers SELECT 1 the connection is already healthy
                # and recreating would needlessly kill that fresh engine. If we
                # do recreate, the entry-snapshot generation lets the wrapper
                # detect a refresh that landed since cycle entry and skip the
                # redundant restart.
                writer: Final = self.writer_db
                if force_recreate is False:
                    try:
                        await writer.query_raw("SELECT 1")
                        verbose_proxy_logger.info(
                            "Writer healthy on probe; skipping recreate (engine "
                            "likely already replaced by a token refresh)."
                        )
                        if isinstance(self.db, RoutingPrismaWrapper):
                            self.db.mark_writer_recovered()
                        await self._start_engine_watcher()
                        return
                    except Exception as probe_err:
                        verbose_proxy_logger.warning(
                            "Writer probe failed (%s); recreating Prisma client.",
                            probe_err,
                        )
                # Fresh Prisma client + new engine subprocess. The previous
                # "lightweight" path called `disconnect()` which blocks the
                # event loop on `subprocess.Popen.wait()`; since that call
                # ends up killing the engine anyway, we do it non-blockingly
                # via `_kill_engine_process` inside `recreate_prisma_client`.
                self._cleanup_engine_watcher()
                recreated: Final = await self.db.recreate_prisma_client(db_url, expected_generation=expected_generation)
                await self._start_engine_watcher()
                # Smoke-test the writer specifically; query_raw on the routing
                # wrapper sends to the reader, which would not validate the
                # newly-recreated writer engine. The reader is left to the
                # caller's own retried query, a stronger check than SELECT 1,
                # and a reader that fails to come back sets `_reader_unavailable`
                # so reads fall through to the writer just recreated here.
                await self.writer_db.query_raw("SELECT 1")
                # A recreate can decline: the optimistic-lock guard no-ops when
                # the writer generation moved since cycle entry, and the routing
                # wrapper then leaves the reader untouched as well. Callers that
                # merely suspect a transport blip are happy either way, but a
                # forced caller asked for this engine to be replaced because its
                # session state is poisoned, and it was not. Do not report that
                # as a success: it would reset the consecutive-failure count and
                # log a repair that never happened.
                if force_recreate is True and recreated is False:
                    raise _ForcedRecreateDeclined(
                        "Forced Prisma recreate declined by the generation guard; "
                        "the engine that failed was not replaced"
                    )

            await asyncio.wait_for(_do_direct_reconnect(), timeout=effective_timeout)

    def _cooldown_applies(self, stale_read_engine: "_StaleReadEngine | None") -> bool:
        """
        Whether the reconnect cooldown should still gate this caller.

        The cooldown collapses a burst of callers onto one recreate, so it
        keeps gating a caller whose named engine has already been replaced:
        that recreate is the one it was waiting for. While that engine is still
        the live one the damage is still being served, so deferring to an
        unrelated reconnect's cooldown would leave it broken until the cooldown
        elapses.

        A named engine always describes the one that served the failing read
        (see `_query_first_with_cached_plan_fallback`), so it is compared
        against `read_db`, identity included: `read_db` can resolve to a
        different wrapper than it did at observation time.

        The waiver is withdrawn once a repair of this same engine has been
        tried and failed. A failed recreate leaves the generation where it was,
        so without this every queued caller would still see its own engine live
        and run its own full recreate serially instead of collapsing onto one
        attempt, which is what the cooldown is for. The record is scoped to the
        engine rather than to a global failure count: an unrelated reconnect
        failing somewhere else says nothing about whether this engine can be
        repaired, and gating on it would suppress the recovery this method
        exists to allow.

        The record is never cleared, and does not need to be. Generations are
        monotonic per wrapper, so once the engine is repaired every later
        caller names a higher one and the entry can never match again. And this
        method is only ever the first half of the gate: the cooldown window
        itself still expires, so an engine that can never be repaired degrades
        to the plain cooldown rather than being suppressed forever.
        """
        if stale_read_engine is None:
            return True
        if self._failed_recreate_generations.get(stale_read_engine.wrapper) == stale_read_engine.generation:
            return True
        return not stale_read_engine.is_still_live(self.read_db)

    async def _attempt_reconnect_inside_lock(
        self,
        force: bool,
        reason: str,
        timeout_seconds: float | None,
        force_recreate: bool = False,
        stale_read_engine: "_StaleReadEngine | None" = None,
    ) -> bool:
        now: Final = time.time()
        if (
            force is False
            and self._cooldown_applies(stale_read_engine)
            and now - self._db_last_reconnect_attempt_ts < self._db_reconnect_cooldown_seconds
        ):
            verbose_proxy_logger.debug(
                "Skipping DB reconnect attempt inside lock due to cooldown. reason=%s",
                reason,
            )
            return False

        # Escalate to heavy reconnect after consecutive lightweight failures.
        # When the Prisma engine process is alive but not accepting connections
        # (e.g., startup race condition), lightweight reconnects (disconnect +
        # connect) will never succeed. Force a full Prisma client recreation
        # to recover from this state.
        if self._consecutive_reconnect_failures >= self._reconnect_escalation_threshold:
            verbose_proxy_logger.warning(
                "Escalating to heavy reconnect after %d consecutive failures. reason=%s",
                self._consecutive_reconnect_failures,
                reason,
            )
            self._engine_confirmed_dead = True

        verbose_proxy_logger.warning("Attempting Prisma DB reconnect. reason=%s", reason)

        reconnect_succeeded = False
        try:
            await self._run_reconnect_cycle(timeout_seconds=timeout_seconds, force_recreate=force_recreate)
            reconnect_succeeded = True
            self._consecutive_reconnect_failures = 0
            verbose_proxy_logger.info("Prisma DB reconnect succeeded. reason=%s", reason)
        except _ForcedRecreateDeclined as declined:
            # A decline is raised only when the recreate returns False, which
            # happens only at the generation guard, and the generation moves
            # only after a replacement has connected. So a decline is proof
            # that a replacement SUCCEEDED, and zeroing a consecutive-failure
            # count on that proof is right by definition rather than by
            # analogy to what a reported success used to do. Note what it
            # proves is that the WRITER was replaced, not that this caller's
            # engine was repaired: on a read replica the reader can still be
            # poisoned, since the wrapper returns before touching it. Leaving
            # the count at the threshold would let the escalation check above
            # re-arm the dead-engine flag on the very next attempt and send a
            # healthy replacement back down the probe-free heavy path.
            self._consecutive_reconnect_failures = 0
            verbose_proxy_logger.warning("Prisma DB reconnect declined. reason=%s detail=%s", reason, declined)
        except Exception as reconnect_err:
            self._consecutive_reconnect_failures += 1
            # Remember WHICH engine could not be repaired, so the rest of this
            # caller's burst collapses onto the cooldown instead of each
            # retrying the recreate that just failed. Recorded only for a
            # caller that named a generation: a watchdog or transport-error
            # reconnect failing here is unrelated to any stale read engine and
            # must not suppress its waiver.
            if stale_read_engine is not None:
                # Key off the wrapper the CALLER named, never a freshly resolved
                # `read_db`. A failed reader recreate is itself what marks the
                # reader unavailable, so re-resolving here would file the
                # reader's failure under the writer: the poisoned reader would
                # lose its record and the healthy writer would gain a spurious
                # one, wrong in both directions at once.
                self._failed_recreate_generations = MappingProxyType(
                    {**self._failed_recreate_generations, stale_read_engine.wrapper: stale_read_engine.generation}
                )
            verbose_proxy_logger.error(
                "Prisma DB reconnect failed (%d consecutive). reason=%s error=%s",
                self._consecutive_reconnect_failures,
                reason,
                reconnect_err,
            )
        finally:
            self._db_last_reconnect_attempt_ts = time.time()

        return reconnect_succeeded

    async def attempt_db_reconnect(
        self,
        reason: str,
        force: bool = False,
        timeout_seconds: float | None = None,
        lock_timeout_seconds: float | None = None,
        force_recreate: bool = False,
        stale_read_engine: "_StaleReadEngine | None" = None,
    ) -> bool:
        """
        Attempt to reconnect the Prisma client in a singleflight manner.

        `force` bypasses the cooldown unconditionally; `force_recreate`
        bypasses the liveness probe that would otherwise skip recreating a
        reachable engine; `stale_read_engine` bypasses the cooldown only while
        the engine that produced the caller's failure is still the live one
        (see `_cooldown_applies`).

        A `force_recreate` caller can also get False for a third reason: the
        generation guard declined because another path had already replaced
        the engine, which is a successful outcome reported as False. Callers
        that branch on the return value (`exception_handler` raises on False,
        `auth_checks` retries only on True) would misread that as a dead end,
        and are safe today only because neither passes `force_recreate`. Do
        not add it to one of them without revisiting how it reads the result.

        Returns:
            bool: True if reconnection succeeded, else False.
        """
        now: Final = time.time()
        if (
            force is False
            and self._cooldown_applies(stale_read_engine)
            and now - self._db_last_reconnect_attempt_ts < self._db_reconnect_cooldown_seconds
        ):
            verbose_proxy_logger.debug(
                "Skipping DB reconnect attempt due to cooldown. reason=%s",
                reason,
            )
            return False

        if lock_timeout_seconds is None:
            async with self._db_reconnect_lock:
                return await self._attempt_reconnect_inside_lock(
                    force, reason, timeout_seconds, force_recreate, stale_read_engine
                )

        lock_acquired_by_timeout_task = False

        async def _acquire_reconnect_lock() -> bool:
            nonlocal lock_acquired_by_timeout_task
            await self._db_reconnect_lock.acquire()
            lock_acquired_by_timeout_task = True
            return True

        acquire_task: Final = asyncio.create_task(_acquire_reconnect_lock())
        done, _pending = await asyncio.wait(
            {acquire_task},
            timeout=lock_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acquire_task not in done:
            acquire_task.cancel()
            try:
                await acquire_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

            # Defensive cleanup for timeout/cancel race on Python 3.9-3.11.
            if lock_acquired_by_timeout_task:
                try:
                    self._db_reconnect_lock.release()
                except RuntimeError:
                    pass
            verbose_proxy_logger.debug(
                "Skipping DB reconnect attempt due to lock acquisition timeout. reason=%s timeout=%ss",
                reason,
                lock_timeout_seconds,
            )
            return False

        try:
            acquire_task.result()
        except Exception as lock_acquire_err:
            verbose_proxy_logger.debug(
                "Skipping DB reconnect attempt due to lock acquisition error. reason=%s error=%s",
                reason,
                lock_acquire_err,
            )
            return False

        try:
            return await self._attempt_reconnect_inside_lock(
                force, reason, timeout_seconds, force_recreate, stale_read_engine
            )
        finally:
            self._db_reconnect_lock.release()

    async def start_db_health_watchdog_task(self) -> None:
        """Start background tasks that monitor DB health:
        - A periodic SELECT 1 probe that triggers reconnect on network/connection failure.
        - A process-level watcher that detects engine death via waitpid thread, pidfd, or os.kill polling.
        """
        if self._db_health_watchdog_enabled is not True:
            verbose_proxy_logger.debug("Prisma DB health watchdog disabled via PRISMA_HEALTH_WATCHDOG_ENABLED")
            return
        if self._db_health_watchdog_task is not None:
            return
        # Let planned writer-engine restarts (IAM token refresh, guarded
        # reconnect) re-arm the watcher on the new PID instead of being
        # mistaken for a crash (issue #29176). Set on the writer wrapper since
        # the watcher tracks the writer engine.
        self.writer_db.on_engine_replaced = self._handle_writer_engine_replaced
        self._db_health_watchdog_task = asyncio.create_task(self._db_health_watchdog_loop())
        verbose_proxy_logger.info(
            "Started Prisma DB health watchdog (interval=%ss, reconnect_cooldown=%ss, probe_timeout=%ss, reconnect_timeout=%ss)",
            self._db_health_watchdog_interval_seconds,
            self._db_reconnect_cooldown_seconds,
            self._db_health_watchdog_probe_timeout_seconds,
            self._db_watchdog_reconnect_timeout_seconds,
        )
        await self._start_engine_watcher()

    async def stop_db_health_watchdog_task(self) -> None:
        """Stop DB health watchdog task and engine watcher gracefully."""
        self._stop_engine_watcher()
        if self._db_health_watchdog_task is None:
            return
        self._db_health_watchdog_task.cancel()
        try:
            await self._db_health_watchdog_task
        except asyncio.CancelledError:
            pass
        self._db_health_watchdog_task = None
        verbose_proxy_logger.info("Stopped Prisma DB health watchdog")

    async def _db_health_watchdog_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._db_health_watchdog_interval_seconds)
                await asyncio.wait_for(
                    self.db.query_raw("SELECT 1"),
                    timeout=self._db_health_watchdog_probe_timeout_seconds,
                )
                if isinstance(self.db, RoutingPrismaWrapper) and self.db.writer_unavailable:
                    await self.attempt_db_reconnect(
                        reason="db_health_watchdog_writer_unavailable",
                        timeout_seconds=self._db_watchdog_reconnect_timeout_seconds,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                if isinstance(e, asyncio.TimeoutError) or PrismaDBExceptionHandler.is_database_infrastructure_error(e):
                    await self.attempt_db_reconnect(
                        reason="db_health_watchdog_connection_error",
                        timeout_seconds=self._db_watchdog_reconnect_timeout_seconds,
                    )
                else:
                    verbose_proxy_logger.debug("Prisma DB health watchdog observed non-DB error: %s", e)

    def _probe_target_wrapper(self) -> PrismaWrapper:
        """The Prisma wrapper a `SELECT 1` health probe actually reaches.

        `health_check()` issues `query_raw`, which `RoutingPrismaWrapper` sends
        to the reader unless the reader is degraded. The writer's engine state
        therefore says nothing about a probe that failed against the reader, so
        the gate has to follow the same routing rule the probe did.
        """
        if isinstance(self.db, RoutingPrismaWrapper):
            return self.db.writer if self.db.reader_unavailable else self.db.reader
        return self.db

    async def _run_health_probe(self, wrapper: PrismaWrapper) -> object:
        """Issue the `SELECT 1` a health check is made of, against `wrapper`.

        Takes the wrapper rather than re-reading `self.db`, because routing is
        re-resolved on every attribute access: a reader that recovers between
        the caller picking its target and the query going out would send the
        probe to a different engine than the one whose generation the caller is
        about to check, and attribute the failure to the wrong replacement.
        """
        sql_query: Final = "SELECT 1"
        response: Final[object] = await wrapper.query_raw(sql_query)
        return response

    async def _probe_answers_now(self, wrapper: PrismaWrapper) -> bool:
        try:
            await self._run_health_probe(wrapper)
        except Exception as probe_error:  # noqa: BLE001  # any failure means the database is not answering
            verbose_proxy_logger.debug("Prisma health_check() confirmation probe failed: %s", probe_error)
            return False
        return True

    async def _planned_engine_replacement_absorbed(
        self,
        e: Exception,
        wrapper: PrismaWrapper,
        generation_before: int,
    ) -> bool:
        """True iff `e` is a connection-class probe failure that a completed
        planned query-engine replacement explains.

        Planned replacements (RDS IAM token refresh, guarded reconnect) kill the
        running query engine and spawn a new one. A `SELECT 1` probe that races
        that sub-second window fails with a transport error against the engine's
        local HTTP port even though nothing is wrong with the database, and
        reporting it drives a false-positive `db_exceptions` alert on every
        replacement.

        Two things must both hold, because neither is sufficient alone. The
        engine generation must have moved, which says a replacement completed
        rather than merely being attempted: reconnect attempts during a real
        outage hold the same lock for tens of seconds, so gating on an in-flight
        replacement would swallow most of an outage's alerts. And a fresh probe
        must succeed, because `Prisma.connect()` polls the query engine's own
        `/status` endpoint rather than round-tripping to the database, so a
        future engine that binds before it validates its connection pool would
        let the generation advance with the database still unreachable.

        Waiting for an in-flight replacement to settle is what makes the
        generation check meaningful, since the generation has not moved yet at
        the instant the probe fails. The wait is generous against a replacement
        that takes well under a second and short enough that an outage-hung
        reconnect is not waited out; a replacement that has not settled by then
        reports rather than stays silent.
        """
        if not PrismaDBExceptionHandler.is_database_connection_error(e):
            return False
        await wrapper.wait_for_planned_engine_replacement(self.PLANNED_ENGINE_REPLACEMENT_SETTLE_SECONDS)
        if wrapper.engine_generation == generation_before:
            return False
        return await self._probe_answers_now(wrapper)

    async def _report_health_check_failure(
        self,
        e: Exception,
        duration: float,
        traceback_str: str,
        wrapper: PrismaWrapper,
        generation_before: int,
    ) -> None:
        if await self._planned_engine_replacement_absorbed(e, wrapper, generation_before):
            verbose_proxy_logger.info(
                "Prisma health_check() connection error raced a planned query-engine replacement; "
                "not reporting it as a DB exception: %s",
                e,
            )
            return
        await self.proxy_logging_obj.failure_handler(
            original_exception=e,
            duration=duration,
            call_type="health_check",
            traceback_str=traceback_str,
        )

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=10,
        on_backoff=on_backoff,
    )
    async def health_check(self):
        """
        Health check endpoint for the prisma client
        """
        start_time: Final = time.time()
        probe_wrapper: Final = self._probe_target_wrapper()
        generation_before: Final = probe_wrapper.engine_generation
        try:
            return await self._run_health_probe(probe_wrapper)
        except Exception as e:
            import traceback

            error_msg: Final = f"LiteLLM Prisma Client Exception health_check(): {e}"
            verbose_proxy_logger.warning(error_msg)
            error_traceback: Final = error_msg + "\n" + traceback.format_exc()
            end_time: Final = time.time()
            _duration: Final = end_time - start_time
            asyncio.create_task(
                self._report_health_check_failure(
                    e=e,
                    duration=_duration,
                    traceback_str=error_traceback,
                    wrapper=probe_wrapper,
                    generation_before=generation_before,
                )
            )
            raise e

    async def _get_spend_logs_row_count(self) -> int:
        """
        Get the row count from LiteLLM_SpendLogs table using PostgreSQL system statistics.
        """

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=3,
            max_time=10,
            on_backoff=on_backoff,
        )
        async def _fetch_row_count() -> int:
            sql_query: Final = """
            SELECT reltuples::BIGINT
            FROM pg_class
            WHERE oid = '"LiteLLM_SpendLogs"'::regclass;
            """
            result: Final[Sequence[_RelTuplesRow]] = await self.db.query_raw(query=sql_query)
            return result[0]["reltuples"]

        try:
            return await _fetch_row_count()
        except Exception as e:
            verbose_proxy_logger.error("Error getting LiteLLM_SpendLogs row count: %s", e)
            return 0

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=10,
        on_backoff=on_backoff,
    )
    async def _set_spend_logs_row_count_in_proxy_state(self) -> None:
        """
        Set the `LiteLLM_SpendLogs`row count in proxy state.

        This is used later to determine if we should run expensive UI Usage queries.
        """
        from litellm.proxy.proxy_server import proxy_state

        _num_spend_logs_rows: Final = await self._get_spend_logs_row_count()
        proxy_state.set_proxy_state_variable(
            variable_name="spend_logs_row_count",
            value=_num_spend_logs_rows,
        )

    # Health Check Database Methods
    def _validate_response_time(self, response_time_ms: float | None) -> float | None:
        """Validate and clean response time value"""
        if response_time_ms is None:
            return None
        try:
            value: Final = float(response_time_ms)
            return value if value == value and value not in (float("inf"), float("-inf")) else None
        except (ValueError, TypeError):
            verbose_proxy_logger.warning("Invalid response_time_ms value: %s", response_time_ms)
            return None

    def _clean_details(self, details: dict | None) -> dict | None:
        """Clean and validate details JSON"""
        if not isinstance(details, dict):
            return None
        try:
            return safe_json_loads(safe_dumps(details))
        except Exception as e:
            verbose_proxy_logger.warning("Failed to clean details JSON: %s", e)
            return None

    async def save_health_check_result(
        self,
        model_name: str,
        status: str,
        healthy_count: int = 0,
        unhealthy_count: int = 0,
        error_message: str | None = None,
        response_time_ms: float | None = None,
        details: dict | None = None,
        checked_by: str | None = None,
        model_id: str | None = None,
    ):
        """Save health check result to database"""
        try:
            # Build base data with required fields
            health_check_data: Final = {
                "model_name": str(model_name),
                "status": str(status),
                "healthy_count": int(healthy_count),
                "unhealthy_count": int(unhealthy_count),
            }

            # Add optional fields using dict comprehension and helper methods
            optional_fields: Final = {
                "error_message": str(error_message)[:500] if error_message else None,
                "response_time_ms": self._validate_response_time(response_time_ms),
                "details": self._clean_details(details),
                "checked_by": str(checked_by) if checked_by else None,
                "model_id": str(model_id) if model_id else None,
            }

            # Add only non-None optional fields
            health_check_data.update({k: v for k, v in optional_fields.items() if v is not None})

            verbose_proxy_logger.debug("Saving health check data: %s", health_check_data)
            return await HealthCheckRepository(self).table.create(data=health_check_data)

        except Exception as e:
            verbose_proxy_logger.error("Error saving health check result for model %s: %s", model_name, e)
            return None

    async def get_health_check_history(
        self,
        model_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
        status_filter: str | None = None,
    ) -> "Sequence[prisma_models.LiteLLM_HealthCheckTable]":
        """
        Get health check history with optional filtering
        """
        try:
            where_clause: Final[dict[str, str]] = {}
            if model_name:
                where_clause["model_name"] = model_name
            if status_filter:
                where_clause["status"] = status_filter

            results: Final = await HealthCheckRepository(self).table.find_many(
                where=where_clause,
                order={"checked_at": "desc"},
                take=limit,
                skip=offset,
            )
            return results
        except Exception as e:
            verbose_proxy_logger.error("Error getting health check history: %s", e)
            return []

    async def get_all_latest_health_checks(self) -> "Sequence[prisma_models.LiteLLM_HealthCheckTable]":
        """
        Get the latest health check for each model.

        Uses DB-level DISTINCT ON (model_id, model_name) with ORDER BY checked_at DESC
        (via Prisma ``distinct`` + ``order``) so we never load the full history into memory.
        """
        try:
            return await HealthCheckRepository(self).table.find_many(
                distinct=["model_id", "model_name"],
                order=[
                    {"model_id": "asc"},
                    {"model_name": "asc"},
                    {"checked_at": "desc"},
                ],
            )
        except Exception as e:
            verbose_proxy_logger.error("Error getting all latest health checks: %s", e)
            return []


### HELPER FUNCTIONS ###


async def _cache_user_row(user_id: str, cache: DualCache, db: PrismaClient):
    """
    Check if a user_id exists in cache,
    if not retrieve it.
    """
    cache_key: Final = f"{user_id}_user_api_key_user_id"
    response: Final = cache.get_cache(key=cache_key)
    if response is None:  # Cache miss
        user_row: Final = await db.get_data(user_id=user_id)
        if user_row is not None:
            print_verbose(f"User Row: {user_row}, type = {type(user_row)}")
            if hasattr(user_row, "model_dump_json") and callable(getattr(user_row, "model_dump_json")):
                cache_value: Final[str] = user_row.model_dump_json()
                cache.set_cache(key=cache_key, value=cache_value, ttl=600)  # store for 10 minutes


def _should_use_smtp_ssl(smtp_port: int) -> bool:
    """
    Port 465 expects an immediate TLS handshake (implicit SSL), so a plain
    smtplib.SMTP connection hangs waiting for an SMTP banner. Use SMTP_SSL
    there, or when SMTP_USE_SSL is explicitly enabled.
    """
    return os.getenv("SMTP_USE_SSL", "False") == "True" or smtp_port == 465


def _create_smtp_connection(smtp_host: str, smtp_port: int) -> smtplib.SMTP:
    if _should_use_smtp_ssl(smtp_port=smtp_port):
        return smtplib.SMTP_SSL(host=smtp_host, port=smtp_port, context=ssl.create_default_context())
    return smtplib.SMTP(host=smtp_host, port=smtp_port)


async def send_email(
    receiver_email: str | None = None,
    subject: str | None = None,
    html: str | None = None,
):
    """
    smtp_host,
    smtp_port,
    smtp_username,
    smtp_password,
    sender_name,
    sender_email,
    """
    ## SERVER SETUP ##

    smtp_host: Final = os.getenv("SMTP_HOST")
    smtp_port: Final = int(os.getenv("SMTP_PORT", "587"))  # default to port 587
    smtp_username: Final = os.getenv("SMTP_USERNAME")
    smtp_password: Final = os.getenv("SMTP_PASSWORD")
    sender_email: Final = os.getenv("SMTP_SENDER_EMAIL", None)
    if sender_email is None:
        raise ValueError("Trying to use SMTP, but SMTP_SENDER_EMAIL is not set")
    if receiver_email is None:
        raise ValueError(f"No receiver email provided for SMTP email. {receiver_email}")
    if subject is None:
        raise ValueError(f"No subject provided for SMTP email. {subject}")
    if html is None:
        raise ValueError(f"No HTML body provided for SMTP email. {html}")

    ## EMAIL SETUP ##
    email_message: Final = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject
    verbose_proxy_logger.debug("sending email from %s to %s", sender_email, receiver_email)

    if smtp_host is None:
        raise ValueError("Trying to use SMTP, but SMTP_HOST is not set")

    # Attach the body to the email
    email_message.attach(MIMEText(html, "html"))

    try:
        using_ssl: Final = _should_use_smtp_ssl(smtp_port=smtp_port)
        with _create_smtp_connection(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
        ) as server:
            if not using_ssl and os.getenv("SMTP_TLS", "True") != "False":
                server.starttls(context=ssl.create_default_context())

            # Login to your email account only if smtp_username and smtp_password are provided
            if smtp_username and smtp_password:
                server.login(
                    user=smtp_username,
                    password=smtp_password,
                )

            # Send the email
            server.send_message(
                msg=email_message,
                from_addr=sender_email,
                to_addrs=receiver_email,
            )

    except Exception as e:
        verbose_proxy_logger.exception("An error occurred while sending the email:" + str(e))


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token: Final = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


def hash_password(password: str) -> str:
    """Hash a password using scrypt with a random salt."""
    import base64
    import hashlib
    import os

    salt: Final = os.urandom(16)
    dk: Final = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "scrypt:" + base64.b64encode(salt + dk).decode()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash. Supports scrypt and SHA256."""
    import base64
    import hashlib
    import secrets

    if stored.startswith("scrypt:"):
        try:
            raw: Final = base64.b64decode(stored[7:])
            salt, dk = raw[:16], raw[16:]
            dk2: Final = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
            return secrets.compare_digest(dk, dk2)
        except Exception:
            return False
    # SHA256 fallback (not vulnerable to pass-the-hash: checks sha256(input) == stored)
    if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored):
        return secrets.compare_digest(hashlib.sha256(password.encode()).hexdigest().encode(), stored.encode())
    return False


async def migrate_passwords_to_scrypt_async(prisma_client) -> str:
    """
    Migrate plaintext passwords in the DB to scrypt. SHA256 passwords
    are left alone (they migrate on next login via the SHA256 fallback).
    Skips quickly if no plaintext passwords exist.
    """
    all_with_pw: Final = await UserRepository(prisma_client).table.find_many(
        where={"password": {"not": None}},
    )

    def _is_sha256_hex(s: str) -> bool:
        return len(s) == 64 and all(c in "0123456789abcdef" for c in s)

    plaintext_users: Final = [
        (u.user_id, u.password)
        for u in all_with_pw
        if u.password and not u.password.startswith("scrypt:") and not _is_sha256_hex(u.password)
    ]
    if not plaintext_users:
        return "No plaintext passwords found"

    for user_id, plaintext_password in plaintext_users:
        await UserRepository(prisma_client).table.update(
            where={"user_id": user_id},
            data={"password": hash_password(plaintext_password)},
        )
    return f"Migrated {len(plaintext_users)} plaintext passwords to scrypt"


def _hash_token_if_needed(token: str) -> str:
    """
    Hash the token if it's a string and starts with "sk-"

    Else return the token as is
    """
    if token.startswith("sk-"):
        return hash_token(token=token)
    else:
        return token


async def enqueue_spend_logs(
    prisma_client: PrismaClient,
    logs: Sequence[Mapping[str, object]],
    *,
    at_head: bool = False,
    max_bytes: int = SPEND_LOG_QUEUE_MAX_BYTES,
) -> None:
    """Queue spend logs for the next flush, held under ``SPEND_LOG_QUEUE_MAX_BYTES``.

    ``at_head`` replays a batch the DB refused, so it flushes before the logs
    that piled up during the outage. Past the budget the oldest logs are
    dropped, which keeps a long outage from growing the queue until the pod
    dies.
    """
    added: Final = sum(spend_log_row_bytes(row) for row in logs)
    async with prisma_client._spend_log_transactions_lock:
        queued: Final = (
            tuple(logs) + tuple(prisma_client.spend_log_transactions)
            if at_head
            else tuple(prisma_client.spend_log_transactions) + tuple(logs)
        )
        kept, kept_bytes = spend_log_queue_within_budget(queued, PrismaClient.spend_log_queue_bytes + added, max_bytes)
        prisma_client.spend_log_transactions[:] = kept
        PrismaClient.spend_log_queue_bytes = kept_bytes
    if len(kept) < len(queued):
        verbose_proxy_logger.error(
            "Spend tracking - spend log queue is at its %d byte budget; dropped the %d oldest spend logs",
            max_bytes,
            len(queued) - len(kept),
        )


def request_spend_log_flush() -> None:
    """Wake the queue monitor now rather than leaving the rows for its next poll.

    The Responses API hands the client an id it can chain from straight away, and that
    lookup reads the DB, so the row cannot sit in this worker's queue for a poll interval.
    Repeated requests coalesce into the monitor's next pass, so the batching holds.
    """
    PrismaClient.spend_log_flush_requested.set()


async def _wait_for_spend_log_flush_request(interval: float) -> bool:
    """Wait out ``interval``, returning early and True when a flush was requested."""
    try:
        await asyncio.wait_for(PrismaClient.spend_log_flush_requested.wait(), timeout=interval)
    except asyncio.TimeoutError:
        return False
    PrismaClient.spend_log_flush_requested.clear()
    return True


async def dequeue_spend_logs(prisma_client: PrismaClient, limit: int) -> list[dict[str, object]]:
    """Take up to ``limit`` of the oldest queued spend logs off the queue.

    Every enqueue and dequeue goes through this pair so the byte total the
    queue is bounded by stays in step with what the queue actually holds.
    """
    async with prisma_client._spend_log_transactions_lock:
        popped: Final = prisma_client.spend_log_transactions[:limit]
        prisma_client.spend_log_transactions[:] = prisma_client.spend_log_transactions[limit:]
        PrismaClient.spend_log_queue_bytes = max(
            0, PrismaClient.spend_log_queue_bytes - sum(spend_log_row_bytes(row) for row in popped)
        )
    return popped


class ProxyUpdateSpend:
    @staticmethod
    async def update_end_user_spend(
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        end_user_list_transactions: dict[str, float],
    ):
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(timeout=timedelta(seconds=60)) as transaction:
                    batcher: _EndUserSpendBatch
                    async with transaction.batch_() as batcher:
                        # Sort by end_user_id for consistent lock ordering across pods to prevent deadlocks.
                        for end_user_id, response_cost in sorted(end_user_list_transactions.items()):
                            if litellm.max_end_user_budget is not None:
                                pass
                            batcher.litellm_endusertable.upsert(
                                where={"user_id": end_user_id},
                                data={
                                    "create": {
                                        "user_id": end_user_id,
                                        "spend": response_cost,
                                        "blocked": False,
                                    },
                                    "update": {"spend": {"increment": response_cost}},
                                },
                            )

                break
            except Exception as e:
                await DBSpendUpdateWriter._handle_spend_update_failure(
                    e=e,
                    attempt=i,
                    n_retry_times=n_retry_times,
                    start_time=start_time,
                    proxy_logging_obj=proxy_logging_obj,
                )

    @staticmethod
    async def update_spend_logs(
        n_retry_times: int,
        prisma_client: PrismaClient,
        db_writer_client: AsyncHTTPHandler | None,
        proxy_logging_obj: ProxyLogging,
        logs_to_process: list[dict[str, object]] | None = None,
    ):
        BATCH_SIZE: Final = 1000  # Preferred size of each batch to write to the database
        MAX_LOGS_PER_INTERVAL: Final = 10000  # Maximum number of logs to flush in a single interval
        popped_batch = False
        if logs_to_process is None:
            logs_to_process = await dequeue_spend_logs(prisma_client, MAX_LOGS_PER_INTERVAL)
            popped_batch = True
        if len(logs_to_process) > 0:
            verbose_proxy_logger.info(
                "Spend tracking - processing %d spend logs for DB write",
                len(logs_to_process),
            )
        start_time: Final = time.time()
        try:
            for i in range(n_retry_times + 1):
                try:
                    base_url = os.getenv("SPEND_LOGS_URL", None)
                    if len(logs_to_process) > 0 and base_url is not None and db_writer_client is not None:
                        if not base_url.endswith("/"):
                            base_url += "/"
                        verbose_proxy_logger.debug("base_url: %s", base_url)
                        json_data = json.dumps(logs_to_process)
                        response = await db_writer_client.post(
                            url=base_url + "spend/update",
                            data=json_data,
                            headers={"Content-Type": "application/json"},
                        )
                        del json_data
                        if response.status_code == 200:
                            # Items already removed from queue at start of function
                            pass
                    else:
                        for j in range(0, len(logs_to_process), BATCH_SIZE):
                            batch = logs_to_process[j : j + BATCH_SIZE]
                            batch_with_dates = [prisma_client.jsonify_object({**entry}) for entry in batch]
                            isolation_budget = MAX_SPEND_LOG_ISOLATION_FAILURES_PER_BATCH
                            for statement_rows in spend_log_write_batches(
                                batch_with_dates,
                                SPEND_LOG_WRITE_BATCH_MAX_BYTES,
                                SPEND_LOG_WRITE_BATCH_MAX_ROWS,
                            ):
                                isolation_budget = await _create_spend_logs_with_poison_isolation(
                                    SpendLogsRepository(prisma_client),
                                    statement_rows,
                                    isolation_budget,
                                )
                            verbose_proxy_logger.debug("Flushed %s logs to the DB.", len(batch))
                            # Explicitly clear batch memory
                            del batch, batch_with_dates

                        # Items already removed from queue at start of function
                        async with prisma_client._spend_log_transactions_lock:
                            remaining_count = len(prisma_client.spend_log_transactions)
                        verbose_proxy_logger.debug(
                            "%s logs processed. Remaining in queue: %s", len(logs_to_process), remaining_count
                        )
                    break
                except Exception as e:
                    if not PrismaDBExceptionHandler.is_database_transport_error(e):
                        raise
                    verbose_proxy_logger.warning(
                        "Spend tracking - DB connection error writing spend logs, retry %d/%d. logs_count=%d, error=%s",
                        i + 1,
                        n_retry_times,
                        len(logs_to_process),
                        str(e),
                    )
                    if i >= n_retry_times:
                        await enqueue_spend_logs(prisma_client, logs_to_process, at_head=True)
                        raise
                    await asyncio.sleep(2**i)
        except Exception as e:
            _raise_failed_update_spend_exception(e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj)
        finally:
            # Clean up logs_to_process only if we popped it (caller-owned otherwise)
            if popped_batch:
                del logs_to_process

    @staticmethod
    def disable_spend_updates() -> bool:
        """
        returns True if should not update spend in db
        Skips writing spend logs and updates to key, team, user spend to DB
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_spend_updates") is True:
            return True
        return False


async def update_spend(
    prisma_client: PrismaClient,
    db_writer_client: AsyncHTTPHandler | None,
    proxy_logging_obj: ProxyLogging,
):
    """
    Batch write updates to db.

    Triggered every minute.

    NOTE: This job now skips tag spend updates, which are handled by a separate
    scheduler job (update_daily_tag_spend) at a longer interval to reduce contention.

    Requires:
    user_id_list: dict,
    keys_list: list,
    team_list: list,
    spend_logs: list,
    """
    n_retry_times: Final = 3
    await proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler(
        prisma_client=prisma_client,
        n_retry_times=n_retry_times,
        proxy_logging_obj=proxy_logging_obj,
    )

    ### UPDATE SPEND LOGS ###
    # Check queue size with lock protection
    queue_size: Final = await _total_queued_spend_transactions(prisma_client)
    verbose_proxy_logger.debug("Spend Logs transactions: %s", queue_size)

    # Process spend log transactions when called directly.
    # This keeps backwards compatibility with the old behavior.
    # See update_spend_logs_job and _monitor_spend_logs_queue for the new behavior.
    # Safe to keep: under high concurrency this can take up to ~30s to run,
    # so it's unlikely to overlap with monitor_spend_logs_queue.
    if queue_size > 0:
        await update_spend_logs_job(
            prisma_client=prisma_client,
            db_writer_client=db_writer_client,
            proxy_logging_obj=proxy_logging_obj,
        )


async def _total_queued_spend_transactions(prisma_client: PrismaClient) -> int:
    """Pending entries across every request-time spend queue, sized under each queue's
    lock. Every drain trigger reads this one owner, so a queue added later joins the
    direct path, the batch job's emptiness check and the monitor at once."""
    async with prisma_client._spend_log_transactions_lock:
        spend_queue_size: Final = len(prisma_client.spend_log_transactions)
    async with prisma_client._tool_usage_transactions_lock:
        tool_queue_size: Final = len(prisma_client.tool_usage_transactions)
    async with prisma_client._autorouter_turn_transactions_lock:
        autorouter_queue_size: Final = len(prisma_client.autorouter_turn_transactions)
    return spend_queue_size + tool_queue_size + autorouter_queue_size


async def update_daily_tag_spend(
    prisma_client: PrismaClient,
    proxy_logging_obj: ProxyLogging,
):
    """
    Separate scheduler job to commit daily tag spend updates.

    Runs at a longer interval (2.3x default) than the main update_spend job
    to reduce query contention for DailyTagSpend table.

    This is called by a dedicated scheduler job and does NOT process:
    - Regular spend updates (user, key, team, org)
    - End-user spend
    - Agent spend
    - Spend logs

    Only processes tag spend transactions from the daily_tag_spend_update_queue.

    Args:
        prisma_client: PrismaClient instance
        proxy_logging_obj: ProxyLogging instance for error handling
    """
    n_retry_times: Final = 3
    try:
        if proxy_logging_obj.db_spend_update_writer.redis_update_buffer._should_commit_spend_updates_to_redis():
            await proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db_with_redis(
                prisma_client=prisma_client,
                n_retry_times=n_retry_times,
                proxy_logging_obj=proxy_logging_obj,
            )
        else:
            await proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db(
                prisma_client=prisma_client,
                n_retry_times=n_retry_times,
                proxy_logging_obj=proxy_logging_obj,
            )
    except Exception as e:
        # NOTE: keep this as a plain ``error`` (no traceback) to match the
        # historical behavior of this site. ``spend_log_error`` would attach
        # the active exception's traceback whenever the suppression env var
        # is unset, which would be a regression for operators who never saw
        # one here before.
        verbose_proxy_logger.error("Error updating daily tag spend: %s", e)


async def update_spend_logs_job(
    prisma_client: PrismaClient,
    db_writer_client: AsyncHTTPHandler | None,
    proxy_logging_obj: ProxyLogging,
):
    """
    Job to process spend_log_transactions queue.

    This job is triggered based on queue size rather than time.
    Pops the batch once, writes spend logs, then runs guardrail usage tracking.
    """
    n_retry_times: Final = 3
    MAX_LOGS_PER_INTERVAL: Final = 10000

    # Atomically pop batch from queue. The tool usage queue counts toward the
    # emptiness check: a spend-log write failure aborts a run before the tool
    # drain below, and those entries must not strand once the spend queue drains.
    if await _total_queued_spend_transactions(prisma_client) == 0:
        return

    logs_to_process: Final = await dequeue_spend_logs(prisma_client, MAX_LOGS_PER_INTERVAL)

    try:
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            db_writer_client=db_writer_client,
            logs_to_process=logs_to_process,
        )
    except asyncio.CancelledError:
        await enqueue_spend_logs(prisma_client, logs_to_process, at_head=True)
        verbose_proxy_logger.warning(
            "Spend tracking - spend log write cancelled, requeued %d rows for the next flush",
            len(logs_to_process),
        )
        raise

    # Guardrail/policy usage tracking (same batch, outside spend-logs update)
    try:
        from litellm.proxy.guardrails.usage_tracking import (
            process_spend_logs_guardrail_usage,
        )

        await process_spend_logs_guardrail_usage(
            prisma_client=prisma_client,
            logs_to_process=logs_to_process,
        )
    except Exception as guardrail_tracking_err:
        verbose_proxy_logger.warning(
            "Spend tracking - guardrail usage tracking failed (non-fatal): %s",
            guardrail_tracking_err,
        )

    # Tool usage tracking: drain the request-time queue into the tool index and the
    # LiteLLM_DailyToolSpend rollup. Never retried; a dropped batch is permanently
    # absent from the rollup, so failures log at error.
    async with prisma_client._tool_usage_transactions_lock:
        tool_usage_to_process: Final = prisma_client.tool_usage_transactions[:MAX_LOGS_PER_INTERVAL]
        prisma_client.tool_usage_transactions = prisma_client.tool_usage_transactions[len(tool_usage_to_process) :]
    try:
        from litellm.proxy.db.spend_log_tool_index import flush_tool_usage_transactions

        await flush_tool_usage_transactions(
            prisma_client=prisma_client,
            transactions=tool_usage_to_process,
        )
    except Exception as tool_tracking_err:
        verbose_proxy_logger.error(
            "Spend tracking - tool usage flush failed; %s tool usage transactions dropped: %s",
            len(tool_usage_to_process),
            tool_tracking_err,
        )

    async with prisma_client._autorouter_turn_transactions_lock:
        autorouter_turns_to_process: Final = prisma_client.autorouter_turn_transactions[:MAX_LOGS_PER_INTERVAL]
        remaining_autorouter_turns: Final = prisma_client.autorouter_turn_transactions[
            len(autorouter_turns_to_process) :
        ]
        prisma_client.autorouter_turn_transactions = remaining_autorouter_turns  # rebind-ok: drain under lock
    try:
        from litellm.proxy.db.autorouter_session_rollup import flush_autorouter_turn_transactions

        await flush_autorouter_turn_transactions(
            prisma_client=prisma_client,
            transactions=autorouter_turns_to_process,
        )
    except Exception as autorouter_tracking_err:  # noqa: BLE001  # a drain bug must not abort the spend job
        verbose_proxy_logger.error(
            "Spend tracking - auto-router session rollup drain failed; %s turn transactions dropped: %s",
            len(autorouter_turns_to_process),
            autorouter_tracking_err,
        )


MAX_SPEND_LOG_DRAIN_ITERATIONS: Final = 20


async def drain_spend_logs_queue(
    prisma_client: PrismaClient,
    db_writer_client: "AsyncHTTPHandler | None",
    proxy_logging_obj: ProxyLogging,
) -> None:
    monitor_task: Final = prisma_client.spend_logs_queue_monitor_task
    if monitor_task is not None:
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task
        prisma_client.spend_logs_queue_monitor_task = None  # rebind-ok: the client owns its monitor handle

    for _ in range(MAX_SPEND_LOG_DRAIN_ITERATIONS):
        if await _total_queued_spend_transactions(prisma_client) == 0:
            return
        await update_spend_logs_job(
            prisma_client=prisma_client,
            db_writer_client=db_writer_client,
            proxy_logging_obj=proxy_logging_obj,
        )

    remaining: Final = await _total_queued_spend_transactions(prisma_client)
    if remaining > 0:
        spend_log_error(
            "Spend tracking - %d spend log rows still queued after %d drain passes",
            remaining,
            MAX_SPEND_LOG_DRAIN_ITERATIONS,
        )


async def _monitor_spend_logs_queue(
    prisma_client: PrismaClient,
    db_writer_client: AsyncHTTPHandler | None,
    proxy_logging_obj: ProxyLogging,
):
    """
    Background task that monitors the spend_log_transactions queue size
    and triggers processing when the threshold is reached.

    Args:
        prisma_client: Prisma client instance
        db_writer_client: Optional HTTP handler for external spend logs endpoint
        proxy_logging_obj: Proxy logging object
    """
    from litellm.constants import (
        SPEND_LOG_QUEUE_POLL_INTERVAL,
        SPEND_LOG_QUEUE_SIZE_THRESHOLD,
    )

    threshold: Final = SPEND_LOG_QUEUE_SIZE_THRESHOLD
    base_interval: Final = SPEND_LOG_QUEUE_POLL_INTERVAL
    max_backoff: Final = 30.0  # Maximum backoff interval in seconds
    backoff_multiplier: Final = 1.5  # Exponential backoff multiplier
    current_interval = base_interval

    verbose_proxy_logger.info(
        "Starting spend logs queue monitor (threshold: %s, poll_interval: %ss)", threshold, base_interval
    )

    while True:
        try:
            # Check queue sizes with lock protection; the tool usage queue keeps
            # the monitor firing when a prior failed run left it nonempty.
            queue_size = await _total_queued_spend_transactions(prisma_client)

            if queue_size > 0:
                if queue_size >= threshold:
                    verbose_proxy_logger.debug(
                        "Spend logs queue size (%s) reached threshold (%s), triggering processing",
                        queue_size,
                        threshold,
                    )
                    # Reset to base interval when threshold is reached
                    current_interval = base_interval
                else:
                    verbose_proxy_logger.debug(
                        "Spend logs queue size (%s) below threshold (%s), processing with backoff",
                        queue_size,
                        threshold,
                    )
                    # Exponential backoff when below threshold but still processing
                    current_interval = min(current_interval * backoff_multiplier, max_backoff)

                await update_spend_logs_job(
                    prisma_client=prisma_client,
                    db_writer_client=db_writer_client,
                    proxy_logging_obj=proxy_logging_obj,
                )
            else:
                # Exponential backoff when no logs to process
                current_interval = min(current_interval * backoff_multiplier, max_backoff)

            if await _wait_for_spend_log_flush_request(current_interval):
                current_interval = base_interval
        except Exception as e:
            spend_log_error("Error in spend logs queue monitor: %s", str(e), exc=e)
            # Continue monitoring even if there's an error, with exponential backoff
            current_interval = min(current_interval * backoff_multiplier, max_backoff)
            await asyncio.sleep(current_interval)


MAX_SPEND_LOG_ISOLATION_FAILURES_PER_BATCH: Final = 256


async def _create_spend_logs_with_poison_isolation(
    repo: SpendLogsRepository,
    rows: Sequence[Mapping[str, object]],
    failure_budget: int,
) -> int:
    """Write spend-log rows, isolating any row Postgres rejects on its data.

    ``create_many`` writes the whole batch in a single statement, so one row
    carrying bytes Postgres refuses (a residual NUL byte is the canonical case)
    fails the entire insert and drops every good row alongside it. On a genuine
    data-layer rejection the batch is bisected so the good rows still persist
    and only the offending row is dropped and logged. Transport failures,
    including the "can't reach database server" outage that prisma mislabels as
    a ``DataError``, are re-raised unchanged so the caller's connection-retry
    path still runs.

    ``failure_budget`` caps the *failed* inserts the isolation may issue, which
    is the work an authenticated caller flooding poisoned rows can amplify. The
    one insert a statement needs when nothing is poisoned is not charged, so a
    caller can thread a single budget through every statement of a flush and
    bound the whole flush's failed inserts and log lines by the initial value,
    without a large healthy flush ever running out and losing rows. When the
    budget is spent the still-failing remainder is dropped wholesale (the
    pre-existing drop-the-batch behavior) under one log line, and a statement
    reached afterwards is still attempted, so clean rows behind a poison flood
    persist. Returns the budget left after this subtree.
    """
    try:
        await repo.table.create_many(data=rows, skip_duplicates=True)
        return failure_budget
    except Exception as e:
        if not PrismaDBExceptionHandler.is_prisma_data_error(e):
            raise
        if PrismaDBExceptionHandler.is_database_service_unavailable_error(e):
            raise
        budget_left: Final = max(failure_budget - 1, 0)
        if len(rows) == 1:
            request_id: Final = rows[0].get("request_id")
            spend_log_error(
                "Spend tracking - dropping spend log row Postgres rejected. request_id=%s error=%s",
                request_id,
                str(e),
                exc=e,
            )
            return budget_left
        if budget_left <= 0:
            spend_log_error(
                "Spend tracking - dropping %d spend log rows without per-row isolation; "
                "isolation failure budget exhausted for this flush",
                len(rows),
            )
            return 0
        mid: Final = len(rows) // 2
        remaining: Final = await _create_spend_logs_with_poison_isolation(repo, rows[:mid], budget_left)
        if remaining <= 0:
            spend_log_error(
                "Spend tracking - dropping %d spend log rows without per-row isolation; "
                "isolation failure budget exhausted for this flush",
                len(rows) - mid,
            )
            return 0
        return await _create_spend_logs_with_poison_isolation(repo, rows[mid:], remaining)


def _raise_failed_update_spend_exception(e: Exception, start_time: float, proxy_logging_obj: ProxyLogging):
    """
    Raise an exception for failed update spend logs

    - Calls proxy_logging_obj.failure_handler to log the error
    - Ensures error messages says "Non-Blocking"
    """
    import traceback

    error_msg: Final = f"[Non-Blocking]LiteLLM Prisma Client Exception - update spend logs: {e}"
    error_traceback: Final = error_msg + "\n" + traceback.format_exc()
    end_time: Final = time.time()
    _duration: Final = end_time - start_time
    asyncio.create_task(
        proxy_logging_obj.failure_handler(
            original_exception=e,
            duration=_duration,
            call_type="update_spend",
            traceback_str=error_traceback,
        )
    )
    raise e


def _get_month_end_date(today: date) -> date:
    if today.month == 12:
        return date(today.year + 1, 1, 1) - timedelta(days=1)
    return date(today.year, today.month + 1, 1) - timedelta(days=1)


def _is_projected_spend_over_limit(current_spend: float, soft_budget_limit: float | None):
    if soft_budget_limit is None:
        # If there's no limit, we can't exceed it.
        return False

    today: Final = date.today()

    # Finding the first day of the next month, then subtracting one day to get the end of the current month.
    end_month: Final = _get_month_end_date(today)

    remaining_days: Final = (end_month - today).days

    # Check for the start of the month to avoid division by zero
    if today.day == 1:
        daily_spend_estimate = current_spend
    else:
        daily_spend_estimate = current_spend / (today.day - 1)

    # Total projected spend for the month
    projected_spend: Final = current_spend + (daily_spend_estimate * remaining_days)

    if projected_spend > soft_budget_limit:
        print_verbose("Projected spend exceeds soft budget limit!")
        return True
    return False


def _get_projected_spend_over_limit(current_spend: float, soft_budget_limit: float | None) -> tuple | None:
    if soft_budget_limit is None:
        return None

    today: Final = date.today()
    end_month: Final = _get_month_end_date(today)
    remaining_days: Final = (end_month - today).days

    # assuming the current spend till today (not including today)
    if today.day == 1:
        daily_spend = current_spend
    else:
        daily_spend = current_spend / (today.day - 1)
    projected_spend: Final = current_spend + (daily_spend * remaining_days)

    if projected_spend > soft_budget_limit:
        if daily_spend <= 0:
            limit_exceed_date = today
        else:
            remaining_budget: Final = soft_budget_limit - current_spend
            if remaining_budget <= 0:
                limit_exceed_date = today
            else:
                approx_days: Final = remaining_budget / daily_spend
                limit_exceed_date = today + timedelta(days=approx_days)

        # return the projected spend and the date it will exceeded
        return projected_spend, limit_exceed_date

    return None


def _is_valid_team_configs(team_id=None, team_config=None, request_data=None):
    if team_id is None or team_config is None or request_data is None:
        return
    # check if valid model called for team
    if "models" in team_config:
        valid_models: Final = team_config.pop("models")
        model_in_request: Final = request_data["model"]
        if model_in_request not in valid_models:
            raise Exception(
                f"Invalid model for team {team_id}: {model_in_request}.  Valid models for team are: {valid_models}\n"
            )
    return


def _to_ns(dt):
    return int(dt.timestamp() * 1e9)


def _check_and_merge_model_level_guardrails(
    data: dict,
    llm_router: Router | None,
    trust_client_model_info: bool = True,
) -> dict:
    """
    Check if the model has guardrails defined and merge them with existing guardrails in the request data.

    Args:
        data: The request data dict
        llm_router: The LLM router instance to get deployment info from
        trust_client_model_info: If False, ignore metadata.model_info.id and
            resolve guardrails by alias-union only. Set to False on the
            pre_call path because add_litellm_data_to_request preserves
            client-supplied model_info when allow_client_pricing_override is
            set, so a caller could spoof an unguarded model_info.id while
            requesting a guarded alias and bypass guardrails (veria-ai HIGH
            on #29654). Defaults to True for post_call paths where the
            router has populated model_info.id itself.

    Returns:
        Modified data dict with merged guardrails (if any model-level guardrails exist)
    """
    if llm_router is None:
        return data

    metadata: Final = data.get("metadata") or {}
    litellm_metadata: Final = data.get("litellm_metadata") or {}
    model_info: Final = metadata.get("model_info") or {}
    model_id: Final = model_info.get("id") if trust_client_model_info else None
    # route_request resolves team-scoped public model names with the
    # server-populated team id; pre_call lookup must do the same so
    # team-scoped guardrails are not silently skipped (greptile/veria-ai
    # Medium on #29654).
    team_id: Final = metadata.get("user_api_key_team_id") or litellm_metadata.get("user_api_key_team_id")

    model_level_guardrails: list[object] | None = None
    if model_id is not None:
        deployment: Final = llm_router.get_deployment(model_id=model_id)
        if deployment is None:
            return data
        deployment_guardrails: Final = deployment.litellm_params.get("guardrails")
        # Bare-string guardrail names were truthy-accepted before; preserve
        # that contract so post_call callers don't silently lose them.
        if isinstance(deployment_guardrails, list):
            model_level_guardrails = deployment_guardrails
        elif deployment_guardrails:
            model_level_guardrails = [deployment_guardrails]
    else:
        # Pre_call paths run before route_request picks a deployment, so we
        # don't know which deployment's litellm_params.guardrails will apply.
        # Take the UNION across all deployments in the group so a guardrail
        # set on ANY eligible deployment still fires (#29652; addresses
        # veria-ai HIGH on the single-deployment fallback that would skip
        # non-first deployments).
        model_alias: Final = data.get("model")
        if not isinstance(model_alias, str) or not model_alias:
            return data
        # Pass team_id so team-scoped public model names resolve the same way
        # route_request resolves them; otherwise team-scoped deployments are
        # invisible to this lookup and their guardrails are silently dropped.
        deployments: Final = llm_router.get_model_list(model_name=model_alias, team_id=team_id) or []
        seen: Final[set] = set()
        union: Final[list] = []
        for dep in deployments:
            litellm_params_dep = dep.get("litellm_params") or {}
            guardrails = litellm_params_dep.get("guardrails")
            if isinstance(guardrails, str):
                guardrails = [guardrails]
            elif not isinstance(guardrails, list):
                continue
            for g in guardrails:
                key = g if isinstance(g, str) else repr(g)
                if key not in seen:
                    seen.add(key)
                    union.append(g)
        model_level_guardrails = union or None

    if model_level_guardrails is None:
        return data

    # Merge model-level guardrails with existing ones
    return _merge_guardrails_with_existing(data, model_level_guardrails)


def _merge_guardrails_with_existing(data: dict, model_level_guardrails: object) -> dict:
    """
    Merge model-level guardrails with any existing guardrails in the request data.

    Args:
        data: The request data dict
        model_level_guardrails: Guardrails defined at the model level

    Returns:
        Modified data dict with merged guardrails in metadata
    """
    modified_data: Final = data.copy()
    metadata: Final = modified_data.setdefault("metadata", {})
    existing_guardrails = metadata.get("guardrails", [])

    # Ensure existing_guardrails is a list
    if not isinstance(existing_guardrails, list):
        existing_guardrails = [existing_guardrails] if existing_guardrails else []

    # Ensure model_level_guardrails is a list
    if not isinstance(model_level_guardrails, list):
        model_level_guardrails = [model_level_guardrails] if model_level_guardrails else []

    # Combine existing and model-level guardrails
    metadata["guardrails"] = list(set(existing_guardrails + model_level_guardrails))
    return modified_data


def get_error_message_str(e: Exception) -> str:
    error_message = ""
    if isinstance(e, HTTPException):
        if isinstance(e.detail, str):
            error_message = e.detail
        elif isinstance(e.detail, dict):
            error_message = json.dumps(e.detail)
        elif hasattr(e, "message"):
            _error: Final = getattr(e, "message", None)
            if isinstance(_error, str):
                error_message = _error
            elif isinstance(_error, dict):
                error_message = json.dumps(_error)
        else:
            error_message = str(e)
    else:
        error_message = str(e)
    return error_message


def _get_redoc_url() -> str | None:
    """
    Get the Redoc URL from the environment variables.

    - If REDOC_URL is set, return it.
    - If NO_REDOC is True, return None.
    - Otherwise, default to "/redoc".
    """
    if redoc_url := os.getenv("REDOC_URL"):
        return redoc_url

    if str_to_bool(os.getenv("NO_REDOC")) is True:
        return None

    return "/redoc"


def _get_docs_url() -> str | None:
    """
    Get the docs (Swagger UI) URL from the environment variables.

    - If DOCS_URL is set, return it.
    - If NO_DOCS is True, return None.
    - Otherwise, default to "/".
    """
    if docs_url := os.getenv("DOCS_URL"):
        return docs_url

    if str_to_bool(os.getenv("NO_DOCS")) is True:
        return None

    return "/"


def _get_openapi_url() -> str | None:
    """
    Get the OpenAPI JSON URL from the environment variables.

    - If OPENAPI_URL is set, return it.
    - If NO_OPENAPI is True, return None.
    - Otherwise, default to "/openapi.json".
    """
    if openapi_url := os.getenv("OPENAPI_URL"):
        return openapi_url

    if str_to_bool(os.getenv("NO_OPENAPI")) is True:
        return None

    return "/openapi.json"


def handle_exception_on_proxy(e: Exception) -> ProxyException:
    """
    Returns an Exception as ProxyException, this ensures all exceptions are OpenAI API compatible
    """
    from fastapi import status

    verbose_proxy_logger.exception("Exception: %s", e)

    if isinstance(e, HTTPException):
        return ProxyException(
            message=getattr(e, "detail", f"error({e})"),
            type=ProxyErrorTypes.internal_server_error,
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
        )
    elif isinstance(e, ProxyException):
        return e
    _status_code: Final = getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    return ProxyException(
        message=str(e),
        type=ProxyErrorTypes.internal_server_error,
        param=getattr(e, "param", "None"),
        code=_status_code,
    )


def _premium_user_check(feature: str | None = None):
    """
    Raises an HTTPException if the user is not a premium user
    """
    from litellm.proxy.proxy_server import premium_user

    if feature:
        detail_msg = f"This feature is only available for LiteLLM Enterprise users: {feature}. {CommonProxyErrors.not_premium_user.value}"
    else:
        detail_msg = (
            f"This feature is only available for LiteLLM Enterprise users. {CommonProxyErrors.not_premium_user.value}"
        )

    if not premium_user:
        raise HTTPException(
            status_code=403,
            detail={"error": detail_msg},
        )


def is_known_model(model: str | None, llm_router: Router | None) -> bool:
    """
    Returns True if the model is in the llm_router model names
    """
    if model is None or llm_router is None:
        return False
    model_names: Final = llm_router.get_model_names()

    model_names_set: Final = set(model_names)

    is_in_list = False
    if model in model_names_set:
        is_in_list = True

    return is_in_list


def is_known_vector_store_index(index_name: str) -> bool:
    """
    Returns True if the vector store index is in the llm_router vector store indexes
    """

    if litellm.vector_store_index_registry is None:
        return False
    return index_name in litellm.vector_store_index_registry.get_vector_store_indexes()


def join_paths(base_path: str, route: str) -> str:
    # Remove trailing slashes from base_path and leading slashes from route
    base_path = base_path.rstrip("/")
    route = route.lstrip("/")

    # If base_path is empty, return route with leading slash
    if not base_path:
        return f"/{route}" if route else "/"

    # If route is empty, return just base_path
    if not route:
        return base_path

    # Check if base_path already ends with the route to avoid duplication
    if base_path.endswith(f"/{route}"):
        final_path = base_path
    else:
        # Join with single slash
        final_path = f"{base_path}/{route}"

    return final_path


def get_custom_url(request_base_url: str, route: str | None = None) -> str:
    # Use environment variable value, otherwise use URL from request
    server_base_url: Final = get_proxy_base_url()
    if server_base_url is not None:
        base_url = server_base_url
    else:
        base_url = request_base_url

    server_root_path: Final = get_server_root_path()
    if route is not None:
        if server_root_path != "":
            # First join base_url with server_root_path, then with route
            intermediate_url: Final = join_paths(base_url, server_root_path)
            return join_paths(intermediate_url, route)
        else:
            return join_paths(base_url, route)
    else:
        return join_paths(base_url, server_root_path)


def get_proxy_base_url() -> str | None:
    """
    Get the proxy base url from the environment variables.
    """
    return os.getenv("PROXY_BASE_URL")


def get_server_root_path() -> str:
    """
    Get the server root path from the environment variables.

    - If SERVER_ROOT_PATH is set, return it.
    - Otherwise, default to "/".
    """
    return os.getenv("SERVER_ROOT_PATH", "")


def normalize_route_for_root_path(route: str) -> str | None:
    """Strip SERVER_ROOT_PATH prefix. Returns de-prefixed route, or None if route is not under root path."""
    root_path: Final = get_server_root_path()
    if root_path and root_path != "/":
        if route.startswith(root_path + "/"):
            return route[len(root_path) :]
        return None
    return route


def get_prisma_client_or_throw(message: str):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": message},
        )
    return prisma_client


def is_valid_api_key(key: str) -> bool:
    """
    Validates API key format:
    - sk- keys: must match ^sk-[A-Za-z0-9_-]+$
    - hashed keys: must match ^[a-fA-F0-9]{64}$
    - Length between 20 and 100 characters
    """
    import re

    if not isinstance(key, str):
        return False
    if 3 <= len(key) <= 100:
        if re.match(r"^sk-[A-Za-z0-9_-]+$", key):
            return True
        if re.match(r"^[a-fA-F0-9]{64}$", key):
            return True
    return False


def construct_database_url_from_env_vars() -> str | None:
    """
    Construct a DATABASE_URL from individual environment variables.
    Returns:
        Optional[str]: The constructed DATABASE_URL or None if required variables are missing
    """
    import urllib.parse

    # Check if all required variables are provided
    database_host: Final = os.getenv("DATABASE_HOST")
    database_username: Final = os.getenv("DATABASE_USERNAME")
    database_password: Final = os.getenv("DATABASE_PASSWORD")
    database_name: Final = os.getenv("DATABASE_NAME")
    database_schema: Final = os.getenv("DATABASE_SCHEMA")

    if database_host and database_username and database_name:
        # Handle the problem of special character escaping in the database URL
        database_username_enc: Final = urllib.parse.quote_plus(database_username)
        database_password_enc: Final = urllib.parse.quote_plus(database_password) if database_password else ""
        database_name_enc: Final = urllib.parse.quote_plus(database_name)

        # Construct DATABASE_URL from the provided variables
        if database_password:
            database_url = (
                f"postgresql://{database_username_enc}:{database_password_enc}@{database_host}/{database_name_enc}"
            )
        else:
            database_url = f"postgresql://{database_username_enc}@{database_host}/{database_name_enc}"

        if database_schema:
            database_url += f"?schema={database_schema}"

        return database_url

    return None


async def _get_validated_team_object(
    user_api_key_dict: "UserAPIKeyAuth",
    team_id: str,
    prisma_client: "PrismaClient",
    user_api_key_cache: "UserApiKeyCache",
    proxy_logging_obj: "ProxyLogging",
) -> "LiteLLM_TeamTableCachedObj":
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.management_endpoints.team_endpoints import validate_membership

    team_object: Final = await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    await validate_membership(user_api_key_dict=user_api_key_dict, team_table=team_object)
    return team_object


async def _get_team_object_for_access_groups(
    team_id: str | None,
    prisma_client: Optional["PrismaClient"],
    user_api_key_cache: Optional["UserApiKeyCache"],
    proxy_logging_obj: Optional["ProxyLogging"],
) -> Optional["LiteLLM_TeamTableCachedObj"]:
    from litellm.proxy.auth.auth_checks import get_team_object

    if team_id is None or prisma_client is None or user_api_key_cache is None or proxy_logging_obj is None:
        return None
    try:
        return await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except HTTPException:
        verbose_proxy_logger.debug("Could not fetch team %s while listing models", team_id)
        return None


async def _get_access_group_models(
    user_api_key_dict: "UserAPIKeyAuth",
    team_object: Optional["LiteLLM_TeamTableCachedObj"],
    prisma_client: Optional["PrismaClient"],
    user_api_key_cache: Optional["UserApiKeyCache"],
    proxy_logging_obj: Optional["ProxyLogging"],
) -> tuple[str, ...]:
    from litellm.proxy.auth.auth_checks import (
        _get_models_from_access_groups,
        get_authorized_resources_from_key_access_groups,
    )

    team_group_models: Final = await _get_models_from_access_groups(
        access_group_ids=(team_object.access_group_ids or ()) if team_object is not None else (),
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    key_group_models: Final = await get_authorized_resources_from_key_access_groups(
        valid_token=user_api_key_dict,
        team_object=team_object,
        resource_field="access_model_names",
    )
    return tuple(dict.fromkeys((*team_group_models, *key_group_models)))


async def get_available_models_for_user(
    user_api_key_dict: "UserAPIKeyAuth",
    llm_router: Optional["Router"],
    general_settings: dict,
    user_model: str | None,
    prisma_client: Optional["PrismaClient"] = None,
    proxy_logging_obj: Optional["ProxyLogging"] = None,
    team_id: str | None = None,
    include_model_access_groups: bool = False,
    only_model_access_groups: bool = False,
    return_wildcard_routes: bool = False,
    user_api_key_cache: Optional["UserApiKeyCache"] = None,
) -> list[str]:
    """
    Get the list of models available to a user based on their API key and team permissions.

    Args:
        user_api_key_dict: User API key authentication object
        llm_router: LiteLLM router instance
        general_settings: General settings from config
        user_model: User-specific model
        prisma_client: Prisma client for database operations
        proxy_logging_obj: Proxy logging object
        team_id: Specific team ID to check (optional)
        include_model_access_groups: Whether to include model access groups
        only_model_access_groups: Whether to only return model access groups
        return_wildcard_routes: Whether to return wildcard routes

    Returns:
        List of model names available to the user
    """
    from litellm.proxy.auth.model_checks import (
        get_complete_model_list,
        get_key_models,
        get_team_models,
    )

    # Get proxy model list and access groups
    if llm_router is None:
        proxy_model_list = []
        model_access_groups = {}
    else:
        proxy_model_list = llm_router.get_model_names()
        model_access_groups = llm_router.get_model_access_groups()

    requested_team_object: Final = (
        await _get_validated_team_object(
            user_api_key_dict=user_api_key_dict,
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if team_id and prisma_client and proxy_logging_obj and user_api_key_cache
        else None
    )

    key_models: Final[Sequence[str]] = (
        ()
        if requested_team_object is not None
        else get_key_models(
            user_api_key_dict=user_api_key_dict,
            proxy_model_list=proxy_model_list,
            model_access_groups=model_access_groups,
            include_model_access_groups=include_model_access_groups,
        )
    )

    team_models: Final = get_team_models(
        team_models=(
            requested_team_object.models if requested_team_object is not None else user_api_key_dict.team_models
        ),
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
        include_model_access_groups=include_model_access_groups,
    )

    effective_team_id: Final = team_id or user_api_key_dict.team_id

    access_group_models: Final = (
        await _get_access_group_models(
            user_api_key_dict=user_api_key_dict,
            team_object=requested_team_object
            or await _get_team_object_for_access_groups(
                team_id=effective_team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            ),
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if key_models or team_models
        else ()
    )

    granted_key_models: Final = (*key_models, *access_group_models) if key_models else key_models
    granted_team_models: Final = (*team_models, *access_group_models) if team_models else team_models

    # Get complete model list
    all_models: Final = get_complete_model_list(
        key_models=granted_key_models,
        team_models=granted_team_models,
        proxy_model_list=proxy_model_list,
        user_model=user_model,
        infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
        return_wildcard_routes=return_wildcard_routes,
        llm_router=llm_router,
        model_access_groups=model_access_groups,
        include_model_access_groups=include_model_access_groups,
        only_model_access_groups=only_model_access_groups,
        team_id=effective_team_id,
    )

    return all_models


def create_model_info_response(
    model_id: str,
    provider: str,
    include_metadata: bool = False,
    fallback_type: str | None = None,
    llm_router: Optional["Router"] = None,
    get_model_info: Callable[[str], ModelInfo] = litellm.get_model_info,
) -> ModelInfoResponse:
    """
    Create a standardized OpenAI-compatible model object.

    When include_metadata is true, attaches the model's configured fallbacks
    (resolved via the router under fallback_type, defaulting to "general").
    Raises HTTPException(400) for an unknown fallback_type.
    """
    from litellm.proxy.auth.model_checks import get_all_fallbacks

    base: Final[ModelInfoResponse] = {
        "id": model_id,
        "object": "model",
        "created": DEFAULT_MODEL_CREATED_AT_TIME,
        "owned_by": provider,
    }

    try:
        model_cost_info: ModelInfo | None = get_model_info(model_id)
    except Exception as e:
        verbose_proxy_logger.debug(
            "create_model_info_response: cost map lookup failed for %s: %s",
            model_id,
            e,
        )
        model_cost_info = None

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    if model_cost_info is not None:
        max_input_tokens = coerce_token_limit(model_cost_info.get("max_input_tokens"))
        max_output_tokens = coerce_token_limit(model_cost_info.get("max_output_tokens"))
        mode: Final = model_cost_info.get("mode")
        if isinstance(mode, str):
            base["mode"] = mode

    if llm_router is not None:
        configured_input, configured_output = llm_router.get_configured_token_limits(model_id)
        if configured_input is not None:
            max_input_tokens = configured_input
        if configured_output is not None:
            max_output_tokens = configured_output

    if max_input_tokens is not None:
        base["max_input_tokens"] = max_input_tokens
    if max_output_tokens is not None:
        base["max_output_tokens"] = max_output_tokens

    if not include_metadata:
        return base

    effective_fallback_type: Final = fallback_type if fallback_type is not None else "general"

    valid_fallback_types: Final = ["general", "context_window", "content_policy"]
    if effective_fallback_type not in valid_fallback_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fallback_type. Must be one of: {valid_fallback_types}",
        )

    fallbacks: Final = get_all_fallbacks(
        model=model_id,
        llm_router=llm_router,
        fallback_type=effective_fallback_type,
    )
    return {**base, "metadata": {"fallbacks": fallbacks}}


def validate_model_access(
    model_id: str,
    available_models: list[str],
) -> None:
    """
    Validate that a model is accessible to the user.
    Supports batch requests with comma-separated model IDs.

    Args:
        model_id: The model ID to validate (can be comma-separated for batch requests)
        available_models: List of models available to the user

    Raises:
        HTTPException: If the model is not accessible
    """
    # Handle batch requests with comma-separated models
    if "," in model_id:
        models: Final = [m.strip() for m in model_id.split(",")]
        inaccessible_models: Final = [m for m in models if m not in available_models]
        if inaccessible_models:
            raise HTTPException(
                status_code=404,
                detail="The following model(s) do not exist or are not accessible: {}".format(
                    ", ".join(inaccessible_models)
                ),
            )
    else:
        # Single model validation
        if model_id not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"The model `{model_id}` does not exist or is not accessible",
            )


_PRESERVED_NONE_FIELDS: Final[list[tuple[str, str]]] = [
    ("message", "content"),  # null when tool_calls present (issue #6677)
    ("message", "role"),  # always required by OpenAI spec
    ("delta", "content"),  # null in streaming chunks
]


def model_dump_with_preserved_fields(
    obj: Any,
    preserve_fields: list[str] | None = None,
    exclude_unset: bool = True,
) -> dict[str, object]:
    """
    Serialize a Pydantic model to a dictionary while preserving specific fields
    even if they are None.

    Fields listed in _PRESERVED_NONE_FIELDS are restored after
    model_dump(exclude_none=True) strips them.

    Args:
        obj: The Pydantic BaseModel instance to serialize
        preserve_fields: Deprecated, kept for backward compatibility.
        exclude_unset: Whether to exclude fields that were not explicitly set

    Returns:
        Dictionary representation with None values excluded except for preserved fields
    """
    result: Final = obj.model_dump(exclude_none=True, exclude_unset=exclude_unset)

    choices: Final = result.get("choices")
    if not choices:
        return result

    obj_choices: Final = obj.choices
    for choice_obj, choice_dict in zip(obj_choices, choices):
        for sub_object, field_name in _PRESERVED_NONE_FIELDS:
            sub_dict = choice_dict.get(sub_object)
            if sub_dict is None:
                continue
            if field_name not in sub_dict:
                sub_obj = getattr(choice_obj, sub_object, None)
                if sub_obj is not None and hasattr(sub_obj, field_name):
                    sub_dict[field_name] = getattr(sub_obj, field_name)

    return result
