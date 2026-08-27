import asyncio
import copy
import enum
import importlib
import inspect
import io
import os
import random
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, MutableMapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeAlias,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import anyio
import websockets
import websockets.exceptions
from pydantic import BaseModel, Json, JsonValue
from typing_extensions import NotRequired, assert_never

from litellm._uuid import uuid
from litellm.constants import (
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_CONNECTOR_LIMIT_PER_HOST,
    AIOHTTP_KEEPALIVE_TIMEOUT,
    AIOHTTP_NEEDS_CLEANUP_CLOSED,
    AIOHTTP_TTL_DNS_CACHE,
    AUDIO_SPEECH_CHUNK_SIZE,
    BASE_MCP_ROUTE,
    DAILY_TAG_SPEND_BATCH_MULTIPLIER,
    DEFAULT_MAX_RECURSE_DEPTH,
    DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
    DEFAULT_SHARED_HEALTH_CHECK_TTL,
    DEFAULT_SLACK_ALERTING_THRESHOLD,
    LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS,
    LITELLM_SETTINGS_SAFE_DB_OVERRIDES,
    LITELLM_UI_ALLOW_HEADERS,
    LITELLM_UI_SESSION_DURATION,
)
from litellm.litellm_core_utils.litellm_logging import (
    _init_custom_logger_compatible_class,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import (
    UI_TEAM_ID,
    CallbackDelete,
    CallInfo,
    CommonProxyErrors,
    ConfigFieldDelete,
    ConfigFieldInfo,
    ConfigFieldUpdate,
    ConfigGeneralSettings,
    ConfigList,
    ConfigYAML,
    CoordinationRedisParams,
    EnterpriseLicenseData,
    FieldDetail,
    InvitationClaim,
    InvitationDelete,
    InvitationModel,
    InvitationNew,
    InvitationUpdate,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_JWTAuth,
    LiteLLM_TagTable,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    PassThroughGenericEndpoint,
    ProxyErrorTypes,
    ProxyException,
    RoleBasedPermissions,
    SpecialModelNames,
    SupportedDBObjectType,
    TeamDefaultSettings,
    TokenCountRequest,
    TransformRequestBody,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.callback_utils import (
    is_sensitive_callback_key,
    normalize_callback_names,
    process_callback,
    strip_callback_config,
)
from litellm.proxy.common_utils.realtime_utils import _realtime_request_body
from litellm.router_utils.add_retry_fallback_headers import (
    get_fallback_errors_from_headers,
    get_hidden_params_dict,
)
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    TextCompletionResponse,
    TokenCountResponse,
)
from litellm.utils import (
    _invalidate_model_cost_lowercase_map,
    load_credentials_from_list,
    reapply_runtime_model_cost_registrations,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession
    from fastapi.routing import APIRoute
    from opentelemetry.trace import Span as _Span
    from prisma import models as prisma_models

    from litellm.integrations.opentelemetry import OpenTelemetry

    Span = _Span | Any
else:
    Span = Any
    OpenTelemetry = Any

REALTIME_REQUEST_SCOPE_TEMPLATE: Final[dict[str, object]] = {
    "type": "http",
    "method": "POST",
    "path": "/v1/realtime",
}


def showwarning(message, category, filename, lineno, file=None, line=None):
    traceback_info: Final = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    if file is not None:
        file.write(traceback_info)


warnings.showwarning = showwarning
warnings.filterwarnings("default", category=UserWarning)

# Your client code here


messages: Final[list] = []
sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path - for litellm local dev

try:
    import logging

    import backoff
    import fastapi
    import orjson
    import yaml
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError as e:
    raise ImportError(f"Missing dependency {e}. Run `pip install 'litellm[proxy]'`")

list_of_messages: Final = [
    "'The thing I wish you improved is...'",
    "'A feature I really want is...'",
    "'The worst thing about this product is...'",
    "'This product would be better if...'",
    "'I don't like how this works...'",
    "'It would help me if you could add...'",
    "'This feature doesn't meet my needs because...'",
    "'I get frustrated when the product...'",
]


def generate_feedback_box():
    box_width: Final = 60

    # Select a random message
    message: Final = random.choice(list_of_messages)

    print()  # noqa: T201
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")  # noqa: T201
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")  # noqa: T201
    print("\033[1;37m" + f"# {message:^59} #\033[0m")  # noqa: T201
    print(  # noqa: T201
        "\033[1;37m" + "# {:^59} #\033[0m".format("https://github.com/BerriAI/litellm/issues/new")
    )
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")  # noqa: T201
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")  # noqa: T201
    print()  # noqa: T201
    print(" Thank you for using LiteLLM! - Krrish & Ishaan")  # noqa: T201
    print()  # noqa: T201
    print()  # noqa: T201
    print()  # noqa: T201
    print(  # noqa: T201
        "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"
    )
    print()  # noqa: T201
    print()  # noqa: T201


import contextlib
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache

import litellm
import litellm._redis
from litellm import Router
from litellm._logging import _redact_string, verbose_proxy_logger, verbose_router_logger
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.constants import (
    _REALTIME_BODY_CACHE_SIZE,
    APSCHEDULER_COALESCE,
    APSCHEDULER_MAX_INSTANCES,
    APSCHEDULER_MISFIRE_GRACE_TIME,
    APSCHEDULER_REPLACE_EXISTING,
    CLI_SSO_SESSION_TTL_SECONDS,
    DAYS_IN_A_MONTH,
    DEFAULT_HEALTH_CHECK_INTERVAL,
    DEFAULT_MODEL_CREATED_AT_TIME,
    GLOBAL_PROXY_SPEND_CACHE_KEY,
    LITELLM_PROXY_ADMIN_NAME,
    LITELLM_PROXY_BUDGET_NAME,
    MONTHLY_SPEND_REPORT_JOB_ID,
    PROMETHEUS_FALLBACK_STATS_JOB_ID,
    PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS,
    PROXY_BATCH_POLLING_ENABLED,
    PROXY_BATCH_POLLING_INTERVAL,
    PROXY_BATCH_WRITE_AT,
    PROXY_BUDGET_RESCHEDULER_MAX_TIME,
    PROXY_BUDGET_RESCHEDULER_MIN_TIME,
    PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
    WEEKLY_SPEND_REPORT_JOB_ID,
)
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.litellm_core_utils.agentic_loop_settings import (
    validated_max_agentic_loops,
)
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_litellm_metadata_from_kwargs,
)
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.realtime_errors import (
    realtime_error_event,
    websocket_close_reason,
)
from litellm.litellm_core_utils.sensitive_data_masker import (
    SensitiveDataMasker,
    mask_sensitive_keys,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._lazy_features import attach_lazy_features
from litellm.proxy._types import *
from litellm.proxy.analytics_endpoints.analytics_endpoints import (
    router as analytics_router,
)
from litellm.proxy.auth.auth_checks import (
    ExperimentalUIJWTToken,
    can_key_call_resolved_model,
    get_team_object,
    log_db_metrics,
)
from litellm.proxy.auth.auth_utils import (
    check_response_size_is_safe,
    is_request_body_safe,
    warn_once_if_custom_auth_skips_common_checks,
)
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.auth.model_checks import (
    expand_wildcard_deployments_for_model_info,
    get_all_fallbacks,
    get_complete_model_list,
    get_key_models,
    get_mcp_server_ids,
    get_team_models,
)
from litellm.proxy.auth.user_api_key_auth import (
    _fetch_global_spend_with_event_coordination,
    user_api_key_auth,
    user_api_key_auth_websocket,
)
from litellm.proxy.batches_endpoints.endpoints import router as batches_router

## Import All Misc routes here ##
from litellm.proxy.caching_routes import router as caching_router
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    _is_azure_model_router_request,
    _should_return_raw_model_name,
    create_response,
    open_sse_before_first_byte,
    ttft_keepalive_interval,
)
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AuthCacheInvalidationSubscriber,
)
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy
from litellm.proxy.common_utils.config_sync_pubsub import ConfigSyncSubscriber
from litellm.proxy.common_utils.debug_utils import init_verbose_loggers
from litellm.proxy.common_utils.debug_utils import router as debugging_endpoints_router
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.healthy_model_filter import (
    get_hidden_unhealthy_model_names,
    is_healthy_only_listing_default,
)
from litellm.proxy.common_utils.html_forms.ui_login import build_ui_login_form
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    check_file_size_under_limit,
    get_form_data,
)
from litellm.proxy.common_utils.load_config_utils import (
    get_config_file_contents_from_gcs,
    get_file_contents_from_s3,
)
from litellm.proxy.common_utils.model_deprecation import collect_model_deprecations
from litellm.proxy.common_utils.model_listing_utils import TeamModelNameTranslator
from litellm.proxy.common_utils.openai_endpoint_utils import (
    remove_sensitive_info_from_deployment,
)
from litellm.proxy.common_utils.periodic_reload_schedule import (
    MODEL_COST_MAP_RELOAD_PARAM_NAME,
    clear_reload_interval,
    pod_reload_is_due,
    read_reload_schedule,
    record_manual_reload,
    record_reload_run,
    reload_schedule_status,
    utc_now,
    write_reload_interval,
)
from litellm.proxy.common_utils.proxy_state import ProxyState
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.common_utils.scheduled_job_stagger import (
    apply_scheduled_job_stagger,
    attach_job_timing_logger,
    parse_stagger_settings,
    stagger_trigger,
)
from litellm.proxy.common_utils.swagger_utils import ERROR_RESPONSES
from litellm.proxy.common_utils.timezone_utils import (
    get_budget_reset_settings,
    get_budget_reset_time,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    end_user_cache_key,
    get_management_object_ttl,
    tag_cache_key,
)
from litellm.proxy.config_resolvers import resolve_fields
from litellm.proxy.config_resolvers.alerting import (
    EMAIL_DESCRIPTORS,
    SLACK_DESCRIPTORS,
)
from litellm.proxy.container_endpoints.endpoints import router as container_router
from litellm.proxy.credential_endpoints.endpoints import router as credential_router
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import (
    SPEND_LOG_CLEANUP_BOUND_SETTINGS,
    SpendLogCleanup,
)
from litellm.proxy.db.exception_handler import (
    PrismaDBExceptionHandler,
    call_with_db_reconnect_retry,
)
from litellm.proxy.db.gateway_request_tracking import (
    GatewayRequestAccumulator,
    flush_gateway_requests,
)
from litellm.proxy.db.proxy_worker_heartbeat import (
    PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS,
    ProxyWorkerHeartbeat,
)
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.discovery_endpoints import ui_discovery_endpoints_router
from litellm.proxy.fine_tuning_endpoints.endpoints import router as fine_tuning_router
from litellm.proxy.fine_tuning_endpoints.endpoints import set_fine_tuning_config
from litellm.proxy.google_endpoints.endpoints import router as google_router
from litellm.proxy.guardrails.init_guardrails import (
    init_guardrails_v2,
    initialize_guardrails,
)
from litellm.proxy.health_check import (
    filter_deployments_to_model_groups,
    health_check_filter_kwargs_from_general_settings,
    parse_background_health_check_model_groups,
    perform_health_check,
)
from litellm.proxy.health_endpoints._health_endpoints import router as health_router
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)
from litellm.proxy.hooks.prompt_injection_detection import (
    _OPTIONAL_PromptInjectionDetection,
)
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.image_endpoints.endpoints import router as image_router
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.logging_endpoints.callback_logs_endpoints import (
    rust_control_plane_router,
)
from litellm.proxy.management_endpoints.auto_router_endpoints import (
    router as auto_router_management_router,
)
from litellm.proxy.management_endpoints.budget_management_endpoints import (
    router as budget_management_router,
)
from litellm.proxy.management_endpoints.cache_settings_endpoints import (
    router as cache_settings_router,
)
from litellm.proxy.management_endpoints.callback_management_endpoints import (
    router as callback_management_endpoints_router,
)
from litellm.proxy.management_endpoints.common_utils import (
    _user_has_admin_privileges,
    _user_has_admin_view,
    admin_can_invite_user,
)
from litellm.proxy.management_endpoints.coordination_redis_endpoints import (
    get_persisted_coordination_redis_settings,
)
from litellm.proxy.management_endpoints.coordination_redis_endpoints import (
    router as coordination_redis_settings_router,
)
from litellm.proxy.management_endpoints.cost_tracking_settings import (
    router as cost_tracking_settings_router,
)
from litellm.proxy.management_endpoints.customer_endpoints import (
    router as customer_router,
)
from litellm.proxy.management_endpoints.fallback_management_endpoints import (
    router as fallback_management_router,
)
from litellm.proxy.management_endpoints.gateway_request_endpoints import (
    router as gateway_request_router,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    router as internal_user_router,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    user_update,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_verification_tokens,
    duration_in_seconds,
    generate_key_helper_fn,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    router as key_management_router,
)
from litellm.proxy.management_endpoints.management_v1 import (
    router as management_v1_router,
)
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    problem_response,
)
from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    router as model_access_group_management_router,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _add_model_to_db,
    _add_team_model_to_db,
    _deduplicate_litellm_router_models,
    live_model_ids_snapshot,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    router as model_management_router,
)
from litellm.proxy.management_endpoints.organization_endpoints import (
    router as organization_router,
)
from litellm.proxy.management_endpoints.router_settings_endpoints import (
    router as router_settings_router,
)
from litellm.proxy.management_endpoints.tag_management_endpoints import (
    router as tag_management_router,
)
from litellm.proxy.management_endpoints.team_callback_endpoints import (
    router as team_callback_router,
)
from litellm.proxy.management_endpoints.team_endpoints import router as team_router
from litellm.proxy.management_endpoints.team_endpoints import (
    update_team,
    validate_membership,
)
from litellm.proxy.management_endpoints.ui_sso import (
    get_disabled_non_admin_personal_key_creation,
)
from litellm.proxy.management_endpoints.ui_sso import router as ui_sso_router
from litellm.proxy.management_endpoints.user_agent_analytics_endpoints import (
    router as user_agent_analytics_router,
)
from litellm.proxy.management_endpoints.workflow_management_endpoints import (
    router as workflow_management_router,
)
from litellm.proxy.management_helpers.audit_logs import (
    create_audit_log_for_update,
    create_object_audit_log,
)
from litellm.proxy.management_helpers.team_metadata_validation import (
    TEAM_METADATA_SCHEMA_REGISTRY,
    TEAM_METADATA_VALIDATOR_REGISTRY,
    parse_team_metadata_schema,
)
from litellm.proxy.memory.memory_endpoints import router as memory_router
from litellm.proxy.middleware.billable_request_metrics_middleware import (
    BillableRequestMetricsMiddleware,
    BillingRecorder,
)
from litellm.proxy.plugin_routes import (
    register_plugins_from_config,
)
from litellm.proxy.plugin_routes import (
    router as plugin_router,
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

try:
    from litellm.proxy.enterprise_billing.billing_metrics import (
        build_billing_metrics_recorder as _build_billing_metrics_recorder,
    )
    from litellm.proxy.enterprise_billing.billing_metrics import (
        shutdown_billing_metrics_recorder as _shutdown_billing_metrics_recorder,
    )

    build_billing_metrics_recorder: Callable[..., BillingRecorder | None] | None = _build_billing_metrics_recorder
    shutdown_billing_metrics_recorder: Callable[[], None] | None = _shutdown_billing_metrics_recorder
except ImportError:
    build_billing_metrics_recorder = None
    shutdown_billing_metrics_recorder = None
from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)
from litellm.proxy.middleware.prometheus_auth_middleware import PrometheusAuthMiddleware
from litellm.proxy.middleware.request_size_limit_middleware import (
    RequestSizeLimitMiddleware,
)
from litellm.proxy.middleware.security_headers_middleware import (
    SecurityHeadersMiddleware,
)
from litellm.proxy.ocr_endpoints.endpoints import router as ocr_router
from litellm.proxy.openai_files_endpoints.files_endpoints import (
    router as openai_files_router,
)
from litellm.proxy.openai_files_endpoints.files_endpoints import (
    set_files_config,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    openai_passthrough_router,
    passthrough_endpoint_router,
    vertex_ai_live_websocket_passthrough,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    router as llm_passthrough_router,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    initialize_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    router as pass_through_router,
)
from litellm.proxy.public_endpoints import router as public_endpoints_router
from litellm.proxy.rag_endpoints.endpoints import router as rag_router
from litellm.proxy.rerank_endpoints.endpoints import router as rerank_router
from litellm.proxy.response_api_endpoints.endpoints import router as response_router
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.search_endpoints.endpoints import router as search_router
from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager
from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    router as spend_management_router,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
    router as ui_crud_endpoints_router,
)
from litellm.proxy.ui_crud_endpoints.user_banner_endpoints import (
    router as user_banner_endpoints_router,
)
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    ProxyUpdateSpend,
    _cache_user_row,
    _get_docs_url,
    _get_openapi_url,
    _get_projected_spend_over_limit,
    _get_redoc_url,
    _is_projected_spend_over_limit,
    _is_valid_team_configs,
    evict_config_param,
    get_config_param,
    get_custom_url,
    get_error_message_str,
    get_server_root_path,
    handle_exception_on_proxy,
    hash_password,
    hash_token,
    invalidate_config_param,
    litellm_config_cache,
    migrate_passwords_to_scrypt_async,
    model_dump_with_preserved_fields,
    prefetch_config_params,
    update_spend,
)
from litellm.proxy.video_endpoints.endpoints import router as video_router
from litellm.repositories.base_repository import SupportsModelDump
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.repositories.prisma_protocols import TableActions
from litellm.router import (
    AssistantsTypedDict,
    Deployment,
    LiteLLM_Params,
    ModelGroupInfo,
)
from litellm.scheduler import FlowItem, Scheduler
from litellm.secret_managers.aws_secret_manager import load_aws_kms
from litellm.secret_managers.google_kms import load_google_kms
from litellm.secret_managers.main import (
    get_secret,
    get_secret_bool,
    get_secret_str,
    normalize_nonempty_secret_str,
    secret_manager_would_be_consulted,
    str_to_bool,
)
from litellm.types.integrations.slack_alerting import AlertType, SlackAlertingArgs
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    AnthropicResponse,
    AnthropicResponseContentBlockText,
    AnthropicResponseUsageBlock,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)
from litellm.types.proxy.management_endpoints.ui_sso import (
    DefaultTeamSSOParams,
    LiteLLM_UpperboundKeyGenerateParams,
)
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_WARN_DAYS,
    ModelDeprecationResponse,
)
from litellm.types.realtime import RealtimeQueryParams
from litellm.types.router import (
    ClassifierPlugin,
    DeploymentTypedDict,
    RouterGeneralSettings,
    RoutingPlugin,
    SearchToolTypedDict,
    updateDeployment,
)
from litellm.types.router import ModelInfo as RouterModelInfo
from litellm.types.scheduler import DefaultPriorities
from litellm.types.secret_managers.main import (
    KeyManagementSettings,
    KeyManagementSystem,
)
from litellm.types.utils import CredentialItem, CustomHuggingfaceTokenizer, RawRequestTypedDict, StandardLoggingPayload
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.utils import _add_custom_logger_callback_to_specific_event

try:
    from litellm._version import version
except Exception:
    version = "0.0.0"
litellm.suppress_debug_info = True
import json

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    applications,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    ORJSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.routing import APIRouter
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles

# import enterprise folder
enterprise_router = APIRouter()
try:
    # when using litellm cli
    from litellm.proxy import enterprise
except Exception:
    # when using litellm docker image
    try:
        import enterprise
    except Exception:
        pass

###################
# Import enterprise routes
try:
    from litellm_enterprise.proxy.enterprise_routes import router as _enterprise_router
    from litellm_enterprise.proxy.proxy_server import EnterpriseProxyConfig

    enterprise_router = _enterprise_router
    enterprise_proxy_config: EnterpriseProxyConfig | None = EnterpriseProxyConfig()
except ImportError:
    enterprise_proxy_config = None
###################

server_root_path: Final = get_server_root_path()
_license_check = LicenseCheck()
premium_user: bool = _license_check.is_premium()
premium_user_data: Optional["EnterpriseLicenseData"] = _license_check.airgapped_license_data
global_max_parallel_request_retries_env: Final[str | None] = os.getenv("LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRIES")
proxy_state: Final = ProxyState()
SENSITIVE_DATA_MASKER: Final = SensitiveDataMasker()


# Secret-bearing general_settings fields the segment masker does not match by
# name: database_url and database_extra_connection_params embed DB credentials,
# pass_through_endpoints carry upstream Authorization headers, and
# alert_to_webhook_url is itself a webhook secret
_EXTRA_SECRET_GENERAL_SETTINGS_FIELDS: Final = frozenset(
    {
        "database_url",
        "database_extra_connection_params",
        "pass_through_endpoints",
        "alert_to_webhook_url",
    }
)


def _redact_worker_config_for_logging(worker_config: str | dict[str, JsonValue] | None) -> JsonValue:
    """Mask sensitive fields in the worker config before it enters a log record.

    `worker_config` reaches `proxy_startup_event` as either the JSON blob
    persisted by `save_worker_config` (a string) or the dict passed directly
    to `initialize`. Both shapes can carry `master_key`, `database_url`,
    provider API keys, etc.; passing the raw value to `verbose_proxy_logger`
    leaks them whenever the last-line-of-defense regex filter is bypassed
    (`LITELLM_DISABLE_REDACT_SECRETS=true`, an older log sink, a downstream
    handler that captures records pre-filter). Redact at the source.
    """
    if worker_config is None:
        return None
    if isinstance(worker_config, dict):
        return _redact_secret_values_in_obj(worker_config)
    parsed: Final = safe_json_loads(worker_config, default=None)
    if isinstance(parsed, dict):
        return safe_dumps(_redact_secret_values_in_obj(parsed))
    return worker_config


if global_max_parallel_request_retries_env is None:
    global_max_parallel_request_retries: int = 3
else:
    global_max_parallel_request_retries = int(global_max_parallel_request_retries_env)

global_max_parallel_request_retry_timeout_env: Final[str | None] = os.getenv(
    "LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRY_TIMEOUT"
)
if global_max_parallel_request_retry_timeout_env is None:
    global_max_parallel_request_retry_timeout: float = 60.0
else:
    global_max_parallel_request_retry_timeout = float(global_max_parallel_request_retry_timeout_env)

ui_link: Final = f"{server_root_path}/ui"
fallback_login_link: Final = f"{server_root_path}/fallback/login"
model_hub_link: Final = f"{server_root_path}/ui/model_hub_table"
ui_message = f"👉 [```LiteLLM Admin Panel on /ui```]({ui_link}). Create, Edit Keys with SSO. Having issues? Try [```Fallback Login```]({fallback_login_link})"
ui_message += "\n\n💸 [```LiteLLM Model Cost Map```](https://models.litellm.ai/)."

ui_message += f"\n\n🔎 [```LiteLLM Model Hub```]({model_hub_link}). See available models on the proxy. [**Docs**](https://docs.litellm.ai/docs/proxy/ai_hub)"

custom_swagger_message: Final = (
    "[**Customize Swagger Docs**](https://docs.litellm.ai/docs/proxy/enterprise#swagger-docs---custom-routes--branding)"
)

### CUSTOM BRANDING [ENTERPRISE FEATURE] ###
_title: Final = os.getenv("DOCS_TITLE", "LiteLLM API") if premium_user else "LiteLLM API"
_description: Final = (
    os.getenv(
        "DOCS_DESCRIPTION",
        f"Enterprise Edition \n\nProxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}",
    )
    if premium_user
    else f"Proxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}"
)


def cleanup_router_config_variables():
    global \
        master_key, \
        user_config_file_path, \
        otel_logging, \
        user_custom_auth, \
        user_custom_auth_path, \
        user_custom_key_generate, \
        user_custom_key_update, \
        user_custom_sso, \
        user_custom_ui_sso_sign_in_handler, \
        use_background_health_checks, \
        use_shared_health_check, \
        health_check_interval, \
        health_check_concurrency, \
        prisma_client

    # Set all variables to None
    master_key = None
    user_config_file_path = None
    otel_logging = None
    user_custom_auth = None
    user_custom_auth_path = None
    user_custom_key_generate = None
    user_custom_key_update = None
    TEAM_METADATA_VALIDATOR_REGISTRY.set(None)
    TEAM_METADATA_SCHEMA_REGISTRY.set(())
    user_custom_sso = None
    user_custom_ui_sso_sign_in_handler = None
    use_background_health_checks = None
    use_shared_health_check = None
    health_check_interval = None
    health_check_concurrency = None
    prisma_client = None


async def _flush_spend_logs_queue_on_shutdown() -> None:
    if prisma_client is None:
        return

    try:
        from litellm.proxy.utils import drain_spend_logs_queue

        await drain_spend_logs_queue(
            prisma_client=prisma_client,
            db_writer_client=db_writer_client,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:  # noqa: BLE001  # shutdown must continue even if the drain fails
        verbose_proxy_logger.exception("Error flushing spend logs queue on shutdown: %s", e)


async def proxy_shutdown_event(worker_heartbeat: ProxyWorkerHeartbeat | None = None) -> None:
    global prisma_client, master_key, user_custom_auth, user_custom_key_generate, user_custom_key_update
    verbose_proxy_logger.info("Shutting down LiteLLM Proxy Server")
    if worker_heartbeat is not None and prisma_client:
        await worker_heartbeat.deregister()
    if prisma_client:
        # Drain the SGR fold first: it lives in memory, so an un-drained interval
        # is lost, and a write attempted after disconnect raises
        # ClientNotConnectedError rather than persisting anything. Ordering this
        # inside the same guard is what keeps the two from drifting apart.
        await flush_gateway_requests(prisma_client, gateway_request_accumulator)
        verbose_proxy_logger.debug("Disconnecting from Prisma")
        await prisma_client.disconnect()

    if litellm.cache is not None:
        await litellm.cache.disconnect()

    await jwt_handler.close()

    if db_writer_client is not None:
        await db_writer_client.close()

    # final flush of billable-request counts: without it, up to one export
    # interval of enterprise billing data is dropped on every restart
    if shutdown_billing_metrics_recorder is not None:
        shutdown_billing_metrics_recorder()

    # flush remaining langfuse logs
    if "langfuse" in litellm.success_callback:
        try:
            # flush langfuse logs on shutdow
            from litellm.utils import langFuseLogger

            if langFuseLogger is not None:
                langFuseLogger.Langfuse.flush()
        except Exception:
            # [DO NOT BLOCK shutdown events for this]
            pass

    ## RESET CUSTOM VARIABLES ##
    cleanup_router_config_variables()


_AiohttpAddrInfo: TypeAlias = tuple[int | socket.AddressFamily, int | socket.SocketKind, int, str, tuple[object, ...]]


class _AiohttpConnectorKwargs(TypedDict, total=False):
    keepalive_timeout: float
    ttl_dns_cache: int
    enable_cleanup_closed: bool
    limit: int
    limit_per_host: int
    socket_factory: Callable[[_AiohttpAddrInfo], socket.socket]


async def _initialize_shared_aiohttp_session():
    """Initialize shared aiohttp session for connection reuse with connection limits."""
    try:
        from aiohttp import ClientSession, DummyCookieJar, TCPConnector

        from litellm.llms.custom_httpx.http_handler import (
            _build_aiohttp_keepalive_socket_factory,
        )

        connector_kwargs: Final[_AiohttpConnectorKwargs] = {
            "keepalive_timeout": AIOHTTP_KEEPALIVE_TIMEOUT,
            "ttl_dns_cache": AIOHTTP_TTL_DNS_CACHE,
        }
        if AIOHTTP_NEEDS_CLEANUP_CLOSED:
            connector_kwargs["enable_cleanup_closed"] = True
        if AIOHTTP_CONNECTOR_LIMIT > 0:
            connector_kwargs["limit"] = AIOHTTP_CONNECTOR_LIMIT
        if AIOHTTP_CONNECTOR_LIMIT_PER_HOST > 0:
            connector_kwargs["limit_per_host"] = AIOHTTP_CONNECTOR_LIMIT_PER_HOST
        socket_factory: Final = _build_aiohttp_keepalive_socket_factory()
        if socket_factory is not None:
            connector_kwargs["socket_factory"] = socket_factory

        connector: Final = TCPConnector(**connector_kwargs)
        session: Final = ClientSession(connector=connector, cookie_jar=DummyCookieJar())

        verbose_proxy_logger.info(
            "SESSION REUSE: Created shared aiohttp session for connection pooling (ID: %s, limit=%s, limit_per_host=%s)",
            id(session),
            AIOHTTP_CONNECTOR_LIMIT,
            AIOHTTP_CONNECTOR_LIMIT_PER_HOST,
        )
        return session
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to create shared aiohttp session: %s. Continuing without session reuse.", e
        )
        return None


@asynccontextmanager
async def proxy_startup_event(app: FastAPI) -> AsyncGenerator[None, None]:
    global \
        prisma_client, \
        master_key, \
        use_background_health_checks, \
        llm_router, \
        llm_model_list, \
        general_settings, \
        proxy_budget_rescheduler_min_time, \
        proxy_budget_rescheduler_max_time, \
        litellm_proxy_admin_name, \
        db_writer_client, \
        store_model_in_db, \
        premium_user, \
        _license_check, \
        proxy_batch_polling_interval, \
        shared_aiohttp_session
    import json

    init_verbose_loggers()

    ## RUN WORKER STARTUP HOOKS (e.g., gflags initialization) ##
    _startup_hooks_env: Final = os.environ.get("LITELLM_WORKER_STARTUP_HOOKS", "")
    if _startup_hooks_env:
        for _hook_spec in _startup_hooks_env.split(","):
            _hook_spec = _hook_spec.strip()
            if not _hook_spec:
                continue
            try:
                if ":" not in _hook_spec:
                    raise ValueError(
                        f"Invalid hook spec '{_hook_spec}': expected format is 'module.path:function_name'"
                    )
                _module_path, _func_name = _hook_spec.rsplit(":", 1)
                _module = importlib.import_module(_module_path)
                _hook_fn = getattr(_module, _func_name)
                if inspect.iscoroutinefunction(_hook_fn):
                    await _hook_fn()
                else:
                    _hook_fn()
                verbose_proxy_logger.info("Worker startup hook '%s' executed successfully", _hook_spec)
            except Exception as e:
                verbose_proxy_logger.error("Worker startup hook '%s' failed: %s", _hook_spec, e)
                raise

    ## CHECK PREMIUM USER
    verbose_proxy_logger.debug("litellm.proxy.proxy_server.py::startup() - CHECKING PREMIUM USER - %s", premium_user)
    if premium_user is False:
        premium_user = _license_check.is_premium()

    ## CHECK MASTER KEY IN ENVIRONMENT ##
    master_key = get_secret_str("LITELLM_MASTER_KEY")
    ### LOAD CONFIG ###
    worker_config: str | dict | None = get_secret("WORKER_CONFIG")
    env_config_yaml: Final[str | None] = get_secret_str("CONFIG_FILE_PATH")
    verbose_proxy_logger.debug("worker_config: %s", _redact_worker_config_for_logging(worker_config))
    # check if it's a valid file path
    if env_config_yaml is not None:
        if os.path.isfile(env_config_yaml) and proxy_config.is_yaml(config_file_path=env_config_yaml):
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(router=llm_router, config_file_path=env_config_yaml)
    elif worker_config is not None:
        if (
            (
                isinstance(worker_config, str)
                and os.path.isfile(worker_config)
                and proxy_config.is_yaml(config_file_path=worker_config)
            )
            or os.environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None
            and isinstance(worker_config, str)
        ):
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(router=llm_router, config_file_path=worker_config)
        elif isinstance(worker_config, dict):
            await initialize(**worker_config)
        else:
            # if not, assume it's a json string
            worker_config = json.loads(worker_config)
            if isinstance(worker_config, dict):
                await initialize(**worker_config)

    # check if DATABASE_URL in environment - load from there
    if prisma_client is None:
        _db_url: Final[str | None] = get_secret("DATABASE_URL", None)
        prisma_client = await ProxyStartupEvent._setup_prisma_client(
            database_url=_db_url,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
        )

    if prisma_client is not None:

        async def _run_pw_migration():
            try:
                result: Final = await migrate_passwords_to_scrypt_async(prisma_client)
                verbose_proxy_logger.info("Password migration: %s", result)
            except Exception as e:
                verbose_proxy_logger.warning("Password migration skipped: %s", e)

        asyncio.create_task(_run_pw_migration())

        async def _run_agent_grant_id_migration() -> None:
            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
                object_permission_table,
            )

            for attempt in range(3):
                try:
                    result = await global_agent_registry.migrate_legacy_grant_ids(
                        table=object_permission_table(prisma_client)
                    )
                    if result.rewritten:
                        verbose_proxy_logger.info(
                            "Rewrote %s object_permission rows from legacy config agent ids", result.rewritten
                        )
                    if result.missed == 0:
                        return
                    verbose_proxy_logger.warning(
                        "Legacy agent grant id migration attempt %s/3 left %s rows unmigrated",
                        attempt + 1,
                        result.missed,
                    )
                except Exception as e:  # noqa: BLE001  # startup task must survive any DB error and retry
                    verbose_proxy_logger.warning(
                        "Legacy agent grant id migration attempt %s/3 failed: %s", attempt + 1, e
                    )
                if attempt < 2:
                    await asyncio.sleep(5)

        asyncio.create_task(_run_agent_grant_id_migration())

    ## A coordination_redis block saved from the admin UI lives in the database,
    ## which is only reachable once the prisma client exists. Apply it here, before
    ## the coordination Redis is published to its consumers below.
    db_coordination_redis_cache: Final = await ProxyStartupEvent._init_coordination_redis_from_db(
        litellm_settings=proxy_config.get_config_state().get("litellm_settings") or {},
        llm_router=llm_router,
    )
    if db_coordination_redis_cache is not None:
        _set_redis_usage_cache(db_coordination_redis_cache)

    ## use_redis_transaction_buffer: fall back to a standalone Redis (REDIS_* env)
    ## when the proxy cache backend is not Redis ##
    transaction_buffer_redis_cache = redis_usage_cache
    if transaction_buffer_redis_cache is None:
        transaction_buffer_redis_cache = ProxyStartupEvent._get_transaction_buffer_redis_cache(
            general_settings=general_settings
        )

    ProxyStartupEvent._initialize_startup_logging(
        llm_router=llm_router,
        proxy_logging_obj=proxy_logging_obj,
        redis_usage_cache=transaction_buffer_redis_cache,
    )

    ## V2 OTEL: publish the chosen V2 logger's TracerProvider as the OTel global.
    ## This MUST run after callback initialization above: a preset (arize, langfuse,
    ## …) builds its logger there, folding the OTEL_* base exporter and its own
    ## exporter into one logger. The FastAPI instrumentation mounted at app-creation
    ## binds to the global provider, so reusing that one logger is what makes the
    ## server span and the gen-ai spans share one provider and land in the same
    ## trace, exporting to every configured backend. Running before callback init
    ## (when no logger exists yet) would build a second, generic logger whose
    ## provider became the global, orphaning the gen-ai spans onto a different
    ## backend than the server span. A generic logger is built only when none was
    ## configured.
    try:
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        if is_otel_v2_enabled():
            from opentelemetry import trace as _otel_trace

            from litellm.integrations.otel.logger import (
                OpenTelemetryV2,
                publish_global_otel_v2_provider,
            )
            from litellm.litellm_core_utils.litellm_logging import _in_memory_loggers

            registered: Final = open_telemetry_logger if isinstance(open_telemetry_logger, OpenTelemetryV2) else None
            publish_global_otel_v2_provider(
                _in_memory_loggers,  # any-ok: pre-existing untyped List[Any] global
                _otel_trace.set_tracer_provider,
                registered=registered,
            )
    except Exception as e:
        verbose_proxy_logger.debug("Skipping OTel V2 provider setup: %s", e)

    ## Validate use_redis_transaction_buffer requires Redis cache ##
    ProxyStartupEvent._validate_redis_transaction_buffer_config(
        general_settings=general_settings,
        redis_usage_cache=transaction_buffer_redis_cache,
    )

    ProxyStartupEvent._warn_if_mock_testing_params_enabled(general_settings=general_settings)

    ## SEMANTIC TOOL FILTER ##
    # Read litellm_settings from config for semantic filter initialization
    try:
        verbose_proxy_logger.debug("About to initialize semantic tool filter")
        _config: Final = proxy_config.get_config_state()
        _litellm_settings: Final = _config.get("litellm_settings", {})
        verbose_proxy_logger.debug("litellm_settings keys = %s", list(_litellm_settings.keys()))
        await ProxyStartupEvent._initialize_semantic_tool_filter(
            llm_router=llm_router,
            litellm_settings=_litellm_settings,
        )
        verbose_proxy_logger.debug("After semantic tool filter initialization")
    except Exception as e:
        verbose_proxy_logger.error("Semantic filter init failed: %s", e, exc_info=True)

    ## JWT AUTH ##
    ProxyStartupEvent._initialize_jwt_auth(
        general_settings=general_settings,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )

    if prompt_injection_detection_obj is not None:  # [TODO] - REFACTOR THIS
        prompt_injection_detection_obj.update_environment(router=llm_router)

    verbose_proxy_logger.debug("prisma_client: %s", prisma_client)
    if prisma_client is not None and litellm.max_budget > 0:
        ProxyStartupEvent._add_proxy_budget_to_db()
        asyncio.create_task(
            ProxyStartupEvent._warm_global_spend_cache(
                user_api_key_cache=user_api_key_cache,
                prisma_client=prisma_client,
            )
        )
    ProxyStartupEvent._warn_budget_without_db(
        max_budget=litellm.max_budget,
        prisma_client=prisma_client,
    )

    ### START BATCH WRITING DB + CHECKING NEW MODELS###
    worker_heartbeat: Final = (
        await ProxyStartupEvent.initialize_scheduled_background_jobs(
            general_settings=general_settings,
            prisma_client=prisma_client,
            proxy_budget_rescheduler_min_time=proxy_budget_rescheduler_min_time,
            proxy_budget_rescheduler_max_time=proxy_budget_rescheduler_max_time,
            proxy_batch_write_at=proxy_batch_write_at,
            proxy_logging_obj=proxy_logging_obj,
        )
        if prisma_client is not None
        else None
    )
    if prisma_client is not None:
        await ProxyStartupEvent._update_default_team_member_budget()

        ## SYNC UI SETTINGS ##
        await ProxyStartupEvent._sync_ui_settings_to_general_settings()

    # Start background health checks AFTER models are loaded and index is built
    if use_background_health_checks:
        asyncio.create_task(_run_background_health_check())  # start the background health check coroutine.

    # Start adaptive-router queue flusher unconditionally — adaptive routers
    # may be added later via `/config/reload`, and the flusher is a no-op when
    # `llm_router.adaptive_routers` is empty. Per-router DB state is loaded
    # lazily by the flusher on first tick (see `_state_loaded` flag) so
    # hot-reloaded routers also get their persisted priors.
    if llm_router is not None and getattr(llm_router, "adaptive_routers", None):
        for _tagged_routers in llm_router.adaptive_routers.values():
            for _tagged in _tagged_routers:
                await _tagged.strategy.load_state_from_db(prisma_client)
                _tagged.strategy._state_loaded = True
    asyncio.create_task(_adaptive_router_flusher_loop())

    ## [Optional] Initialize dd tracer
    ProxyStartupEvent._init_dd_tracer()

    ## [Optional] Initialize Pyroscope continuous profiling (env: LITELLM_ENABLE_PYROSCOPE=true)
    ProxyStartupEvent._init_pyroscope()

    ## Initialize shared aiohttp session for connection reuse
    shared_aiohttp_session = await _initialize_shared_aiohttp_session()

    # End of startup event
    yield

    # Shutdown event - drain in-flight requests before tearing down dependencies
    # so SIGTERM (rolling update, scale-down, liveness kill) doesn't drop them.
    GracefulShutdownManager.start_shutdown()
    await GracefulShutdownManager.wait_for_drain()

    # Shutdown event - close shared aiohttp session
    if shared_aiohttp_session is not None:
        try:
            await shared_aiohttp_session.close()
            verbose_proxy_logger.info("SESSION REUSE: Closed shared aiohttp session")
        except Exception as e:
            verbose_proxy_logger.error("Error closing shared aiohttp session: %s", e)

    # Shutdown event - stop RDS IAM token refresh background task
    if (
        prisma_client is not None
        and hasattr(prisma_client, "db")
        and hasattr(prisma_client.db, "stop_token_refresh_task")
    ):
        try:
            await prisma_client.db.stop_token_refresh_task()
        except Exception as e:
            verbose_proxy_logger.error("Error stopping token refresh task: %s", e)

    # Shutdown event - stop Prisma DB health watchdog task
    if prisma_client is not None and hasattr(prisma_client, "stop_db_health_watchdog_task"):
        try:
            await prisma_client.stop_db_health_watchdog_task()
        except Exception as e:
            verbose_proxy_logger.error("Error stopping DB health watchdog task: %s", e)

    await _flush_spend_logs_queue_on_shutdown()

    await proxy_config.stop_config_sync_subscriber()

    await proxy_config.stop_auth_cache_invalidation_subscriber()

    await proxy_shutdown_event(worker_heartbeat=worker_heartbeat)


def _generate_stable_operation_id(route: "APIRoute") -> str:
    operation_id = re.sub(r"\W", "_", f"{route.name}{route.path_format}")
    route_methods: Final = sorted(route.methods or [])
    if len(route_methods) == 1:
        operation_id = f"{operation_id}_{route_methods[0].lower()}"
    return operation_id


_OPENAPI_HTTP_METHODS: Final = {
    "delete",
    "get",
    "head",
    "options",
    "patch",
    "post",
    "put",
    "trace",
}


# Credentials surfaced by `/get/config/callbacks` in the alerting block: the
# full Slack incoming-webhook URL is itself a credential, and the SMTP
# password is a service password. Masked on read so plaintext never reaches
# the UI. Kept here at module scope to match the analogous descriptor
# `is_secret` flags in litellm.proxy.config_resolvers and the
# `_CACHE_SENSITIVE_FIELDS` constant in the cache endpoint file.
_ALERTING_SENSITIVE_VARS: Final[set[str]] = {"SLACK_WEBHOOK_URL", "SMTP_PASSWORD"}


def _strip_operation_id_method_suffix(operation_id: str) -> str:
    base, separator, suffix = operation_id.rpartition("_")
    if separator and suffix in _OPENAPI_HTTP_METHODS:
        return base
    return operation_id


def ensure_unique_openapi_operation_ids(
    openapi_schema: dict[str, Any],
    reserved_operation_ids: set[str] | None = None,
) -> dict[str, Any]:
    operation_entries: Final = []
    operation_id_counts: Final[dict[str, int]] = {}
    for path_item in openapi_schema.get("paths", {}).values():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method not in _OPENAPI_HTTP_METHODS or not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str):
                continue
            operation_entries.append((method, operation, operation_id))
            operation_id_counts[operation_id] = operation_id_counts.get(operation_id, 0) + 1

    used_operation_ids: Final = set(reserved_operation_ids or set())
    seen_operation_ids: Final[set[str]] = set()
    for method, operation, operation_id in operation_entries:
        should_rewrite = (
            operation_id_counts[operation_id] > 1
            or operation_id in used_operation_ids
            or operation_id in seen_operation_ids
        )
        if not should_rewrite:
            seen_operation_ids.add(operation_id)
            used_operation_ids.add(operation_id)
            continue

        base_operation_id = _strip_operation_id_method_suffix(operation_id)
        new_operation_id = f"{base_operation_id}_{method}"
        suffix = 2
        while new_operation_id in used_operation_ids or new_operation_id in seen_operation_ids:
            new_operation_id = f"{base_operation_id}_{method}_{suffix}"
            suffix += 1
        operation["operationId"] = new_operation_id
        seen_operation_ids.add(new_operation_id)
        used_operation_ids.add(new_operation_id)

    if reserved_operation_ids is not None:
        reserved_operation_ids.update(used_operation_ids)

    return openapi_schema


app = FastAPI(
    docs_url=_get_docs_url(),
    redoc_url=_get_redoc_url(),
    openapi_url=_get_openapi_url(),
    title=_title,
    description=_description,
    version=version,
    root_path=server_root_path,
    lifespan=proxy_startup_event,
    generate_unique_id_function=_generate_stable_operation_id,
    strict_content_type=False,
)

## V2 OTEL: instrument the FastAPI app for server spans (gated by
## LITELLM_OTEL_V2). This MUST run at app-creation time — once the lifespan runs,
## the middleware stack is frozen and ``instrument_app`` raises "Cannot add
## middleware after an application has started". See
## ``litellm.integrations.otel.mount`` for the full rationale; the call is a safe
## no-op when the gate is off or the instrumentation package is unavailable.
from litellm.integrations.otel.mount import instrument_fastapi_app

instrument_fastapi_app(app)

vertex_live_passthrough_vertex_base: Final = VertexBase()


### CUSTOM API DOCS [ENTERPRISE FEATURE] ###
# Custom OpenAPI schema generator to include only selected routes
from fastapi.routing import APIWebSocketRoute


def _inject_websocket_stubs_into_openapi_schema(openapi_schema: dict, websocket_routes: list) -> dict:
    """
    Add a synthetic GET stub for each WebSocket route so it appears in Swagger UI.

    Merges into any existing path entry rather than replacing it — a WebSocket route
    that shares its path with an HTTP route must not erase the HTTP operation. If
    a "get" operation is already documented on the path, the WebSocket stub is
    skipped to preserve the real GET.
    """
    for route in websocket_routes:
        base_path = route.path.split("{")[0].rstrip("?")

        parameters = []
        try:
            if hasattr(route, "dependant") and route.dependant is not None:
                # Handle both FastAPI <0.120 and >=0.120
                query_params = getattr(route.dependant, "query_params", [])
                if query_params:
                    for param in query_params:
                        parameters.append(
                            {
                                "name": param.name,
                                "in": "query",
                                "required": param.required,
                                "schema": {"type": "string"},
                            }
                        )
        except (AttributeError, TypeError):
            pass

        path_entry = openapi_schema["paths"].setdefault(base_path, {})
        if "get" not in path_entry:
            path_entry["get"] = {
                "summary": f"WebSocket: {route.name or base_path}",
                "description": "WebSocket connection endpoint",
                "operationId": f"websocket_{route.name or base_path.replace('/', '_')}",
                "parameters": parameters,
                "responses": {"101": {"description": "WebSocket Protocol Switched"}},
                "tags": ["WebSocket"],
            }

    return openapi_schema


def get_openapi_schema():
    if app.openapi_schema:
        return app.openapi_schema

    # Use compatibility wrapper for FastAPI 0.120+ schema generation
    from litellm.proxy.common_utils.openapi_schema_compat import (
        get_openapi_schema_with_compat,
    )

    openapi_schema = get_openapi_schema_with_compat(
        get_openapi_func=get_openapi,
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Find all WebSocket routes
    websocket_routes: Final = [route for route in app.routes if isinstance(route, APIWebSocketRoute)]

    # Add a synthetic GET stub for each so they render in Swagger UI,
    # without clobbering existing HTTP operations on the same path.
    openapi_schema = _inject_websocket_stubs_into_openapi_schema(openapi_schema, websocket_routes)

    # Add LLM API request schema bodies for documentation
    from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec

    openapi_schema = CustomOpenAPISpec.add_llm_api_request_schema_body(openapi_schema)

    # Stub unloaded lazy features so they appear as Swagger sections.
    from litellm.proxy._lazy_features import inject_lazy_stubs, loaded_lazy_modules

    openapi_schema = inject_lazy_stubs(openapi_schema, loaded_lazy_modules(app))
    openapi_schema = ensure_unique_openapi_operation_ids(openapi_schema)

    # Fix Swagger UI execute path error when server_root_path is set
    if server_root_path:
        openapi_schema["servers"] = [{"url": "/" + server_root_path.strip("/")}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi_schema()

    # Filter routes to include only specific ones
    openai_routes: Final = LiteLLMRoutes.openai_routes.value
    paths_to_include: Final[dict] = {}
    for route in openai_routes:
        if route in openapi_schema["paths"]:
            paths_to_include[route] = openapi_schema["paths"][route]
    openapi_schema["paths"] = paths_to_include

    # Add LLM API request schema bodies for documentation
    from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec

    openapi_schema = CustomOpenAPISpec.add_llm_api_request_schema_body(openapi_schema)

    # Stub unloaded lazy features so they appear as Swagger sections.
    from litellm.proxy._lazy_features import inject_lazy_stubs, loaded_lazy_modules

    openapi_schema = inject_lazy_stubs(openapi_schema, loaded_lazy_modules(app))
    openapi_schema = ensure_unique_openapi_operation_ids(openapi_schema)

    # Fix Swagger UI execute path error when server_root_path is set
    if server_root_path:
        openapi_schema["servers"] = [{"url": "/" + server_root_path.strip("/")}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


if os.getenv("DOCS_FILTERED", "False") == "True" and premium_user:
    app.openapi = custom_openapi
else:
    # For regular users, use get_openapi_schema to include LLM API schemas
    app.openapi = get_openapi_schema


class UserAPIKeyCacheTTLEnum(enum.Enum):
    in_memory_cache_ttl = 60  # 1 min ttl ## configure via `general_settings::user_api_key_cache_ttl: <your-value>`


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    # NOTE: DO NOT MODIFY THIS, its crucial to map to Openai exceptions
    headers: Final = exc.headers
    error_dict: Final = exc.to_dict()
    status_code: Final = int(exc.code) if exc.code else status.HTTP_500_INTERNAL_SERVER_ERROR
    _close_dangling_otel_server_span(request, status_code, exc=exc)
    return JSONResponse(
        status_code=status_code,
        content={"error": error_dict},
        headers=headers,
    )


def _close_dangling_otel_server_span(request: Request, status_code: int, exc: Exception | None = None) -> None:
    parent_otel_span: Final[_Span | None] = getattr(request.state, "parent_otel_span", None)
    if parent_otel_span is None:
        return
    if open_telemetry_logger is None:
        return
    # Under OTel V2 the FastAPI instrumentor owns the server span (parent_otel_span
    # is that same span) and ends it itself with the http.* attributes stamped on
    # completion. The instrumentor only records an error when the exception reaches
    # it uncaught, but these handlers swallow it into a JSONResponse, so it never
    # does; stamp the error.* attributes here (without ending or re-statusing the
    # span, which the instrumentor still owns) so pre-call failures carry the error
    # like v1 did. Otherwise close and annotate the dangling span ourselves.
    try:
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        v2_enabled = is_otel_v2_enabled()
    except Exception:
        v2_enabled = False
    try:
        from opentelemetry.trace import Status, StatusCode

        if v2_enabled:
            if status_code >= 400:
                open_telemetry_logger.record_error_attributes_on_span(parent_otel_span, exc, status_code)
            return
        open_telemetry_logger.set_response_status_code_attribute(parent_otel_span, status_code)
        if status_code >= 400:
            open_telemetry_logger.record_error_attributes_on_span(parent_otel_span, exc, status_code)
        parent_otel_span.set_status(Status(StatusCode.ERROR if status_code >= 400 else StatusCode.OK))
        parent_otel_span.end()
    except Exception as e:
        verbose_proxy_logger.debug("Error closing dangling OTEL SERVER span: %s", str(e))
    finally:
        if not v2_enabled:
            request.state.parent_otel_span = None


@app.exception_handler(ManagementProblem)
async def management_problem_exception_handler(request: Request, exc: ManagementProblem):
    _close_dangling_otel_server_span(request, exc.problem.status, exc=exc)
    return problem_response(exc.problem)


class _ConfigParamRow(Protocol):
    param_name: str
    param_value: Mapping[str, JsonValue] | None


class _ConfigOverridesRow(Protocol):
    config_value: Mapping[str, JsonValue] | None


class _SSOConfigRow(Protocol):
    sso_settings: MutableMapping[str, object]


class _UISettingsRow(Protocol):
    ui_settings: Mapping[str, object] | str | None


class _InvitationLinkRow(Protocol):
    user_id: str
    expires_at: datetime
    is_accepted: bool
    accepted_at: datetime | None
    created_by: str


class _UserTableRow(Protocol):
    user_id: str
    user_email: str | None
    user_role: str | None


class _UserTeamsRow(Protocol):
    @property
    def teams(self) -> Sequence[str]: ...


_ProxyModelRow: TypeAlias = "prisma_models.LiteLLM_ProxyModelTable"


def _config_param_table(client: PrismaClient | None) -> TableActions[_ConfigParamRow]:
    return cast(  # cast-ok: this is prisma's LiteLLM_Config actions object, which parses its Json column to a mapping
        "TableActions[_ConfigParamRow]", ConfigRepository(client).table
    )


class _TTFTRow(TypedDict):
    api_base: str
    model: str
    time_to_first_token: float
    request_id: str
    day: str


class _LatencyRow(TypedDict):
    api_base: str | None
    model: str
    day: str
    avg_latency_per_token: float


class _ExceptionRow(TypedDict, total=False):
    combined_model_api_base: str
    total_exceptions: int
    exception_counts: Mapping[str, int]


class _ValidationErrorDetail(TypedDict):
    loc: tuple[int | str, ...]
    msg: str


@app.exception_handler(RequestValidationError)
async def otel_request_validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith(MANAGEMENT_V1_PREFIX):
        _close_dangling_otel_server_span(request, 400, exc=exc)
        validation_errors: Final[Sequence[_ValidationErrorDetail]] = exc.errors()
        return problem_response(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}invalid-query-parameter",
                title="Invalid query parameter",
                status=400,
                detail="; ".join(
                    f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}" for error in validation_errors
                )
                or "The request query parameters are invalid.",
            )
        )
    _close_dangling_otel_server_span(request, 422, exc=exc)
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(Exception)
async def otel_unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (ProxyException, HTTPException, RequestValidationError)):
        raise exc
    verbose_proxy_logger.exception("Unhandled exception in request: %s", type(exc).__name__)
    _close_dangling_otel_server_span(request, 500, exc=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_server_error",
            }
        },
    )


router: Final = APIRouter()


def _get_cors_config(
    cors_origins_env: str | None = None,
    cors_credentials_env: str | None = None,
):
    """
    Compute CORS allowed origins and credentials flag from environment variables.

    Extracted into a function so it can be unit-tested without reloading the module.

    Args:
        cors_origins_env: Value of LITELLM_CORS_ORIGINS (defaults to os.getenv).
        cors_credentials_env: Value of LITELLM_CORS_ALLOW_CREDENTIALS (defaults to os.getenv).

    Returns:
        Tuple[List[str], bool]: (origins, allow_credentials)
    """
    _origins_raw: Final = cors_origins_env if cors_origins_env is not None else os.getenv("LITELLM_CORS_ORIGINS")
    if _origins_raw is None or _origins_raw.strip() == "":
        computed_origins = ["*"]
    else:
        computed_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]

    # Disable credentials by default when wildcard origins are used — combining
    # allow_origins=["*"] with allow_credentials=True causes Starlette to reflect
    # the incoming Origin header, allowing any site to make credentialed requests.
    # Set LITELLM_CORS_ALLOW_CREDENTIALS=true to explicitly restore the old behaviour
    # (e.g. for non-browser clients that relied on the Access-Control-Allow-Credentials
    # header being present regardless of origin).
    _credentials_raw: Final = (
        cors_credentials_env if cors_credentials_env is not None else os.getenv("LITELLM_CORS_ALLOW_CREDENTIALS")
    )
    if _credentials_raw is not None:
        computed_credentials = _credentials_raw.strip().lower() == "true"
    else:
        computed_credentials = "*" not in computed_origins

    return computed_origins, computed_credentials


origins, allow_cors_credentials = _get_cors_config()


# get current directory
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    packaged_ui_path: Final = os.path.join(current_dir, "_experimental", "out")
    ui_path = packaged_ui_path
    litellm_asset_prefix: Final = "/litellm-asset-prefix"

    def _dir_has_content(path: str) -> bool:
        try:
            return os.path.isdir(path) and any(os.scandir(path))
        except FileNotFoundError:
            return False

    def _validate_ui_directory(ui_path: str) -> bool:
        """
        Verify UI directory has minimum required structure.

        Checks for:
        - Directory exists
        - Has index.html (main entry point)
        - Has _next directory (Next.js assets)

        Returns True if UI directory appears valid and servable.
        """
        if not os.path.isdir(ui_path):
            return False

        # Must have main index.html
        if not os.path.exists(os.path.join(ui_path, "index.html")):
            return False

        # Must have _next directory with Next.js assets
        next_dir: Final = os.path.join(ui_path, "_next")
        if not os.path.isdir(next_dir):
            return False

        return True

    def _is_ui_pre_restructured(ui_dir: str) -> bool:
        """
        Detect if UI directory is already pre-restructured and ready to serve.

        Returns True if:
        1. Marker file .litellm_ui_ready exists (created by Dockerfile), OR
        2. Restructuring pattern detected (subdirectories with index.html inside)

        This allows skipping copy/restructure operations on read-only filesystems.
        """
        if not os.path.isdir(ui_dir):
            return False

        # Primary signal: marker file created by Dockerfile
        marker_file: Final = os.path.join(ui_dir, ".litellm_ui_ready")
        if os.path.exists(marker_file):
            verbose_proxy_logger.debug("Found UI ready marker: %s", marker_file)
            return True

        # Fallback signal: Detect restructuring pattern
        # After restructuring, routes exist as directories with index.html inside
        # (e.g., login/index.html instead of login.html)
        # Check for main index.html first (basic UI structure requirement)
        if not os.path.exists(os.path.join(ui_dir, "index.html")):
            return False

        # Look for ANY subdirectory with index.html (proves restructuring happened)
        # Ignore directories starting with _ (Next.js internals like _next)
        try:
            for entry in os.scandir(ui_dir):
                if entry.is_dir() and not entry.name.startswith("_"):
                    index_path = os.path.join(entry.path, "index.html")
                    if os.path.exists(index_path):
                        # Found at least one restructured route - this proves the pattern
                        verbose_proxy_logger.debug(
                            "Detected restructured UI via pattern: found %s/index.html", entry.name
                        )
                        return True
        except (PermissionError, OSError) as e:
            verbose_proxy_logger.debug("Could not scan %s for restructuring detection: %s", ui_dir, e)
            return False

        # No restructured routes found
        return False

    def _try_populate_ui_directory(source_path: str, target_path: str) -> tuple[bool, str]:
        """
        Attempt to populate target UI directory from source.

        Returns: (success: bool, error_message: str)
        """
        try:
            os.makedirs(target_path, exist_ok=True)
            if not _dir_has_content(target_path) and _dir_has_content(source_path):
                shutil.copytree(
                    source_path,
                    target_path,
                    dirs_exist_ok=True,
                )
                verbose_proxy_logger.info("Successfully populated UI at %s", target_path)
                return True, ""
            else:
                return False, "Source or target directory state invalid"
        except (PermissionError, OSError) as e:
            return False, str(e)

    # Use a writable runtime UI directory whenever possible.
    # This prevents mutating the packaged UI directory (e.g. site-packages or the repo checkout)
    # and ensures extensionless routes like /ui/login work via <route>/index.html.
    is_non_root: Final = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"

    # Determine runtime UI path
    # Priority: LITELLM_UI_PATH env var > default path based on is_non_root
    if is_non_root:
        default_runtime_ui_path = "/var/lib/litellm/ui"
    else:
        default_runtime_ui_path = packaged_ui_path

    runtime_ui_path: Final = os.getenv("LITELLM_UI_PATH", default_runtime_ui_path)

    # Validate packaged UI before proceeding
    if not _validate_ui_directory(packaged_ui_path):
        verbose_proxy_logger.error(
            "Packaged UI at %s is invalid or incomplete. UI may not function correctly.", packaged_ui_path
        )

    # Decision tree for UI path selection:
    # 1. If runtime path == packaged path: use packaged UI directly
    # 2. If runtime UI exists and is pre-restructured: use it
    # 3. If runtime UI exists but not restructured: use it (will restructure later)
    # 4. If runtime UI missing: try to populate from packaged UI
    #    4a. If population succeeds: use runtime UI
    #    4b. If population fails: fall back to packaged UI

    should_use_runtime_path: Final = runtime_ui_path != packaged_ui_path

    if should_use_runtime_path:
        is_pre_restructured = _is_ui_pre_restructured(runtime_ui_path)
        has_content: Final = _dir_has_content(runtime_ui_path)

        # Case 2: Runtime UI exists and is ready
        if has_content and is_pre_restructured:
            verbose_proxy_logger.info("Using pre-restructured UI at %s", runtime_ui_path)
            ui_path = runtime_ui_path

        # Case 3: Runtime UI exists but needs restructuring
        elif has_content and not is_pre_restructured:
            verbose_proxy_logger.warning(
                "UI at %s has content but is not properly restructured. Will attempt to restructure in place.",
                runtime_ui_path,
            )
            ui_path = runtime_ui_path

        # Case 4: Runtime UI missing - try to populate
        else:
            verbose_proxy_logger.info("UI not found at %s. Attempting to populate from packaged UI.", runtime_ui_path)

            success, error = _try_populate_ui_directory(packaged_ui_path, runtime_ui_path)

            if success:
                # Case 4a: Population succeeded
                ui_path = runtime_ui_path
            else:
                # Case 4b: Population failed - fall back to packaged UI
                verbose_proxy_logger.warning(
                    "Failed to populate UI at %s: %s. Falling back to packaged UI at %s. For read-only deployments, pre-build UI in Dockerfile or set LITELLM_UI_PATH to a writable emptyDir volume.",
                    runtime_ui_path,
                    error,
                    packaged_ui_path,
                )
                ui_path = packaged_ui_path
    else:
        # Case 1: Using packaged UI directly (local development)
        verbose_proxy_logger.info("Using packaged UI directory: %s", packaged_ui_path)
        ui_path = packaged_ui_path

    # Validate final UI path
    if not _validate_ui_directory(ui_path):
        verbose_proxy_logger.error("Selected UI path %s is invalid or incomplete. UI may not work correctly.", ui_path)

    # Only modify files if a custom server root path is set AND filesystem is writable
    if server_root_path and server_root_path != "/":
        # Check if UI path is writable
        is_writable = os.access(ui_path, os.W_OK)

        if not is_writable:
            verbose_proxy_logger.warning(
                "Cannot apply server_root_path replacements to UI at %s: path is not writable. Ensure server_root_path is '/' or pre-process UI files in Dockerfile with custom server_root_path.",
                ui_path,
            )
        else:
            # Iterate through files in the UI directory
            for root, dirs, files in os.walk(ui_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    # Skip binary files and files that don't need path replacement
                    if filename.endswith(
                        (
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".gif",
                            ".ico",
                            ".woff",
                            ".woff2",
                            ".ttf",
                            ".eot",
                        )
                    ):
                        continue
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Replace the asset prefix with the server root path
                        modified_content = content.replace(
                            f"{litellm_asset_prefix}",
                            f"{server_root_path}",
                        )

                        # Replace the /.well-known/litellm-ui-config with the server root path
                        modified_content = modified_content.replace(
                            "/litellm/.well-known/litellm-ui-config",
                            f"{server_root_path}/.well-known/litellm-ui-config",
                        )

                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(modified_content)
                    except (UnicodeDecodeError, PermissionError, OSError):
                        # Skip binary files or files we can't write to
                        continue

    # # Mount the _next directory at the root level
    app.mount(
        "/_next",
        StaticFiles(directory=os.path.join(ui_path, "_next")),
        name="next_static",
    )
    app.mount(
        f"{litellm_asset_prefix}/_next",
        StaticFiles(directory=os.path.join(ui_path, "_next")),
        name="next_static",
    )
    # print(f"mounted _next at {server_root_path}/ui/_next")

    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")

    def _restructure_ui_html_files(ui_root: str) -> None:
        """Ensure each exported HTML route is available as <route>/index.html."""

        for current_root, _, files in os.walk(ui_root):
            rel_root = os.path.relpath(current_root, ui_root)
            first_segment = "" if rel_root == "." else rel_root.split(os.sep)[0]

            # Ignore Next.js asset directories
            if first_segment in {"_next", "litellm-asset-prefix"}:
                continue

            for filename in files:
                if not filename.endswith(".html") or filename == "index.html":
                    continue

                file_path = os.path.join(current_root, filename)
                target_dir = os.path.splitext(file_path)[0]
                target_path = os.path.join(target_dir, "index.html")

                os.makedirs(target_dir, exist_ok=True)
                try:
                    os.replace(file_path, target_path)
                except FileNotFoundError:
                    # Another process may have already moved this file.
                    continue

    # Handle HTML file restructuring
    # Only restructure if:
    # 1. UI is not already pre-restructured
    # 2. Filesystem is writable
    try:
        is_pre_restructured = _is_ui_pre_restructured(ui_path)
        is_writable = os.access(ui_path, os.W_OK)

        if is_pre_restructured:
            verbose_proxy_logger.info("Skipping UI restructuring: %s is already pre-restructured", ui_path)
        elif not is_writable:
            verbose_proxy_logger.warning(
                "Cannot restructure UI at %s: path is not writable. UI may not work correctly for extensionless routes. Pre-build and restructure UI in Dockerfile for read-only deployments.",
                ui_path,
            )
        else:
            _restructure_ui_html_files(ui_path)
            verbose_proxy_logger.info("Restructured UI directory: %s", ui_path)
    except PermissionError as e:
        verbose_proxy_logger.exception("Permission error while restructuring UI directory %s: %s", ui_path, e)
    except Exception as e:
        verbose_proxy_logger.exception("Error while restructuring UI directory %s: %s", ui_path, e)

except Exception:
    pass
current_dir = os.path.dirname(os.path.abspath(__file__))
# ui_path = os.path.join(current_dir, "_experimental", "out")
# # Mount this test directory instead
# app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=LITELLM_UI_ALLOW_HEADERS,
)

app.add_middleware(PrometheusAuthMiddleware)
# Added before InFlightRequestsMiddleware so it nests *inside* it: Starlette
# makes the last-added middleware outermost. The billable count is recorded
# after the inner app returns, so if this sat outside the in-flight tracker a
# request could be counted as drained while its record() had not yet run, and
# proxy_shutdown_event could flush and stop the exporter underneath it.
app.add_middleware(
    BillableRequestMetricsMiddleware,
    # Factory, not an instance: the recorder is resolved on the first request so
    # it sees premium_user and the billing env vars AFTER proxy_startup_event has
    # loaded the YAML config's environment_variables. Building it here at import
    # time would permanently capture recorder=None for YAML-configured
    # deployments. The lambda reads the module globals at call time.
    recorder_factory=lambda: (
        build_billing_metrics_recorder(
            premium=premium_user,
            # Read from the license check, not the premium_user_data module
            # global: that global is bound once at import and goes stale when
            # the license arrives via the YAML config's environment_variables.
            license_data=_license_check.airgapped_license_data,
            litellm_version=version,
        )
        if build_billing_metrics_recorder is not None
        else None
    ),
    # Unlike the billing recorder this is not license-gated: the admin UI must
    # report SGR on any deployment. Gated only on a database being configured,
    # since without one the fold would never be drained. Read at call time, so
    # it sees prisma_client as of the first request rather than import time.
    sink_factory=lambda: gateway_request_accumulator if prisma_client is not None else None,
)
app.add_middleware(InFlightRequestsMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


def mount_swagger_ui():
    swagger_directory: Final = os.path.join(current_dir, "swagger")
    swagger_path = "/" if server_root_path is None else server_root_path
    if not swagger_path.endswith("/"):
        swagger_path = swagger_path + "/"
    custom_root_path_swagger_path: Final = swagger_path + "swagger"

    app.mount("/swagger", StaticFiles(directory=swagger_directory), name="swagger")

    # On dropdown expand: one-time fetch to the prefix (triggers lazy load),
    # then spec re-download so real routes replace the stub. Raw JS (no
    # <script> tag) since it's injected inside the existing inline script.
    from fastapi.responses import HTMLResponse

    from litellm.proxy._lazy_features import lazy_tag_to_prefix

    _lazy_plugin_js: Final = (
        "const TAG_TO_PREFIX = " + json.dumps(lazy_tag_to_prefix()) + ";"
        "const warmedTags = new Set();"
        "const LAZY_TAGS = new Set(Object.keys(TAG_TO_PREFIX));"
        "const hideStubRows = () => {"
        "document.querySelectorAll('.opblock').forEach(op => {"
        "const d = op.querySelector('.opblock-summary-description');"
        "if (d && LAZY_TAGS.has(d.textContent.trim())) op.style.display = 'none';"
        "});};"
        "const annotateLazyHeaders = () => {"
        "document.querySelectorAll('.opblock-tag').forEach(tagEl => {"
        "const m = (tagEl.id || '').match(/^operations-tag-(.+)$/);"
        "if (!m || !LAZY_TAGS.has(m[1])) return;"
        "const existing = tagEl.querySelector('.lazy-load-hint');"
        "if (warmedTags.has(m[1])) { if (existing) existing.remove(); return; }"
        "if (existing) return;"
        "const hint = document.createElement('small');"
        "hint.className = 'lazy-load-hint';"
        "hint.textContent = ' (expand to load routes)';"
        "hint.style.opacity = '0.6';"
        "hint.style.marginLeft = '6px';"
        "const target = tagEl.querySelector('a span') || tagEl.querySelector('span') || tagEl;"
        "target.appendChild(hint);"
        "});};"
        "setInterval(() => { hideStubRows(); annotateLazyHeaders(); }, 200);"
        "const LazyLoadPlugin = () => ({"
        "afterLoad:function(system){setTimeout(()=>{"
        "for(const tag of LAZY_TAGS)system.layoutActions.show(['operations-tag',tag],false);"
        "},200);},"
        "statePlugins:{layout:{wrapActions:{show:(ori,sys)=>(...args)=>{"
        "const thing=args[0];const shown=args[1];let tag=null;"
        "if(Array.isArray(thing)){for(const t of thing)if(TAG_TO_PREFIX[t])tag=t;}"
        "if(shown!==false&&tag&&!warmedTags.has(tag)){warmedTags.add(tag);"
        "fetch('/lazy/warm/'+tag,{method:'POST',credentials:'include'}).then(r=>r.json()).then(d=>{"
        "if(!d.paths||Object.keys(d.paths).length===0)return;"
        "const cur=sys.specSelectors.specJson().toJS();"
        "const merged={};let inserted=false;"
        "for(const k in (cur.paths||{})){"
        "if(k===d.stub_path){for(const nk in d.paths)merged[nk]=d.paths[nk];inserted=true;}"
        "else{merged[k]=cur.paths[k];}}"
        "if(!inserted)Object.assign(merged,d.paths);"
        "cur.paths=merged;"
        "cur.components=cur.components||{};"
        "cur.components.schemas=Object.assign(cur.components.schemas||{},(d.components||{}).schemas||{});"
        "sys.specActions.updateSpec(JSON.stringify(cur));"
        "}).catch(()=>{});}"
        "return ori(...args);}}}}});"
    )

    def swagger_monkey_patch(*args, **kwargs):
        response: Final = get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_js_url=f"{custom_root_path_swagger_path}/swagger-ui-bundle.js",
            swagger_css_url=f"{custom_root_path_swagger_path}/swagger-ui.css",
            swagger_favicon_url=f"{custom_root_path_swagger_path}/favicon.png",
        )
        body = response.body.decode("utf-8")
        body = body.replace(
            "const ui = SwaggerUIBundle({",
            _lazy_plugin_js + 'const ui = SwaggerUIBundle({plugins:[LazyLoadPlugin],tagsSorter:"alpha",',
            1,
        )
        return HTMLResponse(content=body)

    applications.get_swagger_ui_html = swagger_monkey_patch


mount_swagger_ui()

docs_url: Final = _get_docs_url()
root_redirect_url: Final[str | None] = os.getenv("ROOT_REDIRECT_URL")
if docs_url != "/" and root_redirect_url is not None:

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url=root_redirect_url)


user_api_base = None
user_model = None
user_debug = False
user_max_tokens = None
user_request_timeout = None
user_temperature = None
user_telemetry = True
user_config: Final = None
user_headers = None
user_config_file_path: str | None = None
local_logging: Final = True  # writes logs to a local api_log.json file for debugging
experimental = False
#### GLOBAL VARIABLES ####
llm_router: Router | None = None
llm_model_list: list | None = None
# Serializes every model reconcile (ProxyConfig.add_deployment and clear_cache) so the
# read-modify-write of llm_router above is atomic. Without it, two concurrent model
# writes each reconcile the router against their OWN db snapshot, and the one holding
# the older snapshot evicts the deployment the newer one just added -- the db keeps the
# row, this pod stops serving it. Control-plane only (model create/update/delete and
# the config-sync tick), never on a completion path, so the serialization is free.
# Module-level rather than per-ProxyConfig because llm_router is a module global and a
# second ProxyConfig instance must not get its own independent lock over it.
MODEL_RECONCILE_LOCK: Final = asyncio.Lock()
general_settings: dict = {}
config_passthrough_endpoints: list[dict[str, Any]] | None = None
log_file: Final = "api_log.json"
worker_config: Final = None
master_key: str | None = None
otel_logging = False
prisma_client: PrismaClient | None = None
shared_aiohttp_session: Optional["ClientSession"] = None  # Global shared session for connection reuse
user_api_key_cache: UserApiKeyCache = UserApiKeyCache(
    default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value
)
spend_counter_cache: Final = DualCache(default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value)
cli_sso_session_cache: Final = DualCache(default_in_memory_ttl=CLI_SSO_SESSION_TTL_SECONDS)
model_max_budget_limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=user_api_key_cache)
litellm.logging_callback_manager.add_litellm_callback(model_max_budget_limiter)
redis_usage_cache: RedisCache | None = None  # redis cache used for tracking spend, tpm/rpm limits
polling_via_cache_enabled: Literal["all"] | list[str] | bool = False
native_background_mode: list[str] = []  # Models that should use native provider background mode instead of polling
polling_cache_ttl: int = 3600  # Default 1 hour TTL for polling cache
user_custom_auth = None
user_custom_key_generate = None
# Sentinel: prevents PKCE-no-Redis advisory from re-logging on config hot-reload.
# Tests that need to reset it can patch 'litellm.proxy.proxy_server._pkce_no_redis_warning_emitted'.
_pkce_no_redis_warning_emitted: bool = False
_cp_no_redis_warning_emitted: bool = False
user_custom_key_update = None
user_custom_sso = None
user_custom_ui_sso_sign_in_handler = None
use_background_health_checks = None
use_shared_health_check = None
use_queue = False
health_check_interval = None
health_check_concurrency = None
health_check_details = None
health_check_results: dict[str, int | list[dict[str, Any]]] = {}
background_health_check_loop_active = False
background_health_check_cycle_seq = 0
queue: Final[list] = []
litellm_proxy_budget_name: Final = LITELLM_PROXY_BUDGET_NAME
litellm_proxy_admin_name = LITELLM_PROXY_ADMIN_NAME
ui_access_mode: Literal["admin", "all"] | dict = "all"
proxy_budget_rescheduler_min_time = PROXY_BUDGET_RESCHEDULER_MIN_TIME
proxy_budget_rescheduler_max_time = PROXY_BUDGET_RESCHEDULER_MAX_TIME
proxy_batch_polling_interval = PROXY_BATCH_POLLING_INTERVAL
proxy_batch_write_at = PROXY_BATCH_WRITE_AT
proxy_config_reload_interval_seconds = PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
litellm_master_key_hash = None
disable_spend_logs = False
jwt_handler: Final = JWTHandler()
prompt_injection_detection_obj: _OPTIONAL_PromptInjectionDetection | None = None
store_model_in_db: bool = False
open_telemetry_logger: OpenTelemetry | None = None
### GATEWAY REQUEST COUNTS (SGR) ###
# Folded in memory by BillableRequestMetricsMiddleware, drained to
# LiteLLM_DailyGatewayRequests by the update_gateway_requests scheduler job.
gateway_request_accumulator: Final = GatewayRequestAccumulator()
### INITIALIZE GLOBAL LOGGING OBJECT ###
proxy_logging_obj: ProxyLogging = ProxyLogging(user_api_key_cache=user_api_key_cache, premium_user=premium_user)
### REDIS QUEUE ###
async_result: Final = None
celery_app_conn: Final = None
celery_fn: Final = None  # Redis Queue for handling requests

scheduler = None

# Global variable for anthropic beta headers reload scheduling
last_anthropic_beta_headers_reload = None


### DB WRITER ###
db_writer_client: AsyncHTTPHandler | None = None
### logger ###


def _resolve_typed_dict_type(typ: object):
    """Resolve the actual TypedDict class from a potentially wrapped type."""
    from typing_extensions import _TypedDictMeta

    origin: Final[object] = get_origin(typ)
    if origin is Union or origin is UnionType:  # Check if it's a Union (like Optional)
        union_args: Final[tuple[object, ...]] = get_args(typ)
        for arg in union_args:
            if isinstance(arg, _TypedDictMeta):
                return arg
    elif isinstance(typ, type) and isinstance(typ, dict):
        return typ
    return None


def _resolve_pydantic_type(typ: object) -> list:
    """Resolve the actual TypedDict class from a potentially wrapped type."""
    origin: Final[object] = get_origin(typ)
    typs: Final = []
    if origin is Union or origin is UnionType:  # Check if it's a Union (like Optional)
        union_args: Final[tuple[object, ...]] = get_args(typ)
        for arg in union_args:
            if arg is not None and "NoneType" not in str(arg):
                typs.append(arg)
    elif isinstance(typ, type) and isinstance(typ, BaseModel):
        return [typ]
    return typs


def load_from_azure_key_vault(use_azure_key_vault: bool = False):
    if use_azure_key_vault is False:
        return

    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        # Set your Azure Key Vault URI
        KVUri: Final = os.getenv("AZURE_KEY_VAULT_URI", None)

        if KVUri is None:
            raise Exception("Error when loading keys from Azure Key Vault: AZURE_KEY_VAULT_URI is not set.")

        credential: Final = DefaultAzureCredential()

        # Create the SecretClient using the credential
        client: Final = SecretClient(vault_url=KVUri, credential=credential)

        litellm.secret_manager_client = client
        litellm._key_management_system = KeyManagementSystem.AZURE_KEY_VAULT
    except Exception as e:
        _error_str: Final = str(e)
        verbose_proxy_logger.exception(
            "Error when loading keys from Azure Key Vault: %s .Ensure you run `pip install azure-identity azure-keyvault-secrets`",
            _error_str,
        )


def cost_tracking():
    global prisma_client
    if prisma_client is not None:
        from litellm.integrations.shadow_eval_logger import ShadowEvalLogger

        litellm.logging_callback_manager.add_litellm_callback(_ProxyDBLogger())
        litellm.logging_callback_manager.add_litellm_async_success_callback(_ProxyDBLogger())
        litellm.logging_callback_manager.add_litellm_callback(ShadowEvalLogger())


# Bounds authoritative DB re-reads when enforcing a budget against a
# stale-low spend counter: at most one DB read per counter per window.
SPEND_DB_FLOOR_CACHE_TTL_SECONDS: Final = 5


def _fail_closed_budget_enforcement() -> bool:
    return general_settings.get("fail_closed_budget_enforcement") is True


def _raise_budget_unverifiable(counter_key: str) -> None:
    verbose_proxy_logger.warning(
        "fail_closed_budget_enforcement: rejecting request — spend for %s could "
        "not be verified against Redis or the database",
        counter_key,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": (
                "Budget enforcement unavailable: current spend could not be "
                "verified against Redis or the database, and "
                "fail_closed_budget_enforcement is enabled, so the request was "
                "rejected to avoid exceeding the configured budget. Retry shortly."
            )
        },
    )


async def get_current_spend(
    counter_key: str,
    fallback_spend: float,
    max_budget: float | None = None,
    window_entity_type: str | None = None,
    window_entity_id: str | None = None,
    window_start: datetime | None = None,
    fallback_authoritative: bool = False,
) -> float:
    """
    Read current spend from the cross-pod spend counter.

    Reads Redis FIRST (authoritative cross-pod value), not DualCache's
    async_get_cache which returns in-memory first. This is critical:
    DualCache.async_get_cache returns stale per-pod values because each
    pod's in-memory cache is only updated by that pod's own increments.

    Fallback chain:
    1. Redis counter (cross-pod, authoritative)
    2. In-memory counter (single-instance or Redis failure)
    3. Reseed from authoritative DB spend (counter expired, cross-pod stale)
    4. Caller-supplied fallback (DB unavailable, cold start)

    When ``max_budget`` is supplied, the counter is re-checked against the
    authoritative recorded spend before a request is admitted. A Redis counter
    that survived a Redis restart can return a stale-low value loaded from an
    older RDB snapshot; that read is a hit (not a clean miss), so step 3 never
    runs and a key can leak spend past ``max_budget`` indefinitely. The
    authoritative source depends on the counter: primary key/team/user/org
    counters read the DB row; per-window counters (``window_start`` supplied)
    aggregate spend logs; end-user/tag counters have no DB row, so the caller's
    ``fallback_spend`` (loaded fresh in auth) is authoritative. The DB read is
    skipped for healthy primary counters (counter at or above recorded spend)
    and cached in-process for a few seconds, so a persistently stale counter
    drives at most one read per counter per window rather than one per request.
    """
    current, verified = await _read_spend_counter_estimate(counter_key=counter_key, fallback_spend=fallback_spend)
    if fallback_authoritative:
        verified = True

    if max_budget is None or current >= max_budget:
        return current

    # Cheap staleness signal for primary counters: the counter reads below the
    # spend this caller already knows about. Window counters have no such signal
    # (fallback is 0), so they always re-check, bounded by the cache. Strict mode
    # (fail_closed_budget_enforcement) always re-checks against the authoritative
    # source too, so a counter that is stale-low at the same time as the caller's
    # cached spend cannot slip through; the 5s cache keeps that bounded.
    is_window: Final = window_start is not None
    if fallback_spend > current or is_window or _fail_closed_budget_enforcement():
        authoritative: Final = await _authoritative_floor_spend(
            counter_key=counter_key,
            window_entity_type=window_entity_type,
            window_entity_id=window_entity_id,
            window_start=window_start,
        )
        if authoritative is not None:
            verified = True
            if authoritative > current:
                await _repair_stale_spend_counter(counter_key=counter_key, db_spend=authoritative)
                return authoritative
        elif fallback_spend > current:
            # end-user / tag counters have no DB row; fallback_spend is the
            # authoritative recorded value loaded in auth.
            return fallback_spend

    # Opt-in hard guarantee: when the spend backing this admit decision came
    # only from a per-pod cache (Redis and DB both unreadable), reject rather
    # than admit on an unverifiable budget. No-op unless the flag is set, so
    # default behavior is unchanged.
    if not verified and _fail_closed_budget_enforcement():
        _raise_budget_unverifiable(counter_key)

    return current


async def _repair_stale_spend_counter(counter_key: str, db_spend: float) -> None:
    """Raise a counter that has fallen below the authoritative DB spend (e.g.
    Redis restarted and reloaded an older snapshot) so every worker reads the
    corrected value directly instead of re-deriving it per request, and so a
    worker whose own cached spend is also stale still sees the true total.

    The write is monotonic: it only ever raises the counter, so a repair that
    carries a slightly-stale DB total cannot clobber a concurrent increment that
    already pushed the counter higher (which would let racing requests
    under-count). Redis enforces this atomically via async_set_max; the
    in-memory copy is guarded by a read-compare-write with no await in between,
    so it is atomic within the worker.
    """
    cached: Final = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
    needs_update = True
    if cached is not None:
        try:
            needs_update = float(cached) < db_spend
        except (TypeError, ValueError):
            needs_update = True
    if needs_update:
        spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=db_spend)
    if spend_counter_cache.redis_cache is not None:
        try:
            await spend_counter_cache.redis_cache.async_set_max(key=counter_key, value=db_spend)
        except Exception:
            verbose_proxy_logger.debug(
                "Unable to repair stale spend counter %s in Redis",
                counter_key,
                exc_info=True,
            )


async def reseed_spend_counter_from_db(counter_key: str) -> None:
    """Recover a counter that the reservation reconcile found in an inconsistent
    state (missing, or where applying the reconcile delta would drive it
    negative) by reseeding it from the DB instead of deleting it.

    The DB row is a LAGGING authoritative floor, not post-request truth: the
    entity .spend column is flushed in batches (every PROXY_BATCH_WRITE_AT), so
    it can exclude this request's just-recorded cost and other buffered spend.
    That is fine here: the monotonic set-max can only RAISE a stale-low counter
    toward that floor (never lowers it or clobbers a concurrent increment), and
    the read-time floor (_authoritative_floor_spend) converges to the true total
    as the buffer flushes. The point is to restore enforcement to a real floor
    rather than leave the counter deleted and unenforced (the prior fail-open).
    Counters with no DB row (window/end-user/tag) are left untouched rather than
    deleted, so enforcement keeps reading whatever value they hold.
    """
    db_spend: Final = await SpendCounterReseed.from_db(prisma_client=prisma_client, counter_key=counter_key)
    if db_spend is None:
        return
    await _repair_stale_spend_counter(counter_key=counter_key, db_spend=db_spend)


async def _authoritative_floor_spend(
    counter_key: str,
    window_entity_type: str | None = None,
    window_entity_id: str | None = None,
    window_start: datetime | None = None,
) -> float | None:
    marker_key: Final = f"spend_db_floor:{counter_key}"
    cached: Final = spend_counter_cache.in_memory_cache.get_cache(key=marker_key)
    if cached is not None:
        return float(cached)

    db_spend = await SpendCounterReseed.from_db(prisma_client=prisma_client, counter_key=counter_key)
    if (
        db_spend is None
        and window_entity_type is not None
        and window_entity_id is not None
        and window_start is not None
    ):
        db_spend = await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma_client,
            entity_type=window_entity_type,
            entity_id=window_entity_id,
            window_start=window_start,
        )
    if db_spend is None:
        return None

    # a spend reset that committed during the DB read above wrote the post-reset
    # floor to the marker; keep it over this read's now-stale pre-commit value
    rechecked: Final = spend_counter_cache.in_memory_cache.get_cache(key=marker_key)
    if rechecked is not None:
        return float(rechecked)

    spend_counter_cache.in_memory_cache.set_cache(
        key=marker_key,
        value=db_spend,
        ttl=SPEND_DB_FLOOR_CACHE_TTL_SECONDS,
    )
    return db_spend


async def _read_spend_counter_estimate(counter_key: str, fallback_spend: float) -> tuple[float, bool]:
    """Return (spend, authoritative). ``authoritative`` is True when the value
    came from Redis or a fresh DB read (cross-pod truth), False when it came
    from the per-pod in-memory copy or the caller's fallback. Only the
    fail-closed path reads the flag; normal callers ignore it."""
    # 1. Redis first (cross-pod authoritative). On clean miss, skip
    # in-memory: per-pod in-memory only has this pod's writes, so it
    # would mask cross-pod increments.
    redis_clean_miss = False
    if spend_counter_cache.redis_cache is not None:
        try:
            val = await spend_counter_cache.redis_cache.async_get_cache(key=counter_key)
            if val is not None:
                return float(val), True
            redis_clean_miss = True
        except Exception as e:
            verbose_proxy_logger.debug(
                "get_current_spend: Redis read failed for %s, falling back to in-memory: %s",
                counter_key,
                e,
            )

    # 2. In-memory only when Redis is unreachable.
    if not redis_clean_miss:
        val = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
        if val is not None:
            return float(val), False

    # 3. Reseed from DB - fallback_spend lags cross-pod, would allow bypass.
    db_spend: Final = await SpendCounterReseed.coalesced(
        prisma_client=prisma_client,
        spend_counter_cache=spend_counter_cache,
        counter_key=counter_key,
    )
    if db_spend is not None:
        return db_spend, True

    # 4. Caller-supplied fallback (DB unavailable).
    return fallback_spend, False


async def increment_spend_counters(
    token: str | None,
    team_id: str | None,
    user_id: str | None,
    response_cost: float | None,
    org_id: str | None = None,
    budget_reservation: dict | None = None,
    end_user_id: str | None = None,
    tags: list[str] | None = None,
):
    """
    Atomically increment spend counters for budget enforcement.

    Uses spend_counter_cache (DualCache with Redis backend when available)
    so counters are shared across all pods. Budget check functions read
    from these counters via get_current_spend() (Redis-first).

    Awaited (not create_task) in the cost callback, so the counter is
    updated before the next request's auth check runs.
    """
    reserved_counter_keys: Final = await _reconcile_budget_reservation_for_counter_update(
        budget_reservation=budget_reservation,
        response_cost=response_cost,
    )

    if response_cost is None or response_cost == 0:
        if budget_reservation is not None:
            budget_reservation["finalized"] = True
        return

    cost: Final[float] = response_cost

    async def _key_scope(key_token: str) -> None:
        # key_token arrives pre-hashed from metadata["user_api_key"] (auth flow
        # hashes raw "sk-..." keys before they reach the callback). The
        # startswith("sk-") check is a safety net matching update_cache —
        # if a raw key somehow arrives, hash it; otherwise use as-is to
        # avoid double-hashing (budget checks read valid_token.token which
        # is single-hashed).
        hashed_token: Final = (
            hash_token(token=key_token) if isinstance(key_token, str) and key_token.startswith("sk-") else key_token
        )
        key_counter_key: Final = f"spend:key:{hashed_token}"
        if key_counter_key not in reserved_counter_keys:
            await _init_and_increment_spend_counter(
                counter_key=key_counter_key,
                source_cache_key=hashed_token,
                increment=cost,
            )

        key_obj: Final[object] = await user_api_key_cache.async_get_cache(key=hashed_token)
        if key_obj is None:
            return
        key_budget_limits = getattr(key_obj, "budget_limits", None) or (
            key_obj.get("budget_limits") if isinstance(key_obj, dict) else None
        )
        if isinstance(key_budget_limits, str):
            key_budget_limits = json.loads(key_budget_limits)
        if not isinstance(key_budget_limits, list):
            return
        for window in key_budget_limits:
            duration = window["budget_duration"] if isinstance(window, dict) else window.budget_duration
            key_window_counter = f"spend:key:{hashed_token}:window:{duration}"
            if key_window_counter not in reserved_counter_keys:
                await _init_and_increment_window_spend_counter(
                    counter_key=key_window_counter,
                    entity_type="Key",
                    entity_id=hashed_token,
                    window_start=get_budget_window_start(window),
                    increment=cost,
                )

    async def _team_scope(scope_team_id: str) -> None:
        team_counter_key: Final = f"spend:team:{scope_team_id}"
        if team_counter_key not in reserved_counter_keys:
            await _init_and_increment_spend_counter(
                counter_key=team_counter_key,
                source_cache_key=f"team_id:{scope_team_id}",
                increment=cost,
            )

        team_obj: Final[object] = await user_api_key_cache.async_get_cache(key=f"team_id:{scope_team_id}")
        if team_obj is None:
            return
        team_budget_limits = getattr(team_obj, "budget_limits", None) or (
            team_obj.get("budget_limits") if isinstance(team_obj, dict) else None
        )
        if isinstance(team_budget_limits, str):
            team_budget_limits = json.loads(team_budget_limits)
        if not isinstance(team_budget_limits, list):
            return
        for window in team_budget_limits:
            duration = window["budget_duration"] if isinstance(window, dict) else window.budget_duration
            team_window_counter = f"spend:team:{scope_team_id}:window:{duration}"
            if team_window_counter not in reserved_counter_keys:
                await _init_and_increment_window_spend_counter(
                    counter_key=team_window_counter,
                    entity_type="Team",
                    entity_id=scope_team_id,
                    window_start=get_budget_window_start(window),
                    increment=cost,
                )

    async def _team_member_scope(scope_user_id: str, scope_team_id: str) -> None:
        team_member_counter_key: Final = f"spend:team_member:{scope_user_id}:{scope_team_id}"
        if team_member_counter_key in reserved_counter_keys:
            return
        await _init_and_increment_spend_counter(
            counter_key=team_member_counter_key,
            source_cache_key=f"team_membership:{scope_user_id}:{scope_team_id}",
            increment=cost,
        )

    async def _user_scope(scope_user_id: str) -> None:
        user_counter_key: Final = f"spend:user:{scope_user_id}"
        if user_counter_key in reserved_counter_keys:
            return
        await _init_and_increment_spend_counter(
            counter_key=user_counter_key,
            source_cache_key=scope_user_id,
            increment=cost,
        )

    scope_coros: Final = tuple(
        coro
        for coro in (
            _key_scope(token) if token is not None else None,
            _team_scope(team_id) if team_id is not None else None,
            _team_member_scope(user_id, team_id) if user_id is not None and team_id is not None else None,
            _user_scope(user_id) if user_id is not None else None,
            _increment_end_user_and_tag_spend_counters(
                end_user_id=end_user_id,
                tags=tags,
                response_cost=cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            if end_user_id is not None or tags is not None
            else None,
            _increment_org_spend_counter(
                org_id=org_id,
                response_cost=cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            if org_id is not None
            else None,
        )
        if coro is not None
    )

    # return_exceptions so a failing scope does not leave its siblings running
    # as orphaned tasks that race the caller's reservation-counter invalidation;
    # all scopes settle, then the first error propagates as before.
    scope_results: Final = await asyncio.gather(*scope_coros, return_exceptions=True)
    scope_errors: Final = [r for r in scope_results if isinstance(r, BaseException)]
    if scope_errors:
        raise scope_errors[0]

    if budget_reservation is not None:
        budget_reservation["finalized"] = True


async def _reconcile_budget_reservation_for_counter_update(
    budget_reservation: dict | None,
    response_cost: float | None,
) -> set[str]:
    if budget_reservation is None:
        return set()

    from litellm.proxy.spend_tracking.budget_reservation import (
        get_reserved_counter_keys,
        invalidate_budget_reservation_counters,
        reconcile_budget_reservation,
    )

    reserved_counter_keys: Final = get_reserved_counter_keys(budget_reservation=budget_reservation)
    try:
        await reconcile_budget_reservation(
            budget_reservation=budget_reservation,
            actual_cost=response_cost or 0.0,
            finalize=False,
        )
    except Exception:
        verbose_proxy_logger.warning(
            "Failed to reconcile budget reservation after persisted spend; invalidating reserved counters and falling back to direct increment",
            exc_info=True,
        )
        try:
            await invalidate_budget_reservation_counters(budget_reservation=budget_reservation)
        except Exception:
            verbose_proxy_logger.exception(
                "Failed to invalidate reserved counters after reservation reconciliation failed"
            )
        return set()
    return reserved_counter_keys


async def _increment_end_user_and_tag_spend_counters(
    end_user_id: str | None,
    tags: list[str] | None,
    response_cost: float,
    reserved_counter_keys: set[str],
) -> None:
    if end_user_id is not None:
        await _init_and_increment_unreserved_spend_counter(
            counter_key=f"spend:end_user:{end_user_id}",
            source_cache_key=end_user_cache_key(end_user_id),
            increment=response_cost,
            reserved_counter_keys=reserved_counter_keys,
        )

    if tags is None:
        return

    seen_tags: Final[set[str]] = set()
    for tag_name in tags:
        if not tag_name or not isinstance(tag_name, str) or tag_name in seen_tags:
            continue
        seen_tags.add(tag_name)
        await _init_and_increment_unreserved_spend_counter(
            counter_key=f"spend:tag:{tag_name}",
            source_cache_key=tag_cache_key(tag_name),
            increment=response_cost,
            reserved_counter_keys=reserved_counter_keys,
        )


async def _increment_org_spend_counter(
    org_id: str | None,
    response_cost: float,
    reserved_counter_keys: set[str],
) -> None:
    if org_id is None:
        return

    await _init_and_increment_unreserved_spend_counter(
        counter_key=f"spend:org:{org_id}",
        source_cache_key=[f"org_id:{org_id}:with_budget", f"org_id:{org_id}"],
        increment=response_cost,
        reserved_counter_keys=reserved_counter_keys,
    )


async def _init_and_increment_unreserved_spend_counter(
    counter_key: str,
    source_cache_key: str | list[str],
    increment: float,
    reserved_counter_keys: set[str],
) -> None:
    if counter_key in reserved_counter_keys:
        return

    await _init_and_increment_spend_counter(
        counter_key=counter_key,
        source_cache_key=source_cache_key,
        increment=increment,
    )


async def _init_and_increment_spend_counter(
    counter_key: str,
    source_cache_key: str | list[str],
    increment: float,
):
    """
    Initialize counter from the authoritative DB spend value if not yet
    set, then atomically increment in both in-memory and Redis.

    On first access per pod:
    1. Check spend_counter_cache (in-memory -> Redis via DualCache)
    2. If not found, reseed from the DB via `SpendCounterReseed.coalesced`.
       Falls back to the cached object's `.spend` via user_api_key_cache
       only if prisma is unavailable, since that value can lag the flusher.
    3. Seed counter via async_increment_cache (not async_set_cache) to avoid a
       check-then-set race: if two pods cold-start simultaneously, both may see
       the counter as absent and seed it. Using increment means the worst case
       is over-counting (conservative, blocks slightly early) rather than
       under-counting (would allow overspend).
    4. Increment atomically (both in-memory + Redis)
    """
    await _ensure_spend_counter_initialized(
        counter_key=counter_key,
        source_cache_key=source_cache_key,
    )
    await _increment_spend_counter_cache(counter_key=counter_key, increment=increment)


async def _init_and_increment_window_spend_counter(
    counter_key: str,
    entity_type: str,
    entity_id: str,
    window_start: datetime | None,
    increment: float,
):
    if window_start is None:
        verbose_proxy_logger.warning(
            "Skipping spend counter increment for invalid budget window %s",
            counter_key,
        )
        return

    initialized: Final = await _ensure_window_spend_counter_initialized(
        counter_key=counter_key,
        entity_type=entity_type,
        entity_id=entity_id,
        window_start=window_start,
    )
    if initialized is False:
        return
    await _increment_spend_counter_cache(counter_key=counter_key, increment=increment)


async def _ensure_spend_counter_initialized(
    counter_key: str,
    source_cache_key: str | list[str],
):
    is_warm: Final = await _is_spend_counter_cache_warm(counter_key=counter_key)
    if is_warm is False:
        # Shares the per-counter lock with get_current_spend.
        db_spend: Final = await SpendCounterReseed.coalesced(
            prisma_client=prisma_client,
            spend_counter_cache=spend_counter_cache,
            counter_key=counter_key,
            require_cache_warm=True,
        )
        if db_spend is None:
            # DB unavailable - fall back to in-process cache (may be stale).
            base_spend: Final = await _get_source_cache_base_spend(source_cache_key=source_cache_key)
            if base_spend > 0:
                await _increment_spend_counter_cache(counter_key=counter_key, increment=base_spend)


async def _get_source_cache_base_spend(
    source_cache_key: str | list[str],
) -> float:
    source_cache_keys: Final = [source_cache_key] if isinstance(source_cache_key, str) else source_cache_key
    for cache_key in source_cache_keys:
        source = await user_api_key_cache.async_get_cache(key=cache_key)
        if source is None:
            continue
        if isinstance(source, dict):
            return float(source.get("spend", 0.0) or 0.0)
        return float(getattr(source, "spend", 0.0) or 0.0)
    return 0.0


async def _ensure_window_spend_counter_initialized(
    counter_key: str,
    entity_type: str,
    entity_id: str,
    window_start: datetime,
) -> bool:
    is_warm: Final = await _is_spend_counter_cache_warm(counter_key=counter_key)
    if is_warm is True:
        return True

    window_spend: Final = await SpendCounterReseed.coalesced_window(
        prisma_client=prisma_client,
        spend_counter_cache=spend_counter_cache,
        counter_key=counter_key,
        entity_type=entity_type,
        entity_id=entity_id,
        window_start=window_start,
    )
    if window_spend is None:
        verbose_proxy_logger.warning(
            "Skipping cold spend counter seed for %s because window spend could not be loaded",
            counter_key,
        )
        return False
    return True


async def _is_spend_counter_cache_warm(counter_key: str) -> bool:
    if spend_counter_cache.redis_cache is not None:
        try:
            current_value: Final[object] = await spend_counter_cache.redis_cache.async_get_cache(
                key=counter_key,
            )
            if current_value is None:
                return False
            spend_counter_cache.in_memory_cache.set_cache(
                key=counter_key,
                value=current_value,
            )
            return True
        except Exception as e:
            verbose_proxy_logger.debug(
                "Unable to read Redis spend counter %s before initialization, falling back to in-memory: %s",
                counter_key,
                e,
            )

    return spend_counter_cache.in_memory_cache.get_cache(key=counter_key) is not None


async def increment_spend_counter(counter_key: str, increment: float):
    """Public raw-counter increment for budget domains outside the entity scopes (e.g.
    shadow eval's per-leg spend), sharing the primitive the entity counters use so
    invalidation and read semantics can never drift."""
    return await _increment_spend_counter_cache(counter_key=counter_key, increment=increment)


async def _increment_spend_counter_cache(counter_key: str, increment: float):
    if spend_counter_cache.redis_cache is not None:
        try:
            current_value: Final = await spend_counter_cache.redis_cache.async_increment(
                key=counter_key,
                value=increment,
                refresh_ttl=True,
            )
        except Exception:
            await _invalidate_spend_counter(counter_key=counter_key)
            raise
        spend_counter_cache.in_memory_cache.set_cache(
            key=counter_key,
            value=current_value,
        )
        return current_value

    return await spend_counter_cache.async_increment_cache(
        key=counter_key,
        value=increment,
        refresh_ttl=True,
    )


async def _invalidate_spend_counter(counter_key: str):
    spend_counter_cache.in_memory_cache.delete_cache(key=counter_key)
    if spend_counter_cache.redis_cache is not None:
        try:
            await spend_counter_cache.redis_cache.async_delete_cache(key=counter_key)
        except Exception:
            verbose_proxy_logger.debug(
                "Unable to delete stale spend counter %s after increment failure",
                counter_key,
                exc_info=True,
            )


async def update_cache(
    token: str | None,
    user_id: str | None,
    end_user_id: str | None,
    team_id: str | None,
    response_cost: float | None,
    parent_otel_span: Span | None,
    tags: list[str] | None = None,
):
    """
    Use this to update the cache with new user spend.

    Put any alerting logic in here.
    """

    values_to_update_in_cache: Final[list[tuple[str, object]]] = []

    ### UPDATE KEY SPEND ###
    async def _update_key_cache(token: str, response_cost: float):
        # Fetch the existing cost for the given token
        if isinstance(token, str) and token.startswith("sk-"):
            hashed_token = hash_token(token=token)
        else:
            hashed_token = token
        verbose_proxy_logger.debug("_update_key_cache: hashed_token=%s", hashed_token)
        existing_spend_obj = await user_api_key_cache.async_get_cache(key=hashed_token, model_type=UserAPIKeyAuth)
        verbose_proxy_logger.debug("_update_key_cache: existing_spend_obj=%s", existing_spend_obj)
        if existing_spend_obj is None:
            return

        existing_spend: Final = existing_spend_obj.spend or 0.0
        # Calculate the new cost by adding the existing cost and response_cost
        new_spend: Final = existing_spend + response_cost

        ## CHECK IF USER PROJECTED SPEND > SOFT LIMIT
        if (
            existing_spend_obj.soft_budget_cooldown is False
            and existing_spend_obj.soft_budget is not None
            and (
                _is_projected_spend_over_limit(
                    current_spend=new_spend,
                    soft_budget_limit=existing_spend_obj.soft_budget,
                )
                is True
            )
        ):
            projected_spend, projected_exceeded_date = _get_projected_spend_over_limit(
                current_spend=new_spend,
                soft_budget_limit=existing_spend_obj.soft_budget,
            )
            soft_limit: Final = existing_spend_obj.soft_budget
            call_info: Final = CallInfo(
                token=existing_spend_obj.token or "",
                spend=new_spend,
                key_alias=existing_spend_obj.key_alias,
                max_budget=soft_limit,
                user_id=existing_spend_obj.user_id,
                projected_spend=projected_spend,
                projected_exceeded_date=str(projected_exceeded_date),
                event_group=Litellm_EntityType.KEY,
            )
            # alert user
            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="projected_limit_exceeded",
                    user_info=call_info,
                )
            )
            # set cooldown on alert

    ### UPDATE USER SPEND ###
    async def _update_user_cache():
        ## UPDATE CACHE FOR USER ID + GLOBAL PROXY
        if response_cost is None:
            return
        user_ids: Final = [user_id]
        try:
            for _id in user_ids:
                # Fetch the existing cost for the given user
                if _id is None:
                    continue
                cached_user = await user_api_key_cache.async_get_cache(key=_id)
                if cached_user is None:
                    # do nothing if there is no cache value
                    return
                existing_spend_obj = CacheCodec.deserialize(cached_user, LiteLLM_UserTable)
                if existing_spend_obj is None:
                    return
                verbose_proxy_logger.debug(
                    "_update_user_db: existing spend: %s; response_cost: %s", existing_spend_obj, response_cost
                )

                existing_spend = existing_spend_obj.spend or 0.0
                # Calculate the new cost by adding the existing cost and response_cost
                new_spend = existing_spend + response_cost

                existing_spend_obj.spend = new_spend
                values_to_update_in_cache.append(
                    (
                        _id,
                        CacheCodec.serialize(existing_spend_obj, model_type=LiteLLM_UserTable),
                    )
                )
            ## UPDATE GLOBAL PROXY ##
            global_proxy_spend: Final = await user_api_key_cache.async_get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)
            if global_proxy_spend is None:
                # do nothing if not in cache
                return
            elif response_cost is not None and global_proxy_spend is not None:
                increment: Final = global_proxy_spend + response_cost
                values_to_update_in_cache.append((GLOBAL_PROXY_SPEND_CACHE_KEY, increment))
        except Exception as e:
            verbose_proxy_logger.warning(
                "Spend tracking - failed to update user spend in cache. "
                "Budget enforcement may use stale spend values. "
                "user_id=%s, response_cost=%s - %s\n%s",
                user_id,
                response_cost,
                str(e),
                traceback.format_exc(),
            )

    ### UPDATE END-USER SPEND ###
    async def _update_end_user_cache():
        if end_user_id is None or response_cost is None:
            return

        _id: Final = end_user_cache_key(end_user_id)
        try:
            # Fetch the existing cost for the given user
            cached_end_user: Final = await user_api_key_cache.async_get_cache(key=_id)
            if cached_end_user is None:
                # if user does not exist in LiteLLM_UserTable, create a new user
                # do nothing if end-user not in api key cache
                return
            existing_spend_obj: Final = CacheCodec.deserialize(cached_end_user, LiteLLM_EndUserTable)
            if existing_spend_obj is None:
                return
            verbose_proxy_logger.debug(
                "_update_end_user_db: existing spend: %s; response_cost: %s", existing_spend_obj, response_cost
            )

            existing_spend: Final = existing_spend_obj.spend or 0.0
            # Calculate the new cost by adding the existing cost and response_cost
            new_spend: Final = existing_spend + response_cost

            existing_spend_obj.spend = new_spend
            values_to_update_in_cache.append(
                (
                    _id,
                    CacheCodec.serialize(existing_spend_obj, model_type=LiteLLM_EndUserTable),
                )
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Spend tracking - failed to update end user spend in cache. "
                "Budget enforcement may use stale spend values. "
                "end_user_id=%s, response_cost=%s - %s\n%s",
                end_user_id,
                response_cost,
                str(e),
                traceback.format_exc(),
            )

    ### UPDATE TEAM SPEND ###
    async def _update_team_cache():
        if team_id is None or response_cost is None:
            return

        _id: Final = f"team_id:{team_id}"
        try:
            cached_team: Final = await user_api_key_cache.async_get_cache(key=_id)
            if cached_team is None:
                # do nothing if team not in api key cache
                return
            existing_spend_obj: Final[LiteLLM_TeamTableCachedObj | None] = CacheCodec.deserialize(
                cached_team, LiteLLM_TeamTableCachedObj
            )
            if existing_spend_obj is None:
                return
            verbose_proxy_logger.debug(
                "_update_team_db: existing spend: %s; response_cost: %s", existing_spend_obj, response_cost
            )

            existing_spend: Final[float] = existing_spend_obj.spend or 0.0
            # Calculate the new cost by adding the existing cost and response_cost
            new_spend: Final = existing_spend + response_cost

            existing_spend_obj.spend = new_spend
            values_to_update_in_cache.append(
                (
                    _id,
                    CacheCodec.serialize(existing_spend_obj, model_type=LiteLLM_TeamTableCachedObj),
                )
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Spend tracking - failed to update team spend in cache. "
                "Budget enforcement may use stale spend values. "
                "team_id=%s, response_cost=%s - %s\n%s",
                team_id,
                response_cost,
                str(e),
                traceback.format_exc(),
            )

    ### UPDATE TAG SPEND ###
    async def _update_tag_cache():
        """
        Update the tag cache with the new spend.
        """
        if tags is None or response_cost is None:
            return

        try:
            for tag_name in tags:
                if not tag_name or not isinstance(tag_name, str):
                    continue

                cache_key = tag_cache_key(tag_name)
                # Fetch the existing tag object from cache
                cached_tag = await user_api_key_cache.async_get_cache(key=cache_key)
                if cached_tag is None:
                    # do nothing if tag not in api key cache
                    continue

                existing_tag_obj = CacheCodec.deserialize(cached_tag, LiteLLM_TagTable)
                if existing_tag_obj is None:
                    continue

                verbose_proxy_logger.debug(
                    "_update_tag_cache: existing spend for tag=%s: %s; response_cost: %s",
                    tag_name,
                    existing_tag_obj,
                    response_cost,
                )

                existing_spend = existing_tag_obj.spend or 0.0
                # Calculate the new cost by adding the existing cost and response_cost
                new_spend = existing_spend + response_cost

                existing_tag_obj.spend = new_spend
                values_to_update_in_cache.append(
                    (
                        cache_key,
                        CacheCodec.serialize(existing_tag_obj, model_type=LiteLLM_TagTable),
                    )
                )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Spend tracking - failed to update tag spend in cache. "
                "Budget enforcement may use stale spend values. "
                "tags=%s, response_cost=%s - %s\n%s",
                tags,
                response_cost,
                str(e),
                traceback.format_exc(),
            )

    if token is not None and response_cost is not None:
        await _update_key_cache(token=token, response_cost=response_cost)

    if user_id is not None:
        await _update_user_cache()

    if end_user_id is not None:
        await _update_end_user_cache()

    if team_id is not None:
        await _update_team_cache()

    if tags is not None:
        await _update_tag_cache()

    global_proxy_spend_key: Final = GLOBAL_PROXY_SPEND_CACHE_KEY
    local_object_updates: Final = tuple((k, v) for k, v in values_to_update_in_cache if k != global_proxy_spend_key)
    shared_scalar_updates: Final = tuple((k, v) for k, v in values_to_update_in_cache if k == global_proxy_spend_key)

    if local_object_updates:
        asyncio.create_task(
            user_api_key_cache.async_set_cache_pipeline(
                cache_list=list(local_object_updates),
                ttl=get_management_object_ttl(user_api_key_cache),
                litellm_parent_otel_span=parent_otel_span,
                local_only=True,
            )
        )
    if shared_scalar_updates:
        asyncio.create_task(
            user_api_key_cache.async_set_cache_pipeline(
                cache_list=list(shared_scalar_updates),
                ttl=get_management_object_ttl(user_api_key_cache),
                litellm_parent_otel_span=parent_otel_span,
            )
        )


def run_ollama_serve():
    try:
        command: Final = ["ollama", "serve"]

        with open(os.devnull, "w") as devnull:
            subprocess.Popen(command, stdout=devnull, stderr=devnull)
    except Exception as e:
        verbose_proxy_logger.debug(
            "\n            LiteLLM Warning: proxy started with `ollama` model\n`ollama serve` failed with Exception%s. \nEnsure you run `ollama serve`\n        ",
            e,
        )


def _get_process_rss_mb() -> float | None:
    """
    Get process RSS memory in MB.
    On Linux, ru_maxrss is in KB. On macOS, ru_maxrss is in bytes.
    """
    try:
        import resource

        ru_maxrss: Final = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(ru_maxrss) / (1024 * 1024)
        return float(ru_maxrss) / 1024
    except Exception:
        return None


def _rss_mb_for_log() -> str:
    rss_mb: Final = _get_process_rss_mb()
    if rss_mb is None:
        return "unknown"
    return f"{rss_mb:.2f}"


_UNEXPECTED_KWARG: Final = re.compile(r"unexpected keyword argument '(?P<name>[^']+)'")


async def _run_direct_health_check_with_instrumentation(
    model_list: list,
    details: bool | None,
    max_concurrency: int | None,
    instrumentation_context: dict,
):
    """Call ``perform_health_check``, dropping exactly the optional kwarg each TypeError names.

    A callee that predates an argument rejects it by name, so only that one is dropped. A
    hand-written ladder of combinations would drop working options alongside it, and would
    need a new rung every time an argument is added.
    """
    optional: Mapping[str, object] = MappingProxyType(  # rebind-ok: loses the kwarg the callee rejected
        {
            "router": llm_router,
            "instrumentation_context": instrumentation_context,
            **health_check_filter_kwargs_from_general_settings(general_settings),
        }
    )
    for _ in range(len(optional) + 1):
        try:
            return await perform_health_check(
                model_list=model_list,
                details=details,
                max_concurrency=max_concurrency,
                **optional,
            )
        except TypeError as e:
            rejected = _UNEXPECTED_KWARG.search(str(e))
            if rejected is None or rejected["name"] not in optional:
                raise
            optional = MappingProxyType({k: v for k, v in optional.items() if k != rejected["name"]})
    raise AssertionError("perform_health_check rejected every optional argument")


def _schedule_background_health_check_db_save(
    prisma_client,
    shared_health_manager,
    model_list: list,
    healthy_endpoints: list,
    unhealthy_endpoints: list,
):
    """Fire-and-forget: persist health check results to DB if prisma is available."""
    if prisma_client is None:
        return
    import time as time_module

    from litellm.proxy.health_endpoints._health_endpoints import (
        _save_background_health_checks_to_db,
    )

    checked_by: Final = shared_health_manager.pod_id if shared_health_manager is not None else "background_health_check"
    start_time: Final = time_module.time()
    asyncio.create_task(
        _save_background_health_checks_to_db(
            prisma_client,
            model_list,
            healthy_endpoints,
            unhealthy_endpoints,
            start_time,
            checked_by=checked_by,
        )
    )


def _get_endpoint_exception_status(endpoint: dict, exceptions: dict) -> int:
    """Return the HTTP status code for an unhealthy endpoint.

    Prefers the live exception object in `exceptions` (direct health check path).
    Falls back to the `exception_status` integer stored on the endpoint dict
    (shared-cache path, where exception objects are not available).
    """
    model_id: Final = endpoint.get("model_id")
    exc: Final = exceptions.get(model_id) if model_id else None
    if exc is not None:
        return getattr(exc, "status_code", 500)
    return endpoint.get("exception_status", 500)


def _write_health_state_to_router_cache(
    healthy_endpoints: list,
    unhealthy_endpoints: list,
    exceptions_by_model_id: dict | None = None,
) -> None:
    """
    Write deployment health states to the router's health state cache
    for health-check-driven routing. No-op if the feature is disabled.

    `model_list_healthy_only` reads the same cache to hide unhealthy models from
    the listing endpoints, so it also keeps the cache populated. That is a pure
    write: every routing-time reader is itself gated on
    `enable_health_check_routing`, and the cooldown/failure bookkeeping below
    stays behind that flag, so routing is untouched when only the listing filter
    is on.
    """
    from litellm.proxy.health_check import build_deployment_health_states
    from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        increment_deployment_failures_for_current_minute,
    )

    _exceptions: Final[dict] = exceptions_by_model_id or {}

    try:
        if llm_router is None:
            return
        health_check_routing_enabled: Final = llm_router.enable_health_check_routing
        if not health_check_routing_enabled and not is_healthy_only_listing_default(general_settings):
            return

        # When health_check_ignore_transient_errors is set, treat 429/408
        # endpoints as healthy so they are not filtered from routing.
        _effective_unhealthy = unhealthy_endpoints
        if llm_router.health_check_ignore_transient_errors:
            _effective_unhealthy = [
                ep for ep in unhealthy_endpoints if _get_endpoint_exception_status(ep, _exceptions) not in (429, 408)
            ]

        states: Final = build_deployment_health_states(
            healthy_endpoints=healthy_endpoints,
            unhealthy_endpoints=_effective_unhealthy,
        )
        if states:
            llm_router.health_state_cache.set_deployment_health_states(states)
            verbose_proxy_logger.debug(
                "health_check_routing_state_updated healthy=%d unhealthy=%d",
                sum(1 for s in states.values() if s.get("is_healthy")),
                sum(1 for s in states.values() if not s.get("is_healthy")),
            )

        if not health_check_routing_enabled:
            return

        for endpoint in unhealthy_endpoints:
            model_id = endpoint.get("model_id")
            if not model_id:
                continue

            original_exception = _exceptions.get(model_id)
            if original_exception is None:
                continue

            exception_status = getattr(original_exception, "status_code", 500)

            if llm_router.health_check_ignore_transient_errors and exception_status in (
                429,
                408,
            ):
                continue

            increment_deployment_failures_for_current_minute(
                litellm_router_instance=llm_router,
                deployment_id=model_id,
            )

            _set_cooldown_deployments(
                litellm_router_instance=llm_router,
                original_exception=original_exception,
                exception_status=exception_status,
                deployment=model_id,
                time_to_cooldown=llm_router.cooldown_time,
            )

    except Exception as e:
        verbose_proxy_logger.warning("Failed to write health state to router cache: %s", str(e))


_ADAPTIVE_ROUTER_FLUSH_INTERVAL_SECONDS: Final = 10


async def _adaptive_router_flusher_loop():
    """
    Drain every AdaptiveRouter's in-memory state + session aggregators into
    Postgres on a fixed cadence. Hot-path writes go to memory; this loop is
    the only writer to the adaptive router DB tables.
    """
    global llm_router, prisma_client
    while True:
        try:
            await asyncio.sleep(_ADAPTIVE_ROUTER_FLUSH_INTERVAL_SECONDS)
            adaptive_routers = getattr(llm_router, "adaptive_routers", None) or {}
            if not adaptive_routers or prisma_client is None:
                continue
            for tagged_routers in adaptive_routers.values():
                for tagged in tagged_routers:
                    ar = tagged.strategy
                    # Lazy state load: covers adaptive routers registered via
                    # `/config/reload` after proxy boot.
                    if not getattr(ar, "_state_loaded", False):
                        try:
                            await ar.load_state_from_db(prisma_client)
                        finally:
                            ar._state_loaded = True
                    await ar.queue.flush_state_to_db(prisma_client)
                    await ar.queue.flush_session_to_db(prisma_client)
        except asyncio.CancelledError:
            raise
        except Exception:
            verbose_proxy_logger.exception("adaptive_router flusher iteration failed")


async def _run_background_health_check():
    """
    Periodically run health checks in the background on the endpoints.

    Update health_check_results, based on this.
    Uses shared health check state when Redis is available to coordinate across pods.
    """
    global health_check_results, llm_model_list, health_check_interval
    global health_check_concurrency, health_check_details, use_shared_health_check
    global redis_usage_cache, prisma_client
    global background_health_check_loop_active, background_health_check_cycle_seq

    if health_check_interval is None or not isinstance(health_check_interval, int) or health_check_interval <= 0:
        return

    if background_health_check_loop_active:
        verbose_proxy_logger.warning(
            "background_health_check_loop_overlap_detected existing_loop_active=true interval_seconds=%s max_concurrency=%s shared=%s",
            health_check_interval,
            health_check_concurrency,
            use_shared_health_check,
        )
    background_health_check_loop_active = True
    verbose_proxy_logger.info(
        "background_health_check_loop_started interval_seconds=%s max_concurrency=%s shared=%s details=%s thread_count=%d rss_mb=%s",
        health_check_interval,
        health_check_concurrency,
        use_shared_health_check,
        health_check_details,
        threading.active_count(),
        _rss_mb_for_log(),
    )

    # Initialize shared health check manager if Redis is available and feature is enabled
    shared_health_manager = None
    if use_shared_health_check and redis_usage_cache is not None:
        from litellm.proxy.health_check_utils.shared_health_check_manager import (
            SharedHealthCheckManager,
        )

        shared_health_manager = SharedHealthCheckManager(
            redis_cache=redis_usage_cache,
            health_check_ttl=DEFAULT_SHARED_HEALTH_CHECK_TTL,
            lock_ttl=DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
        )
        verbose_proxy_logger.info("Initialized shared health check manager")

    while True:
        background_health_check_cycle_seq += 1
        cycle_id = f"bg-{background_health_check_cycle_seq}"
        cycle_start_time = time.monotonic()

        # make 1 deep copy of llm_model_list on every health check iteration
        _llm_model_list = copy.deepcopy(llm_model_list) or []
        model_count_total = len(_llm_model_list)

        # filter out models that have disabled background health checks
        _llm_model_list = [
            m for m in _llm_model_list if not m.get("model_info", {}).get("disable_background_health_check", False)
        ]
        scoped_model_groups = llm_router.background_health_check_model_groups if llm_router is not None else None
        _llm_model_list = list(filter_deployments_to_model_groups(_llm_model_list, scoped_model_groups))
        if scoped_model_groups is not None and not _llm_model_list:
            verbose_proxy_logger.warning(
                "background_health_check_model_groups matched no deployments; groups=%s",
                sorted(scoped_model_groups),
            )
        model_count_enabled = len(_llm_model_list)
        expected_peak_in_flight = model_count_enabled
        if isinstance(health_check_concurrency, int) and health_check_concurrency > 0 and model_count_enabled > 0:
            expected_peak_in_flight = min(model_count_enabled, health_check_concurrency)

        verbose_proxy_logger.debug(
            "background_health_check_cycle_start cycle_id=%s model_count_total=%d model_count_enabled=%d interval_seconds=%s max_concurrency=%s expected_peak_in_flight=%d shared=%s thread_count=%d rss_mb=%s",
            cycle_id,
            model_count_total,
            model_count_enabled,
            health_check_interval,
            health_check_concurrency,
            expected_peak_in_flight,
            shared_health_manager is not None,
            threading.active_count(),
            _rss_mb_for_log(),
        )

        instrumentation_context = {
            "enabled": True,
            "source": "proxy_background_loop",
            "cycle_id": cycle_id,
        }

        # Use shared health check if available, otherwise fall back to direct health check
        # Convert health_check_details to bool for perform_shared_health_check (defaults to True if None)
        details_bool = health_check_details if health_check_details is not None else True
        _hc_filter = health_check_filter_kwargs_from_general_settings(general_settings)

        if shared_health_manager is not None:
            try:
                (
                    healthy_endpoints,
                    unhealthy_endpoints,
                    _exceptions_by_model_id,
                ) = await shared_health_manager.perform_shared_health_check(
                    model_list=_llm_model_list,
                    details=details_bool,
                    max_concurrency=health_check_concurrency,
                    router=llm_router,
                    **_hc_filter,
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    "Error in shared health check, falling back to direct health check: %s",
                    str(e),
                )
                (
                    healthy_endpoints,
                    unhealthy_endpoints,
                    _exceptions_by_model_id,
                ) = await _run_direct_health_check_with_instrumentation(
                    _llm_model_list,
                    details_bool,
                    health_check_concurrency,
                    instrumentation_context,
                )
        else:
            (
                healthy_endpoints,
                unhealthy_endpoints,
                _exceptions_by_model_id,
            ) = await _run_direct_health_check_with_instrumentation(
                _llm_model_list,
                details_bool,
                health_check_concurrency,
                instrumentation_context,
            )

        # Update the global variable with the health check results
        health_check_results["healthy_endpoints"] = healthy_endpoints
        health_check_results["unhealthy_endpoints"] = unhealthy_endpoints
        health_check_results["healthy_count"] = len(healthy_endpoints)
        health_check_results["unhealthy_count"] = len(unhealthy_endpoints)
        cycle_duration_ms = (time.monotonic() - cycle_start_time) * 1000
        verbose_proxy_logger.debug(
            "background_health_check_cycle_complete cycle_id=%s model_count_enabled=%d healthy_count=%d unhealthy_count=%d duration_ms=%.2f interval_seconds=%s thread_count=%d rss_mb=%s",
            cycle_id,
            model_count_enabled,
            len(healthy_endpoints),
            len(unhealthy_endpoints),
            cycle_duration_ms,
            health_check_interval,
            threading.active_count(),
            _rss_mb_for_log(),
        )
        if cycle_duration_ms > (health_check_interval * 1000):
            verbose_proxy_logger.warning(
                "background_health_check_cycle_duration_exceeded_interval cycle_id=%s duration_ms=%.2f interval_seconds=%s",
                cycle_id,
                cycle_duration_ms,
                health_check_interval,
            )

        # Save background health checks to database (non-blocking)
        _schedule_background_health_check_db_save(
            prisma_client,
            shared_health_manager,
            _llm_model_list,
            healthy_endpoints,
            unhealthy_endpoints,
        )

        # Write health state to router cache for health-check-driven routing
        _write_health_state_to_router_cache(healthy_endpoints, unhealthy_endpoints, _exceptions_by_model_id)

        await asyncio.sleep(health_check_interval)


class StreamingCallbackError(Exception):
    pass


# Fields in ``litellm_settings`` / ``general_settings`` whose values flow
# into ``get_instance_fn`` during config load. Remote-URL values
# (``s3://`` / ``gcs://``) are scrubbed from these when the value
# originates from a DB-overlay merge: at the point ``get_instance_fn``
# is invoked, ``config_file_path`` is non-None (the YAML load chain is
# active), so the runtime gate cannot distinguish a YAML-sourced value
# from a DB-sourced value. Scrubbing at the merge boundary closes that
# gap without tracking source on every config dict entry.
_DB_OVERLAY_REMOTE_MODULE_STR_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "litellm_settings": ("post_call_rules",),
    "general_settings": (
        "custom_auth",
        "custom_key_generate",
        "custom_key_update",
        "custom_team_metadata_validate",
        "custom_sso",
        "custom_ui_sso_sign_in_handler",
    ),
}
_DB_OVERLAY_REMOTE_MODULE_LIST_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "litellm_settings": (
        "callbacks",
        "success_callback",
        "failure_callback",
        "audit_log_callbacks",
    ),
}


def _is_remote_module_url(value: object) -> bool:
    return isinstance(value, str) and (value.startswith("s3://") or value.startswith("gcs://"))


def _scrub_guardrail_inner(inner: dict[str, JsonValue]) -> None:
    """Strip remote-URL entries from a guardrail's ``callbacks`` list
    and ``guardrail`` (v2 module-path) field. Mutates in place."""
    cbs: Final = inner.get("callbacks")
    if isinstance(cbs, list):
        cleaned: Final = [c for c in cbs if not _is_remote_module_url(c)]
        if len(cleaned) != len(cbs):
            verbose_proxy_logger.warning(
                "Refused %d remote-URL entries from DB-overlay litellm_settings.guardrails[...].callbacks",
                len(cbs) - len(cleaned),
            )
            inner["callbacks"] = cleaned
    if _is_remote_module_url(inner.get("guardrail")):
        verbose_proxy_logger.warning(
            "Refused remote-URL guardrail module from DB-overlay litellm_settings.guardrails[...].guardrail: %r",
            inner.get("guardrail"),
        )
        inner["guardrail"] = None


def _scrub_db_overlay_remote_module_loads(section: str, db_value: JsonValue) -> JsonValue:
    """Strip ``s3://`` / ``gcs://`` entries from the DB-overlay value for
    fields whose contents reach ``get_instance_fn``. The same scheme is
    allowed from a YAML config (the documented operator flow) but a
    DB-overlay write would otherwise smuggle the same payload through
    the YAML-load chain and reach ``_load_instance_from_remote_storage``."""
    if not isinstance(db_value, dict):
        return db_value
    str_fields: Final = _DB_OVERLAY_REMOTE_MODULE_STR_FIELDS.get(section, ())
    list_fields: Final = _DB_OVERLAY_REMOTE_MODULE_LIST_FIELDS.get(section, ())
    if not str_fields and not list_fields and section != "general_settings":
        return db_value
    sanitized: Final = copy.deepcopy(db_value)
    for field in str_fields:
        v = sanitized.get(field)
        if _is_remote_module_url(v):
            verbose_proxy_logger.warning(
                "Refused remote-URL value for DB-overlay %s.%s=%r; only "
                "config.yaml entries may reference s3:// / gcs:// modules.",
                section,
                field,
                v,
            )
            sanitized[field] = None
    for field in list_fields:
        v = sanitized.get(field)
        if isinstance(v, list):
            cleaned = [item for item in v if not _is_remote_module_url(item)]
            if len(cleaned) != len(v):
                verbose_proxy_logger.warning(
                    "Refused %d remote-URL entries from DB-overlay %s.%s; "
                    "only config.yaml entries may reference s3:// / gcs:// "
                    "modules.",
                    len(v) - len(cleaned),
                    section,
                    field,
                )
                sanitized[field] = cleaned
    # ``custom_provider_map`` is a list of dicts with ``custom_handler`` —
    # walk it explicitly.
    if section == "litellm_settings":
        cpm: Final = sanitized.get("custom_provider_map")
        if isinstance(cpm, list):
            for item in cpm:
                if isinstance(item, dict) and _is_remote_module_url(item.get("custom_handler")):
                    verbose_proxy_logger.warning(
                        "Refused remote-URL custom_handler from DB-overlay litellm_settings.custom_provider_map: %r",
                        item.get("custom_handler"),
                    )
                    item["custom_handler"] = None
    # ``litellm_settings.guardrails`` is a list of single-key dicts in
    # v1 ({guardrail_name: {callbacks: [...], default_on: bool}}) or a
    # list of v2 entries ({guardrail_name, litellm_params: {guardrail:
    # "module.path", callbacks: [...]}}). Both shapes terminate in
    # ``callbacks`` (a list) or ``guardrail`` (a single dotted name)
    # that flow into ``get_instance_fn`` during config load.
    if section == "litellm_settings":
        guardrails: Final = sanitized.get("guardrails")
        if isinstance(guardrails, list):
            for entry in guardrails:
                if not isinstance(entry, dict):
                    continue
                for inner in entry.values():
                    if not isinstance(inner, dict):
                        continue
                    _scrub_guardrail_inner(inner)
                lp = entry.get("litellm_params")
                if isinstance(lp, dict):
                    _scrub_guardrail_inner(lp)

    # ``general_settings.litellm_jwtauth.custom_validate`` is a nested
    # string field.
    if section == "general_settings":
        jwt: Final = sanitized.get("litellm_jwtauth")
        if isinstance(jwt, dict) and _is_remote_module_url(jwt.get("custom_validate")):
            verbose_proxy_logger.warning(
                "Refused remote-URL custom_validate from DB-overlay general_settings.litellm_jwtauth: %r",
                jwt.get("custom_validate"),
            )
            jwt["custom_validate"] = None
        # ``pass_through_endpoints`` is a list of dicts whose ``target``
        # is passed through ``create_pass_through_route`` →
        # ``get_instance_fn``. A DB-overlay ``target: "s3://attacker/m.i"``
        # would otherwise reach the loader because the YAML-load chain
        # has ``config_file_path`` set.
        pte: Final = sanitized.get("pass_through_endpoints")
        if isinstance(pte, list):
            for entry in pte:
                if isinstance(entry, dict) and _is_remote_module_url(entry.get("target")):
                    verbose_proxy_logger.warning(
                        "Refused remote-URL target from DB-overlay "
                        "general_settings.pass_through_endpoints "
                        "(path=%r): %r",
                        entry.get("path"),
                        entry.get("target"),
                    )
                    entry["target"] = None
    return sanitized


def _normalize_user_url_validation(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return str_to_bool(value)
    return bool(value)


def _apply_ssrf_general_settings(settings: Mapping[str, object]) -> None:
    if "user_url_allowed_hosts" in settings:
        litellm.user_url_allowed_hosts = cast(list[str], settings["user_url_allowed_hosts"])

    user_url_validation: Final = _normalize_user_url_validation(settings.get("user_url_validation"))
    if user_url_validation is not None:
        litellm.user_url_validation = user_url_validation

    if "provider_url_destination_allowed_hosts" in settings:
        litellm.provider_url_destination_allowed_hosts = cast(
            list[str], settings["provider_url_destination_allowed_hosts"]
        )


def _set_redis_usage_cache(coordination_redis_cache: RedisCache | None) -> None:
    """Publish the resolved coordination Redis to the consumers that read it directly."""
    global redis_usage_cache
    redis_usage_cache = coordination_redis_cache


def _resolve_coordination_redis_env_refs(raw_params: Mapping[str, object]) -> dict[str, object]:
    """Resolve `os.environ/VAR` references in a coordination_redis block."""
    return {
        key: (get_secret(value) if isinstance(value, str) and value.startswith("os.environ/") else value)
        for key, value in raw_params.items()
    }


def _build_redis_usage_cache(redis_params: Mapping[str, object]) -> RedisCache:
    """
    Builds the proxy's coordination Redis client from resolved connection
    params. Cluster-mode targets (explicit `startup_nodes` or the
    REDIS_CLUSTER_NODES env var) get a `RedisClusterCache`, so consumers that
    branch on cluster mode (e.g. the v3 rate limiter) take the cluster path;
    everything else (host/url/sentinel) gets a plain `RedisCache`.
    """
    startup_nodes = redis_params.get("startup_nodes")
    if startup_nodes is None:
        env_cluster_nodes: Final = get_secret_str("REDIS_CLUSTER_NODES")
        if env_cluster_nodes is not None:
            startup_nodes = json.loads(env_cluster_nodes)
    non_node_params: Final = {key: value for key, value in redis_params.items() if key != "startup_nodes"}
    if startup_nodes:
        return RedisClusterCache(startup_nodes=startup_nodes, **non_node_params)
    return RedisCache(**non_node_params)


def _environment_has_redis_connection_target() -> bool:
    """
    Whether the REDIS_* environment variables name a Redis to connect to (host,
    url, cluster nodes, or sentinel nodes). Read-only: callers that only need to
    know whether the env fallback would apply use this instead of building a
    client.
    """
    redis_env_kwargs: Final = litellm._redis._redis_kwargs_from_environment()
    return (
        "host" in redis_env_kwargs
        or "url" in redis_env_kwargs
        or get_secret_str("REDIS_CLUSTER_NODES") is not None
        or get_secret_str("REDIS_SENTINEL_NODES") is not None
    )


def _build_redis_usage_cache_from_environment() -> RedisCache | None:
    """
    Builds a standalone coordination Redis from REDIS_* environment variables.

    Lets the proxy's coordination Redis (cross-pod tpm/rpm rate limits, spend
    tracking, pod lock manager) run when the response-cache backend is not a
    plain Redis KV cache (e.g. a semantic cache, disk, or s3).

    Returns None when the environment carries no connection target (host, url,
    cluster nodes, or sentinel nodes).
    """
    if not _environment_has_redis_connection_target():
        return None
    return _build_redis_usage_cache(litellm._redis._redis_kwargs_from_environment())


def _attach_redis_usage_cache(redis_cache: RedisCache, enable_redis_auth_cache: bool) -> None:
    """
    Wires an established coordination Redis into the proxy-level caches that
    consume it directly: the spend counter cache, the CLI SSO login-session
    cache, the cluster-wide config cache, and (only when opted in) the
    virtual-key auth cache.

    The CLI SSO login-session cache is always backed by Redis when available so
    that the browser SSO flow behind `lite login` survives landing on different
    workers; it must not be gated behind enable_redis_auth_cache.
    """
    spend_counter_cache.attach_redis_cache(
        redis_cache,
        default_redis_ttl=litellm.default_redis_ttl,
    )
    cli_sso_session_cache.attach_redis_cache(
        redis_cache,
        default_redis_ttl=CLI_SSO_SESSION_TTL_SECONDS,
    )
    if enable_redis_auth_cache is True:
        user_api_key_cache.attach_redis_cache(
            redis_cache,
            default_redis_ttl=litellm.default_redis_ttl,
        )
        verbose_proxy_logger.info(
            "enable_redis_auth_cache=True: attached Redis to "
            "user_api_key_cache — virtual-key lookups are now "
            "shared across all proxy workers."
        )
    else:
        verbose_proxy_logger.info(
            "enable_redis_auth_cache is not set: user_api_key_cache "
            "remains in-memory only (per-worker). Set "
            "litellm_settings.enable_redis_auth_cache: true to share "
            "the auth cache across workers and reduce DB load."
        )
    litellm_config_cache.redis_cache = redis_cache


def resolve_routing_plugins(
    plugin_paths: list,
    config_file_path: str | None,
    source_label: str,
) -> list:
    """
    Resolves a list of routing-plugin entries to live `RoutingPlugin` instances.
    Each string entry is resolved through `get_instance_fn` (the same dotted-path
    convention `litellm_settings.callbacks` uses, which resolves both local module
    files next to the config and modules installed as Python packages); non-string
    entries are assumed to already be instances and passed through. Raises at
    config-load time if any entry resolves to something that doesn't implement
    `RoutingPlugin`, rather than deferring to a confusing `AttributeError` on the
    first request that reaches the plugin pipeline. `source_label` names the config
    key being resolved so the error points the operator at the right place.
    """
    resolved_plugins: Final = [
        get_instance_fn(value=plugin_path, config_file_path=config_file_path)
        if isinstance(plugin_path, str)
        else plugin_path
        for plugin_path in plugin_paths
    ]
    for plugin_path, resolved_plugin in zip(plugin_paths, resolved_plugins):
        # `@runtime_checkable` only checks that `run` exists as an attribute, not that
        # it's a coroutine function -- a synchronous `def run(self, context)` would pass
        # isinstance() here and only fail at request time with a confusing `TypeError:
        # object RoutingContext can't be used in 'await' expression`.
        if not isinstance(resolved_plugin, RoutingPlugin) or not inspect.iscoroutinefunction(
            getattr(resolved_plugin, "run", None)
        ):
            raise ValueError(
                f"{source_label} entry {plugin_path!r} resolved to {resolved_plugin!r}, which does "
                "not implement the RoutingPlugin interface (an async `run(context)` method). Fix the "
                "referenced module before starting the proxy."
            )
    return resolved_plugins


def resolve_complexity_router_plugins(
    model_name: str,
    complexity_router_config: dict,
    config_file_path: str | None,
) -> None:
    """
    Resolves `complexity_router_config["plugins"]` dotted-path strings to live
    instances in place, via `resolve_routing_plugins`, and
    `complexity_router_config["classifier_plugin"]` via `resolve_classifier_plugin`.
    """
    plugin_paths: Final = complexity_router_config.get("plugins")
    if isinstance(plugin_paths, list):
        complexity_router_config["plugins"] = resolve_routing_plugins(
            plugin_paths=plugin_paths,
            config_file_path=config_file_path,
            source_label=f"complexity_router_config.plugins on model {model_name!r}",
        )

    classifier_plugin_path: Final = complexity_router_config.get("classifier_plugin")
    if isinstance(classifier_plugin_path, str):
        resolved_classifier: Final = resolve_classifier_plugin(
            plugin_path=classifier_plugin_path,
            config_file_path=config_file_path,
            source_label=f"complexity_router_config.classifier_plugin on model {model_name!r}",
        )
        complexity_router_config["classifier_plugin"] = resolved_classifier  # rebind-ok: out-param, resolved in place


def validate_deployment_max_agentic_loops(model: Mapping[str, object]) -> None:
    """
    Reject a per-deployment `max_agentic_loops` the agentic loop cannot honor.

    Checked here rather than on `LiteLLM_Params` because the proxy builds its
    router with `ignore_invalid_deployments=True`, so a validator down there
    turns a bad value into a silently missing model instead of a refusal to
    start. Left unchecked entirely, a `0` used to read as the default ceiling
    of 3 and a non-integer failed every request to that model instead.
    """
    litellm_params: Final = model.get("litellm_params")
    if not isinstance(litellm_params, Mapping):
        return
    if "max_agentic_loops" not in litellm_params:
        return

    model_name: Final = model.get("model_name", "")
    validated_max_agentic_loops(
        litellm_params["max_agentic_loops"],
        field=f"litellm_params.max_agentic_loops on model {model_name!r}",
    )


def pin_complexity_router_model_id(model: dict) -> None:  # mutable-ok: out-param, model_info is stamped in place
    """
    Stamps `model_info.id` from the raw litellm_params before plugin resolution swaps
    dotted-path strings for live instances. `_delete_deployment` re-reads the raw config
    and re-hashes these params to decide which ids the config wants served; an id the
    Router derived from the resolved params would never match that hash, so the reconcile
    would evict every plugin-bearing deployment one sync after startup.
    """
    litellm_params: Final = model.get("litellm_params")
    if not isinstance(litellm_params, dict) or not isinstance(litellm_params.get("complexity_router_config"), dict):
        return
    model_info = model.get("model_info")
    if not isinstance(model_info, dict):
        model_info = {}  # mutable-ok: fresh model_info stamped onto the raw yaml model dict
        model["model_info"] = model_info  # rebind-ok: out-param, stamped in place
    if model_info.get("id") is None:
        model_info["id"] = litellm.Router.generate_model_id(
            model_group=model.get("model_name", ""),
            litellm_params=litellm_params,
        )


def resolve_classifier_plugin(
    plugin_path: str,
    config_file_path: str | None,
    source_label: str,
) -> ClassifierPlugin:
    """
    Resolves a classifier-plugin dotted path to a live `ClassifierPlugin` instance, with the
    same load-time interface check `resolve_routing_plugins` applies to routing plugins: a
    sync `def classify` passes the runtime_checkable isinstance and would only fail on the
    first classified request, so reject it here where the error names the config key.
    """
    resolved: Final = get_instance_fn(value=plugin_path, config_file_path=config_file_path)
    if not isinstance(resolved, ClassifierPlugin) or not inspect.iscoroutinefunction(
        getattr(resolved, "classify", None)
    ):
        raise ValueError(
            f"{source_label} entry {plugin_path!r} resolved to {resolved!r}, which does not "
            "implement the ClassifierPlugin interface (an async `classify(context)` method). Fix "
            "the referenced module before starting the proxy."
        )
    return resolved


def _swap_in_model_cost_map(new_model_cost_map: dict) -> int:
    """Adopt a freshly fetched cost map into this process's litellm state, return the model count"""
    litellm.model_cost = new_model_cost_map
    # Invalidate case-insensitive lookup map since model_cost was replaced
    _invalidate_model_cost_lowercase_map()
    # Repopulate provider model sets (e.g. litellm.anthropic_models) so that
    # wildcard patterns like "anthropic/*" include any newly added models.
    litellm.add_known_models(model_cost_map=new_model_cost_map)
    # Counted before the re-apply below, which writes into this same dict, so the
    # number reported describes the fetched price data alone.
    fetched_model_count: Final = len(new_model_cost_map) if new_model_cost_map else 0
    # The swap discards everything registered at runtime (deployment model_info,
    # register_model overrides), so put it back on top of the fresh catalog.
    reapply_runtime_model_cost_registrations()
    return fetched_model_count


def should_load_db_object(object_type: str | SupportedDBObjectType) -> bool:
    """
    Check if an object type should be loaded from the database based on general_settings.supported_db_objects.

    Args:
        object_type: Type of object to check (e.g., SupportedDBObjectType.MODELS, "models", etc.)

    Returns:
        True if the object should be loaded, False otherwise
    """
    supported_db_objects: Final = general_settings.get("supported_db_objects", None)

    if supported_db_objects is None:
        return True

    if not isinstance(supported_db_objects, list):
        verbose_proxy_logger.warning(
            "supported_db_objects is not a list, got %s. Loading all objects.", type(supported_db_objects)
        )
        return True

    object_type_str: Final = str(object_type)
    return any(str(obj) == object_type_str for obj in supported_db_objects)


class ProxyConfig:
    """
    Abstraction class on top of config loading/updating logic. Gives us one place to control all config updating logic.
    """

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self._last_semantic_filter_config: dict[str, object] | None = None
        self._last_hashicorp_vault_config: dict[str, object] | None = None
        self.worker_registry: list[WorkerRegistryEntry] = []
        self.config_sync_subscriber: ConfigSyncSubscriber | None = None
        self.auth_cache_invalidation_subscriber: AuthCacheInvalidationSubscriber | None = None
        from litellm.litellm_core_utils.get_model_cost_map import (
            get_model_cost_map_loaded_at,
        )

        self.model_cost_map_loaded_at: datetime = get_model_cost_map_loaded_at() or utc_now()
        # Starts unapplied rather than adopting the published revision: this pod cannot tell
        # whether an existing request predates the prices it just fetched, and re-serving one
        # costs a single fetch where skipping one leaves it priced wrong indefinitely
        self.model_cost_map_applied_revision: int = 0
        # Keys explicitly set in the YAML config file. Used to give YAML
        # precedence over stale DB-cached values for these specific keys
        # during periodic config reloads (_update_general_settings).
        self._yaml_general_settings_keys: set[str] = set()  # mutable-ok: populated once at startup, read-only thereafter  # fmt: skip
        self._yaml_spend_log_cleanup_bounds: dict[str, object] = {}  # mutable-ok: snapshot of YAML bounds at load time  # fmt: skip

    def is_yaml(self, config_file_path: str) -> bool:
        if not os.path.isfile(config_file_path):
            return False

        _, file_extension = os.path.splitext(config_file_path)
        return file_extension.lower() == ".yaml" or file_extension.lower() == ".yml"

    def _load_yaml_file(self, file_path: str) -> dict:
        """
        Load and parse a YAML file
        """
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            raise Exception(f"Error loading yaml file {file_path}: {e}")

    async def _get_config_from_file(self, config_file_path: str | None = None) -> dict:
        """
        Given a config file path, load the config from the file.
        Args:
            config_file_path (str): path to the config file
        Returns:
            dict: config
        """
        global prisma_client, user_config_file_path

        file_path: Final = config_file_path or user_config_file_path
        if config_file_path is not None:
            user_config_file_path = config_file_path
        # Load existing config
        ## Yaml
        if os.path.exists(f"{file_path}"):
            with open(f"{file_path}", "r") as config_file:
                config = yaml.safe_load(config_file)
        elif file_path is not None:
            raise Exception(f"Config file not found: {file_path}")
        else:
            config = {
                "model_list": [],
                "general_settings": {},
                "router_settings": {},
                "litellm_settings": {},
            }

        if config is None:
            raise Exception("Config cannot be None or Empty.")
        # Process includes
        config = self._process_includes(config=config, base_dir=os.path.dirname(os.path.abspath(file_path or "")))

        # verbose_proxy_logger.debug(f"loaded config={json.dumps(config, indent=4)}")
        return config

    def _process_includes(self, config: dict, base_dir: str) -> dict:
        """
        Process includes by appending their contents to the main config

        Handles nested config.yamls with `include` section

        Example config: This will get the contents from files in `include` and append it
        ```yaml
        include:
            - model_config.yaml

        litellm_settings:
            callbacks: ["prometheus"]
        ```
        """
        if "include" not in config:
            return config

        if not isinstance(config["include"], list):
            raise ValueError("'include' must be a list of file paths")

        # Load and append all included files
        for include_file in config["include"]:
            file_path = os.path.join(base_dir, include_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Included file not found: {file_path}")

            included_config = self._load_yaml_file(file_path)
            # Simply update/extend the main config with included config
            for key, value in included_config.items():
                if isinstance(value, list) and key in config:
                    config[key].extend(value)
                else:
                    config[key] = value

        # Remove the include directive
        del config["include"]
        return config

    async def save_config(self, new_config: dict, include_env_vars: bool = False):
        global prisma_client, general_settings, user_config_file_path, store_model_in_db
        # Load existing config
        ## DB - writes valid config to db
        """
        - Do not write restricted params like 'api_key' to the database
        - if api_key is passed, save that to the local environment or connected secret manage (maybe expose `litellm.save_secret()`)
        """

        if prisma_client is not None and (
            general_settings.get("store_model_in_db", False) is True or store_model_in_db
        ):
            # if using - db for config - models are in ModelTable

            # Make a copy to avoid mutating the original config
            config_to_save: Final = new_config.copy()

            # environment_variables are persisted to the DB only when a caller
            # explicitly opts in. Most callers reach save_config after
            # get_config() merged YAML + OS env into new_config (with
            # os.environ/ placeholders already resolved to plaintext), so
            # persisting them here would snapshot file/container env vars into
            # a config row that then shadows those sources on every restart.
            # The dedicated /config/update path writes env vars directly, so
            # no current caller needs include_env_vars=True.
            if not include_env_vars:
                config_to_save.pop("environment_variables", None)

            # SECURITY: Always encrypt environment_variables before DB write.
            # _encrypt_env_variables_for_db is idempotent — a caller that
            # already encrypted the values (or re-submitted ciphertext read
            # back from the DB) will not get a stacked second layer.
            if "environment_variables" in config_to_save and config_to_save["environment_variables"]:
                config_to_save["environment_variables"] = self._encrypt_env_variables_for_db(
                    environment_variables=config_to_save["environment_variables"]
                )

            config_to_save.pop("model_list", None)
            await prisma_client.insert_data(data=config_to_save, table_name="config")
        else:
            # Save the updated config - if user is not using a dB
            ## YAML
            with open(f"{user_config_file_path}", "w") as config_file:
                yaml.dump(new_config, config_file, default_flow_style=False)

    async def save_environment_variables(self, updates: dict[str, str | None]) -> None:
        """Persist specific environment variables to the DB config row.

        Each key in ``updates`` is written to the ``environment_variables``
        config row; a ``None`` value deletes that key. Env vars the caller does
        not name are preserved, so a caller that owns a couple of keys can
        update just those without snapshotting unrelated (YAML/OS-sourced)
        values the way a full ``save_config`` write would. No-op when config is
        not DB-backed.
        """
        global prisma_client, general_settings, store_model_in_db
        if prisma_client is None or not (general_settings.get("store_model_in_db", False) is True or store_model_in_db):
            return

        row: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
            where={"param_name": "environment_variables"}
        )
        existing: Final[dict] = dict(row.param_value) if row is not None and row.param_value is not None else {}

        to_set: Final = {k: v for k, v in updates.items() if v is not None}
        encrypted: Final = self._encrypt_env_variables_for_db(environment_variables=to_set) if to_set else {}
        deleted_keys: Final = {k for k, v in updates.items() if v is None}
        merged: Final = {**{k: v for k, v in existing.items() if k not in deleted_keys}, **encrypted}

        serialized: Final = json.dumps(merged)
        await ConfigRepository(prisma_client).table.upsert(
            where={"param_name": "environment_variables"},
            data={
                "create": {"param_name": "environment_variables", "param_value": serialized},
                "update": {"param_value": serialized},
            },
        )
        await invalidate_config_param("environment_variables")

    def _check_for_os_environ_vars(
        self, config: dict, depth: int = 0, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH
    ) -> dict:
        """
        Check for os.environ/ variables in the config and replace them with the actual values.
        Includes a depth limit to prevent infinite recursion.

        Args:
            config (dict): The configuration dictionary to process.
            depth (int): Current recursion depth.
            max_depth (int): Maximum allowed recursion depth.

        Returns:
            dict: Processed configuration dictionary.
        """
        if depth > max_depth:
            verbose_proxy_logger.warning("Maximum recursion depth (%s) reached while processing config.", max_depth)
            return config

        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self._check_for_os_environ_vars(config=value, depth=depth + 1, max_depth=max_depth)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item = self._check_for_os_environ_vars(config=item, depth=depth + 1, max_depth=max_depth)
            # if the value is a string and starts with "os.environ/" - then it's an environment variable
            elif isinstance(value, str) and value.startswith("os.environ/"):
                resolved = get_secret(value)
                if resolved is None and secret_manager_would_be_consulted(value):
                    verbose_proxy_logger.warning("%s is absent from the configured secret manager", value)
                config[key] = resolved
        return config

    def _initialize_secret_manager_from_raw_config(
        self, config: Mapping[str, object], config_file_path: str | None
    ) -> None:
        """
        Bring the secret manager up before `os.environ/<KEY>` references are resolved.

        `_check_for_os_environ_vars` writes whatever it resolves back into the config, so a key
        held only by the secret manager would otherwise become a permanent `None` that the later
        fallbacks in `load_config` can no longer recover from.

        `get_config` also runs on management-endpoint request paths, so this returns early once a
        manager exists rather than rebuilding the client on every request.

        The manager's own settings can only come from real environment variables, so they are
        resolved against a throwaway copy and the config is left untouched for the main pass.
        """
        if litellm.secret_manager_client is not None:
            return

        general_settings: Final = config.get("general_settings")
        if not isinstance(general_settings, dict):
            return

        raw_system: Final = general_settings.get("key_management_system")
        key_management_system: Final = (
            get_secret(raw_system)
            if isinstance(raw_system, str) and raw_system.startswith("os.environ/")
            else raw_system
        )
        if not isinstance(key_management_system, str):
            return

        raw_settings: Final = general_settings.get("key_management_settings")
        if isinstance(raw_settings, dict):
            litellm._key_management_settings = KeyManagementSettings(
                **self._check_for_os_environ_vars(config=copy.deepcopy(raw_settings))
            )

        self.initialize_secret_manager(
            key_management_system=key_management_system,
            config_file_path=config_file_path,
        )

    def _get_team_config(self, team_id: str, all_teams_config: list[dict]) -> dict:
        team_config: dict = {}
        for team in all_teams_config:
            if "team_id" not in team:
                raise Exception(f"team_id missing from team: {SENSITIVE_DATA_MASKER.mask_dict(team)}")
            if team_id == team["team_id"]:
                team_config = team
                break
        for k, v in team_config.items():
            if isinstance(v, str) and v.startswith("os.environ/"):
                team_config[k] = get_secret(v)
        return team_config

    def load_team_config(self, team_id: str):
        """
        - for a given team id
        - return the relevant completion() call params
        """

        # load existing config
        config: Final = self.get_config_state()

        ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
        litellm_settings: Final = config.get("litellm_settings", {})
        all_teams_config: Final = litellm_settings.get("default_team_settings", None)
        if all_teams_config is None:
            return {}
        team_config: Final = self._get_team_config(team_id=team_id, all_teams_config=all_teams_config)
        return team_config

    def _init_coordination_redis(self, config: dict) -> RedisCache | None:
        """
        Builds the coordination Redis from `general_settings.coordination_redis`
        when present, attaching it to the proxy-level caches. Runs before cache
        init, so an explicit block takes precedence over borrowing the
        response-cache Redis and over the REDIS_* env fallback. Returns the
        built client (None when the block is absent) for the caller to publish.
        """
        settings: Final = config.get("general_settings") or {}
        litellm_settings: Final = config.get("litellm_settings") or {}
        raw_params: Final = settings.get("coordination_redis")
        if raw_params is None:
            return None
        if not isinstance(raw_params, dict):
            raise ValueError("general_settings.coordination_redis must be a mapping of Redis connection params")

        coordination_params: Final = CoordinationRedisParams(**_resolve_coordination_redis_env_refs(raw_params))
        if not coordination_params.has_connection_target():
            raise ValueError(
                "general_settings.coordination_redis needs a connection target: "
                "set one of host, url, startup_nodes, or sentinel_nodes"
            )

        coordination_redis_cache: Final = _build_redis_usage_cache(coordination_params.model_dump(exclude_none=True))
        _attach_redis_usage_cache(
            coordination_redis_cache,
            enable_redis_auth_cache=litellm_settings.get("enable_redis_auth_cache", False) is True,
        )
        verbose_proxy_logger.info(
            "coordination_redis: using a standalone Redis from general_settings "
            "for usage tracking, rate limiting, and cross-pod coordination."
        )
        return coordination_redis_cache

    def _init_cache(
        self,
        cache_params: dict,
        enable_redis_auth_cache: bool = False,
    ) -> RedisCache | None:
        """
        Initializes the response cache and resolves the coordination Redis.

        Returns the coordination Redis for the caller to publish: an explicit
        coordination_redis block already set wins, else a plain-Redis response
        cache backend is borrowed, else the REDIS_* environment fallback applies.
        """
        from litellm import Cache

        if "default_in_memory_ttl" in cache_params:
            litellm.default_in_memory_ttl = cache_params["default_in_memory_ttl"]

        if "default_redis_ttl" in cache_params:
            litellm.default_redis_ttl = cache_params["default_redis_ttl"]

        litellm.cache = Cache(**cache_params)

        resolved_usage_cache = redis_usage_cache
        cache_backend: Final = litellm.cache.cache if litellm.cache is not None else None
        if resolved_usage_cache is None:
            if isinstance(cache_backend, (RedisCache, RedisClusterCache)):
                ## INIT PROXY REDIS USAGE CLIENT ##
                resolved_usage_cache = cache_backend
            else:
                resolved_usage_cache = _build_redis_usage_cache_from_environment()
                if resolved_usage_cache is not None:
                    verbose_proxy_logger.info(
                        "Cache backend %s is not a Redis KV cache; built a standalone "
                        "Redis from REDIS_* environment variables for usage tracking, "
                        "rate limiting, and cross-pod coordination.",
                        type(cache_backend).__name__,
                    )

        if resolved_usage_cache is not None:
            # Note: PKCE verifier storage uses redis_usage_cache directly (not
            # user_api_key_cache) to avoid routing all API-key lookups through Redis.
            _attach_redis_usage_cache(resolved_usage_cache, enable_redis_auth_cache)
        elif litellm_config_cache.redis_cache is None:
            verbose_proxy_logger.info("litellm_config_cache: no Redis configured; cluster-wide cache sharing disabled.")
        return resolved_usage_cache

    def switch_on_llm_response_caching(self):
        """
        Enable caching on the router by setting cache_responses=True.
        This ensures caching works without needing caching=True in request body.
        Router passes caching=self.cache_responses to litellm.completion()
        """
        global llm_router
        import litellm

        if llm_router is not None and litellm.cache is not None and llm_router.cache_responses is not True:
            llm_router.cache_responses = True
            verbose_proxy_logger.debug("Set router.cache_responses=True after initializing cache")

    async def get_config(self, config_file_path: str | None = None) -> dict:
        """
        Load config file
        Supports reading from:
        - .yaml file paths
        - LiteLLM connected DB
        - GCS
        - S3

        Args:
            config_file_path (str): path to the config file
        Returns:
            dict: config

        """
        global prisma_client, store_model_in_db
        # Load existing config

        if os.environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None:
            bucket_name: Final = os.environ.get("LITELLM_CONFIG_BUCKET_NAME")
            object_key: Final = os.environ.get("LITELLM_CONFIG_BUCKET_OBJECT_KEY")
            bucket_type: Final = os.environ.get("LITELLM_CONFIG_BUCKET_TYPE")
            verbose_proxy_logger.debug("bucket_name: %s, object_key: %s", bucket_name, object_key)
            if bucket_type == "gcs":
                config = await get_config_file_contents_from_gcs(bucket_name=bucket_name, object_key=object_key)
            else:
                config = get_file_contents_from_s3(bucket_name=bucket_name, object_key=object_key)

            if config is None:
                raise Exception("Unable to load config from given source.")
        else:
            # default to file

            config = await self._get_config_from_file(config_file_path=config_file_path)

        ## UPDATE CONFIG WITH DB
        if prisma_client is not None and store_model_in_db is True:
            config = await self._update_config_from_db(
                config=config,
                prisma_client=prisma_client,
                store_model_in_db=store_model_in_db,
            )

        ## PRINT YAML FOR CONFIRMING IT WORKS
        printed_yaml: Final = copy.deepcopy(config)
        printed_yaml.pop("environment_variables", None)

        self._initialize_secret_manager_from_raw_config(config=config, config_file_path=config_file_path)

        config = self._check_for_os_environ_vars(config=config)

        self.update_config_state(config=config)

        return config

    def update_config_state(self, config: dict):
        self.config = config

    def get_config_state(self):
        """
        Returns a deep copy of the config,

        Do this, to avoid mutating the config state outside of allowed methods
        """
        try:
            return copy.deepcopy(self.config)
        except Exception as e:
            verbose_proxy_logger.debug(
                "ProxyConfig:get_config_state(): Error returning copy of config state. self.config=%s\nError: %s",
                self.config,
                e,
            )
            return {}

    def load_credential_list(self, config: dict) -> list[CredentialItem]:
        """
        Load the credential list from the database
        """
        credential_list_dict: Final = config.get("credential_list")
        credential_list = []
        if credential_list_dict:
            credential_list = [CredentialItem(**cred) for cred in credential_list_dict]
        return credential_list

    def parse_search_tools(self, config: dict) -> list[SearchToolTypedDict] | None:
        """
        Parse and validate search tools from config.
        Loads environment variables and casts to SearchToolTypedDict.

        Args:
            config: Config dictionary containing search_tools

        Returns:
            List of validated SearchToolTypedDict or None if not configured
        """
        search_tools_raw = config.get("search_tools", None)
        if not search_tools_raw:
            # Check in general_settings
            general_settings = config.get("general_settings", {})
            if general_settings:
                search_tools_raw = general_settings.get("search_tools", None)

        if not search_tools_raw:
            return None

        search_tools_parsed: Final[list[SearchToolTypedDict]] = []

        print(  # noqa: T201
            "\033[32mLiteLLM: Proxy initialized with Search Tools:\033[0m"
        )

        for search_tool in search_tools_raw:
            # Display loaded search tool
            search_tool_name = search_tool.get("search_tool_name", "")
            search_provider = search_tool.get("litellm_params", {}).get("search_provider", "")
            print(  # noqa: T201
                f"\033[32m    {search_tool_name} ({search_provider})\033[0m"
            )

            # Handle os.environ/ variables in litellm_params
            litellm_params = search_tool.get("litellm_params", {})
            if litellm_params:
                for k, v in litellm_params.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        _v = v.replace("os.environ/", "")
                        v = get_secret(_v)
                        litellm_params[k] = v
                search_tool["litellm_params"] = litellm_params

            # Cast to SearchToolTypedDict for type safety
            try:
                search_tool_typed: SearchToolTypedDict = SearchToolTypedDict(**search_tool)
                search_tools_parsed.append(search_tool_typed)
            except Exception as e:
                verbose_proxy_logger.error("Error parsing search tool %s: %s", search_tool_name, e)
                continue

        return search_tools_parsed if search_tools_parsed else None

    # Environment variable keys that must not be overridden via config because
    # they can alter process execution, library loading, or network routing.
    _BLOCKED_ENV_KEYS: set[str] = {
        "PATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "HOME",
        "USER",
        "SHELL",
        "LOGNAME",
        "NO_PROXY",
        "no_proxy",
    }

    def _load_environment_variables(self, config: dict):
        ## ENVIRONMENT VARIABLES
        global premium_user
        environment_variables: Final = config.get("environment_variables", None)
        if environment_variables:
            for key, value in environment_variables.items():
                if key in self._BLOCKED_ENV_KEYS:
                    verbose_proxy_logger.warning("Skipping blocked environment variable key: %s", key)
                    continue
                #########################################################
                # handles this scenario:
                # ```yaml
                # environment_variables:
                #     ARIZE_ENDPOINT: os.environ/ARIZE_ENDPOINT
                # ```
                #########################################################
                if isinstance(value, str) and value.startswith("os.environ/"):
                    resolved_secret_string: str | None = get_secret_str(secret_name=value)
                    if resolved_secret_string is not None:
                        os.environ[key] = resolved_secret_string
                else:
                    #########################################################
                    # handles this scenario:
                    # ```yaml
                    # environment_variables:
                    #     ARIZE_ENDPOINT: https://otlp.arize.com/v1
                    # ```
                    #########################################################
                    os.environ[key] = str(value)

            # check if litellm_license in general_settings
            if "LITELLM_LICENSE" in environment_variables:
                _license_check.license_str = os.getenv("LITELLM_LICENSE", None)
                premium_user = _license_check.is_premium()

    def _warn_on_misplaced_jwt_keys(self, config: dict) -> tuple[str, ...]:
        misplaced_jwt_keys = tuple(key for key in ("enable_jwt_auth", "litellm_jwtauth") if key in config)
        if not misplaced_jwt_keys:
            return misplaced_jwt_keys
        verbose_proxy_logger.warning(
            "Ignoring top-level config key(s) %s. JWT auth settings must live under "
            "'general_settings' (e.g. general_settings.enable_jwt_auth, "
            "general_settings.litellm_jwtauth); placed at the top level they are silently "
            "dropped and JWT auth (and JWT-to-virtual-key mapping) will not engage.",
            ", ".join(misplaced_jwt_keys),
        )
        return misplaced_jwt_keys

    async def load_config(self, router: litellm.Router | None, config_file_path: str):
        """
        Load config values into proxy global state
        """
        global \
            master_key, \
            user_config_file_path, \
            otel_logging, \
            user_custom_auth, \
            user_custom_auth_path, \
            user_custom_key_generate, \
            user_custom_key_update, \
            user_custom_sso, \
            user_custom_ui_sso_sign_in_handler, \
            use_background_health_checks, \
            use_shared_health_check, \
            health_check_interval, \
            health_check_concurrency, \
            use_queue, \
            proxy_budget_rescheduler_max_time, \
            proxy_budget_rescheduler_min_time, \
            ui_access_mode, \
            litellm_master_key_hash, \
            proxy_batch_write_at, \
            disable_spend_logs, \
            prompt_injection_detection_obj, \
            redis_usage_cache, \
            store_model_in_db, \
            premium_user, \
            open_telemetry_logger, \
            health_check_details, \
            proxy_batch_polling_interval, \
            proxy_config_reload_interval_seconds, \
            config_passthrough_endpoints

        config: Final[dict] = await self.get_config(config_file_path=config_file_path)

        self._warn_on_misplaced_jwt_keys(config=config)

        self._load_environment_variables(config=config)

        ## Coordination Redis (before cache init, so the explicit block wins)
        coordination_redis_cache: Final = self._init_coordination_redis(config=config)
        if coordination_redis_cache is not None:
            _set_redis_usage_cache(coordination_redis_cache)

        ## Callback settings
        callback_settings: Final = config.get("callback_settings", {})
        if callback_settings:
            litellm.callback_settings = callback_settings

        ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
        litellm_settings = config.get("litellm_settings", None)
        if litellm_settings is None:
            litellm_settings = {}
        if litellm_settings:
            # Prometheus collectors have fixed label schemas. Load and validate this
            # setting before processing callbacks so YAML key order cannot construct
            # the collectors with the default caller-identity mode, and so an invalid
            # value fails the boot instead of being swallowed by callback init.
            from litellm.types.integrations.prometheus import validate_caller_identity_settings

            validate_caller_identity_settings(litellm_settings)

            # ANSI escape code for blue text
            blue_color_code: Final = "\033[94m"
            reset_color_code: Final = "\033[0m"
            for key, value in litellm_settings.items():
                if key == "cache" and value is True:
                    print(f"{blue_color_code}\nSetting Cache on Proxy")  # noqa: T201
                    from litellm.caching.caching import Cache

                    cache_params = {}
                    if "cache_params" in litellm_settings:
                        cache_params_in_config = litellm_settings["cache_params"]
                        # overwrite cache_params with cache_params_in_config
                        cache_params.update(cache_params_in_config)

                    cache_type = cache_params.get("type", "redis")

                    verbose_proxy_logger.debug("passed cache type=%s", cache_type)

                    if (cache_type == "redis" or cache_type == "redis-semantic") and len(cache_params.keys()) == 0:
                        cache_host = get_secret("REDIS_HOST", None)
                        cache_port = get_secret("REDIS_PORT", None)
                        cache_password = None
                        cache_params.update(
                            {
                                "type": cache_type,
                                "host": cache_host,
                                "port": cache_port,
                            }
                        )

                        if get_secret("REDIS_PASSWORD", None) is not None:
                            cache_password = get_secret("REDIS_PASSWORD", None)
                            cache_params.update(
                                {
                                    "password": cache_password,
                                }
                            )

                        # Assuming cache_type, cache_host, cache_port, and cache_password are strings
                        verbose_proxy_logger.debug(
                            "%sCache Type:%s %s",
                            blue_color_code,
                            reset_color_code,
                            cache_type,
                        )
                        verbose_proxy_logger.debug(
                            "%sCache Host:%s %s",
                            blue_color_code,
                            reset_color_code,
                            cache_host,
                        )
                        verbose_proxy_logger.debug(
                            "%sCache Port:%s %s",
                            blue_color_code,
                            reset_color_code,
                            cache_port,
                        )
                        verbose_proxy_logger.debug(
                            "%sCache Password:%s %s",
                            blue_color_code,
                            reset_color_code,
                            cache_password,
                        )

                    # users can pass os.environ/ variables on the proxy - we should read them from the env
                    for key, value in cache_params.items():
                        if isinstance(value, str) and value.startswith("os.environ/"):
                            cache_params[key] = get_secret(value)

                    ## to pass a complete url, or set ssl=True, etc. just set it as `os.environ[REDIS_URL] = <your-redis-url>`, _redis.py checks for REDIS specific environment variables
                    _set_redis_usage_cache(
                        self._init_cache(
                            cache_params=cache_params,
                            enable_redis_auth_cache=litellm_settings.get("enable_redis_auth_cache", False) is True,
                        )
                    )
                    if litellm.cache is not None:
                        verbose_proxy_logger.debug("%sSet Cache on LiteLLM Proxy%s", blue_color_code, reset_color_code)
                elif key == "cache" and value is False:
                    pass
                elif key == "guardrails":
                    guardrail_name_config_map = initialize_guardrails(
                        guardrails_config=value,
                        premium_user=premium_user,
                        config_file_path=config_file_path,
                        litellm_settings=litellm_settings,
                    )

                    litellm.guardrail_name_config_map = guardrail_name_config_map

                elif key == "global_prompt_directory":
                    from litellm.integrations.dotprompt import (
                        set_global_prompt_directory,
                    )

                    set_global_prompt_directory(value)
                    verbose_proxy_logger.info(
                        "%sSet Global Prompt Directory on LiteLLM Proxy%s", blue_color_code, reset_color_code
                    )
                elif key == "global_bitbucket_config":
                    from litellm.integrations.bitbucket import (
                        set_global_bitbucket_config,
                    )

                    set_global_bitbucket_config(value)
                    verbose_proxy_logger.info(
                        "%sSet Global BitBucket Config on LiteLLM Proxy%s", blue_color_code, reset_color_code
                    )
                elif key == "global_gitlab_config":
                    from litellm.integrations.gitlab import set_global_gitlab_config

                    set_global_gitlab_config(value)
                    verbose_proxy_logger.info(
                        "%sSet Global Gitlab Config on LiteLLM Proxy%s", blue_color_code, reset_color_code
                    )
                elif key == "priority_reservation_settings":
                    from litellm.types.utils import PriorityReservationSettings

                    litellm.priority_reservation_settings = PriorityReservationSettings(**value)
                elif key == "callbacks":
                    initialize_callbacks_on_proxy(
                        value=value,
                        premium_user=premium_user,
                        config_file_path=config_file_path,
                        litellm_settings=litellm_settings,
                        callback_specific_params=callback_settings,
                    )

                elif key == "model_group_settings":
                    from litellm.types.router import ModelGroupSettings

                    litellm.model_group_settings = ModelGroupSettings(**value)

                elif key == "post_call_rules":
                    litellm.post_call_rules = [get_instance_fn(value=value, config_file_path=config_file_path)]
                    verbose_proxy_logger.debug("litellm.post_call_rules: %s", litellm.post_call_rules)
                elif key == "max_budget":
                    litellm.max_budget = float(value)
                elif key == "max_internal_user_budget":
                    litellm.max_internal_user_budget = float(value)
                elif key == "default_max_internal_user_budget":
                    litellm.default_max_internal_user_budget = float(value)
                    if litellm.max_internal_user_budget is None:
                        litellm.max_internal_user_budget = litellm.default_max_internal_user_budget
                elif key == "default_internal_user_params" and isinstance(value, dict):
                    litellm.default_internal_user_params = (
                        {**value, "max_budget": float(value["max_budget"])}
                        if value.get("max_budget") is not None
                        else value
                    )
                    verbose_proxy_logger.debug(
                        "%s setting litellm.%s=%s%s",
                        blue_color_code,
                        key,
                        _redact_general_setting_value(key, litellm.default_internal_user_params, is_full_admin=False),
                        reset_color_code,
                    )
                elif key == "custom_provider_map":
                    from litellm.utils import custom_llm_setup

                    litellm.custom_provider_map = [
                        {
                            "provider": item["provider"],
                            "custom_handler": get_instance_fn(
                                value=item["custom_handler"],
                                config_file_path=config_file_path,
                            ),
                        }
                        for item in value
                    ]

                    custom_llm_setup()
                elif key == "success_callback":
                    litellm.success_callback = []

                    # initialize success callbacks
                    for callback in value:
                        # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                        if "." in callback:
                            litellm.logging_callback_manager.add_litellm_success_callback(
                                get_instance_fn(
                                    value=callback,
                                    config_file_path=config_file_path,
                                )
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.logging_callback_manager.add_litellm_success_callback(callback)
                            if "prometheus" in callback:
                                from litellm.integrations.prometheus import (
                                    PrometheusLogger,
                                )

                                if PrometheusLogger is not None:
                                    verbose_proxy_logger.debug("mounting metrics endpoint")
                                    PrometheusLogger._mount_metrics_endpoint()
                    print(  # noqa: T201
                        f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}"
                    )
                elif key == "failure_callback":
                    litellm.failure_callback = []

                    # initialize success callbacks
                    for callback in value:
                        # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                        if "." in callback:
                            litellm.logging_callback_manager.add_litellm_failure_callback(
                                get_instance_fn(
                                    value=callback,
                                    config_file_path=config_file_path,
                                )
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.logging_callback_manager.add_litellm_failure_callback(callback)
                    print(  # noqa: T201
                        f"{blue_color_code} Initialized Failure Callbacks - {litellm.failure_callback} {reset_color_code}"
                    )
                elif key == "audit_log_callbacks":
                    from litellm.proxy.management_helpers.audit_logs import (
                        is_audit_logging_enabled,
                        reset_audit_log_callback_cache,
                    )

                    reset_audit_log_callback_cache()
                    litellm.audit_log_callbacks = []

                    for callback in value:
                        if "." in callback:
                            litellm.audit_log_callbacks.append(
                                get_instance_fn(
                                    value=callback,
                                    config_file_path=config_file_path,
                                )
                            )
                        else:
                            litellm.audit_log_callbacks.append(callback)

                    _store_audit_logs = litellm_settings.get("store_audit_logs", litellm.store_audit_logs)
                    if is_audit_logging_enabled(store_audit_logs=_store_audit_logs):
                        print(  # noqa: T201
                            f"{blue_color_code} Initialized Audit Log Callbacks - {litellm.audit_log_callbacks} {reset_color_code}"
                        )
                    else:
                        verbose_proxy_logger.warning(
                            "'audit_log_callbacks' is configured but audit logging is not enabled. "
                            "Audit log callbacks will not fire."
                        )
                elif key == "cache_params":
                    # this is set in the cache branch
                    # see usage here: https://docs.litellm.ai/docs/proxy/caching
                    pass
                elif key == "responses":
                    # Initialize global polling via cache settings
                    global polling_via_cache_enabled, native_background_mode, polling_cache_ttl
                    background_mode = value.get("background_mode", {})
                    polling_via_cache_enabled = background_mode.get("polling_via_cache", False)
                    native_background_mode = background_mode.get("native_background_mode", [])
                    polling_cache_ttl = background_mode.get("ttl", 3600)
                    verbose_proxy_logger.debug(
                        "%s Initialized polling via cache: enabled=%s, native_background_mode=%s, ttl=%s%s",
                        blue_color_code,
                        polling_via_cache_enabled,
                        native_background_mode,
                        polling_cache_ttl,
                        reset_color_code,
                    )
                elif key == "max_ui_session_budget":
                    litellm.max_ui_session_budget = float(value) if value is not None else None
                    verbose_proxy_logger.debug(
                        "%s setting litellm.max_ui_session_budget=%s%s",
                        blue_color_code,
                        litellm.max_ui_session_budget,
                        reset_color_code,
                    )
                elif key == "default_team_settings":
                    for idx, team_setting in enumerate(value):  # run through pydantic validation
                        try:
                            TeamDefaultSettings(**team_setting)
                        except Exception:
                            if isinstance(team_setting, dict):
                                raise Exception(
                                    f"team_id missing from default_team_settings at index={idx}\npassed in value={team_setting.keys()}"
                                )
                            raise Exception(
                                f"team_id missing from default_team_settings at index={idx}\npassed in value={type(team_setting)}"
                            )
                    verbose_proxy_logger.debug(
                        "%s setting litellm.%s=%s%s",
                        blue_color_code,
                        key,
                        _redact_general_setting_value(key, value, is_full_admin=False),
                        reset_color_code,
                    )
                    setattr(litellm, key, value)
                elif key == "upperbound_key_generate_params":
                    if value is not None and isinstance(value, dict):
                        for _k, _v in value.items():
                            if isinstance(_v, str) and _v.startswith("os.environ/"):
                                value[_k] = get_secret(_v)
                        litellm.upperbound_key_generate_params = LiteLLM_UpperboundKeyGenerateParams(**value)
                    else:
                        raise Exception(f"Invalid value set for upperbound_key_generate_params - value={value}")
                elif key == "json_logs" and value is True:
                    litellm.json_logs = True
                    litellm._turn_on_json()
                    verbose_proxy_logger.debug(
                        "%s Enabled JSON logging via config%s", blue_color_code, reset_color_code
                    )
                elif key == "budget_reset_time":
                    from litellm.proxy.common_utils.timezone_utils import (
                        parse_budget_reset_time,
                    )

                    parse_budget_reset_time(value)
                    setattr(litellm, key, value)
                else:
                    verbose_proxy_logger.debug(
                        "%s setting litellm.%s=%s%s",
                        blue_color_code,
                        key,
                        _redact_general_setting_value(key, value, is_full_admin=False),
                        reset_color_code,
                    )
                    setattr(litellm, key, value)
                    if key == "request_timeout":
                        litellm.request_timeout_explicitly_set = True
                    if key in {"s3_audit_callback_params", "s3_callback_params"}:
                        from litellm.integrations.s3_v2 import S3Logger as S3V2Logger
                        from litellm.litellm_core_utils.litellm_logging import (
                            _in_memory_loggers,
                        )
                        from litellm.proxy.management_helpers.audit_logs import (
                            reset_audit_log_callback_cache,
                        )

                        reset_audit_log_callback_cache()
                        _in_memory_loggers[:] = [cb for cb in _in_memory_loggers if not isinstance(cb, S3V2Logger)]

        ## GENERAL SERVER SETTINGS (e.g. master key,..) # do this after initializing litellm, to ensure sentry logging works for proxylogging
        general_settings = config.get("general_settings", {})
        if general_settings is None:
            general_settings = {}
        _bg_hc_model_groups: Final = parse_background_health_check_model_groups(general_settings)
        _enable_hc_routing = False
        _hc_staleness = None
        _hc_ignore_transient = False
        if general_settings:
            # Record which keys were explicitly set in the YAML config file.
            # These keys take precedence over DB-cached values during periodic
            # reloads (see _update_general_settings).
            self._yaml_general_settings_keys = set(general_settings.keys())  # mutable-ok: snapshot of YAML keys at load time  # fmt: skip
            # The VALUES matter for the cleanup bounds, not just which keys were
            # set: clearing one from the dashboard has to fall back to what the
            # YAML declared, and a set of names cannot answer that.
            self._yaml_spend_log_cleanup_bounds = {  # mutable-ok: snapshot of YAML bounds at load time  # fmt: skip
                key: general_settings[key] for key in SPEND_LOG_CLEANUP_BOUND_SETTINGS if key in general_settings
            }

            ### LOAD KEY MANAGEMENT SETTINGS ###
            # The secret manager itself is brought up by get_config(), which runs before the
            # `os.environ/` references in this config were resolved. Re-reading the settings here
            # picks up any of them that were themselves secret-manager backed.
            key_management_settings: Final = general_settings.get("key_management_settings", None)
            if key_management_settings is not None:
                litellm._key_management_settings = KeyManagementSettings(**key_management_settings)

            ### [DEPRECATED] LOAD FROM GOOGLE KMS ### old way of loading from google kms
            use_google_kms: Final = general_settings.get("use_google_kms", False)
            load_google_kms(use_google_kms=use_google_kms)
            ### [DEPRECATED] LOAD FROM AZURE KEY VAULT ### old way of loading from azure secret manager
            use_azure_key_vault: Final = general_settings.get("use_azure_key_vault", False)
            load_from_azure_key_vault(use_azure_key_vault=use_azure_key_vault)
            ### ALERTING ###
            self._load_alerting_settings(general_settings=general_settings)
            ### PLUGINS ###
            register_plugins_from_config(general_settings)
            ### CONNECT TO DATABASE ###
            database_url = general_settings.get("database_url", None)
            if database_url and database_url.startswith("os.environ/"):
                verbose_proxy_logger.debug("Resolving database_url via secret manager")
                database_url = get_secret(database_url)
                verbose_proxy_logger.debug("Resolved database_url from secret manager")
            ### MASTER KEY ###
            master_key = general_settings.get("master_key", get_secret("LITELLM_MASTER_KEY", None))

            if master_key and master_key.startswith("os.environ/"):
                master_key = get_secret(master_key)

            if master_key is not None and isinstance(master_key, str):
                litellm_master_key_hash = hash_token(master_key)
            else:
                verbose_proxy_logger.critical(
                    "LITELLM_MASTER_KEY is not set! All requests will be treated as INTERNAL_USER with no admin access. Set LITELLM_MASTER_KEY for production use."
                )
            ### USER API KEY CACHE TTL (in-memory + Redis when Redis auth sharing is enabled) ###
            user_api_key_cache_ttl: Final = general_settings.get("user_api_key_cache_ttl", None)
            if user_api_key_cache_ttl is not None:
                ttl: Final = float(user_api_key_cache_ttl)
                # Mirror TTL on Redis as well when ``litellm_settings.enable_redis_auth_cache``
                # attaches Redis to ``user_api_key_cache``; otherwise DualCache misses in
                # memory fall back to a key that outlasts ``user_api_key_cache_ttl``.
                user_api_key_cache.update_cache_ttl(
                    default_in_memory_ttl=ttl,
                    default_redis_ttl=ttl,
                )

            ### PKCE MULTI-INSTANCE PREREQUISITE CHECK ###
            # PKCE verifiers are stored in redis_usage_cache when available so they can
            # be read back by any instance (not just the one that started the auth flow).
            use_pkce: Final = os.getenv("GENERIC_CLIENT_USE_PKCE", "false").lower() == "true"
            if use_pkce and redis_usage_cache is None:
                global _pkce_no_redis_warning_emitted
                if not _pkce_no_redis_warning_emitted:
                    _pkce_no_redis_warning_emitted = True
                    verbose_proxy_logger.warning(
                        "GENERIC_CLIENT_USE_PKCE=true but Redis is not configured for LiteLLM caching. "
                        "PKCE verifiers will not be shared across instances — callbacks may land on a "
                        "different pod than the login request and fail silently. "
                        "Configure Redis via the 'cache' section in your proxy config, "
                        "or enable sticky sessions for single-instance deployments. "
                        "Set PKCE_STRICT_CACHE_MISS=true to fail fast with a 401 on cache misses "
                        "instead of continuing without a code_verifier."
                    )

            ### CONTROL PLANE CODE-EXCHANGE PREREQUISITE CHECK ###
            cp_url: Final = general_settings.get("control_plane_url")
            if cp_url and redis_usage_cache is None:
                global _cp_no_redis_warning_emitted
                if not _cp_no_redis_warning_emitted:
                    _cp_no_redis_warning_emitted = True
                    verbose_proxy_logger.warning(
                        "control_plane_url is configured but Redis is not configured for LiteLLM caching. "
                        "Login codes (SSO and /v3/login) will not be shared across instances — "
                        "the /v3/login/exchange call may land on a different pod and fail with 401. "
                        "Configure Redis via the 'cache' section in your proxy config, "
                        "or ensure sticky sessions for single-instance deployments."
                    )

            ### STORE MODEL IN DB ### feature flag for `/model/new`
            store_model_in_db = general_settings.get("store_model_in_db", False)
            if store_model_in_db is None:
                store_model_in_db = False
            general_settings["store_model_in_db"] = store_model_in_db
            ### CUSTOM API KEY AUTH ###
            ## pass filepath
            custom_auth: Final = general_settings.get("custom_auth", None)
            if custom_auth is not None:
                user_custom_auth = get_instance_fn(value=custom_auth, config_file_path=config_file_path)
            warn_once_if_custom_auth_skips_common_checks(
                custom_auth_configured=custom_auth is not None,
                run_common_checks=bool(general_settings.get("custom_auth_run_common_checks", False)),
            )

            custom_key_generate: Final = general_settings.get("custom_key_generate", None)
            if custom_key_generate is not None:
                user_custom_key_generate = get_instance_fn(value=custom_key_generate, config_file_path=config_file_path)

            custom_key_update: Final = general_settings.get("custom_key_update", None)
            if custom_key_update is not None:
                user_custom_key_update = get_instance_fn(value=custom_key_update, config_file_path=config_file_path)

            custom_team_metadata_validate: Final = general_settings.get("custom_team_metadata_validate", None)
            TEAM_METADATA_VALIDATOR_REGISTRY.set(
                get_instance_fn(value=custom_team_metadata_validate, config_file_path=config_file_path)
                if custom_team_metadata_validate is not None
                else None
            )
            TEAM_METADATA_SCHEMA_REGISTRY.set(parse_team_metadata_schema(general_settings.get("team_metadata_schema")))

            custom_sso: Final = general_settings.get("custom_sso", None)
            if custom_sso is not None:
                user_custom_sso = get_instance_fn(value=custom_sso, config_file_path=config_file_path)

            custom_ui_sso_sign_in_handler: Final = general_settings.get("custom_ui_sso_sign_in_handler", None)
            if custom_ui_sso_sign_in_handler is not None:
                user_custom_ui_sso_sign_in_handler = get_instance_fn(
                    value=custom_ui_sso_sign_in_handler,
                    config_file_path=config_file_path,
                )

            if enterprise_proxy_config is not None:
                await enterprise_proxy_config.load_enterprise_config(general_settings)

            ## pass through endpoints
            if general_settings.get("pass_through_endpoints", None) is not None:
                config_passthrough_endpoints = general_settings["pass_through_endpoints"]
                await initialize_pass_through_endpoints(
                    pass_through_endpoints=general_settings["pass_through_endpoints"],
                    config_file_path=config_file_path,
                )

            ## ADMIN UI ACCESS ##
            ui_access_mode = general_settings.get("ui_access_mode", "all")  # can be either ["admin_only" or "all"]
            ### ALLOWED IP ###
            allowed_ips: Final = general_settings.get("allowed_ips", None)
            if allowed_ips is not None and premium_user is False:
                raise ValueError(
                    "allowed_ips is an Enterprise Feature. Please add a valid LITELLM_LICENSE to your envionment."
                )
            ## BUDGET RESCHEDULER ##
            proxy_budget_rescheduler_min_time = general_settings.get(
                "proxy_budget_rescheduler_min_time", proxy_budget_rescheduler_min_time
            )
            proxy_budget_rescheduler_max_time = general_settings.get(
                "proxy_budget_rescheduler_max_time", proxy_budget_rescheduler_max_time
            )
            ## BATCH POLLING INTERVAL ##
            proxy_batch_polling_interval = general_settings.get(
                "proxy_batch_polling_interval", proxy_batch_polling_interval
            )
            ## BATCH WRITER ##
            proxy_batch_write_at = general_settings.get("proxy_batch_write_at", proxy_batch_write_at)
            ## DB CONFIG RELOAD INTERVAL ##
            proxy_config_reload_interval_seconds = general_settings.get(
                "proxy_config_reload_interval_seconds", proxy_config_reload_interval_seconds
            )
            ## DISABLE SPEND LOGS ## - gives a perf improvement
            disable_spend_logs = general_settings.get("disable_spend_logs", disable_spend_logs)
            ### BACKGROUND HEALTH CHECKS ###
            # Enable background health checks
            use_background_health_checks = general_settings.get("background_health_checks", False)
            # Enable shared health check state across pods (requires Redis)
            use_shared_health_check = general_settings.get("use_shared_health_check", False)
            health_check_interval = general_settings.get("health_check_interval", DEFAULT_HEALTH_CHECK_INTERVAL)
            health_check_concurrency = general_settings.get("health_check_concurrency", None)
            health_check_details = general_settings.get("health_check_details", True)
            ### INTERACTIONS API SCHEMA ###
            _use_legacy_interactions_schema: Final = general_settings.get("use_legacy_interactions_schema")
            if _use_legacy_interactions_schema is not None:
                if isinstance(_use_legacy_interactions_schema, str):
                    litellm.use_legacy_interactions_schema = _use_legacy_interactions_schema.lower() == "true"
                else:
                    litellm.use_legacy_interactions_schema = bool(_use_legacy_interactions_schema)
            # Health-check-driven routing (opt-in, passes through to Router later)
            _enable_hc_routing = general_settings.get("enable_health_check_routing", False)
            _hc_staleness = general_settings.get("health_check_staleness_threshold", None)
            _hc_ignore_transient = general_settings.get("health_check_ignore_transient_errors", False)
            verbose_proxy_logger.info(
                "background_health_check_config enabled=%s shared=%s interval_seconds=%s max_concurrency=%s details=%s health_check_routing=%s model_groups=%s",
                use_background_health_checks,
                use_shared_health_check,
                health_check_interval,
                health_check_concurrency,
                health_check_details,
                _enable_hc_routing,
                sorted(_bg_hc_model_groups) if _bg_hc_model_groups is not None else None,
            )

            ### RBAC ###
            rbac_role_permissions: Final = general_settings.get("role_permissions", None)
            if rbac_role_permissions is not None:
                general_settings["role_permissions"] = [  # validate role permissions
                    RoleBasedPermissions(**role_permission) for role_permission in rbac_role_permissions
                ]

            ### SSRF URL VALIDATION SETTINGS ###
            _apply_ssrf_general_settings(general_settings)

            ## check if user has set a premium feature in general_settings
            if general_settings.get("enforced_params") is not None and premium_user is not True:
                raise ValueError("Trying to use `enforced_params`" + CommonProxyErrors.not_premium_user.value)

            # check if litellm_license in general_settings
            if "litellm_license" in general_settings:
                _license_check.license_str = general_settings["litellm_license"]
                premium_user = _license_check.is_premium()

        router_params: Final[dict] = {
            "cache_responses": litellm.cache is not None,  # cache if user passed in cache values
        }
        # Health-check-driven routing params (from general_settings)
        if _enable_hc_routing:
            router_params["enable_health_check_routing"] = True
        if _hc_staleness is not None:
            router_params["health_check_staleness_threshold"] = _hc_staleness
        if _hc_ignore_transient:
            router_params["health_check_ignore_transient_errors"] = True
        if _bg_hc_model_groups is not None:
            router_params["background_health_check_model_groups"] = sorted(_bg_hc_model_groups)
        ## MODEL LIST
        model_list: Final = config.get("model_list", None)
        if model_list:
            router_params["model_list"] = model_list
            print(  # noqa: T201
                "\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m"
            )
            for model in model_list:
                ### LOAD FROM os.environ/ ###
                for k, v in model["litellm_params"].items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        model["litellm_params"][k] = get_secret(v)
                validate_deployment_max_agentic_loops(model)
                pin_complexity_router_model_id(model)
                complexity_router_config = model["litellm_params"].get("complexity_router_config")
                if isinstance(complexity_router_config, dict):
                    resolve_complexity_router_plugins(
                        model_name=model.get("model_name", ""),
                        complexity_router_config=complexity_router_config,
                        config_file_path=config_file_path,
                    )
                print(f"\033[32m    {model.get('model_name', '')}\033[0m")  # noqa: T201
                litellm_model_name = model["litellm_params"]["model"]
                litellm_model_api_base = model["litellm_params"].get("api_base", None)
                if "ollama" in litellm_model_name and litellm_model_api_base is None:
                    run_ollama_serve()

        ## ASSISTANT SETTINGS
        assistants_config: AssistantsTypedDict | None = None
        assistant_settings: Final = config.get("assistant_settings", None)
        if assistant_settings:
            for k, v in assistant_settings["litellm_params"].items():
                if isinstance(v, str) and v.startswith("os.environ/"):
                    _v = v.replace("os.environ/", "")
                    v = os.getenv(_v)
                    assistant_settings["litellm_params"][k] = v
            assistants_config = AssistantsTypedDict(**assistant_settings)

        ## SEARCH TOOLS SETTINGS
        search_tools: Final[list[SearchToolTypedDict] | None] = self.parse_search_tools(config)

        ## SANDBOX TOOLS SETTINGS
        from litellm.sandbox.sandbox_tools import register_sandbox_tools

        register_sandbox_tools(config.get("sandbox_tools") or [])

        ## /fine_tuning/jobs endpoints config
        finetuning_config: Final = config.get("finetune_settings", None)
        set_fine_tuning_config(config=finetuning_config)

        ## /files endpoint config
        files_config: Final = config.get("files_settings", None)
        set_files_config(config=files_config)

        ## default config for vertex ai routes
        default_vertex_config: Final = config.get("default_vertex_config", None)
        passthrough_endpoint_router.set_default_vertex_config(config=default_vertex_config)

        ## ROUTER SETTINGS (e.g. routing_strategy, ...)
        router_settings: Final = config.get("router_settings", None)

        if router_settings and isinstance(router_settings, dict):
            # model list and search_tools already set
            exclude_args: Final = {
                "model_list",
                "search_tools",
            }

            available_args: Final = [x for x in litellm.Router.get_valid_args() if x not in exclude_args]

            for k, v in router_settings.items():
                if k in available_args:
                    if k == "plugins" and isinstance(v, list):
                        v = resolve_routing_plugins(
                            plugin_paths=v,
                            config_file_path=config_file_path,
                            source_label="router_settings.plugins",
                        )
                    router_params[k] = v
                elif k in {"health_check_interval", "health_check_concurrency"}:
                    raise ValueError(
                        f"'{k}' is NOT a valid router_settings parameter. Please move it to 'general_settings'."
                    )
                else:
                    verbose_proxy_logger.warning(
                        "Key '%s' is not a valid argument for Router.__init__(). Ignoring this key.", k
                    )
        router = litellm.Router(
            **router_params,
            assistants_config=assistants_config,
            search_tools=search_tools,
            router_general_settings=RouterGeneralSettings(
                async_only_mode=True  # only init async clients
            ),
            ignore_invalid_deployments=True,  # don't raise an error if a deployment is invalid
        )

        if redis_usage_cache is not None and router.cache.redis_cache is None:
            router._update_redis_cache(cache=redis_usage_cache)

        # Guardrail settings
        guardrails_v2: list[dict] | None = None

        if config is not None:
            guardrails_v2 = config.get("guardrails", None)
        if guardrails_v2:
            init_guardrails_v2(
                all_guardrails=guardrails_v2,
                config_file_path=config_file_path,
                llm_router=router,
            )

        # Policy Engine settings
        await self._init_policy_engine(
            config=config,
            prisma_client=prisma_client,
            llm_router=router,
        )

        ## Prompt settings
        prompts: list[dict] | None = None
        if config is not None:
            prompts = config.get("prompts", None)
        if prompts:
            from litellm.proxy.prompts.init_prompts import init_prompts

            init_prompts(all_prompts=prompts, config_file_path=config_file_path)

        ## CREDENTIALS
        credential_list_dict: Final = self.load_credential_list(config=config)
        litellm.credential_list = credential_list_dict

        ## NON-LLM CONFIGS eg. MCP tools, vector stores, etc.
        await self._init_non_llm_configs(config=config, config_file_path=config_file_path)

        return router, router.get_model_list(), general_settings

    async def _init_non_llm_configs(self, config: dict, config_file_path: str | None = None):
        """
        Initialize non-LLM configs eg. MCP tools, vector stores, etc.
        """
        ## MCP TOOLS
        mcp_tools_config: Final = config.get("mcp_tools", None)
        if mcp_tools_config:
            from litellm.proxy._experimental.mcp_server.tool_registry import (
                global_mcp_tool_registry,
            )

            global_mcp_tool_registry.load_tools_from_config(mcp_tools_config, config_file_path=config_file_path)

        ## AGENTS
        agent_config: Final = config.get("agents", config.get("agent_list", None))
        if agent_config is not None:
            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            global_agent_registry.load_agents_from_config(agent_config)

        mcp_servers_config: Final = config.get("mcp_servers", None)
        if mcp_servers_config:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            # Get mcp_aliases from litellm_settings if available
            litellm_settings: Final = config.get("litellm_settings", {})
            mcp_aliases: Final = litellm_settings.get("mcp_aliases", None)

            await global_mcp_server_manager.load_servers_from_config(mcp_servers_config, mcp_aliases)

        ## VECTOR STORES
        vector_store_registry_config: Final = config.get("vector_store_registry", None)
        if vector_store_registry_config:
            from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

            if litellm.vector_store_registry is None:
                litellm.vector_store_registry = VectorStoreRegistry()

            # Load vector stores from config
            litellm.vector_store_registry.load_vector_stores_from_config(vector_store_registry_config)

        ## WORKER REGISTRY (Global Control Plane)
        worker_registry_config: Final = config.get("worker_registry", None)
        if worker_registry_config:
            if premium_user is not True:
                raise ValueError("Trying to use `worker_registry`" + CommonProxyErrors.not_premium_user.value)
            self.worker_registry = [WorkerRegistryEntry(**e) for e in worker_registry_config]
        else:
            self.worker_registry = []

    async def _init_policy_engine(
        self,
        config: dict | None,
        prisma_client: Optional["PrismaClient"],
        llm_router: Optional["Router"],
    ):
        """
        Initialize the policy engine from config.

        Args:
            config: The proxy configuration dictionary
            prisma_client: Optional Prisma client for DB validation
            llm_router: Optional LLM router for model validation
        """

        from litellm.proxy.policy_engine.init_policies import init_policies
        from litellm.proxy.policy_engine.policy_validator import PolicyValidator

        if config is None:
            verbose_proxy_logger.debug("Policy engine: config is None, skipping")
            return

        policies_config: Final = config.get("policies", None)
        if not policies_config:
            verbose_proxy_logger.debug("Policy engine: no policies in config, skipping")
            return

        policy_attachments_config: Final = config.get("policy_attachments", None)

        verbose_proxy_logger.info("Policy engine: found %s policies in config", len(policies_config))

        # Initialize policies
        await init_policies(
            policies_config=policies_config,
            policy_attachments_config=policy_attachments_config,
            prisma_client=prisma_client,
            validate_db=prisma_client is not None,
            fail_on_error=True,
        )

    def _load_alerting_settings(self, general_settings: dict):
        """
        Initialize alerting settings
        """

        _alerting_callbacks: Final = general_settings.get("alerting", None)
        verbose_proxy_logger.debug("_alerting_callbacks: %s", _alerting_callbacks)
        if _alerting_callbacks is None:
            return

        # Ensure proxy_logging_obj.alerting is set for all alerting types
        _alerting_value: Final = general_settings.get("alerting", None)
        verbose_proxy_logger.debug("_load_alerting_settings: Calling update_values with alerting=%s", _alerting_value)
        proxy_logging_obj.update_values(
            alerting=_alerting_value,
            alerting_threshold=general_settings.get("alerting_threshold", 600),
            alert_types=general_settings.get("alert_types", None),
            alert_to_webhook_url=general_settings.get("alert_to_webhook_url", None),
            alerting_args=general_settings.get("alerting_args", None),
            alert_type_config=general_settings.get("alert_type_config", None),
            redis_cache=redis_usage_cache,
        )

        for _alert in _alerting_callbacks:
            if _alert == "slack":
                # [OLD] v0 implementation - already handled by update_values above
                pass
            else:
                # [NEW] v1 implementation - init as a custom logger
                if _alert in litellm._known_custom_logger_compatible_callbacks:
                    _logger = _init_custom_logger_compatible_class(
                        logging_integration=_alert,
                        internal_usage_cache=None,
                        llm_router=None,
                        custom_logger_init_args={"alerting_args": general_settings.get("alerting_args", None)},
                    )
                    if _logger is not None:
                        litellm.logging_callback_manager.add_litellm_callback(_logger)

    def initialize_secret_manager(
        self,
        key_management_system: str | None,
        config_file_path: str | None = None,
    ):
        """
        Initialize the relevant secret manager if `key_management_system` is provided
        """
        if key_management_system is not None:
            if key_management_system == KeyManagementSystem.AZURE_KEY_VAULT.value:
                ### LOAD FROM AZURE KEY VAULT ###
                load_from_azure_key_vault(use_azure_key_vault=True)
            elif key_management_system == KeyManagementSystem.GOOGLE_KMS.value:
                ### LOAD FROM GOOGLE KMS ###
                load_google_kms(use_google_kms=True)
            elif (
                key_management_system == KeyManagementSystem.AWS_SECRET_MANAGER.value  # noqa: F405
            ):
                from litellm.secret_managers.aws_secret_manager_v2 import (
                    AWSSecretsManagerV2,
                )

                AWSSecretsManagerV2.load_aws_secret_manager(
                    use_aws_secret_manager=True,
                    key_management_settings=litellm._key_management_settings,
                )
            elif key_management_system == KeyManagementSystem.AWS_KMS.value:
                load_aws_kms(use_aws_kms=True)
            elif key_management_system == KeyManagementSystem.GOOGLE_SECRET_MANAGER.value:
                from litellm.secret_managers.google_secret_manager import (
                    GoogleSecretManager,
                )

                GoogleSecretManager()
            elif key_management_system == KeyManagementSystem.HASHICORP_VAULT.value:
                from litellm.secret_managers.hashicorp_secret_manager import (
                    HashicorpSecretManager,
                )

                HashicorpSecretManager()
            elif key_management_system == KeyManagementSystem.CYBERARK.value:
                from litellm.secret_managers.cyberark_secret_manager import (
                    CyberArkSecretManager,
                )

                CyberArkSecretManager()
            elif key_management_system == KeyManagementSystem.CUSTOM.value:
                ### LOAD CUSTOM SECRET MANAGER ###
                from litellm.secret_managers.custom_secret_manager_loader import (
                    load_custom_secret_manager,
                )

                load_custom_secret_manager(config_file_path=config_file_path)
            else:
                raise ValueError("Invalid Key Management System selected")

    def get_model_info_with_id(self, model, db_model=False) -> RouterModelInfo:
        """
        Common logic across add + delete router models
        Parameters:
        - deployment
        - db_model -> flag for differentiating model stored in db vs. config -> used on UI

        Return model info w/ id
        """
        _id: Final[str | None] = getattr(model, "model_id", None)
        if _id is not None:
            model.model_info["id"] = _id
            model.model_info["db_model"] = True
            model.model_info["blocked"] = bool(getattr(model, "blocked", False))

        if premium_user is True:
            # seeing "created_at", "updated_at", "created_by", "updated_by" is a LiteLLM Enterprise Feature
            model.model_info["created_at"] = getattr(model, "created_at", None)
            model.model_info["updated_at"] = getattr(model, "updated_at", None)
            model.model_info["created_by"] = getattr(model, "created_by", None)
            model.model_info["updated_by"] = getattr(model, "updated_by", None)

        if model.model_info is not None and isinstance(model.model_info, dict):
            if "id" not in model.model_info:
                model.model_info["id"] = model.model_id
            if "db_model" in model.model_info and model.model_info["db_model"] is False:
                model.model_info["db_model"] = db_model
            _model_info = RouterModelInfo(**model.model_info)

        else:
            _model_info = RouterModelInfo(id=model.model_id, db_model=db_model)
        return _model_info

    async def _delete_deployment(self, db_models: list) -> frozenset[str] | None:
        """
        (Helper function of add deployment) -> combined to reduce prisma db calls

        - Create all up list of model id's (db + config)
        - Compare all up list to router model id's
        - Remove any that are missing

        Return:
        - frozenset[str] - the ids the db + config say should be served after this
          reconcile, so a caller can tell an id this evicted on purpose from one that
          went missing. None when no reconcile ran and that set is therefore unknown.
        """
        global user_config_file_path, llm_router
        combined_id_list: Final = []

        ## BASE CASES ##
        if llm_router is None:
            return None
        # NOTE: db_models may be legitimately empty when all DB models have been deleted.
        # Do NOT short-circuit on len(db_models) == 0 — we must still evict any
        # DB-sourced deployments that are no longer in the DB. The caller
        # (_update_llm_router) already guards against None (transient fetch failure).

        ## DB MODELS ##
        for m in db_models:
            model_info = self.get_model_info_with_id(model=m)
            if model_info.id is not None:
                combined_id_list.append(model_info.id)

        ## CONFIG MODELS ##
        try:
            config: Final = await self.get_config(config_file_path=user_config_file_path)
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to load config in _delete_deployment: %s. "
                "Skipping deployment cleanup to avoid removing valid models.",
                str(e),
            )
            return None
        model_list: Final = config.get("model_list", None)
        if model_list:
            for model in model_list:
                ### LOAD FROM os.environ/ ###
                for k, v in model["litellm_params"].items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        model["litellm_params"][k] = get_secret(v)

                ## check if they have model-id's ##
                model_id = model.get("model_info", {}).get("id", None)
                if model_id is None:
                    ## else - generate stable id's ##
                    model_id = llm_router.generate_model_id(
                        model_group=model["model_name"],
                        litellm_params=model["litellm_params"],
                    )
                else:
                    model_id = str(model_id)
                combined_id_list.append(model_id)  # ADD CONFIG MODEL TO COMBINED LIST

        router_model_ids: Final = llm_router.get_model_ids()
        # Check for model IDs in llm_router not present in combined_id_list and delete them

        for model_id in router_model_ids:
            if model_id not in combined_id_list:
                llm_router.delete_deployment(id=model_id)
        return frozenset(combined_id_list)

    def _resolve_db_litellm_param(self, key: str, value: object) -> object:
        if not isinstance(value, str):
            return value

        decrypted_value: Final = decrypt_value_helper(value=value, key=key, return_original_value=True)
        if isinstance(decrypted_value, str) and decrypted_value.startswith("os.environ/"):
            return get_secret(decrypted_value)
        return decrypted_value

    def _add_deployment(self, db_models: list) -> int:
        """
        Iterate through db models

        for any not in router - add them.

        Return - number of deployments added
        """
        import base64

        if llm_router is None:
            return 0

        added_models = 0
        ## ADD MODEL LOGIC
        for m in db_models:
            _litellm_params = m.litellm_params
            if isinstance(_litellm_params, dict):
                # decrypt values
                for k, v in _litellm_params.items():
                    _litellm_params[k] = self._resolve_db_litellm_param(key=k, value=v)
                _litellm_params = LiteLLM_Params.model_validate(_litellm_params)

            else:
                verbose_proxy_logger.error(
                    "Invalid model added to proxy db. Invalid litellm params. litellm_params=%s", _litellm_params
                )
                continue  # skip to next model
            _model_info = self.get_model_info_with_id(model=m, db_model=True)  ## 👈 FLAG = True for db_models

            added = llm_router.upsert_deployment(
                deployment=Deployment(
                    model_name=m.model_name,
                    litellm_params=_litellm_params,
                    model_info=_model_info,
                )
            )

            if added is not None:
                added_models += 1
        return added_models

    def decrypt_model_list_from_db(self, new_models: list) -> list:
        _model_list: Final[list] = []
        for m in new_models:
            _litellm_params = m.litellm_params
            if isinstance(_litellm_params, BaseModel):
                _litellm_params = _litellm_params.model_dump()
            if isinstance(_litellm_params, dict):
                # decrypt values
                for k, v in _litellm_params.items():
                    _litellm_params[k] = self._resolve_db_litellm_param(key=k, value=v)
                _litellm_params = LiteLLM_Params.model_validate(_litellm_params)
            else:
                verbose_proxy_logger.error(
                    "Invalid model added to proxy db. Invalid litellm params. litellm_params=%s", _litellm_params
                )
                continue  # skip to next model

            _model_info = self.get_model_info_with_id(model=m)
            _model_list.append(
                Deployment(
                    model_name=m.model_name,
                    litellm_params=_litellm_params,
                    model_info=_model_info,
                ).to_json(exclude_none=True)
            )

        return _model_list

    async def _update_llm_router(
        self,
        new_models: Json | None,
        proxy_logging_obj: ProxyLogging,
    ) -> frozenset[str] | None:
        global llm_router, llm_model_list, master_key, general_settings

        still_desired_ids: frozenset[str] | None = None

        # Load config separately so a timeout here doesn't block model loading
        config_data: dict = {}
        search_tools = None
        try:
            config_data = await proxy_config.get_config()
            search_tools = self.parse_search_tools(config_data)
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to load config in _update_llm_router: %s. "
                "Proceeding with model loading using cached/empty config.",
                str(e),
            )

        try:
            # new_models is None when _get_models_from_db failed (transient DB error).
            # Skip the update entirely so we don't evict valid deployments.
            if new_models is None:
                verbose_proxy_logger.warning(
                    "_update_llm_router: DB model fetch returned None (transient failure). "
                    "Skipping router update to preserve existing deployments."
                )
                return

            models_list: Final[list] = new_models if isinstance(new_models, list) else []
            if llm_router is None and master_key is not None:
                verbose_proxy_logger.debug("len new_models: %s", len(models_list))

                _model_list: Final[list] = self.decrypt_model_list_from_db(new_models=models_list)
                # Only create router if we have models or search_tools to route
                # Router can function with model_list=[] if search_tools are configured
                if len(_model_list) > 0 or search_tools:
                    verbose_proxy_logger.debug("_model_list: %s", _model_list)
                    llm_router = litellm.Router(
                        model_list=_model_list,
                        router_general_settings=RouterGeneralSettings(
                            async_only_mode=True  # only init async clients
                        ),
                        search_tools=search_tools,
                        ignore_invalid_deployments=True,
                    )
                    verbose_proxy_logger.debug("updated llm_router: %s", llm_router)
            else:
                verbose_proxy_logger.debug("len new_models: %s", len(models_list))
                if search_tools is not None and llm_router is not None:
                    llm_router.search_tools = search_tools
                ## DELETE MODEL LOGIC
                still_desired_ids = await self._delete_deployment(db_models=models_list)

                ## ADD MODEL LOGIC
                self._add_deployment(db_models=models_list)

        except Exception as e:
            verbose_proxy_logger.exception("Error adding/deleting model to llm_router: %s", e)

        if llm_router is not None:
            llm_model_list = llm_router.get_model_list()

        # check if user set any callbacks in Config Table
        self._add_callbacks_from_db_config(config_data)

        # router settings
        await self._add_router_settings_from_db_config(
            config_data=config_data, llm_router=llm_router, prisma_client=prisma_client
        )

        # general settings
        self._add_general_settings_from_db_config(
            config_data=config_data,
            general_settings=general_settings,
            proxy_logging_obj=proxy_logging_obj,
        )

        return still_desired_ids

    def _add_callback_from_db_to_in_memory_litellm_callbacks(
        self,
        callback: str,
        event_types: list[Literal["success", "failure"]],
        existing_callbacks: list,
    ) -> None:
        """
        Helper method to add a single callback to litellm for specified event types.

        Args:
            callback: The callback name to add
            event_types: List of event types (e.g., ["success"], ["failure"], or ["success", "failure"])
            existing_callbacks: The existing callback list to check against
        """
        if callback in litellm._known_custom_logger_compatible_callbacks:
            for event_type in event_types:
                _add_custom_logger_callback_to_specific_event(callback, event_type)
        elif callback not in existing_callbacks:
            if event_types == ["success"]:
                litellm.logging_callback_manager.add_litellm_success_callback(callback)
            elif event_types == ["failure"]:
                litellm.logging_callback_manager.add_litellm_failure_callback(callback)
            else:  # Both success and failure
                litellm.logging_callback_manager.add_litellm_callback(callback)

    def _add_callbacks_from_db_config(self, config_data: dict) -> None:
        """
        Adds callbacks from DB config to litellm
        """
        litellm_settings: Final = config_data.get("litellm_settings", {}) or {}
        success_callbacks: Final = litellm_settings.get("success_callback", None)
        failure_callbacks: Final = litellm_settings.get("failure_callback", None)
        callbacks: Final = litellm_settings.get("callbacks", None)

        if success_callbacks is not None and isinstance(success_callbacks, list):
            for success_callback in success_callbacks:
                self._add_callback_from_db_to_in_memory_litellm_callbacks(
                    callback=success_callback,
                    event_types=["success"],
                    existing_callbacks=litellm.success_callback,
                )

        if failure_callbacks is not None and isinstance(failure_callbacks, list):
            for failure_callback in failure_callbacks:
                self._add_callback_from_db_to_in_memory_litellm_callbacks(
                    callback=failure_callback,
                    event_types=["failure"],
                    existing_callbacks=litellm.failure_callback,
                )

        if callbacks is not None and isinstance(callbacks, list):
            for callback in callbacks:
                self._add_callback_from_db_to_in_memory_litellm_callbacks(
                    callback=callback,
                    event_types=["success", "failure"],
                    existing_callbacks=litellm.callbacks,
                )

    def _encrypt_env_variables(self, environment_variables: dict, new_encryption_key: str | None = None) -> dict:
        """
        Encrypts a dictionary of environment variables and returns them.
        """
        encrypted_env_vars: Final = {}
        for k, v in environment_variables.items():
            encrypted_value = encrypt_value_helper(value=v, new_encryption_key=new_encryption_key)
            encrypted_env_vars[k] = encrypted_value
        return encrypted_env_vars

    def _decrypt_and_set_db_env_variables(
        self, environment_variables: dict, return_original_value: bool = False
    ) -> dict:
        """
        Decrypts a dictionary of environment variables and then sets them in the environment

        Args:
            environment_variables: dict - dictionary of environment variables to decrypt and set
            eg. `{"LANGFUSE_PUBLIC_KEY": "kFiKa1VZukMmD8RB6WXB9F......."}`
        """
        decrypted_env_vars: Final = {}
        for k, v in environment_variables.items():
            try:
                decrypted_value = decrypt_value_helper(value=v, key=k, return_original_value=return_original_value)
                if decrypted_value is not None:
                    os.environ[k] = decrypted_value
                    decrypted_env_vars[k] = decrypted_value
            except Exception as e:
                verbose_proxy_logger.error("Error setting env variable: %s - %s", k, str(e))
        return decrypted_env_vars

    def _decrypt_db_variables(self, variables_dict: dict) -> dict:
        """
        Decrypts a dictionary of variables and returns them.
        """
        decrypted_variables: Final = {}
        for k, v in variables_dict.items():
            decrypted_value = decrypt_value_helper(value=v, key=k, return_original_value=True)
            decrypted_variables[k] = decrypted_value
        return decrypted_variables

    def _encrypt_env_variables_for_db(self, environment_variables: dict, new_encryption_key: str | None = None) -> dict:
        """
        Idempotently encrypt environment variables for a DB write.

        Config writers may pass either plaintext (first write) or values that
        are already ciphertext — e.g. the Admin UI reads config back via
        /get/config/callbacks (which returns the stored, still-encrypted
        value) and re-POSTs it on the next save. Decrypt first so an
        already-encrypted value is not stacked with a second encryption
        layer, then encrypt exactly once.

        Decryption here deliberately uses _decrypt_db_variables (not
        _decrypt_and_set_db_env_variables): this is a write path, and
        loading values into os.environ is the read path's responsibility.
        """
        decrypted_env_vars: Final = self._decrypt_db_variables(environment_variables)
        return self._encrypt_env_variables(
            environment_variables=decrypted_env_vars,
            new_encryption_key=new_encryption_key,
        )

    @staticmethod
    def _parse_router_settings_value(value: object) -> dict | None:
        """
        Parse a router_settings value that may be a dict or a JSON/YAML string.

        Returns a non-empty dict if valid, otherwise None.
        """
        if value is None:
            return None

        parsed: dict | None = None
        if isinstance(value, dict):
            parsed = value
        elif isinstance(value, str):
            import json

            try:
                parsed = yaml.safe_load(value)
            except (yaml.YAMLError, json.JSONDecodeError):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    pass

        if isinstance(parsed, dict) and parsed:
            return parsed
        return None

    async def _get_hierarchical_router_settings(
        self,
        user_api_key_dict: Optional["UserAPIKeyAuth"],
        prisma_client: PrismaClient | None,
        proxy_logging_obj: Optional["ProxyLogging"] = None,
    ) -> dict | None:
        """
        Get router_settings in priority order: Key > Team

        Uses the already-cached key object and the cached team lookup
        (get_team_object) to avoid direct DB queries on the hot path.

        Global router_settings are NOT looked up here — they are already
        applied to the Router object at config-load / DB-sync time.

        Returns:
            dict: router_settings, or None if no settings found
        """
        # 1. Try key-level router_settings
        # user_api_key_dict is already the cached/authenticated key object —
        # no DB call needed.
        if user_api_key_dict is not None:
            key_settings: Final = self._parse_router_settings_value(getattr(user_api_key_dict, "router_settings", None))
            if key_settings is not None:
                return key_settings

        # 2. Try team-level router_settings using cached team lookup
        # get_team_object checks in-memory cache / Redis first, only falls
        # back to DB on a cache miss.
        if user_api_key_dict is not None and user_api_key_dict.team_id is not None:
            try:
                team_obj: Final = await get_team_object(
                    team_id=user_api_key_dict.team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                team_settings: Final = self._parse_router_settings_value(getattr(team_obj, "router_settings", None))
                if team_settings is not None:
                    return team_settings
            except Exception:
                # If team lookup fails, no team-level settings available
                pass

        return None

    async def _add_router_settings_from_db_config(
        self,
        config_data: dict,
        llm_router: Router | None,
        prisma_client: PrismaClient | None,
    ) -> None:
        """
        Adds router settings from DB config to litellm proxy

        1. Get router settings from DB
        2. Get router settings from config
        3. Combine both
        4. Update router settings
        """
        if llm_router is not None and prisma_client is not None:
            db_router_settings: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
                where={"param_name": "router_settings"}
            )

            config_router_settings: Final = config_data.get("router_settings", {})

            combined_router_settings = {}
            if (
                config_router_settings is not None
                and isinstance(config_router_settings, dict)
                and db_router_settings is not None
                and isinstance(db_router_settings.param_value, dict)
            ):
                from litellm.utils import _update_dictionary

                db_overlay_deferring_empty_lists_to_config: Final = {
                    k: v
                    for k, v in db_router_settings.param_value.items()
                    if not (k in config_router_settings and isinstance(v, list) and len(v) == 0)
                }
                combined_router_settings = _update_dictionary(
                    config_router_settings, db_overlay_deferring_empty_lists_to_config
                )
            elif config_router_settings is not None and isinstance(config_router_settings, dict):
                combined_router_settings = config_router_settings
            elif db_router_settings is not None and isinstance(db_router_settings.param_value, dict):
                combined_router_settings = db_router_settings.param_value

            if combined_router_settings:
                llm_router.update_settings(**combined_router_settings)

    def _add_general_settings_from_db_config(
        self, config_data: dict, general_settings: dict, proxy_logging_obj: ProxyLogging
    ) -> None:
        """
        Adds general settings from DB config to litellm proxy

        Args:
            config_data: dict
            general_settings: dict - global general_settings currently in use
            proxy_logging_obj: ProxyLogging
        """
        _general_settings: Final = config_data.get("general_settings", {})

        if _general_settings is not None and "alerting" in _general_settings:
            if (
                general_settings is not None
                and general_settings.get("alerting", None) is not None
                and isinstance(general_settings["alerting"], list)
                and _general_settings.get("alerting", None) is not None
                and isinstance(_general_settings["alerting"], list)
            ):
                # Merge DB and YAML/config alerting values instead of overriding
                _yaml_alerting: Final = set(general_settings["alerting"])
                _db_alerting: Final = set(_general_settings["alerting"])
                _merged_alerting = list(_yaml_alerting.union(_db_alerting))
                # Preserve order: YAML values first, then DB values
                _merged_alerting = list(general_settings["alerting"]) + [
                    item for item in _general_settings["alerting"] if item not in general_settings["alerting"]
                ]
                verbose_proxy_logger.debug(
                    "Merging alerting values: YAML=%s, DB=%s, Merged=%s",
                    general_settings["alerting"],
                    _general_settings["alerting"],
                    _merged_alerting,
                )
                general_settings["alerting"] = _merged_alerting
                # Use update_values to properly set alerting for both slack and email
                proxy_logging_obj.update_values(
                    alerting=general_settings["alerting"],
                )
            elif general_settings is None:
                general_settings = {}
                general_settings["alerting"] = _general_settings["alerting"]
                # Use update_values to properly set alerting for both slack and email
                proxy_logging_obj.update_values(
                    alerting=general_settings["alerting"],
                )
            elif isinstance(general_settings, dict):
                general_settings["alerting"] = _general_settings["alerting"]
                # Use update_values to properly set alerting for both slack and email
                proxy_logging_obj.update_values(
                    alerting=general_settings["alerting"],
                )

        if _general_settings is not None and "alert_types" in _general_settings:
            general_settings["alert_types"] = _general_settings["alert_types"]
            proxy_logging_obj.alert_types = general_settings["alert_types"]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alert_types=general_settings["alert_types"], llm_router=llm_router
            )

        if _general_settings is not None and "alert_to_webhook_url" in _general_settings:
            general_settings["alert_to_webhook_url"] = _general_settings["alert_to_webhook_url"]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alert_to_webhook_url=general_settings["alert_to_webhook_url"],
                llm_router=llm_router,
            )

        if _general_settings is not None and "plugins" in _general_settings:
            general_settings["plugins"] = _general_settings["plugins"]
            register_plugins_from_config(general_settings)

    async def _reschedule_spend_log_cleanup_job(self):
        """
        Reschedule the spend log cleanup job based on current general_settings.
        This is called when maximum_spend_logs_retention_period is updated dynamically.
        If the retention period is None, the job will be removed.
        """
        global scheduler, general_settings, prisma_client
        if scheduler is None:
            return

        # Remove existing job if it exists
        try:
            scheduler.remove_job("spend_log_cleanup_job")
            verbose_proxy_logger.info("Removed existing spend log cleanup job")
        except Exception:
            pass  # Job might not exist, which is fine

        # Schedule new job if retention period is set (not None)
        retention_period: Final = general_settings.get("maximum_spend_logs_retention_period")
        autorouter_retention: Final = general_settings.get("maximum_autorouter_session_retention_period")
        health_check_retention: Final = general_settings.get("maximum_health_check_retention_period")
        if retention_period is not None or autorouter_retention is not None or health_check_retention is not None:
            from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import (
                SpendLogCleanup,
            )

            spend_log_cleanup: Final = SpendLogCleanup()
            cleanup_cron: Final = general_settings.get("maximum_spend_logs_cleanup_cron")

            if cleanup_cron:
                from apscheduler.triggers.cron import CronTrigger

                try:
                    cron_trigger: Final = CronTrigger.from_crontab(cleanup_cron)
                    scheduler.add_job(
                        spend_log_cleanup.cleanup_old_spend_logs,
                        cron_trigger,
                        args=[prisma_client],
                        id="spend_log_cleanup_job",
                        replace_existing=True,
                        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                    )
                    verbose_proxy_logger.info("Spend log cleanup rescheduled with cron: %s", cleanup_cron)
                except ValueError:
                    verbose_proxy_logger.error("Invalid maximum_spend_logs_cleanup_cron value: %s", cleanup_cron)
            else:
                # Interval-based scheduling (existing behavior)
                from litellm.litellm_core_utils.duration_parser import (
                    duration_in_seconds,
                )

                retention_interval: Final = general_settings.get("maximum_spend_logs_retention_interval", "1d")
                try:
                    interval_seconds: Final = duration_in_seconds(retention_interval)
                    # this runs against a started scheduler, which the startup stagger sweep
                    # cannot reach, so the offset is applied here or the job reconverges across
                    # replicas the first time an admin edits the retention settings
                    scheduler.add_job(
                        spend_log_cleanup.cleanup_old_spend_logs,
                        stagger_trigger(
                            job_id="spend_log_cleanup_job",
                            trigger=IntervalTrigger(seconds=interval_seconds),
                            period_seconds=interval_seconds,
                            settings=parse_stagger_settings(general_settings),
                        ),
                        args=[prisma_client],
                        id="spend_log_cleanup_job",
                        replace_existing=True,
                        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                    )
                    verbose_proxy_logger.info("Spend log cleanup rescheduled with interval: %s", retention_interval)
                except ValueError:
                    verbose_proxy_logger.error("Invalid maximum_spend_logs_retention_interval value")

    async def _update_general_settings(self, db_general_settings: Json | None):
        """
        Pull from DB, read general settings value
        """
        global general_settings, store_model_in_db
        if db_general_settings is None:
            return
        _general_settings: Final = dict(db_general_settings)
        ## MAX PARALLEL REQUESTS ##
        if "max_parallel_requests" in _general_settings:
            general_settings["max_parallel_requests"] = _general_settings["max_parallel_requests"]

        if "global_max_parallel_requests" in _general_settings:
            general_settings["global_max_parallel_requests"] = _general_settings["global_max_parallel_requests"]

        if "max_batch_file_size_mb" not in self._yaml_general_settings_keys:
            general_settings["max_batch_file_size_mb"] = _general_settings.get("max_batch_file_size_mb")

        ## ALERTING ARGS ##
        if "alerting_args" in _general_settings:
            general_settings["alerting_args"] = _general_settings["alerting_args"]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alerting_args=general_settings["alerting_args"],
            )

        ## PASS-THROUGH ENDPOINTS ##
        if "pass_through_endpoints" in _general_settings:
            general_settings["pass_through_endpoints"] = _general_settings["pass_through_endpoints"]
            await initialize_pass_through_endpoints(pass_through_endpoints=general_settings["pass_through_endpoints"])

        ## UI ACCESS MODE ##
        if "ui_access_mode" in _general_settings:
            general_settings["ui_access_mode"] = _general_settings["ui_access_mode"]

        ## STORE PROMPTS IN SPEND LOGS ##
        if "store_prompts_in_spend_logs" in _general_settings:
            # If the YAML config explicitly set this key, prefer the YAML value
            # over the DB-cached value. This ensures config changes deployed via
            # CI/CD take effect without requiring a manual /config/update call.
            # When YAML does not set this key, the DB value is used (preserving
            # admin UI runtime changes).
            if "store_prompts_in_spend_logs" in self._yaml_general_settings_keys:
                value = general_settings.get("store_prompts_in_spend_logs")
            else:
                value = _general_settings["store_prompts_in_spend_logs"]
            # Normalize case: handle True/true/TRUE, False/false/FALSE, None/null
            if value is None:
                general_settings["store_prompts_in_spend_logs"] = None
            elif isinstance(value, bool):
                general_settings["store_prompts_in_spend_logs"] = value
            elif isinstance(value, str):
                # Case-insensitive string comparison
                general_settings["store_prompts_in_spend_logs"] = value.lower() == "true"
            else:
                # For other types, convert to bool
                general_settings["store_prompts_in_spend_logs"] = bool(value)

        if "disable_auto_add_proxy_admin_to_teams" in _general_settings:
            value = _general_settings["disable_auto_add_proxy_admin_to_teams"]
            if isinstance(value, str):
                general_settings["disable_auto_add_proxy_admin_to_teams"] = value.lower() == "true"
            else:
                general_settings["disable_auto_add_proxy_admin_to_teams"] = value if value is None else bool(value)

        if "apply_user_budget_to_team_keys" in _general_settings and (
            "apply_user_budget_to_team_keys" not in self._yaml_general_settings_keys
        ):
            db_value: Final = _general_settings["apply_user_budget_to_team_keys"]
            if isinstance(db_value, str):
                general_settings["apply_user_budget_to_team_keys"] = db_value.lower() == "true"
            else:
                general_settings["apply_user_budget_to_team_keys"] = db_value if db_value is None else bool(db_value)

        ## STORE MODEL IN DB ##
        if "store_model_in_db" in _general_settings:
            value = _general_settings["store_model_in_db"]
            if value is None:
                pass  # Don't change store_model_in_db to None; keep current value
            elif isinstance(value, bool):
                store_model_in_db = value
            elif isinstance(value, str):
                store_model_in_db = value.lower() == "true"
            else:
                store_model_in_db = bool(value)
            general_settings["store_model_in_db"] = store_model_in_db

        ## MAXIMUM SPEND LOGS RETENTION PERIOD ##
        if "maximum_spend_logs_retention_period" in _general_settings:
            old_value: Final = general_settings.get("maximum_spend_logs_retention_period")
            new_value: Final = _general_settings["maximum_spend_logs_retention_period"]
            general_settings["maximum_spend_logs_retention_period"] = new_value
            # Reschedule cleanup job if value changed (including when set to None)
            if old_value != new_value:
                await self._reschedule_spend_log_cleanup_job()

        if "maximum_autorouter_session_retention_period" in _general_settings:
            old_session_value: Final = general_settings.get("maximum_autorouter_session_retention_period")
            new_session_value: Final = _general_settings["maximum_autorouter_session_retention_period"]
            general_settings["maximum_autorouter_session_retention_period"] = new_session_value
            if old_session_value != new_session_value:
                await self._reschedule_spend_log_cleanup_job()

        if "maximum_health_check_retention_period" in _general_settings:
            old_health_check_value: Final = general_settings.get("maximum_health_check_retention_period")
            new_health_check_value: Final = _general_settings["maximum_health_check_retention_period"]
            general_settings["maximum_health_check_retention_period"] = new_health_check_value
            if old_health_check_value != new_health_check_value:
                await self._reschedule_spend_log_cleanup_job()

        ## SPEND LOG CLEANUP BOUNDS ##
        # The dashboard writes these straight to the DB, so without copying them
        # here the running cleanup job never sees them. A key the DB no longer
        # carries was cleared from the dashboard, and falls back to whatever
        # config.yaml declared, or to None (the shipped default) when it declared
        # nothing. Leaving the deleted DB value in memory would keep enforcing the
        # bound the operator just removed.
        for cleanup_key in SPEND_LOG_CLEANUP_BOUND_SETTINGS:
            general_settings[cleanup_key] = _general_settings.get(
                cleanup_key, self._yaml_spend_log_cleanup_bounds.get(cleanup_key)
            )

        for key in (
            "user_url_allowed_hosts",
            "user_url_validation",
            "provider_url_destination_allowed_hosts",
        ):
            if key in _general_settings:
                general_settings[key] = _general_settings[key]
        _apply_ssrf_general_settings(_general_settings)

    def _update_config_fields(
        self,
        current_config: dict,
        param_name: Literal[
            "general_settings",
            "router_settings",
            "litellm_settings",
            "environment_variables",
        ],
        db_param_value: Any,
    ) -> dict:
        """
        Updates the config fields with the new values from the DB

        Args:
            current_config (dict): Current configuration dictionary to update
            param_name (Literal): Name of the parameter to update
            db_param_value (Any): New value from the database

        Returns:
            dict: Updated configuration dictionary
        """

        def _deep_merge_dicts(dst: dict, src: dict) -> None:
            """
            Deep-merge src into dst, skipping None values and empty lists from src.
            On conflicts, src (DB) wins, but empty lists are treated as "no value" and don't overwrite.
            """
            stack: Final = [(dst, src)]
            while stack:
                d, s = stack.pop()
                for k, v in s.items():
                    if v is None:
                        # Preserve existing config when DB value is None (matches prior behavior)
                        continue
                    # Skip empty lists - treat them as "no value" to preserve file config
                    if isinstance(v, list) and len(v) == 0:
                        continue
                    if isinstance(v, dict) and isinstance(d.get(k), dict):
                        stack.append((d[k], v))
                    else:
                        d[k] = v

        # Strip remote-URL module loads from the DB-overlay before merge —
        # the YAML-load callsites have ``config_file_path`` set, so a
        # DB-sourced ``s3://`` value would otherwise reach
        # ``_load_instance_from_remote_storage`` without going through
        # the runtime gate.
        db_param_value = _scrub_db_overlay_remote_module_loads(section=param_name, db_value=db_param_value)

        if param_name == "environment_variables":
            decrypted_env_vars = self._decrypt_and_set_db_env_variables(db_param_value, return_original_value=True)
            # Normalize keys when loading from DB so services expecting uppercase
            # (e.g. Datadog) can read them even if stored in lowercase.
            merged_env_vars: Final[dict] = {}
            for key, value in decrypted_env_vars.items():
                merged_env_vars[key] = value
                upper_key = key.upper()
                merged_env_vars[upper_key] = value
                os.environ[upper_key] = value

            current_config.setdefault("environment_variables", {}).update(merged_env_vars)
            return current_config
        elif param_name == "litellm_settings" and isinstance(db_param_value, dict):
            for key, value in db_param_value.items():
                if key in LITELLM_SETTINGS_SAFE_DB_OVERRIDES:  # params that are safe to override with db values
                    setattr(litellm, key, value)

        # If param doesn't exist in config, add it
        if param_name not in current_config:
            current_config[param_name] = db_param_value

            return current_config

        # For dictionary values, update only non-none values
        if isinstance(current_config[param_name], dict) and isinstance(db_param_value, dict):
            _deep_merge_dicts(current_config[param_name], db_param_value)
        else:
            # Non-dict or mismatched types: DB value replaces config (unchanged behavior)
            current_config[param_name] = db_param_value

        return current_config

    async def _update_config_from_db(
        self,
        prisma_client: PrismaClient,
        config: dict,
        store_model_in_db: bool | None,
    ):
        if store_model_in_db is not True:
            verbose_proxy_logger.info("'store_model_in_db' is not True, skipping db updates")
            return config

        _tasks: Final = []
        keys: Final = [
            "general_settings",
            "router_settings",
            "litellm_settings",
            "environment_variables",
        ]
        for k in keys:
            _tasks.append(get_config_param(prisma_client, k))

        responses: Final = await asyncio.gather(*_tasks)
        for response in responses:
            if response is None:
                continue

            param_name = getattr(response, "param_name", None)
            param_value = getattr(response, "param_value", None)
            verbose_proxy_logger.debug(
                "param_name=%s, param_value=%s",
                param_name,
                _redact_config_param_value_for_logging(param_name, param_value),
            )

            if param_name is not None and param_value is not None:
                config = self._update_config_fields(
                    current_config=config,
                    param_name=param_name,
                    db_param_value=param_value,
                )

        return config

    def _should_load_db_object(self, object_type: str | SupportedDBObjectType) -> bool:
        return should_load_db_object(object_type=object_type)

    async def _get_models_from_db(self, prisma_client: PrismaClient) -> Sequence[_ProxyModelRow] | None:
        """
        Fetch all model deployments from the DB.

        Returns:
        - list: the rows (may be empty if no models exist)
        - None: signals a DB fetch *failure* — callers must not treat this
          as "all models deleted" and must not evict existing router deployments.
        """
        try:
            new_models: Final[Sequence[_ProxyModelRow]] = await ModelRepository(prisma_client).table.find_many()
            return new_models
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy_server.py::add_deployment() - Error getting new models from DB - %s", e
            )
            return None

    async def add_deployment(
        self,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
    ) -> ReconcileOutcome:
        """
        - Check db for new models
        - Check if model id's in router already
        - If not, add to router

        Serialized against every other model reconcile by MODEL_RECONCILE_LOCK, because
        the work below is a read-modify-write of the shared ``llm_router`` global: it
        reads the db into a snapshot and then makes the router match that snapshot. Two
        of those interleaving is not a lost update but an eviction -- the request whose
        snapshot predates the other's commit reconciles the newer model *out* of the
        router, since _delete_deployment removes every live deployment absent from the
        snapshot it was handed. The model stays in the db and this pod stops serving it
        until some later reload puts it back.

        Returns what the reconcile saw, captured before the lock is released so a
        caller's verdict cannot be corrupted by the next reconcile's own in-flight
        window. See ReconcileOutcome.
        """
        async with MODEL_RECONCILE_LOCK:
            return await self._add_deployment_locked(prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj)

    async def _add_deployment_locked(
        self,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
    ) -> ReconcileOutcome:
        """add_deployment's body, minus the locking. MODEL_RECONCILE_LOCK MUST already
        be held. Split out for the one caller that has to hold the lock across more than
        this reconcile -- clear_cache, which un-serves every db model before calling it
        and would deadlock on a re-acquire."""
        global llm_router, llm_model_list, master_key, general_settings

        still_desired_ids: frozenset[str] | None = None

        try:
            # warm the config cache so the per-param reads below all hit
            await prefetch_config_params(
                prisma_client,
                [
                    "general_settings",
                    "router_settings",
                    "litellm_settings",
                    "environment_variables",
                    "anthropic_beta_headers_reload_config",
                ],
            )

            # Only load models from DB if "models" is in supported_db_objects (or if supported_db_objects is not set)
            if self._should_load_db_object(object_type="models"):
                new_models: Final = await self._get_models_from_db(prisma_client=prisma_client)

                # update llm router
                still_desired_ids = await self._update_llm_router(
                    new_models=new_models, proxy_logging_obj=proxy_logging_obj
                )

            db_general_settings: Final[_ConfigParamRow | None] = await get_config_param(
                prisma_client, "general_settings"
            )

            # update general settings
            if db_general_settings is not None:
                await self._update_general_settings(
                    db_general_settings=db_general_settings.param_value,
                )

            # initialize vector stores, guardrails, etc. table in db
            await self._init_non_llm_objects_in_db(prisma_client=prisma_client)

        except Exception as e:
            verbose_proxy_logger.exception("litellm.proxy.proxy_server.py::ProxyConfig:add_deployment - %s", e)

        # Read while the lock is still held: once it is released the next reconcile can
        # begin, and clear_cache's leading wipe would make this look like a mass drop.
        return ReconcileOutcome(
            still_desired=still_desired_ids,
            live_after=None if still_desired_ids is None else live_model_ids_snapshot(),
        )

    def start_config_sync_subscriber(
        self,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        redis_cache: RedisCache | None,
    ) -> None:
        if redis_cache is None or self.config_sync_subscriber is not None:
            return

        async def _resync_config_from_db() -> None:
            await self.add_deployment(prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj)

        async def _resync_credentials_from_db() -> None:
            await self.get_credentials(prisma_client=prisma_client)

        subscriber: Final = ConfigSyncSubscriber(
            redis_cache=redis_cache,
            resync_callbacks=(_resync_config_from_db, _resync_credentials_from_db),
        )
        self.config_sync_subscriber = subscriber
        subscriber.start()

    async def stop_config_sync_subscriber(self) -> None:
        subscriber: Final = self.config_sync_subscriber
        if subscriber is None:
            return
        self.config_sync_subscriber = None
        try:
            await subscriber.stop()
        except Exception as e:
            verbose_proxy_logger.error("Error stopping config sync subscriber: %s", e)

    def start_auth_cache_invalidation_subscriber(
        self,
        redis_cache: RedisCache | None,
        user_api_key_cache: UserApiKeyCache,
    ) -> None:
        if redis_cache is None or self.auth_cache_invalidation_subscriber is not None:
            return
        subscriber: Final = AuthCacheInvalidationSubscriber(
            redis_cache=redis_cache,
            user_api_key_cache=user_api_key_cache,
            additional_in_memory_caches=(spend_counter_cache.in_memory_cache,),
        )
        self.auth_cache_invalidation_subscriber = subscriber
        subscriber.start()

    async def stop_auth_cache_invalidation_subscriber(self) -> None:
        subscriber: Final = self.auth_cache_invalidation_subscriber
        if subscriber is None:
            return
        self.auth_cache_invalidation_subscriber = None
        try:
            await subscriber.stop()
        except Exception as e:  # noqa: BLE001  # best-effort: a failing stop must not break proxy shutdown
            verbose_proxy_logger.error("Error stopping auth cache invalidation subscriber: %s", e)

    async def _init_non_llm_objects_in_db(self, prisma_client: PrismaClient):
        """
        Use this to read non-llm objects from the db and initialize them

        ex. Vector Stores, Guardrails, MCP tools, etc.
        """
        if self._should_load_db_object(object_type="guardrails"):
            await self._init_guardrails_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="policies"):
            await self._init_policies_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="vector_stores"):
            await self._init_vector_stores_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="vector_store_indexes"):
            await self._init_vector_store_indexes_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="mcp"):
            await self._init_mcp_servers_in_db()

        if self._should_load_db_object(object_type="agents"):
            await self._init_agents_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="pass_through_endpoints"):
            await self._init_pass_through_endpoints_in_db()

        if self._should_load_db_object(object_type="prompts"):
            await self._init_prompts_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="search_tools"):
            await self._init_search_tools_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="tools"):
            await self._init_tool_policy_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="anthropic_beta_headers"):
            await self._check_and_reload_anthropic_beta_headers(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="sso_settings"):
            await self._init_sso_settings_in_db(prisma_client=prisma_client)
        if self._should_load_db_object(object_type="cache_settings"):
            from litellm.proxy.management_endpoints.cache_settings_endpoints import (
                CacheSettingsManager,
            )

            await CacheSettingsManager.init_cache_settings_in_db(prisma_client=prisma_client, proxy_config=self)

        if self._should_load_db_object(object_type="semantic_filter_settings"):
            await self._init_semantic_filter_settings_in_db(prisma_client=prisma_client)

        if self._should_load_db_object(object_type="config_overrides"):
            await self._init_hashicorp_vault_config_override(prisma_client=prisma_client)

        await self._apply_safe_litellm_settings_overrides_from_db(prisma_client=prisma_client)

    async def _apply_safe_litellm_settings_overrides_from_db(self, prisma_client: PrismaClient) -> None:
        config_record: Final = await get_config_param(prisma_client, "litellm_settings")
        if config_record is None or config_record.param_value is None:
            return
        raw_settings: Final = config_record.param_value
        litellm_settings: Final = json.loads(raw_settings) if isinstance(raw_settings, str) else raw_settings
        if not isinstance(litellm_settings, dict):
            return
        for key, value in litellm_settings.items():
            if key in LITELLM_SETTINGS_SAFE_DB_OVERRIDES:
                setattr(litellm, key, value)

    async def _init_semantic_filter_settings_in_db(self, prisma_client: PrismaClient):
        """
        Initialize MCP semantic filter settings from database.
        Called periodically (approximately every 10 seconds) by background task to hot-reload settings across all pods.
        """
        import json

        import litellm
        from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook

        try:
            config_record: Final = await get_config_param(prisma_client, "litellm_settings")

            if config_record is None or config_record.param_value is None:
                return

            litellm_settings = config_record.param_value
            if isinstance(litellm_settings, str):
                litellm_settings = json.loads(litellm_settings)

            mcp_semantic_filter_config: Final = litellm_settings.get("mcp_semantic_tool_filter", None)

            if mcp_semantic_filter_config is None:
                return

            # Check if settings have changed (compare with in-memory state)
            if hasattr(self, "_last_semantic_filter_config"):
                if self._last_semantic_filter_config == mcp_semantic_filter_config:
                    # If hook is missing or router isn't built yet, reinitialize anyway
                    active_hooks = litellm.logging_callback_manager.get_custom_loggers_for_type(SemanticToolFilterHook)
                    if active_hooks:
                        for active_hook in active_hooks:
                            if isinstance(active_hook, SemanticToolFilterHook):
                                if active_hook.filter is not None and active_hook.filter.tool_router is not None:
                                    verbose_proxy_logger.debug(
                                        "Semantic filter settings unchanged, skipping reinitialization"
                                    )
                                    return
                    verbose_proxy_logger.info(
                        "Semantic filter settings unchanged, but hook is missing or uninitialized. Reinitializing."
                    )

            # Remove old hooks using logging callback manager
            litellm.logging_callback_manager.remove_callbacks_by_type(litellm.callbacks, SemanticToolFilterHook)

            # Initialize new hook if enabled
            if mcp_semantic_filter_config.get("enabled", False):
                global llm_router
                hook: Final = await SemanticToolFilterHook.initialize_from_config(
                    config=mcp_semantic_filter_config,
                    llm_router=llm_router,
                )
                if hook:
                    litellm.logging_callback_manager.add_litellm_callback(hook)
                    verbose_proxy_logger.info("MCP Semantic Filter reinitialized from DB")
            else:
                verbose_proxy_logger.info("MCP Semantic Filter disabled")

            # Store current config for comparison next time
            self._last_semantic_filter_config = mcp_semantic_filter_config.copy()

        except Exception as e:
            verbose_proxy_logger.exception("Error initializing semantic filter settings from DB: %s", e)

    async def _init_sso_settings_in_db(self, prisma_client: PrismaClient):
        """
        Initialize SSO settings from database into the router on startup.
        """

        try:
            sso_settings: Final[_SSOConfigRow | None] = cast(  # cast-ok: prisma Json stub is `str`, runtime is a dict
                "_SSOConfigRow | None",
                await call_with_db_reconnect_retry(
                    prisma_client,
                    lambda: SSOConfigRepository(prisma_client).table.find_unique(where={"id": "sso_config"}),
                    reason="init_sso_settings_in_db_lookup_failure",
                ),
            )
            if sso_settings is not None:
                sso_settings.sso_settings.pop("role_mappings", None)
                sso_settings.sso_settings.pop("team_mappings", None)
                sso_settings.sso_settings.pop("ui_access_mode", None)
                uppercase_sso_settings: Final = {key.upper(): value for key, value in sso_settings.sso_settings.items()}
                self._decrypt_and_set_db_env_variables(environment_variables=uppercase_sso_settings)
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_sso_settings_in_db - %s", e
            )

    async def _init_hashicorp_vault_config_override(self, prisma_client: PrismaClient):
        """
        Load Hashicorp Vault config override from DB.
        Decrypts sensitive fields, sets HCP_VAULT_* env vars, and reinitializes the secret manager.
        Called periodically via _init_non_llm_objects_in_db to sync config across pods.
        """
        from litellm.proxy.management_endpoints.config_override_endpoints import (
            HASHICORP_ENV_VAR_MAPPING,
            _clear_hashicorp_vault_state,
            _get_current_env_values,
            _parse_config_value,
            _set_env_vars,
        )

        try:
            db_record: Final[_ConfigOverridesRow | None] = cast(  # cast-ok: prisma Json stub is `str`, runtime dict
                "_ConfigOverridesRow | None",
                await call_with_db_reconnect_retry(
                    prisma_client,
                    lambda: ConfigOverridesRepository(prisma_client).table.find_unique(
                        where={"config_type": "hashicorp_vault"}
                    ),
                    reason="init_hashicorp_vault_config_override_lookup_failure",
                ),
            )

            if db_record is None or db_record.config_value is None:
                if self._last_hashicorp_vault_config is not None:
                    _clear_hashicorp_vault_state(self)
                return

            config_data: Final = _parse_config_value(db_record.config_value)

            # Skip reinit if config hasn't changed since last poll
            if self._last_hashicorp_vault_config == config_data:
                return

            # Decrypt all fields and set env vars
            decrypted_data: Final = self._decrypt_db_variables(config_data)

            # Snapshot current env vars so we can restore on failure
            previous_env: Final = _get_current_env_values(HASHICORP_ENV_VAR_MAPPING)
            _set_env_vars(decrypted_data)

            # Reinitialize the secret manager
            try:
                self.initialize_secret_manager(key_management_system="hashicorp_vault")
            except Exception:
                # Restore previous working env vars instead of wiping all
                _set_env_vars(previous_env)
                raise

            self._last_hashicorp_vault_config = config_data.copy()
            verbose_proxy_logger.debug("Hashicorp Vault config override loaded from DB")
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error loading Hashicorp Vault config override from DB: %s",
                str(e),
            )

    async def check_periodic_reloads(self, prisma_client: PrismaClient):
        """
        Run the admin-configured periodic model cost map reload.

        Scheduled on its own job so a schedule configured from the Admin UI fires whether
        or not `store_model_in_db` is enabled.
        """
        if self._should_load_db_object(object_type="model_cost_map"):
            await self._check_and_reload_model_cost_map(prisma_client=prisma_client)

    async def _check_and_reload_model_cost_map(self, prisma_client: PrismaClient):
        """
        Check if model cost map needs to be reloaded based on database configuration.
        Runs on the periodic reload job, independently of `store_model_in_db`.
        """
        try:
            schedule = await read_reload_schedule(prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME)
            if schedule is None:
                return

            current_time = utc_now()
            is_due = pod_reload_is_due(
                schedule=schedule,
                pod_applied_revision=self.model_cost_map_applied_revision,
                pod_data_loaded_at=self.model_cost_map_loaded_at,
                current_time=current_time,
                description="Model cost map",
            )
            if not is_due:
                return

            from litellm.litellm_core_utils.get_model_cost_map import (
                ModelCostMapReloadUnavailable,
                refetch_model_cost_map,
            )

            reload_result = await refetch_model_cost_map(url=litellm.model_cost_map_url)
            if isinstance(reload_result, ModelCostMapReloadUnavailable):
                verbose_proxy_logger.warning(
                    "Model cost map reload failed (%s); keeping current pricing data. The revision stays "
                    "unapplied so this pod retries on its next poll",
                    reload_result.reason,
                )
                return

            models_count = _swap_in_model_cost_map(reload_result.model_cost_map)
            self.model_cost_map_loaded_at = current_time
            await record_reload_run(prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME, current_time)
            # Adopted last, so neither a failed fetch nor a failed status write is recorded
            # as served; either way the next poll retries instead of leaving the card
            # reporting a run that never landed
            self.model_cost_map_applied_revision = schedule.reload_revision

            verbose_proxy_logger.info("Model cost map reloaded successfully. Models count: %s", models_count)

        except Exception as e:
            verbose_proxy_logger.exception("Error in _check_and_reload_model_cost_map: %s", e)

    async def _check_and_reload_anthropic_beta_headers(self, prisma_client: PrismaClient):
        """
        Check if anthropic beta headers config needs to be reloaded based on database configuration.
        This function runs every 10 seconds as part of _init_non_llm_objects_in_db.
        """
        try:
            # Get anthropic beta headers reload configuration from database
            config_record: Final = await get_config_param(prisma_client, "anthropic_beta_headers_reload_config")

            if config_record is None or config_record.param_value is None:
                return  # No configuration found, skip reload

            config: Final = config_record.param_value
            interval_hours: Final = config.get("interval_hours")
            force_reload: Final = config.get("force_reload", False)

            if interval_hours is None and force_reload is False:
                return  # No interval configured, skip reload

            current_time: Final = datetime.utcnow()

            # Check if we need to reload based on interval or force reload
            should_reload = False

            if force_reload:
                should_reload = True
                verbose_proxy_logger.info("Anthropic beta headers reload triggered by force reload flag")
            elif interval_hours is not None:
                # Use pod's in-memory last reload time
                global last_anthropic_beta_headers_reload
                if last_anthropic_beta_headers_reload is not None:
                    try:
                        last_reload_time: Final = datetime.fromisoformat(last_anthropic_beta_headers_reload)
                        time_since_last_reload: Final = current_time - last_reload_time
                        hours_since_last_reload: Final = time_since_last_reload.total_seconds() / 3600

                        if hours_since_last_reload >= interval_hours:
                            should_reload = True
                            verbose_proxy_logger.info(
                                f"Anthropic beta headers reload triggered by interval. Hours since last reload: {hours_since_last_reload:.2f}, Interval: {interval_hours}"
                            )
                    except Exception as e:
                        verbose_proxy_logger.warning("Error parsing last reload time: %s", e)
                        # If we can't parse the last reload time, reload anyway
                        should_reload = True
                else:
                    # No last reload time recorded, reload now
                    should_reload = True
                    verbose_proxy_logger.info(
                        "Anthropic beta headers reload triggered - no previous reload time recorded"
                    )

            if should_reload:
                # Perform the reload
                from litellm.anthropic_beta_headers_manager import (
                    reload_beta_headers_config,
                )

                new_config: Final = reload_beta_headers_config()

                # Update pod's in-memory last reload time
                last_anthropic_beta_headers_reload = current_time.isoformat()

                # Clear force reload flag in database
                await ConfigRepository(prisma_client).table.upsert(
                    where={"param_name": "anthropic_beta_headers_reload_config"},
                    data={
                        "create": {
                            "param_name": "anthropic_beta_headers_reload_config",
                            "param_value": safe_dumps(
                                {
                                    "interval_hours": interval_hours,
                                    "force_reload": False,
                                }
                            ),
                        },
                        "update": {
                            "param_value": safe_dumps(
                                {
                                    "interval_hours": interval_hours,
                                    "force_reload": False,
                                }
                            )
                        },
                    },
                )
                await evict_config_param("anthropic_beta_headers_reload_config")

                # Count providers in config
                provider_count = sum(1 for k in new_config if k != "provider_aliases" and k != "description")
                verbose_proxy_logger.info(
                    "Anthropic beta headers config reloaded successfully. Providers: %s", provider_count
                )

        except Exception as e:
            verbose_proxy_logger.exception("Error in _check_and_reload_anthropic_beta_headers: %s", e)

    def _get_prompt_spec_for_db_prompt(self, db_prompt):
        """
        Convert a DB prompt object to a PromptSpec object.

        Handles the versioning of the prompt, if the DB prompt has a version, it will be used to create the versioned prompt_id.

        Args:
            db_prompt: The DB prompt object

        Returns:
            The PromptSpec object
        """
        from litellm.proxy.prompts.prompt_endpoints import create_versioned_prompt_spec

        return create_versioned_prompt_spec(db_prompt=db_prompt)

    async def _init_prompts_in_db(self, prisma_client: PrismaClient):
        from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
        from litellm.types.prompts.init_prompts import PromptSpec

        def parse_row(db_prompt: object) -> PromptSpec | None:
            try:
                return self._get_prompt_spec_for_db_prompt(db_prompt=db_prompt)
            except Exception as row_error:  # noqa: BLE001  # a malformed row must not block syncing the remaining prompts
                verbose_proxy_logger.exception(
                    "litellm.proxy.proxy_server.py::ProxyConfig:_init_prompts_in_db - failed to parse prompt row %s: %s",
                    getattr(db_prompt, "prompt_id", None),
                    row_error,
                )
                return None

        try:
            prompt_ids_loaded_before_db_read: Final = frozenset(IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS)
            prompts_in_db: Final[Sequence[object]] = await PromptRepository(prisma_client).table.find_many()
            parsed_specs: Final[tuple[PromptSpec, ...]] = tuple(
                spec for row in prompts_in_db if (spec := parse_row(row)) is not None
            )
            newest_spec_per_id: Final[Mapping[str, PromptSpec]] = MappingProxyType(
                {
                    spec.prompt_id: spec
                    for spec in sorted(
                        parsed_specs,
                        key=lambda s: s.updated_at.timestamp() if s.updated_at else float("-inf"),
                    )
                }
            )
            for prompt_spec in newest_spec_per_id.values():
                try:
                    IN_MEMORY_PROMPT_REGISTRY.sync_prompt_from_db(prompt=prompt_spec)
                except Exception as prompt_sync_error:  # noqa: BLE001  # one poisoned row must not block syncing the remaining prompts
                    verbose_proxy_logger.exception(
                        "litellm.proxy.proxy_server.py::ProxyConfig:_init_prompts_in_db - failed to sync prompt %s: %s",
                        prompt_spec.prompt_id,
                        prompt_sync_error,
                    )
            # An unparsable row still exists in the DB, so skip the sweep rather than unload its in-memory copy
            every_row_parsed: Final = len(parsed_specs) == len(prompts_in_db)
            if every_row_parsed:
                deleted_db_prompt_ids: Final = tuple(
                    prompt_id
                    for prompt_id in prompt_ids_loaded_before_db_read
                    if (loaded_spec := IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.get(prompt_id)) is not None
                    and loaded_spec.prompt_info.prompt_type == "db"
                    and prompt_id not in newest_spec_per_id
                )
                for deleted_prompt_id in deleted_db_prompt_ids:
                    IN_MEMORY_PROMPT_REGISTRY.remove_prompt(prompt_id=deleted_prompt_id)
        except Exception as e:
            verbose_proxy_logger.debug("litellm.proxy.proxy_server.py::ProxyConfig:_init_prompts_in_db - %s", e)

    async def _init_guardrails_in_db(self, prisma_client: PrismaClient):
        from litellm.proxy.guardrails.guardrail_registry import (
            GUARDRAIL_RECONCILE_LOCK,
            IN_MEMORY_GUARDRAIL_HANDLER,
            Guardrail,
            GuardrailRegistry,
        )

        try:
            async with GUARDRAIL_RECONCILE_LOCK:
                guardrails_in_db: Final[list[Guardrail]] = await GuardrailRegistry.get_all_guardrails_from_db(
                    prisma_client=prisma_client
                )
                verbose_proxy_logger.debug("guardrails from the DB %s", str(guardrails_in_db))
                db_guardrail_ids: Final[set] = set()
                for guardrail in guardrails_in_db:
                    guardrail_id = guardrail.get("guardrail_id")
                    if guardrail_id:
                        db_guardrail_ids.add(guardrail_id)
                    try:
                        IN_MEMORY_GUARDRAIL_HANDLER.sync_guardrail_from_db(
                            guardrail=cast(Guardrail, guardrail),
                        )
                    except Exception as e:  # noqa: BLE001  # one unloadable row must not stop the remaining guardrails
                        verbose_proxy_logger.error(
                            "litellm.proxy.proxy_server.py::ProxyConfig:_init_guardrails_in_db - "
                            "skipping guardrail '%s' (ID: %s): %s: %s",
                            guardrail.get("guardrail_name"),
                            guardrail_id,
                            type(e).__name__,
                            e,
                        )

                # Drop in-memory DB-backed entries whose row was deleted on another
                # pod. Config-loaded entries are never touched.
                IN_MEMORY_GUARDRAIL_HANDLER.reconcile_db_guardrails(db_guardrail_ids=db_guardrail_ids)
        except Exception as e:
            verbose_proxy_logger.exception("litellm.proxy.proxy_server.py::ProxyConfig:_init_guardrails_in_db - %s", e)

    async def _init_policies_in_db(self, prisma_client: PrismaClient):
        """
        Initialize policies and policy attachments from database into the in-memory registries.
        """
        from litellm.proxy.policy_engine.attachment_registry import (
            get_attachment_registry,
        )
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        try:
            # Get the global singleton instances
            policy_registry: Final = get_policy_registry()
            attachment_registry: Final = get_attachment_registry()

            # Sync policies from DB to in-memory registry
            await policy_registry.sync_policies_from_db(prisma_client=prisma_client)

            # Sync attachments from DB to in-memory registry
            await attachment_registry.sync_attachments_from_db(prisma_client=prisma_client)

            verbose_proxy_logger.debug("Successfully synced policies and attachments from DB")
        except Exception as e:
            verbose_proxy_logger.exception("litellm.proxy.proxy_server.py::ProxyConfig:_init_policies_in_db - %s", e)

    async def _init_tool_policy_in_db(self, prisma_client: PrismaClient):
        """
        Initialize tool policy from database into the in-memory registry.
        Synced periodically by add_deployment -> _init_non_llm_objects_in_db.
        """
        from litellm.proxy.db.tool_registry_writer import get_tool_policy_registry

        try:
            registry: Final = get_tool_policy_registry()
            await registry.sync_tool_policy_from_db(prisma_client=prisma_client)
            verbose_proxy_logger.debug("Successfully synced tool policy from DB")
        except Exception as e:
            verbose_proxy_logger.exception("litellm.proxy.proxy_server.py::ProxyConfig:_init_tool_policy_in_db - %s", e)

    async def _init_vector_stores_in_db(self, prisma_client: PrismaClient):
        from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

        try:
            # read vector stores from db table
            vector_stores: Final = await VectorStoreRegistry._get_vector_stores_from_db(prisma_client=prisma_client)
            if len(vector_stores) <= 0:
                return

            if litellm.vector_store_registry is None:
                litellm.vector_store_registry = VectorStoreRegistry(vector_stores=vector_stores)
            else:
                for vector_store in vector_stores:
                    litellm.vector_store_registry.add_vector_store_to_registry(vector_store=vector_store)
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_vector_stores_in_db - %s", e
            )

    async def _init_vector_store_indexes_in_db(self, prisma_client: PrismaClient):
        from litellm.vector_stores.vector_store_registry import VectorStoreIndexRegistry

        try:
            # read vector stores from db table
            vector_store_indexes: Final = await VectorStoreIndexRegistry._get_vector_store_indexes_from_db(
                prisma_client=prisma_client
            )

            if len(vector_store_indexes) <= 0:
                return

            if litellm.vector_store_index_registry is None:
                litellm.vector_store_index_registry = VectorStoreIndexRegistry(
                    vector_store_indexes=vector_store_indexes
                )
            else:
                for vector_store_index in vector_store_indexes:
                    litellm.vector_store_index_registry.upsert_vector_store_index(vector_store_index=vector_store_index)
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_vector_stores_in_db - %s", e
            )

    async def _init_mcp_servers_in_db(self):
        from litellm.proxy._experimental.mcp_server.utils import is_mcp_available

        if not is_mcp_available():
            verbose_proxy_logger.debug("MCP module not available, skipping MCP server initialization")
            return

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.oauth2_flow_backfill import (
            backfill_null_oauth2_flows,
        )
        from litellm.proxy._experimental.mcp_server.oauth_issuer_stamp_backfill import (
            backfill_discovery_stamped_issuers,
        )

        try:
            if prisma_client is not None:
                await backfill_null_oauth2_flows(prisma_client)
        except Exception as e:  # noqa: BLE001
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_mcp_servers_in_db backfill - %s", e
            )

        try:
            if prisma_client is not None:
                await backfill_discovery_stamped_issuers(prisma_client)
        except Exception as e:  # noqa: BLE001
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_mcp_servers_in_db issuer stamp backfill - %s", e
            )

        try:
            await global_mcp_server_manager.reload_servers_from_database()
        except Exception as e:
            verbose_proxy_logger.exception("litellm.proxy.proxy_server.py::ProxyConfig:_init_mcp_servers_in_db - %s", e)

    async def init_mcp_servers_from_db(self) -> None:
        if self._should_load_db_object(object_type="mcp"):
            await self._init_mcp_servers_in_db()

    async def reload_mcp_servers_from_db(self) -> None:
        """Registry refresh only, for the periodic job in store_model_in_db-off deployments.

        Deliberately narrower than ``init_mcp_servers_from_db``: the oauth2_flow backfill is a write
        path that only needs to run once at startup, so the cadence here is purely the read-side
        reload whose fast-path exemption retries failed OAuth discovery. Gated the same way, so an
        admin who excluded mcp from supported_db_objects opts out of this too.
        """
        if not self._should_load_db_object(object_type="mcp"):
            return
        from litellm.proxy._experimental.mcp_server.utils import is_mcp_available

        if not is_mcp_available():
            return
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        try:
            await global_mcp_server_manager.reload_servers_from_database()
        except Exception as e:  # noqa: BLE001  # scheduled job: a reload failure must not kill the recurring retry
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:reload_mcp_servers_from_db - %s", e
            )

    async def _init_agents_in_db(self, prisma_client: PrismaClient):
        from litellm.proxy.agent_endpoints.agent_registry import (
            AGENT_RECONCILE_LOCK,
        )
        from litellm.proxy.agent_endpoints.agent_registry import (
            global_agent_registry as AGENT_REGISTRY,
        )

        try:
            async with AGENT_RECONCILE_LOCK:
                db_agents: Final = await AGENT_REGISTRY.get_all_agents_from_db(prisma_client=prisma_client)
                AGENT_REGISTRY.load_agents_from_db_and_config(db_agents=db_agents)
        except Exception as e:
            verbose_proxy_logger.exception("litellm.proxy.proxy_server.py::ProxyConfig:_init_agents_in_db - %s", e)

    async def _init_search_tools_in_db(self, prisma_client: PrismaClient):
        """
        Initialize search tools from database into the router on startup.
        """
        global llm_router

        from litellm.proxy.search_endpoints.search_tool_registry import (
            SearchToolRegistry,
        )
        from litellm.router_utils.search_api_router import SearchAPIRouter

        try:
            db_search_tools: Final = await SearchToolRegistry.get_all_search_tools_from_db(prisma_client=prisma_client)

            parsed_tools: Final = self.parse_search_tools(self.get_config_state())
            config_search_tools: Final = parsed_tools or []

            search_tools: Final = self._merge_config_and_db_search_tools(
                config_search_tools=config_search_tools,
                db_search_tools=[dict(tool) for tool in db_search_tools],
            )

            verbose_proxy_logger.info(
                "Loading %s search tool(s) into router (%s from config, %s from database)",
                len(search_tools),
                len(config_search_tools),
                len(db_search_tools),
            )

            if llm_router is not None:
                await SearchAPIRouter.update_router_search_tools(router_instance=llm_router, search_tools=search_tools)
                verbose_proxy_logger.info("Successfully loaded %s search tool(s) into router", len(search_tools))
            else:
                verbose_proxy_logger.debug(
                    "Router not initialized yet, search tools will be added when router is created"
                )

        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_search_tools_in_db - %s", e
            )

    async def reload_search_tools_from_db(self) -> None:
        """Refresh this worker's router from the search tools table.

        Driven by the management endpoints so the worker that served the write is correct
        immediately, and by the periodic job in store_model_in_db-off deployments. Gated the same
        way as startup, so an admin who excluded search_tools from supported_db_objects opts out.

        Serialized by MODEL_RECONCILE_LOCK for the reason add_deployment documents: the body is a
        read-modify-write of the shared ``llm_router`` global, so two of them interleaving lets the
        older snapshot's wholesale assignment land last and restore a tool the newer one deleted.
        The lock belongs here rather than in _init_search_tools_in_db, which _init_non_llm_objects_in_db
        already calls while holding it.
        """
        if not self._should_load_db_object(object_type="search_tools"):
            return
        if prisma_client is None:
            return
        async with MODEL_RECONCILE_LOCK:
            await self._init_search_tools_in_db(prisma_client=prisma_client)

    @staticmethod
    def _merge_config_and_db_search_tools(
        config_search_tools: list[SearchToolTypedDict],
        db_search_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        db_tool_names: Final = {tool.get("search_tool_name") for tool in db_search_tools}
        return [
            *[
                dict(config_search_tool)
                for config_search_tool in config_search_tools
                if config_search_tool.get("search_tool_name") not in db_tool_names
            ],
            *db_search_tools,
        ]

    async def _init_pass_through_endpoints_in_db(self):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            initialize_pass_through_endpoints_in_db,
        )

        await initialize_pass_through_endpoints_in_db()

    def decrypt_credentials(self, credential: dict | BaseModel) -> CredentialItem:
        if isinstance(credential, dict):
            credential_object = CredentialItem(**credential)
        elif isinstance(credential, BaseModel):
            credential_object = CredentialItem(**credential.model_dump())

        decrypted_credential_values: Final = {}
        for k, v in credential_object.credential_values.items():
            decrypted_credential_values[k] = decrypt_value_helper(value=v, key=k) or v

        credential_object.credential_values = decrypted_credential_values
        return credential_object

    async def delete_credentials(self, db_credentials: list[CredentialItem]):
        """
        Create all-up list of db credentials + local credentials
        Compare to the litellm.credential_list
        Delete any from litellm.credential_list that are not in the all-up list
        """
        ## CONFIG credentials ##
        config: Final = await self.get_config(config_file_path=user_config_file_path)
        credential_list: Final = self.load_credential_list(config=config)

        ## COMBINED LIST ##
        combined_list: Final = db_credentials + credential_list

        ## DELETE ##
        idx_to_delete: Final = []
        for idx, credential in enumerate(litellm.credential_list):
            if credential.credential_name not in [cred.credential_name for cred in combined_list]:
                idx_to_delete.append(idx)
        for idx in sorted(idx_to_delete, reverse=True):
            litellm.credential_list.pop(idx)

    async def get_credentials(self, prisma_client: PrismaClient):
        try:
            credentials = await CredentialsRepository(prisma_client).find_all()
            credentials = [self.decrypt_credentials(cred) for cred in credentials]
            await self.delete_credentials(credentials)  # delete credentials that are not in the all-up list
            CredentialAccessor.upsert_credentials(credentials)  # upsert credentials that are in the all-up list
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy_server.py::get_credentials() - Error getting credentials from DB - %s", e
            )
            return []


proxy_config = ProxyConfig()


def save_worker_config(**data):
    import json

    os.environ["WORKER_CONFIG"] = json.dumps(data)


async def initialize(
    model=None,
    alias=None,
    api_base=None,
    api_version=None,
    debug=False,
    detailed_debug=False,
    temperature=None,
    max_tokens=None,
    request_timeout=600,
    max_budget=None,
    telemetry=False,
    drop_params=True,
    add_function_to_prompt=True,
    headers=None,
    save=False,
    use_queue=False,
    config=None,
):
    global \
        user_model, \
        user_api_base, \
        user_debug, \
        user_detailed_debug, \
        user_user_max_tokens, \
        user_request_timeout, \
        user_temperature, \
        user_telemetry, \
        user_headers, \
        experimental, \
        llm_model_list, \
        llm_router, \
        general_settings, \
        master_key, \
        user_custom_auth, \
        prisma_client
    from litellm.proxy.common_utils.banner import show_banner

    show_banner()
    if os.getenv("LITELLM_DONT_SHOW_FEEDBACK_BOX", "").lower() != "true":
        generate_feedback_box()
    user_model = model
    user_debug = debug
    if debug is True:  # this needs to be first, so users can see Router init debugg
        import logging

        from litellm._logging import (
            verbose_logger,
            verbose_proxy_logger,
            verbose_router_logger,
        )

        # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS
        verbose_logger.setLevel(level=logging.INFO)  # sets package logs to info
        verbose_router_logger.setLevel(level=logging.INFO)  # set router logs to info
        verbose_proxy_logger.setLevel(level=logging.INFO)  # set proxy logs to info
    if detailed_debug is True:
        import logging

        from litellm._logging import (
            verbose_logger,
            verbose_proxy_logger,
            verbose_router_logger,
        )

        verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
        verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
        verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug
    elif debug is False and detailed_debug is False:
        # users can control proxy debugging using env variable = 'LITELLM_LOG'
        litellm_log_setting: Final = os.environ.get("LITELLM_LOG", "")
        if litellm_log_setting is not None:
            if litellm_log_setting.upper() == "INFO":
                import logging

                from litellm._logging import (
                    verbose_logger,
                    verbose_proxy_logger,
                    verbose_router_logger,
                )

                # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS

                verbose_logger.setLevel(level=logging.INFO)  # set package log to info
                verbose_router_logger.setLevel(level=logging.INFO)  # set router logs to info
                verbose_proxy_logger.setLevel(level=logging.INFO)  # set proxy logs to info
            elif litellm_log_setting.upper() == "DEBUG":
                import logging

                from litellm._logging import (
                    verbose_logger,
                    verbose_proxy_logger,
                    verbose_router_logger,
                )

                verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
                verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
                verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug
    dynamic_config: Final = {"general": {}, user_model: {}}
    if config:
        (
            llm_router,
            llm_model_list,
            general_settings,
        ) = await proxy_config.load_config(router=llm_router, config_file_path=config)
    if headers:  # model-specific param
        user_headers = headers
        dynamic_config[user_model]["headers"] = headers
    if api_base:  # model-specific param
        user_api_base = api_base
        dynamic_config[user_model]["api_base"] = api_base
    if api_version:
        os.environ["AZURE_API_VERSION"] = api_version  # set this for azure - litellm can read this from the env
    if max_tokens:  # model-specific param
        dynamic_config[user_model]["max_tokens"] = max_tokens
    if temperature:  # model-specific param
        user_temperature = temperature
        dynamic_config[user_model]["temperature"] = temperature
    if request_timeout:
        user_request_timeout = request_timeout
        dynamic_config[user_model]["request_timeout"] = request_timeout
    if alias:  # model-specific param
        dynamic_config[user_model]["alias"] = alias
    if drop_params is True:  # litellm-specific param
        litellm.drop_params = True
        dynamic_config["general"]["drop_params"] = True
    if add_function_to_prompt is True:  # litellm-specific param
        litellm.add_function_to_prompt = True
        dynamic_config["general"]["add_function_to_prompt"] = True
    if max_budget:  # litellm-specific param
        litellm.max_budget = float(max_budget)
        dynamic_config["general"]["max_budget"] = litellm.max_budget
    if experimental:
        pass
    user_telemetry = telemetry


# for streaming
def data_generator(response):
    verbose_proxy_logger.debug("inside generator")
    for chunk in response:
        verbose_proxy_logger.debug("returned chunk: %s", chunk)
        try:
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except Exception:
            yield f"data: {json.dumps(chunk)}\n\n"


async def async_assistants_data_generator(response, user_api_key_dict: UserAPIKeyAuth, request_data: dict):
    verbose_proxy_logger.debug("inside generator")
    try:
        time.time()
        async with response as chunk:
            ### CALL HOOKS ### - modify outgoing data
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict,
                response=chunk,
                data=request_data,
            )

            # chunk = chunk.model_dump_json(exclude_none=True)
            async for c in chunk:
                c = c.model_dump_json(exclude_none=True)
                try:
                    yield f"data: {c}\n\n"
                except Exception as e:
                    yield f"data: {e}\n\n"

        # Streaming is done, yield the [DONE] chunk
        done_message: Final = "[DONE]"
        yield f"data: {done_message}\n\n"
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.async_assistants_data_generator(): Exception occured - %s", e
        )
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_data,
        )
        verbose_proxy_logger.debug(
            "\x1b[1;31mAn error occurred: %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
        )
        if isinstance(e, HTTPException):
            raise e
        else:
            # Only include the error message, not the traceback.
            # The traceback is already logged above via verbose_proxy_logger.exception().
            # Including it in the SSE response leaks internal details to clients.
            error_msg: Final = str(e)

        proxy_exception: Final = ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )
        error_returned: Final = json.dumps({"error": proxy_exception.to_dict()})
        yield f"data: {error_returned}\n\n"


def _get_client_requested_model_for_streaming(request_data: dict) -> str:
    """
    Prefer the original client-requested model (pre-alias mapping) when available.

    Pre-call processing can rewrite `request_data["model"]` for aliasing/routing purposes.
    The OpenAI-compatible public `model` field should reflect what the client sent.
    """
    requested_model = request_data.get("_litellm_client_requested_model")
    if isinstance(requested_model, str):
        return requested_model

    requested_model = request_data.get("model")
    return requested_model if isinstance(requested_model, str) else ""


def _is_positive_int_like(value: str | float | None) -> bool:
    try:
        return value is not None and int(value) > 0
    except (TypeError, ValueError):
        return False


def _should_include_fallback_errors(request_data: dict[str, object]) -> bool:
    if not general_settings.get("expose_fallback_errors_to_caller"):
        return False
    return request_data.get("include_fallback_errors") is True


def _get_streaming_fallback_metadata(
    response_obj: object,
) -> tuple[bool, str | None, list[dict[str, object]]]:
    additional_headers: Final = get_hidden_params_dict(response_obj).get("additional_headers")
    if not isinstance(additional_headers, dict):
        return False, None, []

    if not _is_positive_int_like(additional_headers.get("x-litellm-attempted-fallbacks")):
        return False, None, []

    fallback_model: Final = additional_headers.get("x-litellm-model-group")
    fallback_errors: Final = get_fallback_errors_from_headers(additional_headers)
    if isinstance(fallback_model, str) and fallback_model:
        return True, fallback_model, fallback_errors
    return True, None, fallback_errors


def _format_fallback_metadata_sse_event(
    *,
    fallback_model: str | None,
    fallback_errors: list[dict[str, object]],
) -> str:
    import time

    payload: Final = {
        "id": "litellm-fallback-metadata",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": fallback_model or "",
        "choices": [],
        "litellm_fallback": {
            "fallback_model": fallback_model,
            "errors": fallback_errors,
        },
    }
    return f"data: {json.dumps(payload)}\n\n"


def _restamp_streaming_chunk_model(
    *,
    chunk: Any,
    requested_model_from_client: str,
    request_data: dict,
    model_mismatch_logged: bool,
    fallback_was_attempted: bool = False,
    fallback_model_from_metadata: str | None = None,
) -> tuple[Any, bool]:
    if _should_return_raw_model_name(request_data):
        return chunk, model_mismatch_logged

    target_model: Final = fallback_model_from_metadata if fallback_was_attempted else requested_model_from_client
    # Always return the client-requested model name (not provider-prefixed internal identifiers)
    # on streaming chunks.
    # On fallback, use the public OpenAI-compatible model name. This keeps
    # provider-prefixed internal identifiers from leaking into the public API.
    #
    # Note: This warning is intentionally verbose. A mismatch is a useful signal that an
    # internal provider/deployment identifier is leaking into the public API, and helps
    # maintainers/operators catch regressions while preserving OpenAI-compatible output.
    if not target_model or not isinstance(chunk, (BaseModel, dict)):
        return chunk, model_mismatch_logged

    # For Azure Model Router, preserve the actual model used in each chunk
    if not fallback_was_attempted and _is_azure_model_router_request(requested_model_from_client):
        return chunk, model_mismatch_logged

    # For fastest_response batch completions, preserve the winning model's name
    # instead of stamping the comma-separated list the client sent.
    if not fallback_was_attempted and request_data.get("fastest_response", False):
        return chunk, model_mismatch_logged

    downstream_model: Final = chunk.get("model") if isinstance(chunk, dict) else getattr(chunk, "model", None)
    if downstream_model == target_model:
        return chunk, model_mismatch_logged

    if not model_mismatch_logged and downstream_model != target_model:
        verbose_proxy_logger.debug(
            "litellm_call_id=%s: streaming chunk model mismatch - target=%r downstream=%r fallback_was_attempted=%s. Overriding chunk model to target.",
            request_data.get("litellm_call_id"),
            target_model,
            downstream_model,
            fallback_was_attempted,
        )
        model_mismatch_logged = True

    if isinstance(chunk, dict):
        chunk["model"] = target_model
        return chunk, model_mismatch_logged

    try:
        chunk.model = target_model
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm_call_id=%s: failed to override chunk.model=%r on chunk_type=%s. error=%s",
            request_data.get("litellm_call_id"),
            target_model,
            type(chunk),
            str(e),
            exc_info=True,
        )

    return chunk, model_mismatch_logged


def _fast_serialize_simple_model_response_stream(
    chunk: ModelResponseStream,
) -> bytes | None:
    """
    Serialize the common OpenAI text streaming chunk without the full Pydantic
    serializer. Fall back for richer chunks so tool calls, logprobs, usage, and
    provider-specific fields keep the canonical model_dump_json behavior.
    """
    if (
        getattr(chunk, "provider_specific_fields", None) is not None
        or getattr(chunk, "system_fingerprint", None) is not None
        or getattr(chunk, "usage", None) is not None
    ):
        return None

    choices: Final = getattr(chunk, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        return None

    choice: Final = choices[0]
    if getattr(choice, "logprobs", None) is not None or getattr(choice, "enhancements", None) is not None:
        return None

    delta: Final = getattr(choice, "delta", None)
    if delta is None:
        return None

    unsupported_delta_fields: Final = (
        "function_call",
        "tool_calls",
        "audio",
        "images",
        "annotations",
        "reasoning_content",
        "thinking_blocks",
        "provider_specific_fields",
        "refusal",
    )
    if any(getattr(delta, field, None) is not None for field in unsupported_delta_fields):
        return None

    delta_dict: Final[dict] = {}
    role: Final = getattr(delta, "role", None)
    content: Final = getattr(delta, "content", None)
    if role is not None:
        delta_dict["role"] = role
    if content is not None:
        delta_dict["content"] = content

    choice_dict: Final = {"index": getattr(choice, "index", 0), "delta": delta_dict}
    finish_reason: Final = getattr(choice, "finish_reason", None)
    if finish_reason is not None:
        choice_dict["finish_reason"] = finish_reason

    # Match the canonical ``model_dump_json(exclude_none=True)`` shape — if a
    # field is None, omit it entirely rather than emitting ``"key": null``.
    # Strict OpenAI-compatible clients reject ``null`` for optional fields like
    # ``model``, so diverging here would surface as a client-side regression
    # only on the fast path. Fall back to the slow path if a required-looking
    # top-level identifier is missing.
    model: Final = getattr(chunk, "model", None)
    if model is None:
        return None

    payload: Final[dict] = {
        "id": getattr(chunk, "id", None),
        "object": getattr(chunk, "object", None),
        "created": getattr(chunk, "created", None),
        "model": model,
        "choices": [choice_dict],
    }
    for top_level_key in ("id", "object", "created"):
        if payload[top_level_key] is None:
            payload.pop(top_level_key)
    return orjson.dumps(payload)


def _serialize_streaming_chunk(chunk: BaseModel) -> str | bytes:
    if isinstance(chunk, ModelResponseStream):
        serialized_chunk: Final = _fast_serialize_simple_model_response_stream(chunk)
        if serialized_chunk is not None:
            return serialized_chunk

    return chunk.model_dump_json(exclude_none=True, exclude_unset=True)


def _is_injected_stream_usage_artifact(chunk: object) -> bool:
    if not isinstance(chunk, ModelResponseStream):
        return False
    if chunk.provider_specific_fields is not None:
        return False
    return all(_is_empty_streaming_choice(choice) for choice in chunk.choices or [])


def _is_empty_streaming_choice(choice: StreamingChoices) -> bool:
    if choice.finish_reason is not None:
        return False
    if getattr(choice, "logprobs", None) is not None:
        return False
    delta: Final = getattr(choice, "delta", None)
    if delta is None:
        return True
    return all(value is None for value in delta.model_dump().values())


async def _apply_streaming_chunk_hooks(
    *,
    chunk: Any,
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict,
    str_so_far: str,
) -> tuple[Any, str]:
    chunk = await proxy_logging_obj.async_post_call_streaming_hook(
        user_api_key_dict=user_api_key_dict,
        response=chunk,
        data=request_data,
        str_so_far=str_so_far if str_so_far else None,
    )

    if isinstance(chunk, (ModelResponse, ModelResponseStream)):
        response_str: Final = litellm.get_response_string(response_obj=chunk)
        str_so_far += response_str

    return chunk, str_so_far


def _format_streaming_sse_chunk(chunk: str | bytes) -> str | bytes:
    if isinstance(chunk, bytes):
        return b"data: " + chunk + b"\n\n"
    return f"data: {chunk}\n\n"


_SSE_FRAME_DELIMITERS: Final = ("\r\n\r\n", "\n\n", "\r\r")
_MAX_RAW_SSE_BUFFER_CHARS: Final = 8 * 1024 * 1024


def _pop_complete_sse_frame(buffer: str) -> tuple[str | None, str]:
    delimiter_positions: Final = [
        (position, delimiter) for delimiter in _SSE_FRAME_DELIMITERS if (position := buffer.find(delimiter)) != -1
    ]
    if not delimiter_positions:
        return None, buffer

    position, delimiter = min(delimiter_positions, key=lambda item: item[0])
    frame_end: Final = position + len(delimiter)
    return buffer[:frame_end], buffer[frame_end:]


_STREAM_KEEPALIVE: Final = object()

_KEEPALIVE_MIN_SECONDS: Final = 1.0
_KEEPALIVE_MAX_SECONDS: Final = 300.0
_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})


async def _iter_with_keepalive(
    aiter: AsyncIterator[Any],
    resolve_keepalive_seconds: Callable[[object], float],
    keepalive_seconds: float,
) -> AsyncGenerator[Any, None]:
    """Wrap `aiter` with idle-gap heartbeats, re-resolving the interval after each
    real chunk via `resolve_keepalive_seconds`. A mid-stream router fallback can
    swap in a deployment with a different keepalive policy, including one that
    newly enables or newly disables heartbeats, partway through the same stream;
    re-resolving against each chunk's own identity (rather than trusting the
    interval picked before iteration started, or picked the last time it went
    inactive) keeps the heartbeat behavior in sync with whichever deployment
    actually produced it, in both directions. While the interval is <= 0, no
    task is created and no timeout is awaited: a chunk is forwarded the moment
    it arrives, at the same cost as a bare `async for`."""
    pending: asyncio.Task[Any] | None = None  # rebind-ok: rebound each loop iteration
    current_keepalive_seconds = keepalive_seconds  # rebind-ok: re-resolved after each chunk
    try:
        while True:
            if current_keepalive_seconds <= 0:
                try:
                    item = await aiter.__anext__()
                except StopAsyncIteration:
                    break
                yield item
                current_keepalive_seconds = resolve_keepalive_seconds(item)
                continue

            if pending is None:
                pending = asyncio.create_task(aiter.__anext__())
            done, _ = await asyncio.wait((pending,), timeout=current_keepalive_seconds)
            if not done:
                yield _STREAM_KEEPALIVE
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                break
            finally:
                pending = None
            yield item
            current_keepalive_seconds = resolve_keepalive_seconds(item)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass


class _DeploymentKeepaliveConfig(NamedTuple):
    keepalive_seconds: object
    allow_client_override: bool


def _keepalive_from_deployment_config(
    request_data: Mapping[str, Any], response: object
) -> _DeploymentKeepaliveConfig | None:
    if llm_router is None:
        return None

    hidden: Final = get_hidden_params_dict(response)
    model_id: Final = hidden.get("model_id")
    if isinstance(model_id, str) and model_id:
        deployment: Final = llm_router.get_deployment(model_id=model_id)
        # A populated model_id names the specific deployment that served this
        # stream. If it no longer resolves (e.g. removed by a config reload
        # mid-stream), that's a stale identity, not an absent one: don't fall
        # through to guessing via model_name below, since a currently-live
        # sibling deployment's config was never what actually served this
        # stream.
        if deployment is None:
            return None
        return _DeploymentKeepaliveConfig(
            keepalive_seconds=getattr(deployment.litellm_params, "keepalive_seconds", None),
            allow_client_override=bool(getattr(deployment.litellm_params, "allow_client_keepalive_override", False)),
        )

    # No model_id at all to pin down which deployment actually served this
    # stream: only trust the fallback when every deployment under this
    # model_name agrees on both keepalive_seconds and
    # allow_client_keepalive_override (including deployments that leave either
    # field unset), so a stream never inherits a sibling deployment's policy.
    configs: Final = frozenset(
        (
            (deployment_dict.get("litellm_params") or _EMPTY_MAPPING).get("keepalive_seconds"),
            bool(
                (deployment_dict.get("litellm_params") or _EMPTY_MAPPING).get("allow_client_keepalive_override", False)
            ),
        )
        for deployment_dict in llm_router.get_model_list(model_name=request_data.get("model")) or ()
    )
    if len(configs) == 1:
        keepalive_seconds, allow_client_override = next(iter(configs))
        return _DeploymentKeepaliveConfig(
            keepalive_seconds=keepalive_seconds, allow_client_override=allow_client_override
        )
    return None


def _is_explicit_keepalive_disable(raw: object) -> bool:
    if not isinstance(raw, (int, float, str)):
        return False
    try:
        return float(raw) <= 0
    except ValueError:
        return False


def _resolve_keepalive_seconds(request_data: Mapping[str, object], response: object = None) -> float:
    deployment_config: Final = _keepalive_from_deployment_config(request_data, response)
    deployment_raw: Final = deployment_config.keepalive_seconds if deployment_config is not None else None
    allow_client_override: Final = deployment_config.allow_client_override if deployment_config is not None else False

    # An operator setting keepalive_seconds: 0 on a deployment is an explicit hard
    # disable: an authenticated client must not be able to re-enable heartbeats
    # (and the idle-timeout evasion that comes with them) for a deployment the
    # operator opted out of, regardless of what the request body asks for.
    if _is_explicit_keepalive_disable(deployment_raw):
        return 0.0

    # keepalive_seconds is operator-only unless the deployment explicitly opts in:
    # a client can't unilaterally enable heartbeats (and the LB-idle-timeout
    # evasion that comes with them) for a deployment that never configured this.
    # When neither the request nor the deployment sets a value, the operator's
    # global `litellm_settings.sse_keepalive_ping_interval_seconds` applies; a
    # deployment's explicit `keepalive_seconds: 0` above still hard-disables it.
    client_supplied: Final = request_data.get("keepalive_seconds") if allow_client_override else None
    raw: Final = (
        client_supplied
        if client_supplied is not None
        else deployment_raw
        if deployment_raw is not None
        else litellm.sse_keepalive_ping_interval_seconds
    )
    try:
        value: Final = float(raw) if isinstance(raw, (int, float, str)) else 0.0
    except ValueError:
        return 0.0
    if value <= 0:
        return 0.0
    clamped: Final = max(_KEEPALIVE_MIN_SECONDS, min(value, _KEEPALIVE_MAX_SECONDS))
    if clamped != value:
        verbose_proxy_logger.info(
            "keepalive_seconds=%s clamped to %s [min=%s, max=%s]",
            value,
            clamped,
            _KEEPALIVE_MIN_SECONDS,
            _KEEPALIVE_MAX_SECONDS,
        )
    return clamped


_KEEPALIVE_CACHE_TTL_SECONDS: Final = 5.0


def _make_keepalive_resolver(request_data: Mapping[str, object]) -> Callable[[object], float]:
    """Wrap `_resolve_keepalive_seconds` with a memo keyed on the serving
    deployment's model_id. The steady-state case (no mid-stream fallback, the
    overwhelming majority of streams) sees the same model_id on every chunk, so
    this turns the per-chunk cost from a full `llm_router.get_deployment()`
    Pydantic rebuild into a cheap hidden-params read once per
    `_KEEPALIVE_CACHE_TTL_SECONDS` for that model_id. The cache expires on its
    own rather than living for the life of the stream, so an operator's live
    config change (disabling keepalive, revoking client override, or removing
    the deployment) is observed within a bounded window instead of being able
    to be evaded by an already-in-flight stream indefinitely. A missing/empty
    model_id can't be trusted as a cache key (see
    `_keepalive_from_deployment_config`'s model_name fallback, which reflects
    current router state rather than one deployment's fixed identity), so
    those chunks always resolve fresh, matching prior behavior exactly.
    """
    last_model_id: str | None = None  # rebind-ok: memoized identity of the last-resolved chunk
    last_value: float = 0.0  # rebind-ok: cached resolution for last_model_id
    last_resolved_at: float = float("-inf")  # rebind-ok: monotonic timestamp of the last real resolution

    def _resolve(item: object) -> float:
        nonlocal last_model_id, last_value, last_resolved_at
        model_id = get_hidden_params_dict(item).get("model_id")
        now: Final = time.monotonic()
        if (
            isinstance(model_id, str)
            and model_id
            and model_id == last_model_id
            and now - last_resolved_at < _KEEPALIVE_CACHE_TTL_SECONDS
        ):
            return last_value
        value: Final = _resolve_keepalive_seconds(request_data, item)
        if isinstance(model_id, str) and model_id:
            last_model_id, last_value, last_resolved_at = model_id, value, now
        return value

    return _resolve


async def async_data_generator(
    response,
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict,
    request: Request | None = None,
):
    verbose_proxy_logger.debug("inside generator")
    stream_completed = False
    client_disconnected = False
    try:
        error_message: str | None = None
        requested_model_from_client: Final = _get_client_requested_model_for_streaming(request_data=request_data)
        (
            fallback_was_attempted,
            fallback_model_from_metadata,
            fallback_errors,
        ) = _get_streaming_fallback_metadata(response)
        model_mismatch_logged = False
        fallback_metadata_event_sent = False
        include_fallback_errors: Final = _should_include_fallback_errors(request_data)
        # Use a running string instead of list + join to avoid O(n^2) overhead.
        # Previously "".join(str_so_far_parts) was called every chunk, re-joining
        # the entire accumulated response. String += is O(n) amortized total.
        _str_so_far: str = ""
        # Separate iterator-level vs per-chunk hook decisions. The iterator
        # wrap is needed when any callback overrides
        # ``async_post_call_streaming_iterator_hook`` or has
        # ``apply_guardrail``; the per-chunk hook (which builds ``str_so_far``
        # and calls ``async_post_call_streaming_hook``) is only needed when
        # there is an active CustomGuardrail or a class that overrides the
        # per-chunk hook. Coalescing them into a single flag forced wasted
        # ``get_response_string`` work per chunk on every deployment that
        # happened to ship a streaming-iterator override (the default).
        needs_iterator_wrap: Final = proxy_logging_obj.needs_iterator_wrap()
        needs_per_chunk_hook: Final = proxy_logging_obj.needs_per_chunk_streaming_hook()
        is_raw_sse_stream: Final = bool(request_data.get("_litellm_raw_sse_stream"))
        strip_stream_usage: Final = bool(request_data.get("_litellm_strip_stream_usage"))
        raw_sse_buffer = ""

        if needs_iterator_wrap:
            stream_iterator = proxy_logging_obj.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=response,
                request_data=request_data,
            )
        else:
            stream_iterator = response

        # A stream can start on a deployment with keepalive off and fall back
        # mid-stream to one that enables it: only skip wrapping altogether when
        # there's no router to ever fall back through AND the resolved interval
        # (including the global sse_keepalive_ping_interval_seconds fallback)
        # starts disabled, not merely because the first chunk's deployment
        # happens to start with it off.
        resolve_keepalive_seconds: Final = _make_keepalive_resolver(request_data)
        initial_keepalive_seconds: Final = resolve_keepalive_seconds(response)
        stream_source: Final = (
            _iter_with_keepalive(
                stream_iterator.__aiter__(),
                resolve_keepalive_seconds,
                initial_keepalive_seconds,
            )
            if llm_router is not None or initial_keepalive_seconds > 0
            else stream_iterator
        )

        async for item in stream_source:
            if item is _STREAM_KEEPALIVE:
                yield ": ping\n\n"
                continue
            chunk = cast(Any, item)  # cast-ok: sentinel already handled above, item is a real chunk here
            if needs_per_chunk_hook:
                ### CALL HOOKS ### - modify outgoing data
                chunk, _str_so_far = await _apply_streaming_chunk_hooks(
                    chunk=chunk,
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                    str_so_far=_str_so_far,
                )

            # Mid-stream fallbacks surface metadata on individual chunks rather than
            # the response wrapper. Keep scanning chunks until a fallback model is
            # resolved, then latch it for the rest of the stream.
            if fallback_model_from_metadata is None:
                (
                    chunk_fallback_was_attempted,
                    chunk_fallback_model,
                    chunk_fallback_errors,
                ) = _get_streaming_fallback_metadata(chunk)
                if chunk_fallback_was_attempted:
                    fallback_was_attempted = True
                    fallback_model_from_metadata = chunk_fallback_model
                    fallback_errors = fallback_errors or chunk_fallback_errors

            pending_fallback_event = (
                include_fallback_errors
                and fallback_was_attempted
                and fallback_errors
                and not fallback_metadata_event_sent
            )

            chunk, model_mismatch_logged = _restamp_streaming_chunk_model(
                chunk=chunk,
                requested_model_from_client=requested_model_from_client,
                request_data=request_data,
                model_mismatch_logged=model_mismatch_logged,
                fallback_was_attempted=fallback_was_attempted,
                fallback_model_from_metadata=fallback_model_from_metadata,
            )

            if strip_stream_usage and _is_injected_stream_usage_artifact(chunk):
                if pending_fallback_event:
                    yield _format_fallback_metadata_sse_event(
                        fallback_model=fallback_model_from_metadata,
                        fallback_errors=fallback_errors,
                    )
                    fallback_metadata_event_sent = True
                continue

            raw_passthrough = False
            if isinstance(chunk, BaseModel):
                chunk = _serialize_streaming_chunk(chunk)
            elif isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
                if is_raw_sse_stream:
                    raw_sse_buffer += chunk
                    while True:
                        frame, raw_sse_buffer = _pop_complete_sse_frame(raw_sse_buffer)
                        if frame is None:
                            break
                        yield frame
                    if len(raw_sse_buffer) > _MAX_RAW_SSE_BUFFER_CHARS:
                        raise ValueError("Raw SSE stream exceeded maximum buffered size without a frame delimiter")
                    raw_passthrough = True
                elif chunk.startswith(("data:", "event:", ":")):
                    yield (chunk if chunk.endswith(_SSE_FRAME_DELIMITERS) else chunk + "\n\n")
                    raw_passthrough = True
            elif isinstance(chunk, str) and is_raw_sse_stream:
                raw_sse_buffer += chunk
                while True:
                    frame, raw_sse_buffer = _pop_complete_sse_frame(raw_sse_buffer)
                    if frame is None:
                        break
                    yield frame
                if len(raw_sse_buffer) > _MAX_RAW_SSE_BUFFER_CHARS:
                    raise ValueError("Raw SSE stream exceeded maximum buffered size without a frame delimiter")
                raw_passthrough = True
            elif isinstance(chunk, str) and chunk.startswith("data: "):
                error_message = chunk
                break

            if not raw_passthrough:
                try:
                    yield _format_streaming_sse_chunk(chunk=chunk)
                except Exception as e:
                    yield f"data: {e}\n\n"

            if pending_fallback_event:
                yield _format_fallback_metadata_sse_event(
                    fallback_model=fallback_model_from_metadata,
                    fallback_errors=fallback_errors,
                )
                fallback_metadata_event_sent = True

        stream_completed = True
        if not needs_iterator_wrap:
            # The iterator-wrap path fires deferred logging itself; fire it
            # here for the no-wrap fast path so non-callback deployments
            # still flush their post-stream logging.
            ProxyLogging._fire_deferred_stream_logging(request_data)

        if raw_sse_buffer:
            yield (raw_sse_buffer if raw_sse_buffer.endswith(_SSE_FRAME_DELIMITERS) else raw_sse_buffer + "\n\n")

        if error_message is not None:
            yield error_message
        # OpenAI-compatible streams terminate with data: [DONE]; Google GenAI (?alt=sse) does not.
        if not request_data.get("_litellm_skip_openai_stream_done"):
            done_message: Final = "[DONE]"
            yield f"data: {done_message}\n\n"
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected mid-stream. CancelledError / GeneratorExit are
        # BaseException, so they bypass the success/failure logging callbacks
        # that normally release the pre-call max_parallel_requests +1. Flag the
        # disconnect; the shielded cleanup in `finally` owns the slot release
        # so it can coordinate with disconnect-time success billing and release
        # exactly once. This is the outermost generator Starlette closes on
        # disconnect, so it fires reliably regardless of needs_iterator_wrap
        # (a nested iterator hook would only see GeneratorExit on GC).
        if not stream_completed:
            client_disconnected = True
        raise
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.async_data_generator(): Exception occured - %s", e)
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_data,
        )
        verbose_proxy_logger.debug(
            "\x1b[1;31mAn error occurred: %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
        )

        if isinstance(e, HTTPException):
            raise e
        elif isinstance(e, StreamingCallbackError):
            error_msg = str(e)
        else:
            # Only include the error message, not the traceback.
            # The traceback is already logged above via verbose_proxy_logger.exception().
            # Including it in the SSE response leaks internal details to clients.
            error_msg = str(e)

        proxy_exception: Final = ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )
        error_returned: Final = json.dumps({"error": proxy_exception.to_dict()})
        stream_completed = True
        yield f"data: {error_returned}\n\n"
    finally:
        await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
            request=request,
            request_data=request_data,
            response=response,
            stream_completed=stream_completed,
            client_disconnected=client_disconnected,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


def select_data_generator(
    response,
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict,
    request: Request | None = None,
):
    return async_data_generator(
        response=response,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
        request=request,
    )


def get_litellm_model_info(model: dict = {}):
    model_info: Final = model.get("model_info", {})
    model_to_lookup = model.get("litellm_params", {}).get("model", None)
    try:
        if "azure" in model_to_lookup or model_info.get("base_model"):
            model_to_lookup = model_info.get("base_model", None)
        litellm_model_info: Final = litellm.get_model_info(model_to_lookup)
        return litellm_model_info
    except Exception:
        # this should not block returning on /model/info
        # if litellm does not have info on the model it should return {}
        return {}


def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    verbose_proxy_logger.debug("Backing off... this was attempt # %s", details["tries"])


def giveup(e):
    result: Final = not (
        isinstance(e, ProxyException)
        and getattr(e, "message", None) is not None
        and isinstance(e.message, str)
        and "Max parallel request limit reached" in e.message
    )

    if general_settings.get("disable_retry_on_max_parallel_request_limit_error") is True:
        return True  # giveup if queuing max parallel request limits is disabled

    if result:
        verbose_proxy_logger.debug(json.dumps({"event": "giveup", "exception": str(e)}))
    return result


class ProxyStartupEvent:
    @staticmethod
    def _warn_budget_without_db(max_budget: float | None, prisma_client: PrismaClient | None) -> None:
        if prisma_client is not None or not max_budget or max_budget <= 0:
            return

        verbose_proxy_logger.warning(
            "A proxy-wide budget (litellm.max_budget=%s) is configured but no database is connected, "
            "so the budget will NOT be enforced and requests will never be blocked. Set DATABASE_URL or "
            "general_settings.database_url and restart. Redis and fail_closed_budget_enforcement do not "
            "cover the proxy-wide budget because there is no global spend counter; Redis alone is not a substitute.",
            max_budget,
        )

    @classmethod
    def _initialize_startup_logging(
        cls,
        llm_router: Router | None,
        proxy_logging_obj: ProxyLogging,
        redis_usage_cache: RedisCache | None,
    ):
        """Initialize logging and alerting on startup"""
        ## COST TRACKING ##
        cost_tracking()

        proxy_logging_obj.startup_event(llm_router=llm_router, redis_usage_cache=redis_usage_cache)

    @staticmethod
    def _warn_if_mock_testing_params_enabled(general_settings: dict) -> None:
        """Announce, loudly, that any caller may inject synthetic failures."""
        from litellm.proxy.route_llm_request import (
            GATED_MOCK_PARAM_NAMES,
            MOCK_TESTING_CONFIG_KEY,
        )

        if general_settings.get(MOCK_TESTING_CONFIG_KEY, False) is not True:
            return

        verbose_proxy_logger.warning(
            "\n%s\n"
            " DANGEROUS SETTING ENABLED\n"
            " general_settings.%s = true\n"
            "\n"
            " Any caller with a valid key on this proxy can now inject synthetic\n"
            " failures and latency into their own requests using these body params:\n"
            "%s\n"
            "\n"
            " A request using them consumes a connection and a concurrency slot\n"
            " without reaching a provider, and returns an error the caller chose.\n"
            "\n"
            " Intended for testing fallback chains. Do not leave enabled.\n"
            "%s",
            "=" * 72,
            MOCK_TESTING_CONFIG_KEY,
            "\n".join(f"   {name}" for name in GATED_MOCK_PARAM_NAMES),
            "=" * 72,
        )

    @staticmethod
    def _validate_redis_transaction_buffer_config(
        general_settings: dict,
        redis_usage_cache: RedisCache | None,
    ):
        """
        Validates that when use_redis_transaction_buffer is enabled,
        a Redis cache is properly configured in litellm_settings.
        """
        from litellm.secret_managers.main import str_to_bool

        _use_redis_transaction_buffer: bool | str | None = general_settings.get("use_redis_transaction_buffer", False)
        if isinstance(_use_redis_transaction_buffer, str):
            _use_redis_transaction_buffer = str_to_bool(_use_redis_transaction_buffer)

        if _use_redis_transaction_buffer and redis_usage_cache is None:
            raise ValueError(
                "`use_redis_transaction_buffer` is enabled in general_settings "
                "but no Redis is configured. This will cause spend updates "
                "to not be tracked. Add a Redis cache in litellm_settings:\n\n"
                "litellm_settings:\n"
                "  cache: true\n"
                "  cache_params:\n"
                "    type: redis\n"
                "    url: os.environ/REDIS_URL\n\n"
                "or set REDIS_* environment variables (e.g. REDIS_HOST, "
                "REDIS_PORT, REDIS_PASSWORD, or REDIS_URL) to use a standalone "
                "Redis for the transaction buffer."
            )

    @staticmethod
    async def _init_coordination_redis_from_db(
        litellm_settings: Mapping[str, object],
        llm_router: Router | None,
    ) -> RedisCache | None:
        """
        Applies a coordination_redis block saved to the database, which the admin
        UI writes and the config file therefore never carries.

        Returns None when nothing is persisted or the persisted block names no
        connection target, leaving the file/env resolution untouched.
        """
        try:
            persisted: Final = await get_persisted_coordination_redis_settings()
        except Exception as e:  # noqa: BLE001  # a config-row read failure must not block proxy startup
            verbose_proxy_logger.warning("Could not read coordination_redis from the database: %s", e)
            return None
        if persisted is None:
            return None

        coordination_params: Final = CoordinationRedisParams(**_resolve_coordination_redis_env_refs(persisted))
        if not coordination_params.has_connection_target():
            verbose_proxy_logger.warning(
                "coordination_redis saved in the database names no connection target; ignoring it."
            )
            return None

        coordination_redis_cache: Final = _build_redis_usage_cache(coordination_params.model_dump(exclude_none=True))
        _attach_redis_usage_cache(
            coordination_redis_cache,
            enable_redis_auth_cache=litellm_settings.get("enable_redis_auth_cache", False) is True,
        )
        if llm_router is not None and llm_router.cache.redis_cache is None:
            llm_router._update_redis_cache(cache=coordination_redis_cache)
        verbose_proxy_logger.info(
            "coordination_redis: using the standalone Redis saved in the database "
            "for usage tracking, rate limiting, and cross-pod coordination."
        )
        return coordination_redis_cache

    @staticmethod
    def _get_transaction_buffer_redis_cache(
        general_settings: dict,
    ) -> RedisCache | None:
        """
        Builds a standalone Redis cache from REDIS_* environment variables so
        use_redis_transaction_buffer can run when the proxy cache backend is not
        Redis (e.g. disk, s3).

        Returns None when the buffer is disabled, or when no Redis host or url
        is set in the environment.
        """
        from litellm.secret_managers.main import str_to_bool

        _use_redis_transaction_buffer: bool | str | None = general_settings.get("use_redis_transaction_buffer", False)
        if isinstance(_use_redis_transaction_buffer, str):
            _use_redis_transaction_buffer = str_to_bool(_use_redis_transaction_buffer)

        if not _use_redis_transaction_buffer:
            return None

        return _build_redis_usage_cache_from_environment()

    @classmethod
    async def _initialize_semantic_tool_filter(
        cls,
        llm_router: Router | None,
        litellm_settings: dict[str, Any],
    ):
        """Initialize MCP semantic tool filter if configured"""
        from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook

        mcp_semantic_filter_config: Final = litellm_settings.get("mcp_semantic_tool_filter", None)

        # Only proceed if the feature is configured and enabled
        if not mcp_semantic_filter_config or not mcp_semantic_filter_config.get("enabled", False):
            verbose_proxy_logger.debug("Semantic tool filter not configured or not enabled, skipping initialization")
            return

        verbose_proxy_logger.debug(
            "Initializing semantic tool filter: llm_router=%s, config=%s",
            llm_router is not None,
            mcp_semantic_filter_config,
        )
        hook: Final = await SemanticToolFilterHook.initialize_from_config(
            config=mcp_semantic_filter_config,
            llm_router=llm_router,
        )

        if hook:
            verbose_proxy_logger.debug("Semantic tool filter hook registered")
            litellm.logging_callback_manager.add_litellm_callback(hook)
        else:
            # Only warn if the feature was configured but failed to initialize
            verbose_proxy_logger.warning("Semantic tool filter hook was configured but failed to initialize")

    @classmethod
    def _initialize_jwt_auth(
        cls,
        general_settings: dict,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache,
    ):
        """Initialize JWT auth on startup"""
        if general_settings.get("litellm_jwtauth", None) is not None:
            for k, v in general_settings["litellm_jwtauth"].items():
                if isinstance(v, str) and v.startswith("os.environ/"):
                    general_settings["litellm_jwtauth"][k] = get_secret(v)
            # ``user_config_file_path`` is set by ``ProxyConfig._get_config_from_file``
            # during startup. Threading it through lets an operator-
            # configured ``custom_validate: s3://...`` resolve through
            # the runtime gate; admin-API JWT config writes (no config
            # file context) hit the gate and refuse remote loads.
            litellm_jwtauth = LiteLLM_JWTAuth(
                config_file_path=user_config_file_path,
                **general_settings["litellm_jwtauth"],
            )
        else:
            litellm_jwtauth = LiteLLM_JWTAuth()
        jwt_handler.update_environment(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_jwtauth=litellm_jwtauth,
        )

    @classmethod
    def _add_proxy_budget_to_db(cls):
        """Adds a global proxy budget to db"""
        if litellm.budget_duration is None:
            raise Exception("budget_duration not set on Proxy. budget_duration is required to use max_budget.")

        asyncio.create_task(cls._upsert_proxy_budget_with_reset_at_backfill())

    @classmethod
    async def _upsert_proxy_budget_with_reset_at_backfill(cls) -> None:
        """
        Upsert the proxy budget aggregate user row with the configured
        max_budget / budget_duration, then backfill budget_reset_at if
        currently NULL.

        The backfill uses `WHERE budget_reset_at IS NULL` so it only fires
        when the row pre-existed without a reset schedule (e.g. row created
        via a different path before the proxy budget was configured). On
        subsequent restarts it no-ops, so an active reset window is never
        slid forward. It also zeroes spend at that moment: a row that was
        never on a reset schedule holds lifetime accrual, which must not
        gate the first duration window.
        """
        await generate_key_helper_fn(
            request_type="user",
            table_name="user",
            user_id=LITELLM_PROXY_BUDGET_NAME,
            duration=None,
            models=[],
            aliases={},
            config={},
            spend=0,
            max_budget=litellm.max_budget,
            budget_duration=litellm.budget_duration,
            query_type="update_data",
            update_key_values={
                "max_budget": litellm.max_budget,
                "budget_duration": litellm.budget_duration,
            },
        )

        # Without this, the upsert leaves budget_reset_at=NULL on rows that
        # took the UPDATE path, and reset_budget_for_litellm_users never
        # matches them (NULL < now() is unknown in SQL) — so the proxy-wide
        # spend cap blocks forever once it's hit.
        if prisma_client is not None and litellm.budget_duration is not None:
            try:
                await UserRepository(prisma_client).table.update_many(
                    where={
                        "user_id": LITELLM_PROXY_BUDGET_NAME,
                        "budget_reset_at": None,
                    },
                    data={
                        "budget_reset_at": get_budget_reset_time(budget_duration=litellm.budget_duration),
                        "spend": 0,
                    },
                )
            except Exception as e:
                verbose_proxy_logger.warning("Failed to backfill budget_reset_at on proxy admin row: %s", e)

    @classmethod
    async def _warm_global_spend_cache(
        cls,
        user_api_key_cache: UserApiKeyCache,
        prisma_client: PrismaClient,
    ) -> None:
        """Warm global spend cache once at startup to reduce impact of first wave of requests."""
        try:
            cache_key: Final = GLOBAL_PROXY_SPEND_CACHE_KEY
            await _fetch_global_spend_with_event_coordination(
                cache_key=cache_key,
                user_api_key_cache=user_api_key_cache,
                prisma_client=prisma_client,
            )
        except Exception as e:
            verbose_proxy_logger.debug("Global spend cache warm-up at startup skipped or failed: %s", e)

    @classmethod
    async def _update_default_team_member_budget(cls):
        """Update the default team member budget"""
        if litellm.default_internal_user_params is None:
            return

        _teams: Final = litellm.default_internal_user_params.get("teams") or []
        if _teams and all(isinstance(team, dict) for team in _teams):
            from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
                update_default_team_member_budget,
            )

            teams_pydantic_obj: Final = [NewUserRequestTeam(**team) for team in _teams]
            await update_default_team_member_budget(
                teams=teams_pydantic_obj,
                user_api_key_dict=UserAPIKeyAuth(token=hash_token(master_key)),
            )

    @classmethod
    async def _sync_ui_settings_to_general_settings(cls):
        """
        Load persisted UI settings from the database and sync runtime flags
        into general_settings so they take effect immediately after startup.
        """
        try:
            import json

            from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
                _RUNTIME_GENERAL_SETTINGS_FLAGS,
            )

            if prisma_client is None:
                return
            db_record: Final[_UISettingsRow | None] = cast(  # cast-ok: prisma Json stub is `str`, runtime is a dict
                "_UISettingsRow | None",
                await UISettingsRepository(prisma_client).table.find_unique(where={"id": "ui_settings"}),
            )
            if db_record and db_record.ui_settings:
                raw: Final = db_record.ui_settings
                ui_settings: Final = json.loads(raw) if isinstance(raw, str) else dict(raw)
                flags_to_sync: Final = {k: ui_settings[k] for k in _RUNTIME_GENERAL_SETTINGS_FLAGS if k in ui_settings}
                if flags_to_sync:
                    general_settings.update(flags_to_sync)
                    verbose_proxy_logger.info(
                        "Synced UI settings to general_settings on startup: %s",
                        list(flags_to_sync.keys()),
                    )
        except Exception as e:
            verbose_proxy_logger.debug("UI settings sync on startup skipped or failed: %s", e)

    @classmethod
    async def initialize_scheduled_background_jobs(
        cls,
        general_settings: dict,
        prisma_client: PrismaClient,
        proxy_budget_rescheduler_min_time: int,
        proxy_budget_rescheduler_max_time: int,
        proxy_batch_write_at: int,
        proxy_logging_obj: ProxyLogging,
    ) -> ProxyWorkerHeartbeat:
        """Initializes scheduled background jobs"""
        global store_model_in_db, scheduler

        # MEMORY LEAK FIX: Configure scheduler with optimized settings
        # Memray analysis showed APScheduler's normalize() and _apply_jitter() causing
        # massive memory allocations (35GB with 483M allocations)
        # Key fixes:
        # 1. Remove/minimize jitter to avoid normalize() memory explosion
        # 2. Use larger misfire_grace_time to prevent backlog calculations
        # 3. Set replace_existing=True to avoid duplicate jobs
        from apscheduler.executors.asyncio import AsyncIOExecutor
        from apscheduler.jobstores.memory import MemoryJobStore

        scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": APSCHEDULER_COALESCE,
                "misfire_grace_time": APSCHEDULER_MISFIRE_GRACE_TIME,
                "max_instances": APSCHEDULER_MAX_INSTANCES,
                # Note: replace_existing is NOT a valid job_default in APScheduler
                # It must be passed individually when calling add_job()
            },
            # Limit job store size to prevent memory growth
            jobstores={"default": MemoryJobStore()},  # explicitly use memory job store
            # Use simple executor to minimize overhead
            executors={
                "default": AsyncIOExecutor(),
            },
            # Disable timezone awareness to reduce computation
            timezone=None,
        )

        # Use fixed intervals with small random offset instead of jitter
        # This avoids the expensive jitter calculations in APScheduler
        budget_interval: Final = proxy_budget_rescheduler_min_time + random.randint(
            0,
            min(
                30,
                proxy_budget_rescheduler_max_time - proxy_budget_rescheduler_min_time,
            ),
        )

        # Ensure minimum interval of 30 seconds for batch writing to prevent memory issues
        batch_writing_interval: Final = proxy_batch_write_at + random.randint(0, 5)

        ### PROXY WORKER HEARTBEAT ###
        worker_heartbeat: Final = ProxyWorkerHeartbeat(prisma_client=prisma_client)
        await worker_heartbeat.beat()
        scheduler.add_job(
            worker_heartbeat.beat,
            "interval",
            seconds=PROXY_WORKER_HEARTBEAT_INTERVAL_SECONDS,
            id="proxy_worker_heartbeat_job",
            replace_existing=True,
            misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
        )

        ### RESET BUDGET ###
        if general_settings.get("disable_reset_budget", False) is False:
            budget_reset_job: Final = ResetBudgetJob(
                proxy_logging_obj=proxy_logging_obj,
                prisma_client=prisma_client,
                reset_settings=get_budget_reset_settings(),
                pod_lock_manager=proxy_logging_obj.db_spend_update_writer.pod_lock_manager,
            )

            scheduler.add_job(
                budget_reset_job.reset_budget,
                "interval",
                seconds=budget_interval,
                # REMOVED jitter parameter - major cause of memory leak
                id="reset_budget_job",
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )

        ### UPDATE SPEND ###
        scheduler.add_job(
            update_spend,
            "interval",
            seconds=batch_writing_interval,
            # REMOVED jitter parameter - major cause of memory leak
            args=[prisma_client, db_writer_client, proxy_logging_obj],
            id="update_spend_job",
            replace_existing=True,
            misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
        )

        ### UPDATE DAILY TAG SPEND (separate scheduler job with longer interval) ###
        ## Reduces QPS as there are more tags for a single request
        tag_spend_update_interval: Final = int(batch_writing_interval * DAILY_TAG_SPEND_BATCH_MULTIPLIER)
        from litellm.proxy.utils import update_daily_tag_spend

        scheduler.add_job(
            update_daily_tag_spend,
            "interval",
            seconds=tag_spend_update_interval,
            args=[prisma_client, proxy_logging_obj],
            id="update_daily_tag_spend_job",
            replace_existing=True,
            misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
        )
        verbose_proxy_logger.info(
            f"Tag spend update job scheduled at {tag_spend_update_interval}s interval "
            f"({tag_spend_update_interval / batch_writing_interval:.1f}x main job interval)"
        )

        ### UPDATE GATEWAY REQUEST COUNTS (SGR) ###
        scheduler.add_job(
            flush_gateway_requests,
            "interval",
            seconds=batch_writing_interval,
            args=(prisma_client, gateway_request_accumulator),
            id="update_gateway_requests_job",
            replace_existing=True,
            misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
        )

        ### MONITOR SPEND LOGS QUEUE (queue-size-based job) ###
        if general_settings.get("disable_spend_logs", False) is False:
            from litellm.proxy.utils import _monitor_spend_logs_queue

            monitor_task: Final = asyncio.create_task(
                _monitor_spend_logs_queue(
                    prisma_client=prisma_client,
                    db_writer_client=db_writer_client,
                    proxy_logging_obj=proxy_logging_obj,
                )
            )
            prisma_client.spend_logs_queue_monitor_task = monitor_task  # rebind-ok: the client owns its monitor handle

        ### ADD NEW MODELS ###
        store_model_in_db = get_secret_bool("STORE_MODEL_IN_DB", store_model_in_db) or store_model_in_db

        # If store_model_in_db is still False, check DB for override.
        # This breaks the chicken-and-egg where DB has store_model_in_db=True
        # but YAML config has False.
        if store_model_in_db is not True and prisma_client is not None:
            try:
                _db_gs_record: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
                    where={"param_name": "general_settings"}
                )
                if _db_gs_record is not None and isinstance(_db_gs_record.param_value, dict):
                    _db_val: Final = _db_gs_record.param_value.get("store_model_in_db")
                    if _db_val is True or (isinstance(_db_val, str) and _db_val.lower() == "true"):
                        store_model_in_db = True
                        verbose_proxy_logger.info("store_model_in_db=True loaded from DB, overriding config/env")
            except Exception as e:
                verbose_proxy_logger.debug("Failed to check DB for store_model_in_db: %s", str(e))

        config_reload_interval_seconds = proxy_config_reload_interval_seconds
        if not isinstance(config_reload_interval_seconds, int) or config_reload_interval_seconds <= 0:
            verbose_proxy_logger.warning(
                "proxy_config_reload_interval_seconds=%s must be a positive integer; falling back to 30s",
                config_reload_interval_seconds,
            )
            config_reload_interval_seconds = 30

        ### PERIODIC RELOADS (model cost map, anthropic beta headers) ###
        scheduler.add_job(
            proxy_config.check_periodic_reloads,
            "interval",
            seconds=config_reload_interval_seconds,
            args=[prisma_client],
            id="periodic_reload_job",
            replace_existing=True,
            misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
        )

        proxy_config.start_auth_cache_invalidation_subscriber(
            redis_cache=redis_usage_cache,
            user_api_key_cache=user_api_key_cache,
        )

        if store_model_in_db is True:
            ### GET STORED CREDENTIALS ###
            scheduler.add_job(
                proxy_config.get_credentials,
                "interval",
                seconds=config_reload_interval_seconds,
                # REMOVED jitter parameter - major cause of memory leak
                args=[prisma_client],
                id="get_credentials_job",
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )
            await proxy_config.get_credentials(prisma_client=prisma_client)

            # MEMORY LEAK FIX: Increase interval from 10s to 30s minimum
            # Frequent polling was causing excessive memory allocations
            scheduler.add_job(
                proxy_config.add_deployment,
                "interval",
                seconds=config_reload_interval_seconds,
                # REMOVED jitter parameter - major cause of memory leak
                args=[prisma_client, proxy_logging_obj],
                id="add_deployment_job",
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )

            # this will load all existing models on proxy startup
            await proxy_config.add_deployment(prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj)

            proxy_config.start_config_sync_subscriber(
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
                redis_cache=redis_usage_cache,
            )

        if store_model_in_db is not True:
            await proxy_config.init_mcp_servers_from_db()
            # Without this branch's own refresh, a UI-created search tool never reaches the router:
            # the add_deployment job that carries it in store_model_in_db=True mode is not scheduled.
            await proxy_config.reload_search_tools_from_db()
            if prisma_client is not None:
                scheduler.add_job(
                    proxy_config.reload_search_tools_from_db,
                    "interval",
                    seconds=config_reload_interval_seconds,
                    id="reload_search_tools_job",
                    replace_existing=True,
                    misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                )
                # DB-backed MCP servers are live objects in every mode, so the registry refresh that
                # store_model_in_db=True deployments get via the add_deployment job must run here
                # too; without it, a server whose OAuth discovery failed at startup is rebuilt only
                # by a management write, since the reload fast path is the retry's only driver.
                mcp_reload_interval_seconds = proxy_config_reload_interval_seconds
                if not isinstance(mcp_reload_interval_seconds, int) or mcp_reload_interval_seconds <= 0:
                    mcp_reload_interval_seconds = 30
                scheduler.add_job(
                    proxy_config.reload_mcp_servers_from_db,
                    "interval",
                    seconds=mcp_reload_interval_seconds,
                    id="reload_mcp_servers_job",
                    replace_existing=True,
                    misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                )

        await cls._initialize_slack_alerting_jobs(
            scheduler=scheduler,
            general_settings=general_settings,
            proxy_logging_obj=proxy_logging_obj,
            prisma_client=prisma_client,
        )

        await cls._initialize_spend_tracking_background_jobs(scheduler=scheduler)

        ### PTU DAILY ROLLUP ###
        from litellm.proxy.spend_tracking.ptu_feature_flag import (
            is_ptu_cost_attribution_enabled,
        )

        if is_ptu_cost_attribution_enabled():
            from litellm.proxy.spend_tracking.ptu_flat_cost_rollup import (
                PTU_ROLLUP_JOB_ID,
                run_scheduled_ptu_rollup,
            )

            async def _alert_ptu_rollup_failure(message: str) -> None:
                await proxy_logging_obj.alerting_handler(
                    message=message,
                    level="High",
                    alert_type=AlertType.failed_tracking_spend,
                )

            async def _scheduled_ptu_rollup() -> None:
                # Reuse the PodLockManager from db_spend_update_writer so only one pod
                # reconciles a day; a multi-pod race could prune another pod's fresh rows
                await run_scheduled_ptu_rollup(
                    prisma_client,
                    pod_lock_manager=proxy_logging_obj.db_spend_update_writer.pod_lock_manager,
                    alert=_alert_ptu_rollup_failure,
                )

            scheduler.add_job(
                _scheduled_ptu_rollup,
                "cron",
                hour=0,
                minute=15,
                timezone="UTC",
                id=PTU_ROLLUP_JOB_ID,
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )
            verbose_proxy_logger.info(
                "PTU rollup job scheduled at 00:15 UTC daily (only models with PTU config accrue flat cost)"
            )

        ### SPEND LOG CLEANUP ###
        if (
            general_settings.get("maximum_spend_logs_retention_period") is not None
            or general_settings.get("maximum_autorouter_session_retention_period") is not None
            or general_settings.get("maximum_health_check_retention_period") is not None
        ):
            spend_log_cleanup: Final = SpendLogCleanup()
            cleanup_cron: Final = general_settings.get("maximum_spend_logs_cleanup_cron")

            if cleanup_cron:
                from apscheduler.triggers.cron import CronTrigger

                try:
                    cron_trigger: Final = CronTrigger.from_crontab(cleanup_cron)
                    scheduler.add_job(
                        spend_log_cleanup.cleanup_old_spend_logs,
                        cron_trigger,
                        args=[prisma_client],
                        id="spend_log_cleanup_job",
                        replace_existing=True,
                        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                    )
                    verbose_proxy_logger.info("Spend log cleanup scheduled with cron: %s", cleanup_cron)
                except ValueError:
                    verbose_proxy_logger.error("Invalid maximum_spend_logs_cleanup_cron value: %s", cleanup_cron)
            else:
                # Interval-based scheduling (existing behavior)
                retention_interval: Final = general_settings.get("maximum_spend_logs_retention_interval", "1d")
                try:
                    interval_seconds: Final = duration_in_seconds(retention_interval)
                    scheduler.add_job(
                        spend_log_cleanup.cleanup_old_spend_logs,
                        "interval",
                        seconds=interval_seconds + random.randint(0, 60),
                        args=[prisma_client],
                        id="spend_log_cleanup_job",
                        replace_existing=True,
                        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                    )
                except ValueError:
                    verbose_proxy_logger.error("Invalid maximum_spend_logs_retention_interval value")
        ### CHECK BATCH COST ###
        if llm_router is not None and PROXY_BATCH_POLLING_ENABLED:
            try:
                from litellm_enterprise.proxy.common_utils.check_batch_cost import (
                    CheckBatchCost,
                )

                check_batch_cost_job: Final = CheckBatchCost(
                    proxy_logging_obj=proxy_logging_obj,
                    prisma_client=prisma_client,
                    llm_router=llm_router,
                    track_unmanaged_batch_cost=general_settings.get("track_unmanaged_batch_cost", False),
                )
                await check_batch_cost_job.confirm_batch_processed_support()
                scheduler.add_job(
                    check_batch_cost_job.check_batch_cost,
                    "interval",
                    seconds=proxy_batch_polling_interval + random.randint(0, 30),  # Add small random offset
                    # REMOVED jitter parameter - major cause of memory leak
                    id="check_batch_cost_job",
                    replace_existing=True,
                    misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                )
                verbose_proxy_logger.info("Batch cost check job scheduled successfully")

            except Exception as e:
                verbose_proxy_logger.debug("Failed to setup batch cost checking: %s", e)
                verbose_proxy_logger.debug(
                    "Checking batch cost for LiteLLM Managed Files is an Enterprise Feature. Skipping..."
                )

        ### CHECK RESPONSES COST ###
        if llm_router is not None and PROXY_BATCH_POLLING_ENABLED:
            try:
                from litellm_enterprise.proxy.common_utils.check_responses_cost import (
                    CheckResponsesCost,
                )

                check_responses_cost_job: Final = CheckResponsesCost(
                    proxy_logging_obj=proxy_logging_obj,
                    prisma_client=prisma_client,
                    llm_router=llm_router,
                )
                scheduler.add_job(
                    check_responses_cost_job.check_responses_cost,
                    "interval",
                    seconds=proxy_batch_polling_interval + random.randint(0, 30),  # Add small random offset
                    # REMOVED jitter parameter - major cause of memory leak
                    id="check_responses_cost_job",
                    replace_existing=True,
                    misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                )
                verbose_proxy_logger.info("Responses cost check job scheduled successfully")

            except Exception as e:
                verbose_proxy_logger.debug("Failed to setup responses cost checking: %s", e)
                verbose_proxy_logger.debug(
                    "Checking responses cost for LiteLLM Managed Files is an Enterprise Feature. Skipping..."
                )

        # MEMORY LEAK FIX: Start scheduler with paused=False to avoid backlog processing
        # Do NOT reset job times to "now" as this can trigger the memory leak
        # The misfire_grace_time and coalesce settings will handle any missed runs properly

        # Every job above anchors on this process's start instant, so without a phase offset
        # they all fire together, on every replica the rollout brought up at the same time
        attach_job_timing_logger(scheduler)
        apply_scheduled_job_stagger(
            scheduler=scheduler,
            settings=parse_stagger_settings(general_settings),
        )

        # Start the scheduler immediately without processing backlogs
        scheduler.start(paused=False)
        verbose_proxy_logger.info(
            "APScheduler started with memory leak prevention settings: removed jitter, increased intervals, misfire_grace_time=%s",
            APSCHEDULER_MISFIRE_GRACE_TIME,
        )
        return worker_heartbeat

    @classmethod
    async def _initialize_spend_tracking_background_jobs(cls, scheduler: AsyncIOScheduler):
        """
        Initialize the spend tracking and other background jobs
        1. CloudZero Background Job
        2. Focus Background Job
        3. Prometheus Background Job
        4. Key Rotation Background Job

        Args:
            scheduler: The scheduler to add the background jobs to
        """
        global prisma_client
        global proxy_logging_obj
        global user_api_key_cache

        ########################################################
        # CloudZero Background Job
        ########################################################
        from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger
        from litellm.integrations.focus.focus_logger import FocusLogger
        from litellm.proxy.spend_tracking.cloudzero_endpoints import is_cloudzero_setup

        if await is_cloudzero_setup():
            await CloudZeroLogger.init_cloudzero_background_job(scheduler=scheduler)

        ########################################################
        # Focus Background Job
        ########################################################
        await FocusLogger.init_focus_export_background_job(scheduler=scheduler)

        ########################################################
        # Vantage Background Job
        ########################################################
        from litellm.integrations.vantage.vantage_logger import VantageLogger
        from litellm.proxy.spend_tracking.vantage_endpoints import (
            _get_vantage_settings,
            is_vantage_setup,
            is_vantage_setup_in_config,
            is_vantage_setup_in_db,
        )

        if await is_vantage_setup():
            # If configured via DB but not in config.yaml callbacks,
            # instantiate and register a VantageLogger so the scheduler
            # can find it.
            if not is_vantage_setup_in_config() and await is_vantage_setup_in_db():
                try:
                    db_settings: Final = await _get_vantage_settings()
                    if db_settings:
                        vantage_logger: Final = VantageLogger(
                            api_key=db_settings.get("api_key"),
                            integration_token=db_settings.get("integration_token"),
                            base_url=db_settings.get("base_url"),
                        )
                        litellm.logging_callback_manager.add_litellm_callback(vantage_logger)
                except Exception as e:
                    verbose_proxy_logger.warning("Failed to register VantageLogger from DB settings: %s", e)
            await VantageLogger.init_vantage_background_job(scheduler=scheduler)

        ########################################################
        # Mavvrik FOCUS Background Job
        ########################################################
        from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (  # noqa: PLC0415
            MavvrikFocusLogger,
        )

        await MavvrikFocusLogger.init_mavvrik_focus_background_job(scheduler=scheduler)

        ########################################################
        # Prometheus Background Job
        ########################################################
        if litellm.prometheus_initialize_budget_metrics is True:
            from litellm.integrations.prometheus import PrometheusLogger

            PrometheusLogger.initialize_budget_metrics_cron_job(scheduler=scheduler)
        ########################################################
        # Key Rotation Background Job
        ########################################################
        from litellm.constants import (
            LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS,
            LITELLM_KEY_ROTATION_ENABLED,
        )

        key_rotation_enabled: Final[bool | None] = str_to_bool(LITELLM_KEY_ROTATION_ENABLED)
        verbose_proxy_logger.debug("key_rotation_enabled: %s", key_rotation_enabled)

        if key_rotation_enabled is True:
            try:
                from litellm.proxy.common_utils.key_rotation_manager import (
                    KeyRotationManager,
                )

                # Get prisma_client and proxy_logging_obj from global scope
                if prisma_client is not None:
                    # Reuse the PodLockManager from db_spend_update_writer
                    pod_lock_manager: Final = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
                    key_rotation_manager: Final = KeyRotationManager(
                        prisma_client,
                        pod_lock_manager=pod_lock_manager,
                    )
                    verbose_proxy_logger.debug(
                        "Key rotation background job scheduled every %s seconds (LITELLM_KEY_ROTATION_ENABLED=true)",
                        LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS,
                    )
                    scheduler.add_job(
                        key_rotation_manager.process_rotations,
                        "interval",
                        seconds=LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS,
                        id="key_rotation_job",
                    )
                else:
                    verbose_proxy_logger.warning("Key rotation enabled but prisma_client not available")
            except Exception as e:
                verbose_proxy_logger.warning("Failed to setup key rotation job: %s", e)
        else:
            verbose_proxy_logger.debug("Key rotation disabled (set LITELLM_KEY_ROTATION_ENABLED=true to enable)")

        await cls._initialize_expired_ui_session_key_cleanup_background_job(scheduler=scheduler)

    @classmethod
    async def _initialize_expired_ui_session_key_cleanup_background_job(cls, scheduler: AsyncIOScheduler):
        """
        Initialize the expired UI session key cleanup background job.
        """
        global prisma_client
        global proxy_logging_obj
        global user_api_key_cache

        ########################################################
        # Expired UI Session Key Cleanup Background Job
        ########################################################
        from litellm.constants import (
            EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
            LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_ENABLED,
            LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_INTERVAL_SECONDS,
        )

        expired_ui_session_key_cleanup_enabled: Final[bool | None] = str_to_bool(
            LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_ENABLED
        )
        verbose_proxy_logger.debug("expired_ui_session_key_cleanup_enabled: %s", expired_ui_session_key_cleanup_enabled)

        if expired_ui_session_key_cleanup_enabled is True:
            try:
                from litellm.proxy.common_utils.expired_ui_session_key_cleanup_manager import (
                    ExpiredUISessionKeyCleanupManager,
                )

                if prisma_client is not None:
                    pod_lock_manager: Final = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
                    expired_ui_session_key_cleanup_manager: Final = ExpiredUISessionKeyCleanupManager(
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        pod_lock_manager=pod_lock_manager,
                    )
                    verbose_proxy_logger.debug(
                        "Expired UI session key cleanup background job scheduled every %s seconds (LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_ENABLED=true)",
                        LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_INTERVAL_SECONDS,
                    )
                    scheduler.add_job(
                        expired_ui_session_key_cleanup_manager.cleanup_expired_keys,
                        "interval",
                        seconds=LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_INTERVAL_SECONDS,
                        id=EXPIRED_UI_SESSION_KEY_CLEANUP_JOB_NAME,
                    )
                else:
                    verbose_proxy_logger.warning(
                        "Expired UI session key cleanup enabled but prisma_client not available"
                    )
            except Exception as e:
                verbose_proxy_logger.warning("Failed to setup expired UI session key cleanup job: %s", e)
        else:
            verbose_proxy_logger.debug(
                "Expired UI session key cleanup disabled (set "
                "LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_ENABLED=true to enable)"
            )

    @classmethod
    async def _initialize_slack_alerting_jobs(
        cls,
        scheduler: AsyncIOScheduler,
        general_settings: dict,
        proxy_logging_obj: ProxyLogging,
        prisma_client: PrismaClient,
    ):
        """Initialize Slack alerting background jobs for spend reports."""
        if (
            proxy_logging_obj is not None
            and proxy_logging_obj.slack_alerting_instance.alerting is not None
            and prisma_client is not None
        ):
            print("Alerting: Initializing Weekly/Monthly Spend Reports")  # noqa: T201
            spend_report_frequency: Final[str] = general_settings.get("spend_report_frequency", "7d") or "7d"

            days: Final = int(spend_report_frequency[:-1])
            if spend_report_frequency[-1].lower() != "d" or days <= 0:
                raise ValueError("spend_report_frequency must be a positive number of days, e.g., '1d', '7d'")

            pod_lock_manager: Final = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
            weekly_lock_ttl: Final = duration_in_seconds(spend_report_frequency) - 3600

            async def _scheduled_weekly_spend_report() -> None:
                # TTL spans the whole reporting window: each pod's interval anchor is its own
                # boot time + jitter, so a shorter lock would let a later pod re-send the report.
                # Minus an hour so the next window's first firer finds a free key
                if (
                    await pod_lock_manager.acquire_lock(
                        cronjob_id=WEEKLY_SPEND_REPORT_JOB_ID, ttl=weekly_lock_ttl, allow_reentrant=False
                    )
                    is False
                ):
                    return
                await proxy_logging_obj.slack_alerting_instance.send_weekly_spend_report(spend_report_frequency)

            async def _scheduled_monthly_spend_report() -> None:
                if (
                    await pod_lock_manager.acquire_lock(
                        cronjob_id=MONTHLY_SPEND_REPORT_JOB_ID, ttl=3600, allow_reentrant=False
                    )
                    is False
                ):
                    return
                await proxy_logging_obj.slack_alerting_instance.send_monthly_spend_report()

            scheduler.add_job(
                _scheduled_weekly_spend_report,
                "interval",
                days=days,
                next_run_time=datetime.now() + timedelta(seconds=10 + random.randint(0, 300)),
                id=WEEKLY_SPEND_REPORT_JOB_ID,
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )

            scheduler.add_job(
                _scheduled_monthly_spend_report,
                "cron",
                day=1,
                id=MONTHLY_SPEND_REPORT_JOB_ID,
                replace_existing=True,
            )

            if os.getenv("PROMETHEUS_URL"):
                from zoneinfo import ZoneInfo

                async def _scheduled_fallback_stats() -> None:
                    if (
                        await pod_lock_manager.acquire_lock(
                            cronjob_id=PROMETHEUS_FALLBACK_STATS_JOB_ID, ttl=3600, allow_reentrant=False
                        )
                        is False
                    ):
                        return
                    await proxy_logging_obj.slack_alerting_instance.send_fallback_stats_from_prometheus()

                scheduler.add_job(
                    _scheduled_fallback_stats,
                    "cron",
                    hour=PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS,
                    minute=0,
                    timezone=ZoneInfo("America/Los_Angeles"),
                    id=PROMETHEUS_FALLBACK_STATS_JOB_ID,
                    replace_existing=True,
                )
                await _scheduled_fallback_stats()

    @classmethod
    async def _setup_prisma_client(
        cls,
        database_url: str | None,
        proxy_logging_obj: ProxyLogging,
        user_api_key_cache: UserApiKeyCache,
    ) -> PrismaClient | None:
        """
        - Sets up prisma client
        - Adds necessary views to proxy
        """
        connected_client: PrismaClient | None = None
        try:
            if database_url is None:
                return None

            prisma_client = PrismaClient(database_url=database_url, proxy_logging_obj=proxy_logging_obj)

            try:
                await prisma_client.connect()
            except Exception as e:
                if "P3018" in str(e) or "P3009" in str(e):
                    verbose_proxy_logger.debug("CRITICAL: DATABASE MIGRATION FAILED")
                    verbose_proxy_logger.debug("Your database is in a 'dirty' state.")
                    verbose_proxy_logger.debug("FIX: Run 'prisma migrate resolve --applied <migration_name>'")
                raise e

            connected_client = prisma_client

            ## Start RDS IAM token refresh background task if enabled ##
            # This proactively refreshes IAM tokens before they expire,
            # preventing the 15-minute connection failure bug (#16220)
            if hasattr(prisma_client, "db") and hasattr(prisma_client.db, "start_token_refresh_task"):
                await prisma_client.db.start_token_refresh_task()

            ## Add necessary views to proxy ##
            asyncio.create_task(
                prisma_client.check_view_exists()
            )  # check if all necessary views exist. Don't block execution

            asyncio.create_task(
                prisma_client._set_spend_logs_row_count_in_proxy_state()
            )  # set the spend logs row count in proxy state. Don't block execution

            if hasattr(prisma_client, "start_db_health_watchdog_task"):
                await prisma_client.start_db_health_watchdog_task()

            # run a health check to ensure the DB is ready
            if get_secret_bool("DISABLE_PRISMA_HEALTH_CHECK_ON_STARTUP", False) is not True:
                await prisma_client.health_check()

            return prisma_client
        except Exception as e:
            PrismaDBExceptionHandler.handle_db_exception(e)
            if connected_client is not None:
                verbose_proxy_logger.warning(
                    "Retaining the connected Prisma client after a post-connect startup step failed: %s. "
                    "The DB health watchdog keeps probing and reconnects once the database recovers.",
                    e,
                )
            return connected_client

    @classmethod
    def _init_dd_tracer(cls):
        """
        Initialize dd tracer - if `USE_DDTRACE=true` in .env

        DD tracer is used to trace Python applications.
        Doc: https://docs.datadoghq.com/tracing/trace_collection/automatic_instrumentation/dd_libraries/python/
        """
        from litellm.litellm_core_utils.dd_tracing import (
            _should_use_dd_profiler,
            _should_use_dd_tracer,
        )

        if _should_use_dd_tracer():
            import ddtrace

            ddtrace.patch_all(logging=True, openai=False)

        if _should_use_dd_profiler():
            from ddtrace.profiling import Profiler

            prof: Final = Profiler()
            prof.start()
            verbose_proxy_logger.debug("Datadog Profiler started......")

    @classmethod
    def _init_pyroscope(cls):
        """
        Optional continuous profiling via Grafana Pyroscope.

        Off by default. Enable with LITELLM_ENABLE_PYROSCOPE=true.
        Requires: pip install pyroscope-io (optional dependency).
        When enabled, PYROSCOPE_SERVER_ADDRESS and PYROSCOPE_APP_NAME are required (no defaults).
        Optional: PYROSCOPE_SAMPLE_RATE (parsed as integer) to set the sample rate.
        Optional: PYROSCOPE_GRAFANA_USER and PYROSCOPE_GRAFANA_API_TOKEN for Grafana Cloud basic auth.
        """
        if not get_secret_bool("LITELLM_ENABLE_PYROSCOPE", False):
            verbose_proxy_logger.debug(
                "LiteLLM: Pyroscope profiling is disabled (set LITELLM_ENABLE_PYROSCOPE=true to enable)."
            )
            return
        try:
            import pyroscope

            app_name: Final = os.getenv("PYROSCOPE_APP_NAME")
            if not app_name:
                raise ValueError(
                    "LITELLM_ENABLE_PYROSCOPE is true but PYROSCOPE_APP_NAME is not set. "
                    "Set PYROSCOPE_APP_NAME when enabling Pyroscope."
                )
            server_address: Final = os.getenv("PYROSCOPE_SERVER_ADDRESS")
            if not server_address:
                raise ValueError(
                    "LITELLM_ENABLE_PYROSCOPE is true but PYROSCOPE_SERVER_ADDRESS is not set. "
                    "Set PYROSCOPE_SERVER_ADDRESS when enabling Pyroscope."
                )
            tags: Final = {}
            env_name: Final = os.getenv("OTEL_ENVIRONMENT_NAME") or os.getenv(
                "LITELLM_DEPLOYMENT_ENVIRONMENT",
            )
            if env_name:
                tags["environment"] = env_name
            sample_rate_env: Final = os.getenv("PYROSCOPE_SAMPLE_RATE")

            grafana_pyroscope_user: Final = normalize_nonempty_secret_str(
                get_secret_str("PYROSCOPE_GRAFANA_USER", default_value=None)
            )
            grafana_api_token: Final = normalize_nonempty_secret_str(
                get_secret_str("PYROSCOPE_GRAFANA_API_TOKEN", default_value=None)
            )
            if grafana_api_token and not grafana_pyroscope_user:
                raise ValueError(
                    "PYROSCOPE_GRAFANA_API_TOKEN is set but PYROSCOPE_GRAFANA_USER is not set. "
                    "Set PYROSCOPE_GRAFANA_USER to the Grafana Cloud Pyroscope user/tenant id."
                )
            if grafana_pyroscope_user and not grafana_api_token:
                raise ValueError(
                    "PYROSCOPE_GRAFANA_USER is set but PYROSCOPE_GRAFANA_API_TOKEN is not set. "
                    "Set PYROSCOPE_GRAFANA_API_TOKEN to the Grafana Cloud API/access policy token."
                )
            configure_kwargs: Final = {
                "application_name": app_name,
                "server_address": server_address,
                "tags": tags if tags else None,
            }
            if grafana_api_token and grafana_pyroscope_user:
                configure_kwargs["basic_auth_username"] = grafana_pyroscope_user
                configure_kwargs["basic_auth_password"] = grafana_api_token
            if sample_rate_env is not None:
                try:
                    # pyroscope-io expects sample_rate as an integer
                    configure_kwargs["sample_rate"] = int(float(sample_rate_env))
                except (ValueError, TypeError):
                    raise ValueError(f"PYROSCOPE_SAMPLE_RATE must be a number, got: {sample_rate_env!r}")
            pyroscope.configure(**configure_kwargs)
            msg = (
                f"LiteLLM: Pyroscope profiling started (app_name={app_name}, server_address={server_address}). "
                f"View CPU profiles at the Pyroscope UI and select application '{app_name}'."
            )
            if "sample_rate" in configure_kwargs:
                msg += f" sample_rate={configure_kwargs['sample_rate']}"
            verbose_proxy_logger.info(msg)
        except ImportError:
            verbose_proxy_logger.warning(
                "LiteLLM: LITELLM_ENABLE_PYROSCOPE is set but the 'pyroscope-io' package is not installed. "
                "Pyroscope profiling will not run. Install with: pip install pyroscope-io"
            )


#### API ENDPOINTS ####
@router.get("/v1/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"])
@router.get(
    "/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)  # if project requires model list
async def model_list(
    request: Request = None,  # pyright: ignore[reportArgumentType]  # FastAPI always injects the Request; the None default only serves direct in-process callers
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    return_wildcard_routes: bool | None = False,
    team_id: str | None = None,
    include_model_access_groups: bool | None = False,
    only_model_access_groups: bool | None = False,
    include_metadata: bool | None = False,
    fallback_type: str | None = None,
    scope: str | None = None,
    healthy_only: bool | None = False,
):
    """
    Use `/model/info` - to get detailed model information, example - pricing, mode, etc.

    This is just for compatibility with openai projects like aider.

    Query Parameters:
    - include_metadata: Include additional metadata in the response with fallback information
    - fallback_type: Type of fallbacks to include ("general", "context_window", "content_policy")
                    Defaults to "general" when include_metadata=true
    - scope: Optional scope parameter. Currently only accepts "expand".
             When scope=expand is passed, proxy admins, team admins, and org admins
             will receive all proxy models as if they are a proxy admin.
    - healthy_only: When true, hide models whose backing deployments are all marked
                    unhealthy by background health checks. Set
                    `general_settings.model_list_healthy_only: true` to apply this
                    to every caller without the query parameter. Requires
                    `background_health_checks: true` in general_settings, plus
                    either `model_list_healthy_only` or `enable_health_check_routing`
                    to keep deployment health state cached; without health state
                    the listing is returned unfiltered (fail open).
                    Models expanded from wildcard routes (e.g. `openai/*`) are not
                    filtered, and nothing is hidden when `allowed_fails_policy` is
                    configured (cooldown remains the sole exclusion mechanism).
                    Hiding is presentation-only: a hidden model can still be
                    called directly.
    """
    global llm_model_list, general_settings, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj

    settings: Final = cast(dict[str, object], general_settings)  # any-ok: legacy settings

    from litellm.llms.anthropic.common_utils import (
        create_anthropic_model_list_response,
    )
    from litellm.proxy.management_endpoints.common_utils import (
        _user_has_admin_privileges,
    )
    from litellm.proxy.utils import (
        create_model_info_response,
        get_available_models_for_user,
    )
    from litellm.types.proxy.model_listing import ModelInfoResponse

    http_request: Final = cast(Request | None, request)  # cast-ok: in-process callers pass no request
    wants_anthropic_format: Final = (
        http_request is not None and http_request.headers.get("anthropic-version") is not None
    )

    # Validate scope parameter if provided
    if scope is not None and scope != "expand":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope parameter. Only 'expand' is currently supported. Received: {scope}",
        )

    # Check if scope=expand is requested and user has admin privileges
    should_expand_scope = False
    if scope == "expand":
        should_expand_scope = _user_has_admin_view(user_api_key_dict) or await _user_has_admin_privileges(
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    # Compute once — used in both branches below to hide paused models from the listing.
    blocked_names: Final = llm_router.get_fully_blocked_model_names() if llm_router is not None else set()

    # Opt-in: also hide models whose deployments are all unhealthy per background
    # health checks. Empty when health state is unavailable or stale (fail open).
    unhealthy_names: Final = await get_hidden_unhealthy_model_names(
        healthy_only=healthy_only,
        general_settings=settings,
        llm_router=llm_router,
    )

    hidden_names: Final = blocked_names | unhealthy_names

    # If scope=expand and user has admin privileges, return all proxy models
    if should_expand_scope:
        # Get all proxy models as if user is a proxy admin
        if llm_router is None:
            proxy_model_list = []
            model_access_groups = {}
        else:
            proxy_model_list = llm_router.get_model_names()
            model_access_groups = llm_router.get_model_access_groups()

        # Include model access groups if requested
        if include_model_access_groups:
            proxy_model_list = list(set(proxy_model_list + list(model_access_groups.keys())))

        # Get complete model list including wildcard routes if requested
        from litellm.proxy.auth.model_checks import get_complete_model_list

        all_models = get_complete_model_list(
            key_models=[],
            team_models=[],
            proxy_model_list=proxy_model_list,
            user_model=None,
            infer_model_from_keys=False,
            return_wildcard_routes=return_wildcard_routes or False,
            llm_router=llm_router,
            model_access_groups=model_access_groups,
            include_model_access_groups=include_model_access_groups or False,
            only_model_access_groups=only_model_access_groups or False,
        )

        # Hide paused/unhealthy models from the public listing
        if hidden_names:
            all_models = [m for m in all_models if m not in hidden_names]

        # Surface the public team name by default; legacy internal keys via flag.
        # The internal routing key drives the metadata/fallback lookup, while the
        # public name is what the client sees as the model id.
        model_data = []
        for response_id, lookup_id in TeamModelNameTranslator.listing_entries(all_models, llm_router, settings):
            model_info = create_model_info_response(
                model_id=lookup_id,
                provider="openai",
                include_metadata=include_metadata or False,
                fallback_type=fallback_type,
                llm_router=llm_router,
            )
            model_info["id"] = response_id
            model_data.append(model_info)

        if wants_anthropic_format:
            admin_listing: Final = cast(Sequence[ModelInfoResponse], model_data)  # cast-ok: rows built above
            return create_anthropic_model_list_response(admin_listing)

        return dict(
            data=model_data,
            object="list",
        )

    # Otherwise, use the normal behavior (current implementation)
    # Get available models for the user
    all_models = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
        general_settings=general_settings,
        user_model=user_model,
        prisma_client=prisma_client,
        proxy_logging_obj=proxy_logging_obj,
        team_id=team_id,
        include_model_access_groups=include_model_access_groups or False,
        only_model_access_groups=only_model_access_groups or False,
        return_wildcard_routes=return_wildcard_routes or False,
        user_api_key_cache=user_api_key_cache,
    )

    # Hide paused/unhealthy models from the public listing
    if hidden_names:
        all_models = [m for m in all_models if m not in hidden_names]

    # Surface the public team name by default; legacy internal keys via flag.
    # The internal routing key drives the metadata/fallback lookup, while the
    # public name is what the client sees as the model id.
    model_data = []
    for response_id, lookup_id in TeamModelNameTranslator.listing_entries(all_models, llm_router, settings):
        model_info = create_model_info_response(
            model_id=lookup_id,
            provider="openai",
            include_metadata=include_metadata or False,
            fallback_type=fallback_type,
            llm_router=llm_router,
        )
        model_info["id"] = response_id
        model_data.append(model_info)

    if wants_anthropic_format:
        listing: Final = cast(Sequence[ModelInfoResponse], model_data)  # cast-ok: rows built above
        return create_anthropic_model_list_response(listing)

    return dict(
        data=model_data,
        object="list",
    )


@router.get(
    "/v1/models/{model_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"],
)
@router.get(
    "/models/{model_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["model management"],
)
async def model_info(
    model_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    team_id: str | None = None,
    healthy_only: bool | None = False,
):
    """
    Retrieve information about a specific model accessible to your API key.

    Returns model details only if the model is available to your API key/team.
    Returns 404 if the model doesn't exist or is not accessible.

    Follows OpenAI API specification for individual model retrieval.
    https://platform.openai.com/docs/api-reference/models/retrieve

    Query parameters mirror `/v1/models` so the same caller context (team
    scoping, health filtering, paused deployments) drives both endpoints; the
    listing's public id must resolve to the same internal deployment here.
    """
    global llm_model_list, general_settings, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj

    settings: Final = cast(dict[str, object], general_settings)  # any-ok: legacy settings

    from litellm.proxy.utils import (
        create_model_info_response,
        get_available_models_for_user,
        validate_model_access,
    )

    all_models = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
        general_settings=general_settings,
        user_model=user_model,
        prisma_client=prisma_client,
        proxy_logging_obj=proxy_logging_obj,
        team_id=team_id,
        include_model_access_groups=False,
        only_model_access_groups=False,
        return_wildcard_routes=False,
        user_api_key_cache=user_api_key_cache,
    )

    # Mirror /v1/models' visibility filter so first-occurrence resolution
    # cannot land on a deployment the listing had hidden.
    blocked_names: Final = llm_router.get_fully_blocked_model_names() if llm_router is not None else set()
    unhealthy_names: Final = await get_hidden_unhealthy_model_names(
        healthy_only=healthy_only,
        general_settings=settings,
        llm_router=llm_router,
    )
    hidden_names: Final = blocked_names | unhealthy_names
    if hidden_names:
        all_models = [m for m in all_models if m not in hidden_names]

    internal_to_public: Final = TeamModelNameTranslator.build_internal_to_public_map(llm_router, settings)
    resolved_model_id: Final = TeamModelNameTranslator.resolve_public_name(
        model_id=model_id,
        available_models=all_models,
        llm_router=llm_router,
        general_settings=settings,
    )

    # Validate that the requested model is accessible
    validate_model_access(model_id=resolved_model_id, available_models=all_models)

    # Get provider information from the router deployment
    if llm_router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")

    deployment: Final = llm_router.get_deployment_by_model_group_name(resolved_model_id)
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in router configuration",
        )

    # Use the actual litellm model from the deployment to get provider info
    _, provider, _, _ = litellm.get_llm_provider(model=deployment.litellm_params.model)

    response_id: Final = internal_to_public.get(resolved_model_id, model_id)
    return create_model_info_response(
        model_id=response_id,
        provider=provider,
        include_metadata=False,
        fallback_type=None,
        llm_router=llm_router,
    )


def _blocked_response_usage(original_response: object | None) -> "litellm.Usage":
    """
    Token usage for a synthetic guardrail-blocked response.

    A post-call block replaces the LLM's response with the violation message,
    but the upstream call already consumed tokens -- report that real usage
    (carried on ``ModifyResponseException.original_response``) rather than
    discarding it. Pre-call blocks never invoked the LLM (no original_response),
    so usage is zero.
    """
    usage: Final = getattr(original_response, "usage", None) if original_response is not None else None
    if isinstance(usage, litellm.Usage):
        return usage
    return litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["chat/completions"],
)
@router.post(
    "/chat/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["chat/completions"],
)
@router.post(
    "/engines/{model:path}/chat/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["chat/completions"],
)
@router.post(
    "/openai/deployments/{model:path}/chat/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["chat/completions"],
    responses={200: {"description": "Successful response"}, **ERROR_RESPONSES},
)  # azure compatible endpoint
async def chat_completion(
    request: Request,
    fastapi_response: Response,
    model: str | None = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """

    Follows the exact same API spec as `OpenAI's Chat API https://platform.openai.com/docs/api-reference/chat`

    ```bash
    curl -X POST http://localhost:4000/v1/chat/completions \

    -H "Content-Type: application/json" \

    -H "Authorization: Bearer sk-1234" \

    -d '{
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'
    ```

    """
    global general_settings, user_debug, proxy_logging_obj, llm_model_list
    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
    data: Final = await _read_request_body(request=request)
    if user_api_key_dict is not None:
        if not isinstance(data.get("metadata"), dict):
            # Covers both missing and JSON-string metadata (multipart /
            # extra_body); otherwise `data["metadata"][k] = v` below raises
            # TypeError on a string value and 500s the request.
            data["metadata"] = {}
        if hasattr(user_api_key_dict, "user_id") and user_api_key_dict.user_id is not None:
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        if hasattr(user_api_key_dict, "team_id") and user_api_key_dict.team_id is not None:
            data["metadata"]["user_api_key_team_id"] = user_api_key_dict.team_id
        if hasattr(user_api_key_dict, "org_id") and user_api_key_dict.org_id is not None:
            data["metadata"]["user_api_key_org_id"] = user_api_key_dict.org_id
        if hasattr(user_api_key_dict, "organization_alias") and user_api_key_dict.organization_alias is not None:
            data["metadata"]["user_api_key_org_alias"] = user_api_key_dict.organization_alias
        if hasattr(user_api_key_dict, "agent_id") and user_api_key_dict.agent_id is not None:
            data["metadata"]["agent_id"] = user_api_key_dict.agent_id

    base_llm_response_processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result: Final = await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acompletion",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        if isinstance(result, BaseModel):
            return model_dump_with_preserved_fields(result, exclude_unset=True)
        else:
            return result
    except ModifyResponseException as e:
        # Guardrail flagged content in passthrough mode - return 200 with violation message
        _data = e.request_data
        # Capture logging_obj before post_call_failure_hook pops it from _data.
        _logging_obj: Final = _data.get("litellm_logging_obj")
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        _chat_response = litellm.ModelResponse()
        _chat_response.model = e.model
        _chat_response.choices[0].message.content = e.message
        _chat_response.choices[0].finish_reason = "content_filter"
        # Report the blocked LLM response's real usage (set before the stream
        # branch so both paths carry it); zero for pre-call blocks.
        _chat_response.usage = _blocked_response_usage(e.original_response)

        if data.get("stream", None) is not None and data["stream"] is True:
            _iterator = litellm.utils.ModelResponseIterator(model_response=_chat_response, convert_to_delta=True)
            _streaming_response = litellm.CustomStreamWrapper(
                completion_stream=_iterator,
                model=e.model,
                custom_llm_provider="cached_response",
                logging_obj=_logging_obj,
            )
            selected_data_generator = select_data_generator(
                response=_streaming_response,
                user_api_key_dict=user_api_key_dict,
                request_data=_data,
                request=request,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                status_code=200,  # Return 200 for passthrough mode
            )
        return _chat_response
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        _chat_response = litellm.ModelResponse()
        _chat_response.choices[0].message.content = e.message

        if data.get("stream", None) is not None and data["stream"] is True:
            _iterator = litellm.utils.ModelResponseIterator(model_response=_chat_response, convert_to_delta=True)
            _streaming_response = litellm.CustomStreamWrapper(
                completion_stream=_iterator,
                model=data.get("model", ""),
                custom_llm_provider="cached_response",
                logging_obj=_data.get("litellm_logging_obj", None),
            )
            selected_data_generator = select_data_generator(
                response=_streaming_response,
                user_api_key_dict=user_api_key_dict,
                request_data=_data,
                request=request,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                status_code=(e.status_code if hasattr(e, "status_code") else status.HTTP_400_BAD_REQUEST),
            )
        _usage: Final = litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        _chat_response.usage = _usage
        return _chat_response
    except Exception as e:
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


@router.post("/v1/completions", dependencies=[Depends(user_api_key_auth)], tags=["completions"])
@router.post("/completions", dependencies=[Depends(user_api_key_auth)], tags=["completions"])
@router.post(
    "/engines/{model:path}/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["completions"],
)
@router.post(
    "/openai/deployments/{model:path}/completions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["completions"],
)
async def completion(
    request: Request,
    fastapi_response: Response,
    model: str | None = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Follows the exact same API spec as `OpenAI's Completions API https://platform.openai.com/docs/api-reference/completions`

    ```bash
    curl -X POST http://localhost:4000/v1/completions \

    -H "Content-Type: application/json" \

    -H "Authorization: Bearer sk-1234" \

    -d '{
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.7
    }'
    ```
    """
    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
    data = {}
    try:
        data = await _read_request_body(request=request)
        if user_api_key_dict is not None:
            if data.get("metadata") is None:
                data["metadata"] = {}
            if hasattr(user_api_key_dict, "user_id") and user_api_key_dict.user_id is not None:
                data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
            if hasattr(user_api_key_dict, "team_id") and user_api_key_dict.team_id is not None:
                data["metadata"]["user_api_key_team_id"] = user_api_key_dict.team_id
            if hasattr(user_api_key_dict, "org_id") and user_api_key_dict.org_id is not None:
                data["metadata"]["user_api_key_org_id"] = user_api_key_dict.org_id
            if hasattr(user_api_key_dict, "organization_alias") and user_api_key_dict.organization_alias is not None:
                data["metadata"]["user_api_key_org_alias"] = user_api_key_dict.organization_alias
            if hasattr(user_api_key_dict, "agent_id") and user_api_key_dict.agent_id is not None:
                data["metadata"]["agent_id"] = user_api_key_dict.agent_id
        base_llm_response_processor: Final = ProxyBaseLLMRequestProcessing(data=data)
        return await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="atext_completion",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except ModifyResponseException as e:
        # Guardrail flagged content in passthrough mode - return 200 with violation message
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )

        if _data.get("stream", None) is not None and _data["stream"] is True:
            _text_response: Final = litellm.ModelResponse()
            # Set text attribute dynamically for text completion format
            setattr(_text_response.choices[0], "text", e.message)
            _text_response.model = e.model
            _usage = _blocked_response_usage(e.original_response)
            # Set usage attribute dynamically (ModelResponse accepts usage in __init__ but it's not in type definition)
            setattr(_text_response, "usage", _usage)
            _iterator = litellm.utils.ModelResponseIterator(model_response=_text_response, convert_to_delta=True)
            _streaming_response = litellm.TextCompletionStreamWrapper(
                completion_stream=_iterator,
                model=e.model,
            )

            selected_data_generator = select_data_generator(
                response=_streaming_response,
                user_api_key_dict=user_api_key_dict,
                request_data=_data,
                request=request,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                status_code=200,  # Return 200 for passthrough mode
            )
        else:
            _response = litellm.TextCompletionResponse()
            _response.choices[0].text = e.message
            _response.model = e.model
            _usage = _blocked_response_usage(e.original_response)
            _response.usage = _usage
            return _response
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        if _data.get("stream", None) is not None and _data["stream"] is True:
            _chat_response: Final = litellm.ModelResponse()
            _usage = litellm.Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            _chat_response.usage = _usage
            _chat_response.choices[0].message.content = e.message
            _iterator = litellm.utils.ModelResponseIterator(model_response=_chat_response, convert_to_delta=True)
            _streaming_response = litellm.TextCompletionStreamWrapper(
                completion_stream=_iterator,
                model=_data.get("model", ""),
            )

            selected_data_generator = select_data_generator(
                response=_streaming_response,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
                request=request,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                headers={},
                status_code=(e.status_code if hasattr(e, "status_code") else status.HTTP_400_BAD_REQUEST),
            )
        else:
            _response = litellm.TextCompletionResponse()
            _response.choices[0].text = e.message
            return _response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.completion(): Exception occured - %s", e)
        error_msg: Final = f"{e}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            openai_code=getattr(e, "code", None),
            code=getattr(e, "status_code", 500),
        )


@router.post(
    "/v1/embeddings",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["embeddings"],
)
@router.post(
    "/embeddings",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["embeddings"],
)
@router.post(
    "/engines/{model:path}/embeddings",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["embeddings"],
)  # azure compatible endpoint
@router.post(
    "/openai/deployments/{model:path}/embeddings",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["embeddings"],
)  # azure compatible endpoint
async def embeddings(
    request: Request,
    fastapi_response: Response,
    model: str | None = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Follows the exact same API spec as `OpenAI's Embeddings API https://platform.openai.com/docs/api-reference/embeddings`

    ```bash
    curl -X POST http://localhost:4000/v1/embeddings \

    -H "Content-Type: application/json" \

    -H "Authorization: Bearer sk-1234" \

    -d '{
        "model": "text-embedding-ada-002",
        "input": "The quick brown fox jumps over the lazy dog"
    }'
    ```

"""
    global proxy_logging_obj
    data: Final = await _read_request_body(request=request)
    base_llm_response_processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        ### HANDLE TOKEN ARRAY INPUT DECODING ###
        # This must happen BEFORE base_process_llm_request() since it modifies the input
        router_model_names: Final = llm_router.model_names if llm_router is not None else []
        if (
            "input" in data
            and isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
            and isinstance(data["input"][0][0], int)
        ):  # check if array of tokens passed in
            # check if provider accept list of tokens as input - e.g. for langchain integration
            if llm_router is not None and data.get("model") in router_model_names:
                # Use router's O(1) lookup instead of O(N) iteration through llm_model_list
                deployment: Final = llm_router.get_deployment_by_model_group_name(model_group_name=data["model"])
                if deployment is not None:
                    litellm_params: Final = deployment.get("litellm_params", {}) or {}
                    litellm_model: Final = litellm_params.get("model", "")
                    # Check if this provider supports token arrays
                    supports_token_arrays: Final = litellm_model in litellm.open_ai_embedding_models or any(
                        litellm_model.startswith(provider)
                        for provider in LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS
                    )
                    if not supports_token_arrays:
                        # non-openai/azure embedding model called with token input - decode tokens
                        input_list: Final = []
                        for i in data["input"]:
                            input_list.append(litellm.decode(model="gpt-3.5-turbo", tokens=i))
                        data["input"] = input_list

        if user_api_key_dict is not None:
            if data.get("metadata") is None:
                data["metadata"] = {}
            if hasattr(user_api_key_dict, "user_id") and user_api_key_dict.user_id is not None:
                data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
            if hasattr(user_api_key_dict, "team_id") and user_api_key_dict.team_id is not None:
                data["metadata"]["user_api_key_team_id"] = user_api_key_dict.team_id
            if hasattr(user_api_key_dict, "org_id") and user_api_key_dict.org_id is not None:
                data["metadata"]["user_api_key_org_id"] = user_api_key_dict.org_id
            if hasattr(user_api_key_dict, "organization_alias") and user_api_key_dict.organization_alias is not None:
                data["metadata"]["user_api_key_org_alias"] = user_api_key_dict.organization_alias
            if hasattr(user_api_key_dict, "agent_id") and user_api_key_dict.agent_id is not None:
                data["metadata"]["agent_id"] = user_api_key_dict.agent_id

        response: Final = await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aembedding",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        return response
    except Exception as e:
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/moderations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["moderations"],
)
@router.post(
    "/moderations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["moderations"],
)
async def moderations(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    The moderations endpoint is a tool you can use to check whether content complies with an LLM Providers policies.
    Quick Start
    ```
    curl --location 'http://0.0.0.0:4000/moderations' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --data '{"input": "Sample text goes here", "model": "text-moderation-stable"}'
    ```
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body: Final = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        data["model"] = (
            general_settings.get("moderation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model")  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="moderation"
        )

        time.time()

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call: Final = await route_request(
            data=data,
            route_type="amoderation",
            llm_router=llm_router,
            user_model=user_model,
        )
        response: Final = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.moderations(): Exception occured - %s", e)
        if isinstance(e, ProxyException):
            raise
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


async def _audio_speech_chunk_generator(
    _response: HttpxBinaryResponseContent,
) -> AsyncGenerator[bytes, None]:
    # chunk_size has a big impact on latency, it can't be too small or too large
    # too small: latency is high
    # too large: latency is low, but memory usage is high
    # 8192 is a good compromise
    _generator: Final = await _response.aiter_bytes(chunk_size=AUDIO_SPEECH_CHUNK_SIZE)
    async for chunk in _generator:
        yield chunk


@router.post(
    "/v1/audio/speech",
    dependencies=[Depends(user_api_key_auth)],
    tags=["audio"],
)
@router.post(
    "/audio/speech",
    dependencies=[Depends(user_api_key_auth)],
    tags=["audio"],
)
async def audio_speech(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Same params as:

    https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body: Final = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if user_model:
            data["model"] = user_model

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="aspeech"
        )

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call: Final = await route_request(
            data=data,
            route_type="aspeech",
            llm_router=llm_router,
            user_model=user_model,
        )
        response: Final = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""
        response_cost: Final = hidden_params.get("response_cost", None) or ""
        litellm_call_id: Final = hidden_params.get("litellm_call_id", None) or ""

        custom_headers: Final = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            model_id=model_id,
            cache_key=cache_key,
            api_base=api_base,
            version=version,
            response_cost=response_cost,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            fastest_response_batch_completion=None,
            call_id=litellm_call_id,
            request_data=data,
            hidden_params=hidden_params,
        )

        # Call response headers hook (matches audio_transcription behavior)
        callback_headers: Final = await proxy_logging_obj.post_call_response_headers_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=dict(request.headers),
        )
        if callback_headers:
            custom_headers.update(callback_headers)

        # Determine media type based on model type
        media_type = "audio/mpeg"  # Default for OpenAI TTS
        request_model: Final = data.get("model", "")
        if request_model:
            request_model_lower: Final = request_model.lower()
            if "gemini" in request_model_lower and (
                "tts" in request_model_lower or "preview-tts" in request_model_lower
            ):
                media_type = "audio/wav"  # Gemini TTS returns WAV format after conversion

        return StreamingResponse(
            _audio_speech_chunk_generator(response),
            media_type=media_type,
            headers=custom_headers,
        )

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=data,
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.audio_speech(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        raise e


@router.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["audio"],
)
@router.post(
    "/audio/transcriptions",
    dependencies=[Depends(user_api_key_auth)],
    tags=["audio"],
)
async def audio_transcriptions(
    request: Request,
    fastapi_response: Response,
    file: UploadFile = File(...),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Same params as:

    https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=curl
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        form_data: Final = await get_form_data(request)
        data = {key: value for key, value in form_data.items() if key != "file"}

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        data["model"] = (
            general_settings.get("moderation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        router_model_names: Final = llm_router.model_names if llm_router is not None else []

        if file.filename is None:
            raise ProxyException(
                message="File name is None. Please check your file name",
                code=status.HTTP_400_BAD_REQUEST,
                type="bad_request",
                param="file",
            )

        # Check if File can be read in memory before reading
        check_file_size_under_limit(
            request_data=data,
            file=file,
            router_model_names=router_model_names,
        )

        file_content: Final = await file.read()
        file_object: Final = io.BytesIO(file_content)
        file_object.name = file.filename
        data["file"] = file_object

        try:
            ### CALL HOOKS ### - modify incoming data / reject request before calling the model
            data = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                data=data,
                call_type="transcription",
            )

            ## ROUTE TO CORRECT ENDPOINT ##
            llm_call: Final = await route_request(
                data=data,
                route_type="atranscription",
                llm_router=llm_router,
                user_model=user_model,
            )
            response: Final = await llm_call
        except Exception as e:
            raise e
        finally:
            file_object.close()  # close the file read in by io library

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""
        response_cost: Final = hidden_params.get("response_cost", None) or ""
        litellm_call_id: Final = hidden_params.get("litellm_call_id", None) or ""
        additional_headers: Final[dict] = hidden_params.get("additional_headers", {}) or {}

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
                hidden_params=hidden_params,
                **additional_headers,
            )
        )

        # Call response headers hook (matches base_process_llm_request behavior)
        callback_headers: Final = await proxy_logging_obj.post_call_response_headers_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=dict(request.headers),
        )
        if callback_headers:
            fastapi_response.headers.update(callback_headers)

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.audio_transcription(): Exception occured - %s", e)
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                openai_code=getattr(e, "code", None),
                code=getattr(e, "status_code", 500),
            )


######################################################################

#                 Vertex AI Live API WebSocket Pass-through

######################################################################


@app.websocket("/vertex_ai/live")
async def vertex_ai_live_passthrough_endpoint(
    websocket: WebSocket,
    model: str | None = fastapi.Query(
        None,
        description="Optional model name, used to determine Vertex region for global models.",
    ),
    vertex_project: str | None = fastapi.Query(
        None,
        description="Override the Vertex AI project id used for the upstream connection.",
    ),
    vertex_location: str | None = fastapi.Query(
        None,
        description="Override the Vertex AI region (for example, 'us-central1').",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth_websocket),
):
    """
    Vertex AI Live API WebSocket Pass-through Endpoint

    This endpoint delegates to the WebSocket function defined in llm_passthrough_endpoints.py
    """
    return await vertex_ai_live_websocket_passthrough(
        websocket=websocket,
        model=model,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        user_api_key_dict=user_api_key_dict,
    )


######################################################################

#                          /v1/realtime Endpoints

######################################################################


@lru_cache(maxsize=_REALTIME_BODY_CACHE_SIZE)
def _realtime_query_params_template(model: str | None, intent: str | None) -> tuple[tuple[str, str], ...]:
    """
    Build a hashable representation of the realtime query params so we can cache
    the repetitive model/intent combinations.
    """
    params: Final[list[tuple[str, str]]] = []
    if model is not None:
        params.append(("model", model))
    if intent is not None:
        params.append(("intent", intent))
    return tuple(params)


@app.websocket("/openai/v1/realtime")
@app.websocket("/v1/realtime")
@app.websocket("/realtime")
async def realtime_websocket_endpoint(
    websocket: WebSocket,
    model: str | None = fastapi.Query(None, description="The model to use for the websocket connection."),
    intent: str | None = fastapi.Query(None, description="The intent of the websocket connection."),
    guardrails: str | None = fastapi.Query(
        None,
        description="Comma-separated list of guardrail names to apply to this request.",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth_websocket),
):
    requested_protocols: Final = [
        p.strip() for p in (websocket.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()
    ]
    accept_kwargs: Final[dict] = {}
    if requested_protocols:
        accept_kwargs["subprotocol"] = requested_protocols[0]

    route_model = model
    if route_model is None:
        if intent == "transcription":
            route_model = "gpt-realtime-whisper"
        else:
            await websocket.close(code=1008, reason="model query parameter is required")
            return
    assert route_model is not None
    try:
        await can_key_call_resolved_model(
            model=route_model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_dict,
            llm_router=llm_router,
        )
    except ProxyException as e:
        await websocket.close(code=1008, reason=e.message[:120])
        return
    await websocket.accept(**accept_kwargs)

    # Only use explicit parameters, not all query params
    query_params: Final = cast(RealtimeQueryParams, dict(_realtime_query_params_template(model, intent)))

    data: dict[str, object] = {
        "model": route_model,
        "websocket": websocket,
        "query_params": query_params,  # Only explicit params
    }

    # Pass guardrails into data so pre-call guardrail processing picks them up
    if guardrails:
        data["guardrails"] = [g.strip() for g in guardrails.split(",") if g.strip()]

    # Use raw ASGI headers (already lowercase bytes) to avoid extra work
    headers_list: Final = list(websocket.scope.get("headers") or [])

    scope: Final = REALTIME_REQUEST_SCOPE_TEMPLATE.copy()
    scope["headers"] = headers_list

    request: Final = Request(scope=scope)

    request._url = websocket.url

    async def return_body():
        return _realtime_request_body(route_model)

    request.body = return_body

    ### ROUTE THE REQUEST ###
    base_llm_response_processor: Final = ProxyBaseLLMRequestProcessing(data=data)

    # Phase 1: pre-call processing (auth, guardrails, rate limits).
    # Errors here (e.g. guardrail block) are sent back to the client as an
    # error event before closing, so the caller knows what happened.
    try:
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            model=route_model,
            route_type="_arealtime",
        )
    except Exception as e:
        verbose_proxy_logger.exception("Realtime pre-call error")
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "guardrail_error",
                            "message": str(e),
                        },
                    }
                )
            )
        except Exception:
            pass
        await websocket.close(code=1011, reason="Pre-call error")
        return

    # Phase 2: route to upstream LLM.
    try:
        data["user_api_key_dict"] = user_api_key_dict
        llm_call: Final = await route_request(
            data=data,
            route_type="_arealtime",
            llm_router=llm_router,
            user_model=user_model,
        )
        await llm_call
    except websockets.exceptions.InvalidStatusCode as e:
        verbose_proxy_logger.exception("Invalid status code")
        await websocket.close(code=e.status_code, reason="Invalid status code")
    except Exception as e:
        verbose_proxy_logger.exception("Internal server error")
        redacted_error: Final = _redact_string(str(e))
        try:
            await websocket.send_text(realtime_error_event(redacted_error, error_type="server_error"))
        except Exception:  # noqa: BLE001  # best-effort notice: a dead client socket must not skip the close below
            verbose_proxy_logger.debug("Could not send realtime error event to client; closing anyway")
        try:
            await websocket.close(
                code=1011,
                reason=websocket_close_reason(redacted_error, fallback="Internal server error"),
            )
        except Exception:  # noqa: BLE001  # the lower layer may have closed the socket already; closing twice is not an error
            verbose_proxy_logger.debug("Could not close realtime client websocket; it is already gone")


######################################################################

#                          /v1/assistant Endpoints


######################################################################


@router.get(
    "/v1/assistants",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.get(
    "/assistants",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def get_assistants(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Returns a list of assistants.

    API Reference docs - https://platform.openai.com/docs/api-reference/assistants/listAssistants
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        await request.body()

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.aget_assistants(**data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.get_assistants(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                openai_code=getattr(e, "code", None),
                code=getattr(e, "status_code", 500),
            )


@router.post(
    "/v1/assistants",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.post(
    "/assistants",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def create_assistant(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create assistant

    API Reference docs - https://platform.openai.com/docs/api-reference/assistants/createAssistant
    """
    global proxy_logging_obj
    data = {}  # ensure data always dict
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body: Final = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.acreate_assistants(**data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.create_assistant(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


@router.delete(
    "/v1/assistants/{assistant_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.delete(
    "/assistants/{assistant_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def delete_assistant(
    request: Request,
    assistant_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete assistant

    API Reference docs - https://platform.openai.com/docs/api-reference/assistants/createAssistant
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.adelete_assistant(assistant_id=assistant_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.delete_assistant(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


@router.post(
    "/v1/threads",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.post(
    "/threads",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def create_threads(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a thread.

    API Reference - https://platform.openai.com/docs/api-reference/threads/createThread
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        await request.body()

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.acreate_thread(**data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.create_threads(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


@router.get(
    "/v1/threads/{thread_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.get(
    "/threads/{thread_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def get_thread(
    request: Request,
    thread_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieves a thread.

    API Reference - https://platform.openai.com/docs/api-reference/threads/getThread
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.aget_thread(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.get_thread(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


@router.post(
    "/v1/threads/{thread_id}/messages",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.post(
    "/threads/{thread_id}/messages",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def add_messages(
    request: Request,
    thread_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a message.

    API Reference - https://platform.openai.com/docs/api-reference/messages/createMessage
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body: Final = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.a_add_message(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.add_messages(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


@router.get(
    "/v1/threads/{thread_id}/messages",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.get(
    "/threads/{thread_id}/messages",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def get_messages(
    request: Request,
    thread_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Returns a list of messages for a given thread.

    API Reference - https://platform.openai.com/docs/api-reference/messages/listMessages
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        response: Final = await llm_router.aget_messages(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.get_messages(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


@router.post(
    "/v1/threads/{thread_id}/runs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.post(
    "/threads/{thread_id}/runs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
async def run_thread(
    request: Request,
    thread_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a run.

    API Reference: https://platform.openai.com/docs/api-reference/runs/createRun
    """
    global proxy_logging_obj
    data: dict = {}
    try:
        body: Final = await request.body()
        data = orjson.loads(body)
        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})
        router: Final = llm_router

        if "stream" in data and data["stream"] is True:  # use generate_responses to stream responses

            async def produce_run_stream() -> StreamingResponse | JSONResponse:
                run_stream: Final = await router.arun_thread(thread_id=thread_id, **data)
                return await create_response(
                    generator=async_assistants_data_generator(
                        user_api_key_dict=user_api_key_dict,
                        response=run_stream,
                        request_data=data,
                    ),
                    media_type="text/event-stream",
                    headers={},  # Added empty headers dict, original call missed this argument
                    request=request,
                )

            async def audit_late_failure(exc: Exception) -> HTTPException | None:
                # Once a keepalive is on the wire this can no longer raise, so the
                # handler's own `except` never runs its post_call_failure_hook.
                return await proxy_logging_obj.post_call_failure_hook(
                    user_api_key_dict=user_api_key_dict, original_exception=exc, request_data=data
                )

            # The upstream withholds its first event for the whole time-to-first-token
            # and `create_response` buffers that first chunk before it can build a
            # response, so the run writes zero bytes until the model answers.
            return await open_sse_before_first_byte(
                produce_run_stream(),
                ping_interval_seconds=ttft_keepalive_interval(data, router),
                on_late_failure=audit_late_failure,
            )

        response: Final = await router.arun_thread(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.run_thread(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "code", getattr(e, "status_code", 500)),
            )


#### DEV UTILS ####

# @router.get(
#     "/utils/available_routes",
#     tags=["llm utils"],
#     dependencies=[Depends(user_api_key_auth)],
# )
# async def get_available_routes(user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.model_repository import ModelRepository
from litellm.repositories.table_repositories import (
    AccessGroupRepository,
    ConfigOverridesRepository,
    InvitationLinkRepository,
    PromptRepository,
    SSOConfigRepository,
    UISettingsRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository


def _get_provider_token_counter(
    deployment: dict, model_to_use: str
) -> tuple[BaseTokenCounter | None, str | None, str | None]:
    """
    Auto-route to the correct provider's token counter based on model/deployment.
    Uses the existing get_provider_model_info infrastructure with switch-case pattern.
    """
    if deployment is None:
        return None

    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    full_model: Final = deployment.get("litellm_params", {}).get("model", "")
    model: str | None = None
    custom_llm_provider: str | None = None

    try:
        # Use existing LiteLLM logic to determine provider
        model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
            model=full_model,
            custom_llm_provider=deployment.get("litellm_params", {}).get("custom_llm_provider"),
            api_base=deployment.get("litellm_params", {}).get("api_base"),
            api_key=deployment.get("litellm_params", {}).get("api_key"),
        )

        # Switch case pattern using existing get_provider_model_info
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        # Convert string provider to LlmProviders enum
        llm_provider_enum: Final = LlmProviders(custom_llm_provider)
        # Add more provider mappings as needed

        if llm_provider_enum:
            provider_model_info: Final = ProviderConfigManager.get_provider_model_info(
                model=full_model, provider=llm_provider_enum
            )
            if provider_model_info is not None:
                return (
                    provider_model_info.get_token_counter(),
                    model,
                    custom_llm_provider,
                )

    except Exception:
        # If provider detection fails, fall back to manual checks
        if full_model.startswith("anthropic/") or "anthropic" in full_model.lower():
            from litellm.llms.anthropic.common_utils import AnthropicModelInfo

            anthropic_model_info: Final = AnthropicModelInfo()
            return anthropic_model_info.get_token_counter(), model, custom_llm_provider

    return None, None, None


async def _try_provider_token_count(
    provider_counter: "BaseTokenCounter",
    custom_llm_provider: str | None,
    model_to_use: str,
    messages: list | None,
    contents: list | None,
    deployment: dict[str, Any] | None,
    request_model: str,
    tools: list | None = None,
    system: str | None = None,
) -> Optional["TokenCountResponse"]:
    """Attempt provider-specific token counting. Returns result on success, None to fall through to local counting."""
    if not provider_counter.should_use_token_counting_api(custom_llm_provider=custom_llm_provider):
        return None
    try:
        result: Final = await provider_counter.count_tokens(
            model_to_use=model_to_use or "",
            messages=messages,
            contents=contents,
            deployment=deployment,
            request_model=request_model,
            tools=tools,
            system=system,
        )
    except httpx.HTTPStatusError as e:
        error_message: Final = getattr(e, "message", None) or str(e)
        status_code: Final = getattr(e, "status_code", None) or e.response.status_code
        raise ProxyException(
            message=error_message,
            type="token_counting_error",
            param="model",
            code=status_code,
        )
    if result is not None and result.error is True:
        if litellm.disable_token_counter is True:
            raise ProxyException(
                message=result.error_message or "Token counting failed",
                type="token_counting_error",
                param="model",
                code=result.status_code or 500,
            )
        verbose_proxy_logger.warning(
            "Provider token counting failed (%s): %s. Falling back to local tokenizer.",
            result.status_code,
            result.error_message,
        )
        return None
    return result


@router.post(
    "/utils/token_counter",
    tags=["llm utils"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountResponse,
)
async def token_counter(request: TokenCountRequest, call_endpoint: bool = False):
    """
    Args:
        request: TokenCountRequest
        call_endpoint: bool - When set to "True" it will call the token counting endpoint - e.g Anthropic or Google AI Studio Token Counting APIs.

    Returns:
        TokenCountResponse
    """
    global llm_router

    prompt: Final = request.prompt
    messages: Final = request.messages
    contents: Final = request.contents
    tools: Final = request.tools
    system: Final = request.system

    #########################################################
    # Validate request
    #########################################################
    if prompt is None and messages is None and contents is None:
        raise HTTPException(status_code=400, detail="prompt or messages or contents must be provided")

    deployment: dict[str, Any] | None = None
    litellm_model_name = None
    model_info: ModelMapInfo | None = None
    if llm_router is not None:
        # get 1 deployment corresponding to the model
        try:
            deployment = await llm_router.async_get_available_deployment(
                model=request.model,
                request_kwargs={},
            )
        except Exception:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.token_counter(): Exception occured while getting deployment"
            )
    if deployment is not None:
        litellm_model_name = deployment.get("litellm_params", {}).get("model")
        model_info = deployment.get("model_info", {})
        load_credentials_from_list(deployment.get("litellm_params", {}))
        # remove the custom_llm_provider_prefix in the litellm_model_name
        if "/" in litellm_model_name:
            litellm_model_name = litellm_model_name.split("/", 1)[1]

    model_to_use: str = (
        litellm_model_name or request.model
    )  # use litellm model name, if it's not avalable then fallback to request.model

    # Try provider-specific token counting first - only for non-direct requests (from provider endpoints)
    provider_counter: BaseTokenCounter | None = None
    custom_llm_provider: str | None = None
    if call_endpoint is True and deployment is not None:
        # Auto-route to the correct provider based on model
        provider_counter, _model, custom_llm_provider = _get_provider_token_counter(deployment, model_to_use)
        if _model is not None:
            model_to_use = _model

    if provider_counter is not None:
        result: Final = await _try_provider_token_count(
            provider_counter=provider_counter,
            custom_llm_provider=custom_llm_provider,
            model_to_use=model_to_use,
            messages=messages,
            contents=contents,
            deployment=deployment,
            request_model=request.model,
            tools=tools,
            system=system,
        )
        if result is not None:
            return result

    # Check if token counter is disabled before fallback
    if litellm.disable_token_counter is True:
        raise ProxyException(
            message="Token counting is disabled and no provider API result available",
            type="token_counting_disabled",
            param="model",
            code=503,
        )

    # Default LiteLLM token counting
    custom_tokenizer: CustomHuggingfaceTokenizer | None = None
    if model_info is not None:
        custom_tokenizer = cast(
            CustomHuggingfaceTokenizer | None,
            model_info.get("custom_tokenizer", None),
        )
    _tokenizer_used: Final = litellm.utils._select_tokenizer(model=model_to_use, custom_tokenizer=custom_tokenizer)

    tokenizer_used: Final = str(_tokenizer_used["type"])
    total_tokens: Final = await asyncify(litellm.token_counter)(
        model=model_to_use,
        text=prompt,
        messages=messages,
        custom_tokenizer=_tokenizer_used,
    )
    return TokenCountResponse(
        total_tokens=total_tokens,
        request_model=request.model,
        model_used=model_to_use,
        tokenizer_type=tokenizer_used,
    )


@router.get(
    "/utils/supported_openai_params",
    tags=["llm utils"],
    dependencies=[Depends(user_api_key_auth)],
)
async def supported_openai_params(model: str):
    """
    Returns supported openai params for a given litellm model name

    e.g. `gpt-4` vs `gpt-3.5-turbo`

    Example curl:
    ```
    curl -X GET --location 'http://localhost:4000/utils/supported_openai_params?model=gpt-3.5-turbo-16k' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    try:
        model, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
        return {
            "supported_openai_params": litellm.get_supported_openai_params(
                model=model, custom_llm_provider=custom_llm_provider
            )
        }
    except Exception:
        raise HTTPException(status_code=400, detail={"error": f"Could not map model={model}"})


@router.post(
    "/utils/transform_request",
    tags=["llm utils"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RawRequestTypedDict,
)
async def transform_request(request: TransformRequestBody):
    from litellm.utils import return_raw_request

    try:
        is_request_body_safe(
            request_body=request.request_body,
            general_settings=general_settings,
            llm_router=llm_router,
            model=request.request_body.get("model", ""),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    return return_raw_request(endpoint=request.call_type, kwargs=request.request_body)


async def _check_if_model_is_user_added(
    models: list[dict],
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
) -> list[dict]:
    """
    Check if model is in db

    Check if db model is 'created_by' == user_api_key_dict.user_id

    Only return models that match
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    filtered_models: Final = []
    for model in models:
        id = model.get("model_info", {}).get("id", None)
        if id is None:
            continue
        db_model: _ProxyModelRow | None = await ModelRepository(prisma_client).table.find_unique(where={"model_id": id})
        if db_model is not None:
            if db_model.created_by == user_api_key_dict.user_id:
                filtered_models.append(model)
    return filtered_models


def _check_if_model_is_team_model(models: list[DeploymentTypedDict], user_row: _UserTeamsRow) -> list[dict]:
    """
    Check if model is a team model

    Check if user is a member of the team that the model belongs to
    """

    user_team_models: Final[list[dict]] = []
    for model in models:
        model_team_id = model.get("model_info", {}).get("team_id", None)

        if model_team_id is not None:
            if model_team_id in user_row.teams:
                user_team_models.append(cast(dict, model))

    return user_team_models


async def non_admin_all_models(
    all_models: list[dict],
    llm_router: Router,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
):
    """
    Check if model is in db

    Check if db model is 'created_by' == user_api_key_dict.user_id

    Only return models that match
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Get all models that are user-added, when model created_by == user_api_key_dict.user_id
    all_models = await _check_if_model_is_user_added(
        models=all_models,
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )

    if user_api_key_dict.user_id:
        try:
            user_row: Final = await UserRepository(prisma_client).table.find_unique(
                where={"user_id": user_api_key_dict.user_id}
            )
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "User not found"})

        # Get all models that are team models, when model team_id == user_row.teams
        if user_row is not None:
            all_models += _check_if_model_is_team_model(
                models=llm_router.get_model_list() or [],
                user_row=user_row,
            )

    # de-duplicate models. Only return unique model ids
    unique_models: Final = _deduplicate_litellm_router_models(models=all_models)
    return unique_models


def _add_team_models_to_all_models(
    team_db_objects_typed: list[LiteLLM_TeamTable],
    llm_router: Router,
) -> dict[str, set[str]]:
    """
    Add team models to all models
    """
    team_models: Final[dict[str, set[str]]] = {}
    proxy_model_list: Final = llm_router.get_model_names()
    model_access_groups: Final = llm_router.get_model_access_groups()

    for team_object in team_db_objects_typed:
        if (
            not team_object.models  # None or empty list = all model access
            or SpecialModelNames.all_proxy_models.value in team_object.models
        ):
            model_list = llm_router.get_model_list()
            if model_list is not None:
                for model in model_list:
                    model_id = model.get("model_info", {}).get("id", None)
                    if model_id is None:
                        continue
                    # if team model id set, check if team id in user_teams
                    team_model_id = model.get("model_info", {}).get("team_id", None)
                    can_add_model = False
                    if team_model_id is None or team_model_id in team_object.team_id:
                        can_add_model = True

                    if can_add_model:
                        team_models.setdefault(model_id, set()).add(team_object.team_id)
        else:
            resolved_model_names = get_team_models(
                team_models=team_object.models,
                proxy_model_list=proxy_model_list,
                model_access_groups=model_access_groups,
            )
            for model_name in resolved_model_names:
                _models = llm_router.get_model_list(model_name=model_name, team_id=team_object.team_id)
                if _models is not None:
                    for model in _models:
                        model_id = model.get("model_info", {}).get("id", None)
                        if model_id is not None:
                            team_models.setdefault(model_id, set()).add(team_object.team_id)
    return team_models


async def _add_access_group_models_to_team_models(
    team_db_objects_typed: list[LiteLLM_TeamTable],
    llm_router: Router,
    prisma_client: PrismaClient,
    team_models: dict[str, set[str]],
) -> dict[str, set[str]]:
    """
    Resolve models reachable via team access groups and merge them into team_models.

    Batch-fetches all distinct access groups in a single DB query, then resolves
    each eligible team's access group models via the pre-fetched map.

    This ensures models associated with a team only through access groups
    (not directly in team.models) are included in the UI model listing.
    """
    # First pass: identify eligible teams and collect all distinct access group IDs
    eligible_teams: Final[list[LiteLLM_TeamTable]] = []
    all_access_group_ids: Final[set[str]] = set()

    for team_object in team_db_objects_typed:
        if not team_object.access_group_ids:
            continue

        # Skip teams with empty models list — they already have access to everything
        # (handled by _add_team_models_to_all_models)
        if not team_object.models or SpecialModelNames.all_proxy_models.value in team_object.models:
            continue

        eligible_teams.append(team_object)
        all_access_group_ids.update(team_object.access_group_ids)

    if not eligible_teams:
        return team_models

    # Single batch fetch for all access groups
    access_group_rows: Final = await AccessGroupRepository(prisma_client).table.find_many(
        where={"access_group_id": {"in": list(all_access_group_ids)}}
    )
    ag_model_map: Final[dict[str, list[str]]] = {
        row.access_group_id: row.access_model_names or [] for row in access_group_rows
    }

    # Second pass: resolve deployments for each eligible team
    for team_object in eligible_teams:
        model_names: set[str] = set()
        for ag_id in team_object.access_group_ids or []:
            model_names.update(ag_model_map.get(ag_id, []))

        for model_name in model_names:
            deployments = llm_router.get_model_list(model_name=model_name, team_id=team_object.team_id)
            if deployments is not None:
                for deployment in deployments:
                    model_id = deployment.get("model_info", {}).get("id", None)
                    if model_id is not None:
                        team_models.setdefault(model_id, set()).add(team_object.team_id)

    return team_models


async def get_all_team_models(
    user_teams: list[str] | Literal["*"],
    prisma_client: PrismaClient,
    llm_router: Router,
) -> dict[str, list[str]]:
    """
    Get all models across all teams user is in.

    1. Get all teams user is in
    2. Get all models across all teams
    3. Return {"model_id": ["team_id1", "team_id2"]}
    """

    team_db_objects_typed: list[LiteLLM_TeamTable] = []

    if user_teams == "*":
        team_db_objects: Sequence[SupportsModelDump] = await TeamRepository(prisma_client).table.find_many()
        team_db_objects_typed = [
            LiteLLM_TeamTable.model_validate(team_db_object.model_dump()) for team_db_object in team_db_objects
        ]
    else:
        team_db_objects = await TeamRepository(prisma_client).table.find_many(where={"team_id": {"in": user_teams}})

        team_db_objects_typed = [
            LiteLLM_TeamTable.model_validate(team_db_object.model_dump()) for team_db_object in team_db_objects
        ]

    team_models = _add_team_models_to_all_models(
        team_db_objects_typed=team_db_objects_typed,
        llm_router=llm_router,
    )

    # Also resolve models reachable via team access groups
    team_models = await _add_access_group_models_to_team_models(
        team_db_objects_typed=team_db_objects_typed,
        llm_router=llm_router,
        prisma_client=prisma_client,
        team_models=team_models,
    )

    # convert set to list
    returned_team_models: Final[dict[str, list[str]]] = {}
    for model_id, team_ids in team_models.items():
        returned_team_models[model_id] = list(team_ids)

    return returned_team_models


def get_direct_access_models(
    user_db_object: LiteLLM_UserTable,
    llm_router: Router,
) -> list[str]:
    """
    Get all models that user has direct access to.

    The 'all-proxy-models' sentinel grants direct access to every non-team
    deployment, mirroring how get_key_models expands it for the key/team path.
    """
    if SpecialModelNames.all_proxy_models.value in user_db_object.models:
        return llm_router.get_model_ids(exclude_team_models=True)

    return [
        model_id
        for model in user_db_object.models
        for deployment in (llm_router.get_model_list(model_name=model) or [])
        if (model_id := deployment.get("model_info", {}).get("id", None)) is not None
    ]


def _filter_models_to_user_accessible(all_models: list[dict]) -> list[dict]:
    """Keep only deployments the caller can use via direct access or team membership."""
    return [
        _model
        for _model in all_models
        if _model.get("model_info", {}).get("direct_access", False)
        or _model.get("model_info", {}).get("access_via_team_ids", [])
    ]


async def _populate_team_access_on_models(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    llm_router: Router,
    all_models: list[dict],
) -> list[dict]:
    """
    Populate `model_info.access_via_team_ids` and `model_info.direct_access`
    without filtering the model list.
    """
    user_teams: list[str] | Literal["*"] | None = None
    direct_access_models: list[str] = []
    if _user_has_admin_view(user_api_key_dict):
        user_teams = "*"
        direct_access_models = llm_router.get_model_ids(exclude_team_models=True)  # has access to all models
    elif user_api_key_dict.user_id is not None:
        user_db_object: Final[SupportsModelDump | None] = await UserRepository(prisma_client).table.find_unique(
            where={"user_id": user_api_key_dict.user_id}
        )
        if user_db_object is not None:
            user_object: Final = LiteLLM_UserTable.model_validate(user_db_object.model_dump())
            user_teams = user_object.teams or []
            direct_access_models = get_direct_access_models(
                user_db_object=user_object,
                llm_router=llm_router,
            )
    if user_teams is not None:
        team_models: Final = await get_all_team_models(
            user_teams=user_teams,
            prisma_client=prisma_client,
            llm_router=llm_router,
        )
        for _model in all_models:
            model_id = _model.get("model_info", {}).get("id", None)
            team_only_model_id = _model.get("model_info", {}).get("team_id", None)
            if model_id is not None:
                can_use_model = False
                if team_only_model_id is not None:
                    team_ids = team_models.get(model_id, [])
                    if team_ids and team_only_model_id in team_ids:
                        can_use_model = True
                else:
                    can_use_model = True
                if can_use_model:
                    _model["model_info"]["access_via_team_ids"] = team_models.get(model_id, [])

    direct_access_model_ids: Final = set(direct_access_models)
    for _model in all_models:
        model_id = _model.get("model_info", {}).get("id", None)
        if model_id is not None:
            _model["model_info"]["direct_access"] = model_id in direct_access_model_ids

    return all_models


async def get_all_team_and_direct_access_models(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    llm_router: Router,
    all_models: list[dict],
) -> list[dict]:
    """
    Get all models across all teams user is in.
    """
    all_models = await _populate_team_access_on_models(
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
        llm_router=llm_router,
        all_models=all_models,
    )
    return _filter_models_to_user_accessible(all_models)


def _enrich_model_info_with_litellm_data(
    model: dict[str, Any], debug: bool = False, llm_router: Router | None = None
) -> dict[str, Any]:
    """
    Enrich a model dictionary with litellm model info (pricing, context window, etc.)
    and remove sensitive information.

    Args:
        model: Model dictionary to enrich
        debug: Whether to include debug information like openai_client
        llm_router: Optional router instance for debug info

    Returns:
        Enriched model dictionary with sensitive info removed
    """
    # provided model_info in config.yaml
    model_info: Final = model.get("model_info", {})
    if debug is True:
        _openai_client = "None"
        if llm_router is not None:
            _openai_client = llm_router._get_client(deployment=model, kwargs={}, client_type="async") or "None"
        else:
            _openai_client = "llm_router_is_None"
        openai_client: Final = str(_openai_client)
        model["openai_client"] = openai_client

    # read litellm model_prices_and_context_window.json to get the following:
    # input_cost_per_token, output_cost_per_token, max_tokens
    litellm_model_info = get_litellm_model_info(model=model)

    # 2nd pass on the model, try seeing if we can find model in litellm model_cost map
    if litellm_model_info == {}:
        # use litellm_param model_name to get model_info
        litellm_params = model.get("litellm_params", {})
        litellm_model = litellm_params.get("model", None)
        try:
            litellm_model_info = litellm.get_model_info(model=litellm_model)
        except Exception:
            litellm_model_info = {}
    # 3rd pass on the model, try seeing if we can find model but without the "/" in model cost map
    if litellm_model_info == {}:
        # use litellm_param model_name to get model_info
        litellm_params = model.get("litellm_params", {})
        litellm_model = litellm_params.get("model", None)
        if litellm_model:
            split_model: Final = litellm_model.split("/")
            if len(split_model) > 0:
                litellm_model = split_model[-1]
            try:
                litellm_model_info = litellm.get_model_info(model=litellm_model, custom_llm_provider=split_model[0])
            except Exception:
                litellm_model_info = {}
    for k, v in litellm_model_info.items():
        if k not in model_info:
            model_info[k] = v
    model["model_info"] = model_info
    # don't return the api key / vertex credentials
    # don't return the llm credentials
    model = remove_sensitive_info_from_deployment(model, excluded_keys={"litellm_credential_name"})
    return model


async def _get_caller_byok_team_scope(
    user_api_key_dict: UserAPIKeyAuth | None,
    prisma_client: PrismaClient | None,
) -> set[str] | None:
    """
    Return the team IDs whose BYOK rows the caller is allowed to see via
    `/v2/model/info` search results.

    `None` means "no scoping" — used for admins and for callers/paths that
    have already been scoped upstream (or in tests that supply their own
    pre-filtered input set). A returned set (possibly empty) means BYOK rows
    must have `model_info.team_id` ∈ that set, otherwise they belong to a
    team the caller is not a member of and must be dropped.
    """
    if user_api_key_dict is None or prisma_client is None:
        return None
    if user_api_key_dict.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        return None
    key_team_scope: Final[set[str]] = {user_api_key_dict.team_id} if user_api_key_dict.team_id else set()
    user_id: Final = user_api_key_dict.user_id
    if user_id is None:
        return key_team_scope
    try:
        user_row: Final = await UserRepository(prisma_client).table.find_unique(where={"user_id": user_id})
    except Exception:
        verbose_proxy_logger.exception(
            "Failed to look up caller teams while scoping BYOK search; defaulting to key team scope only."
        )
        return key_team_scope
    if user_row is None:
        return key_team_scope
    return key_team_scope | set(user_row.teams or [])


def _byok_row_outside_caller_teams(model_info_dict: dict[str, JsonValue], allowed_team_ids: set[str] | None) -> bool:
    """Whether a team BYOK row belongs to a team the caller is not a member of.

    `team_id` is only set on team BYOK rows; non-team rows fall through
    unaffected. `allowed_team_ids is None` means no scoping (e.g. admins).
    """
    if allowed_team_ids is None:
        return False
    team_id: Final = model_info_dict.get("team_id")
    if team_id is None:
        return False
    return team_id not in allowed_team_ids


# Hard cap on rows the DB-side BYOK search may pull when results need to be
# sorted across the full match set. Without this, an authenticated caller
# can hit `/v2/model/info?search=<broad>&sortBy=<field>` and force the
# proxy to materialize and decrypt every matching BYOK row on each request.
_SORTED_SEARCH_DB_FETCH_CAP: Final = 500


async def _fetch_db_models_for_search(
    prisma_client: PrismaClient,
    proxy_config: ProxyConfig,
    search_lower: str,
    db_model_ids_in_router: set[str],
    router_models_count: int,
    page: int,
    size: int,
    sort_by: str | None,
    is_byok_outside_caller_teams: Callable[[dict[str, JsonValue]], bool],
) -> tuple[list[dict[str, Any]], int]:
    """
    Run the bounded DB query that backs `/v2/model/info?search=`. Returns
    `(decrypted_models, total_count)` where `total_count` is the cheap
    `count(...)` of rows matching `search` (not yet team-scoped) so the
    UI's pagination stays accurate without materializing every row.

    Earlier iterations also OR'd a JSON-path match on
    `model_info.team_public_model_name` to surface BYOK rows that live
    only in the DB. That branch fell back to `string_contains: ""`
    because Prisma's JSON `string_contains` is case-sensitive on
    Postgres, which let any authenticated caller force a full BYOK-table
    read via `/v2/model/info?search=x`. We rely on the router-side
    filter for `team_public_model_name` instead and keep the DB cost
    bounded by `search`.
    """
    db_where_condition: Final[dict[str, Any]] = {"model_name": {"contains": search_lower, "mode": "insensitive"}}
    if db_model_ids_in_router:
        db_where_condition["model_id"] = {"not": {"in": list(db_model_ids_in_router)}}

    # Unsorted searches only need enough DB rows to fill the current
    # page after counting router-side matches. Sorted searches need
    # ordering across the full match set, so fall back to a hard cap.
    if sort_by:
        take_limit = _SORTED_SEARCH_DB_FETCH_CAP
    else:
        take_limit = max(0, page * size - router_models_count)

    db_models_total_count: Final = await ModelRepository(prisma_client).table.count(where=db_where_condition)

    db_models_raw: Sequence[_ProxyModelRow] = []
    if take_limit > 0:
        db_models_raw = await ModelRepository(prisma_client).table.find_many(
            where=db_where_condition,
            take=take_limit,
        )

    # Scope BYOK rows to the caller's allowed teams so non-admin callers
    # can't enumerate other teams' BYOK metadata via `?search=...`.
    matching_db_rows: Final = [
        m
        for m in db_models_raw
        if not is_byok_outside_caller_teams(m.model_info if isinstance(m.model_info, dict) else {})
    ]

    decrypted: Final[list[dict[str, object]]] = []
    for db_model in matching_db_rows:
        decrypted_models = proxy_config.decrypt_model_list_from_db([db_model])
        if decrypted_models:
            decrypted.extend(decrypted_models)

    return decrypted, db_models_total_count


async def _apply_search_filter_to_models(
    all_models: list[dict[str, Any]],
    search: str,
    prisma_client: PrismaClient | None,
    proxy_config: ProxyConfig,
    user_api_key_dict: UserAPIKeyAuth | None = None,
    page: int = 1,
    size: int = 50,
    sort_by: str | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    """
    Apply search filter to models, querying database for additional matching models.

    Args:
        all_models: List of models to filter
        search: Search term (case-insensitive)
        prisma_client: Prisma client for database queries
        proxy_config: Proxy config for decrypting models
        user_api_key_dict: Caller identity used to scope BYOK matches to
            teams the caller belongs to. When omitted (None), no team
            scoping is applied — pass it from request handlers that expose
            this function to non-admin callers.
        page: Current page number (1-indexed). Used with ``size`` to bound
            the DB ``find_many(take=...)`` so a broad search term can't
            force a full table read + decrypt on every request.
        size: Page size. See ``page``.
        sort_by: Sort field. When set, results must be sorted across the
            full match set, so the DB fetch is capped at
            ``_SORTED_SEARCH_DB_FETCH_CAP`` instead of one page.

    Returns:
        Tuple of (filtered_models, total_count). total_count is None if not searching.
    """
    if not search or not search.strip():
        return all_models, None

    search_lower: Final = search.lower().strip()

    allowed_team_ids: Final = await _get_caller_byok_team_scope(
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )

    def _is_byok_outside_caller_teams(model_info_dict: dict[str, JsonValue]) -> bool:
        return _byok_row_outside_caller_teams(model_info_dict, allowed_team_ids)

    def _model_matches_search(m: dict[str, Any]) -> bool:
        # Team BYOK models persist an internal `model_name`
        # (e.g. `model_name_{team_id}_{uuid}`) and expose the user-facing
        # name via `model_info.team_public_model_name`. Match both so the
        # name shown in the UI is searchable.
        if search_lower in (m.get("model_name") or "").lower():
            return True
        team_public_model_name: Final = (m.get("model_info") or {}).get("team_public_model_name") or ""
        return search_lower in team_public_model_name.lower()

    # Filter models in router by search term, dropping BYOK rows that
    # belong to teams the caller is not a member of so search can't leak
    # other teams' models when the request omits `include_team_models` /
    # `teamId`.
    filtered_router_models: Final = [
        m
        for m in all_models
        if _model_matches_search(m) and not _is_byok_outside_caller_teams(m.get("model_info") or {})
    ]

    # Separate filtered models into config vs db models, and track db model IDs
    filtered_config_models: Final = []
    db_model_ids_in_router: Final = set()

    for m in filtered_router_models:
        model_info = m.get("model_info", {})
        is_db_model = model_info.get("db_model", False)
        model_id = model_info.get("id")

        if is_db_model and model_id:
            db_model_ids_in_router.add(model_id)
        else:
            filtered_config_models.append(m)

    config_models_count: Final = len(filtered_config_models)
    db_models_in_router_count: Final = len(db_model_ids_in_router)
    router_models_count: Final = config_models_count + db_models_in_router_count

    # Query database for additional models with search term
    db_models: list[dict[str, Any]] = []
    if prisma_client is not None:
        try:
            db_models, db_models_total_count = await _fetch_db_models_for_search(
                prisma_client=prisma_client,
                proxy_config=proxy_config,
                search_lower=search_lower,
                db_model_ids_in_router=db_model_ids_in_router,
                router_models_count=router_models_count,
                page=page,
                size=size,
                sort_by=sort_by,
                is_byok_outside_caller_teams=_is_byok_outside_caller_teams,
            )
            search_total_count = router_models_count + db_models_total_count
        except Exception as e:
            verbose_proxy_logger.exception("Error querying database models with search: %s", e)
            search_total_count = router_models_count
    else:
        search_total_count = router_models_count

    return filtered_router_models + db_models, search_total_count


def _normalize_datetime_for_sorting(dt: object) -> datetime | None:
    """
    Normalize a datetime value to a timezone-aware UTC datetime for sorting.

    This function handles:
    - None values: returns None
    - String values: parses ISO format strings and converts to UTC-aware datetime
    - Datetime objects: converts naive datetimes to UTC-aware, and aware datetimes to UTC

    Args:
        dt: Datetime value (None, str, or datetime object)

    Returns:
        UTC-aware datetime object, or None if input is None or cannot be parsed
    """
    if dt is None:
        return None

    if isinstance(dt, str):
        try:
            # Handle ISO format strings, including 'Z' suffix
            dt_str: Final = dt.replace("Z", "+00:00") if dt.endswith("Z") else dt
            parsed_dt = datetime.fromisoformat(dt_str)
            # Ensure it's UTC-aware
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            else:
                parsed_dt = parsed_dt.astimezone(timezone.utc)
            return parsed_dt
        except (ValueError, AttributeError):
            return None

    if isinstance(dt, datetime):
        # If naive, assume UTC and make it aware
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        # If aware, convert to UTC
        return dt.astimezone(timezone.utc)

    return None


def _sort_models(
    all_models: list[dict[str, Any]],
    sort_by: str | None,
    sort_order: str = "asc",
) -> list[dict[str, Any]]:
    """
    Sort models by the specified field and order.

    Args:
        all_models: List of models to sort
        sort_by: Field to sort by (model_name, created_at, updated_at, costs, status)
        sort_order: Sort order (asc or desc)

    Returns:
        Sorted list of models
    """
    if not sort_by or sort_by not in [
        "model_name",
        "created_at",
        "updated_at",
        "costs",
        "status",
    ]:
        return all_models

    reverse: Final = sort_order.lower() == "desc"

    def get_sort_key(model: dict[str, Any]) -> Any:
        model_info: Final = model.get("model_info", {})

        if sort_by == "model_name":
            # Team BYOK models persist an internal `model_name` (e.g.
            # `model_name_{team_id}_{uuid}`) and expose the user-facing
            # name via `model_info.team_public_model_name` — same as the
            # UI's getDisplayModelName. Sort by the displayed name so
            # BYOK rows interleave alphabetically with non-BYOK rows
            # instead of clumping at the end on their opaque IDs.
            team_public_model_name: Final = model_info.get("team_public_model_name")
            if team_public_model_name:
                return str(team_public_model_name).lower()
            return model.get("model_name", "").lower()

        elif sort_by == "created_at":
            created_at: Final = model_info.get("created_at")
            normalized_dt = _normalize_datetime_for_sorting(created_at)
            if normalized_dt is None:
                # Put None values at the end for asc, at the start for desc
                return (
                    datetime.max.replace(tzinfo=timezone.utc)
                    if not reverse
                    else datetime.min.replace(tzinfo=timezone.utc)
                )
            return normalized_dt

        elif sort_by == "updated_at":
            updated_at: Final = model_info.get("updated_at")
            normalized_dt = _normalize_datetime_for_sorting(updated_at)
            if normalized_dt is None:
                return (
                    datetime.max.replace(tzinfo=timezone.utc)
                    if not reverse
                    else datetime.min.replace(tzinfo=timezone.utc)
                )
            return normalized_dt

        elif sort_by == "costs":
            input_cost: Final = model_info.get("input_cost_per_token", 0) or 0
            output_cost: Final = model_info.get("output_cost_per_token", 0) or 0
            total_cost: Final = input_cost + output_cost
            # Put 0 or None costs at the end for asc, at the start for desc
            if total_cost == 0:
                return float("inf") if not reverse else float("-inf")
            return total_cost

        elif sort_by == "status":
            # False (config) comes before True (db) for asc
            db_model: Final = model_info.get("db_model", False)
            return db_model

        return None

    try:
        sorted_models: Final = sorted(all_models, key=get_sort_key, reverse=reverse)
        return sorted_models
    except Exception as e:
        verbose_proxy_logger.exception("Error sorting models by %s: %s", sort_by, e)
        return all_models


def _is_auto_router_model(model: Mapping[str, object]) -> bool:
    """
    True for any auto-router deployment, i.e. every `auto_router/*` strategy
    (semantic, complexity, adaptive, quality).

    Router._is_auto_router_deployment is deliberately narrower; it answers "is this the
    *semantic* auto-router strategy" and returns False for the complexity and adaptive
    prefixes, so it is not reusable here.
    """
    litellm_params: Final = model.get("litellm_params")
    if not isinstance(litellm_params, Mapping):
        return False
    litellm_model: Final = litellm_params.get("model")
    return isinstance(litellm_model, str) and litellm_model.startswith("auto_router/")


def _paginate_models_response(
    all_models: list[dict[str, Any]],
    page: int,
    size: int,
    total_count: int | None,
    search: str | None,
) -> dict[str, object]:
    """
    Paginate models and return response dictionary.

    Args:
        all_models: List of all models
        page: Current page number
        size: Page size
        total_count: Total count (if None, uses len(all_models))
        search: Search term (for logging)

    Returns:
        Paginated response dictionary
    """
    if total_count is None:
        total_count = len(all_models)

    skip: Final = (page - 1) * size
    total_pages: Final = -(-total_count // size) if total_count > 0 else 0
    paginated_models: Final = all_models[skip : skip + size]

    verbose_proxy_logger.debug(
        "Pagination: skip=%s, take=%s, total_count=%s, total_pages=%s, search=%s",
        skip,
        size,
        total_count,
        total_pages,
        search,
    )

    return {
        "data": paginated_models,
        "total_count": total_count,
        "current_page": page,
        "total_pages": total_pages,
        "size": size,
    }


def _team_models_resolve_to_names(team_models: list[str], access_groups: Mapping[str, Sequence[str]]) -> list[str]:
    """Expand team model entries (including access group names) to concrete model names."""
    resolved: Final[list[str]] = []
    for name in team_models:
        if name in access_groups:
            resolved.extend(access_groups[name])
        else:
            resolved.append(name)
    return resolved


async def _load_team_object_for_model_filter(team_id: str, prisma_client: PrismaClient) -> LiteLLM_TeamTable | None:
    """Load team row from DB; returns None if missing or on error."""
    try:
        team_db_object: Final[SupportsModelDump | None] = await TeamRepository(prisma_client).table.find_unique(
            where={"team_id": team_id}
        )
        if team_db_object is None:
            verbose_proxy_logger.warning("Team %s not found in database", team_id)
            return None
        return LiteLLM_TeamTable.model_validate(team_db_object.model_dump())
    except Exception as e:
        verbose_proxy_logger.exception("Error fetching team %s: %s", team_id, e)
        return None


async def _gather_team_accessible_model_ids(
    team_object: LiteLLM_TeamTable,
    team_id: str,
    prisma_client: PrismaClient,
    llm_router: Router,
) -> set[str]:
    """Collect model IDs the team can use from router config and DB."""
    team_accessible_model_ids: Final[set[str]] = set()
    access_groups: Final = llm_router.get_model_access_groups() if llm_router else {}

    if not team_object.models or SpecialModelNames.all_proxy_models.value in team_object.models:
        model_list: Final = llm_router.get_model_list() if llm_router else []
        if model_list is not None:
            for model in model_list:
                model_id = model.get("model_info", {}).get("id", None)
                if model_id is None:
                    continue
                team_model_id = model.get("model_info", {}).get("team_id", None)
                if team_model_id is None or team_model_id == team_id:
                    team_accessible_model_ids.add(model_id)
    else:
        resolved_model_names: Final[set[str]] = set()
        for model_name in team_object.models:
            if model_name in access_groups:
                resolved_model_names.update(access_groups[model_name])
            else:
                resolved_model_names.add(model_name)

        for model_name in resolved_model_names:
            _models = llm_router.get_model_list(model_name=model_name, team_id=team_id) if llm_router else []
            if _models is not None:
                for model in _models:
                    model_id = model.get("model_info", {}).get("id", None)
                    if model_id is not None:
                        team_accessible_model_ids.add(model_id)

    try:
        if team_object.models and SpecialModelNames.all_proxy_models.value not in team_object.models:
            _resolved_names: Final = _team_models_resolve_to_names(team_object.models, access_groups)
            db_models: Final[Sequence[_ProxyModelRow]] = await ModelRepository(prisma_client).table.find_many(
                where={"model_name": {"in": _resolved_names}}
            )
            for db_model in db_models:
                if db_model.model_id:
                    team_accessible_model_ids.add(db_model.model_id)
    except Exception as e:
        verbose_proxy_logger.debug("Error querying database models for team %s: %s", team_id, e)

    return team_accessible_model_ids


async def _authorize_team_id_query(
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> None:
    """
    `teamId` arrives untrusted via the /v2/model/info query string and the
    filter below includes BYOK rows solely on `model_info.team_id == team_id`.
    Without this guard, any authenticated user who knows (or guesses) another
    team's id could enumerate that team's BYOK model metadata. Allow only
    proxy admins or members of the requested team.
    """
    if user_api_key_dict.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        return

    user_id: Final = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to view this team's models"},
        )
    try:
        user_row: Final = await UserRepository(prisma_client).table.find_unique(where={"user_id": user_id})
    except Exception:
        verbose_proxy_logger.exception("Failed to look up caller teams while authorizing teamId filter")
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to view this team's models"},
        )

    if user_row is None or team_id not in (user_row.teams or []):
        raise HTTPException(
            status_code=403,
            detail={"error": "Not authorized to view this team's models"},
        )


async def _filter_models_by_team_id(
    all_models: list[dict[str, Any]],
    team_id: str,
    prisma_client: PrismaClient,
    llm_router: Router,
    user_api_key_dict: UserAPIKeyAuth | None = None,
) -> list[dict[str, Any]]:
    """
    Filter models by team ID. Returns models where:
    - team_id matches the model's BYOK team_id, OR
    - team_id is in access_via_team_ids, OR
    - model_id is reachable via team.models / access groups

    Args:
        all_models: List of models to filter
        team_id: Team ID to filter by
        prisma_client: Prisma client for database queries
        llm_router: Router instance for config queries
        user_api_key_dict: Caller auth context. When provided, the caller must
            be a proxy admin or a member of `team_id`; otherwise raises 403.

    Returns:
        Filtered list of models
    """
    if user_api_key_dict is not None:
        await _authorize_team_id_query(
            team_id=team_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )

    team_object: Final = await _load_team_object_for_model_filter(team_id, prisma_client)
    if team_object is None:
        return []

    team_accessible_model_ids = await _gather_team_accessible_model_ids(team_object, team_id, prisma_client, llm_router)

    # When filtering by a specific team we want exactly the models that team
    # can use: its BYOK rows and the deployments resolved from team.models /
    # access groups. `direct_access` describes the viewer's own permissions
    # (the admin path sets it on every non-team model) and must NOT widen the
    # team's visible set, otherwise selecting a team in the UI still shows
    # every public model the admin can call.
    filtered_models: Final = []
    for _model in all_models:
        model_info = _model.get("model_info", {})
        model_id = model_info.get("id", None)

        # BYOK rows owned by this team are always accessible to it, even if
        # they haven't been re-added to team.models for some reason.
        if model_info.get("team_id") == team_id:
            filtered_models.append(_model)
            continue

        access_via_team_ids = model_info.get("access_via_team_ids", [])
        if isinstance(access_via_team_ids, list) and team_id in access_via_team_ids:
            filtered_models.append(_model)
            continue

        # Catches models resolved from team.models / access groups that
        # weren't enriched with access_via_team_ids upstream.
        if model_id and model_id in team_accessible_model_ids:
            filtered_models.append(_model)

    return filtered_models


async def _find_model_by_id(
    model_id: str,
    search: str | None,
    llm_router: Router | None,
    prisma_client: PrismaClient | None,
    proxy_config: "ProxyConfig",
) -> tuple[list, int | None]:
    """Find a model by its ID and optionally filter by search term."""
    found_model = None

    # First, search in config
    if llm_router is not None:
        found_model = llm_router.get_model_info(id=model_id)
        if found_model:
            found_model = copy.deepcopy(found_model)

    # If not found in config, search in database
    if found_model is None:
        try:
            db_model: Final = await ModelRepository(prisma_client).table.find_unique(where={"model_id": model_id})
            if db_model:
                # Convert database model to router format
                decrypted_models: Final = proxy_config.decrypt_model_list_from_db([db_model])
                if decrypted_models:
                    found_model = decrypted_models[0]
        except Exception as e:
            verbose_proxy_logger.exception("Error querying database for modelId %s: %s", model_id, e)

    # If model found, verify search filter if provided
    if found_model is not None:
        if search is not None and search.strip():
            search_lower: Final = search.lower().strip()
            model_name: Final = found_model.get("model_name", "")
            if search_lower not in model_name.lower():
                # Model found but doesn't match search filter
                found_model = None

    # Set all_models to the found model or empty list
    all_models: Final = [found_model] if found_model is not None else []
    search_total_count: Final[int | None] = len(all_models)
    return all_models, search_total_count


@router.get(
    "/v2/model/info",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def model_info_v2(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: str | None = fastapi.Query(None, description="Specify the model name (optional)"),
    user_models_only: bool | None = fastapi.Query(False, description="Only return models added by this user"),
    include_team_models: bool | None = fastapi.Query(
        False, description="Return all models across all teams user is in."
    ),
    debug: bool | None = False,
    page: int = Query(1, description="Page number", ge=1),
    size: int = Query(50, description="Page size", ge=1),
    search: str | None = fastapi.Query(None, description="Search model names (case-insensitive partial match)"),
    modelId: str | None = fastapi.Query(None, description="Search for a specific model by its unique ID"),
    teamId: str | None = fastapi.Query(
        None,
        description="Filter models by team ID. Returns models with direct_access=True or teamId in access_via_team_ids",
    ),
    sortBy: str | None = fastapi.Query(
        None,
        description="Field to sort by. Options: model_name, created_at, updated_at, costs, status",
    ),
    sortOrder: str | None = fastapi.Query(
        "asc",
        description="Sort order. Options: asc, desc",
    ),
    exclude_auto_routers: bool | None = fastapi.Query(
        False,
        description=(
            "Omit auto-router deployments (litellm model prefixed `auto_router/`). "
            "They select among deployments rather than being deployments themselves, so a "
            "caller rendering a deployment list can leave them out. Defaults to false, so "
            "existing callers are unaffected"
        ),
    ),
):
    """
    Paginated model metadata for proxy deployments (pricing, provider, team access).

    Returns configured router deployments with enriched `model_info` (costs, provider,
    context window, etc.). Sensitive fields such as API keys and api_base are omitted.

    Query parameters:
        model: Filter to a single public `model_name`.
        user_models_only: When true, only return models created by the calling user.
        include_team_models: When true, populate `access_via_team_ids` and `direct_access`
            on each model and filter to deployments the caller can use.
        page / size: Pagination controls (defaults: page=1, size=50).
        search: Case-insensitive partial match on model name or team public name.
        modelId: Return a single deployment by LiteLLM model id.
        teamId: Filter to models with direct access or team membership for this team id.
        sortBy / sortOrder: Sort by model_name, created_at, updated_at, costs, or status.

    Example request:
    ```
    curl -X GET 'http://localhost:4000/v2/model/info?include_team_models=true&page=1&size=50' \\
    --header 'Authorization: Bearer sk-1234'
    ```

    Example response:
    ```json
    {
        "data": [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4.1"},
                "model_info": {
                    "id": "abc123",
                    "litellm_provider": "openai",
                    "access_via_team_ids": ["team-1"],
                    "direct_access": true
                }
            }
        ],
        "total_count": 1,
        "current_page": 1,
        "total_pages": 1,
        "size": 50
    }
    ```
    """
    global llm_model_list, general_settings, user_config_file_path, proxy_config, llm_router

    # Return empty data array when no models are configured (graceful handling for fresh installs)
    if llm_router is None or not llm_router.model_list:
        return {
            "data": [],
            "total_count": 0,
            "current_page": page,
            "total_pages": 0,
            "size": size,
        }

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Load existing config
    await proxy_config.get_config()

    # If modelId is provided, search for the specific model
    if modelId is not None:
        all_models, search_total_count = await _find_model_by_id(
            model_id=modelId,
            search=search,
            llm_router=llm_router,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )
    else:
        # Normal flow when modelId is not provided
        all_models = copy.deepcopy(llm_router.model_list)

        if user_model is not None:
            # if user does not use a config.yaml, https://github.com/BerriAI/litellm/issues/2061
            all_models += [user_model]

        if model is not None:
            all_models = [m for m in all_models if m["model_name"] == model]

        # Apply search filter if provided
        all_models, search_total_count = await _apply_search_filter_to_models(
            all_models=all_models,
            search=search or "",
            prisma_client=prisma_client,
            proxy_config=proxy_config,
            user_api_key_dict=user_api_key_dict,
            page=page,
            size=size,
            sort_by=sortBy,
        )

    if user_models_only:
        all_models = await non_admin_all_models(
            all_models=all_models,
            llm_router=llm_router,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )

    if include_team_models:
        all_models = await get_all_team_and_direct_access_models(
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            llm_router=llm_router,
            all_models=all_models,
        )

    # Fill in model info based on config.yaml and litellm model_prices_and_context_window.json
    # This must happen before teamId filtering so that direct_access and access_via_team_ids are populated
    for i, _model in enumerate(all_models):
        all_models[i] = _enrich_model_info_with_litellm_data(
            model=_model,
            debug=debug if debug is not None else False,
            llm_router=llm_router,
        )

    # Apply teamId filter if provided
    if teamId is not None and teamId.strip():
        all_models = await _filter_models_by_team_id(
            all_models=all_models,
            team_id=teamId.strip(),
            prisma_client=prisma_client,
            llm_router=llm_router,
            user_api_key_dict=user_api_key_dict,
        )
        # Update search_total_count after teamId filter is applied
        search_total_count = len(all_models)

    # If modelId was provided, update search_total_count after filters are applied
    # to ensure pagination reflects the final filtered result (0 or 1)
    if modelId is not None:
        search_total_count = len(all_models)

    # Apply sorting before pagination
    if sortBy:
        # Validate sortOrder
        if sortOrder and sortOrder.lower() not in ["asc", "desc"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sortOrder: {sortOrder}. Must be 'asc' or 'desc'",
            )
        all_models = _sort_models(
            all_models=all_models,
            sort_by=sortBy,
            sort_order=sortOrder or "asc",
        )

    verbose_proxy_logger.debug("all_models: %s", all_models)

    # Append A2A agents to models list
    from litellm.proxy.agent_endpoints.model_list_helpers import (
        append_agents_to_model_info,
    )

    all_models = await append_agents_to_model_info(
        models=all_models,
        user_api_key_dict=user_api_key_dict,
    )

    # `is True` because direct-call tests bypass FastAPI, so the Query default arrives as a
    # truthy sentinel object rather than False.
    if exclude_auto_routers is True:
        all_models = [m for m in all_models if not _is_auto_router_model(m)]

    # Update total count to include agents
    search_total_count = len(all_models)

    # Translate `model_name` to the public name for team-scoped rows.
    all_models = [_translate_model_name_for_response(m) for m in all_models]

    return _paginate_models_response(
        all_models=all_models,
        page=page,
        size=size,
        total_count=search_total_count,
        search=search,
    )


@router.get(
    "/model/streaming_metrics",
    description="View time to first token for models in spend logs",
    tags=["model management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def model_streaming_metrics(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    _selected_model_group: str | None = None,
    startTime: datetime | None = None,
    endTime: datetime | None = None,
):
    global prisma_client, llm_router
    if prisma_client is None:
        raise ProxyException(
            message=CommonProxyErrors.db_not_connected_error.value,
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    startTime = startTime or datetime.now() - timedelta(days=7)  # show over past week
    endTime = endTime or datetime.now()

    is_same_day: Final = startTime.date() == endTime.date()
    if is_same_day:
        sql_query = """
            SELECT
                api_base,
                model_group,
                model,
                "startTime",
                request_id,
                EXTRACT(epoch FROM ("completionStartTime" - "startTime")) AS time_to_first_token
            FROM
                "LiteLLM_SpendLogs"
            WHERE
                "model_group" = $1 AND "cache_hit" != 'True'
                AND "completionStartTime" IS NOT NULL
                AND "completionStartTime" != "endTime"
                AND DATE("startTime") = DATE($2::timestamp)
            GROUP BY
                api_base,
                model_group,
                model,
                request_id
            ORDER BY
                time_to_first_token DESC;
        """
    else:
        sql_query = """
            SELECT
                api_base,
                model_group,
                model,
                DATE_TRUNC('day', "startTime")::DATE AS day,
                AVG(EXTRACT(epoch FROM ("completionStartTime" - "startTime"))) AS time_to_first_token
            FROM
                "LiteLLM_SpendLogs"
            WHERE
                "startTime" BETWEEN $2::timestamp AND $3::timestamp
                AND "model_group" = $1 AND "cache_hit" != 'True'
                AND "completionStartTime" IS NOT NULL
                AND "completionStartTime" != "endTime"
            GROUP BY
                api_base,
                model_group,
                model,
                day
            ORDER BY
                time_to_first_token DESC;
        """

    _all_api_bases: Final = set()
    db_response: Final[Sequence[_TTFTRow] | None] = await prisma_client.db.query_raw(
        sql_query, _selected_model_group, startTime, endTime
    )
    _daily_entries: dict = {}  # {"Jun 23": {"model1": 0.002, "model2": 0.003}}
    if db_response is not None:
        for model_data in db_response:
            _api_base = model_data["api_base"]
            _model = model_data["model"]
            time_to_first_token = model_data["time_to_first_token"]
            unique_key = ""
            if is_same_day:
                _request_id = model_data["request_id"]
                unique_key = _request_id
                if _request_id not in _daily_entries:
                    _daily_entries[_request_id] = {}
            else:
                _day = model_data["day"]
                unique_key = _day
                time_to_first_token = model_data["time_to_first_token"]
                if _day not in _daily_entries:
                    _daily_entries[_day] = {}
            _combined_model_name = str(_model)
            if "https://" in _api_base:
                _combined_model_name = str(_api_base)
            if "/openai/" in _combined_model_name:
                _combined_model_name = _combined_model_name.split("/openai/")[0]

            _all_api_bases.add(_combined_model_name)

            _daily_entries[unique_key][_combined_model_name] = time_to_first_token

        """
        each entry needs to be like this:
        {
            date: 'Jun 23',
            'gpt-4-https://api.openai.com/v1/': 0.002,
            'gpt-43-https://api.openai.com-12/v1/': 0.002,
        }
        """
        # convert daily entries to list of dicts

        response: Final[list[dict]] = []

        # sort daily entries by date
        _daily_entries = dict(sorted(_daily_entries.items(), key=lambda item: item[0]))
        for day in _daily_entries:
            entry = {"date": str(day)}
            for model_key, latency in _daily_entries[day].items():
                entry[model_key] = latency
            response.append(entry)

        return {
            "data": response,
            "all_api_bases": list(_all_api_bases),
        }


@router.get(
    "/model/metrics",
    description="View number of requests & avg latency per model on config.yaml",
    tags=["model management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def model_metrics(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    _selected_model_group: str | None = "gpt-4-32k",
    startTime: datetime | None = None,
    endTime: datetime | None = None,
    api_key: str | None = None,
    customer: str | None = None,
):
    global prisma_client, llm_router
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    startTime = startTime or datetime.now() - timedelta(days=DAYS_IN_A_MONTH)
    endTime = endTime or datetime.now()

    if api_key is None or api_key == "undefined":
        api_key = "null"

    if customer is None or customer == "undefined":
        customer = "null"

    sql_query: Final = """
        SELECT
            api_base,
            model_group,
            model,
            DATE_TRUNC('day', "startTime")::DATE AS day,
            AVG(EXTRACT(epoch FROM ("endTime" - "startTime")) / NULLIF("completion_tokens", 0)) AS avg_latency_per_token
        FROM
            "LiteLLM_SpendLogs"
        WHERE
            "startTime" >= $2::timestamp AND "startTime" <= $3::timestamp
            AND "model_group" = $1 AND "cache_hit" != 'True'
            AND (
                CASE
                    WHEN $4 != 'null' THEN "api_key" = $4
                    ELSE TRUE
                END
            )
            AND (
                CASE
                    WHEN $5 != 'null' THEN "end_user" = $5
                    ELSE TRUE
                END
            )
        GROUP BY
            api_base,
            model_group,
            model,
            day
        HAVING
            SUM(completion_tokens) > 0
        ORDER BY
            avg_latency_per_token DESC;
    """
    _all_api_bases: Final = set()
    db_response: Final[Sequence[_LatencyRow] | None] = await prisma_client.db.query_raw(
        sql_query, _selected_model_group, startTime, endTime, api_key, customer
    )
    _daily_entries: dict = {}  # {"Jun 23": {"model1": 0.002, "model2": 0.003}}

    if db_response is not None:
        for model_data in db_response:
            _api_base = model_data["api_base"]
            _model = model_data["model"]
            _day = model_data["day"]
            _avg_latency_per_token = model_data["avg_latency_per_token"]
            if _day not in _daily_entries:
                _daily_entries[_day] = {}
            _combined_model_name = str(_model)
            if _api_base is not None and "https://" in _api_base:
                _combined_model_name = str(_api_base)
            if _combined_model_name is not None and "/openai/" in _combined_model_name:
                _combined_model_name = _combined_model_name.split("/openai/")[0]

            _all_api_bases.add(_combined_model_name)
            _daily_entries[_day][_combined_model_name] = _avg_latency_per_token

        """
        each entry needs to be like this:
        {
            date: 'Jun 23',
            'gpt-4-https://api.openai.com/v1/': 0.002,
            'gpt-43-https://api.openai.com-12/v1/': 0.002,
        }
        """
        # convert daily entries to list of dicts

        response: Final[list[dict]] = []

        # sort daily entries by date
        _daily_entries = dict(sorted(_daily_entries.items(), key=lambda item: item[0]))
        for day in _daily_entries:
            entry = {"date": str(day)}
            for model_key, latency in _daily_entries[day].items():
                entry[model_key] = latency
            response.append(entry)

        return {
            "data": response,
            "all_api_bases": list(_all_api_bases),
        }


@router.get(
    "/model/metrics/slow_responses",
    description="View number of hanging requests per model_group",
    tags=["model management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def model_metrics_slow_responses(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    _selected_model_group: str | None = "gpt-4-32k",
    startTime: datetime | None = None,
    endTime: datetime | None = None,
    api_key: str | None = None,
    customer: str | None = None,
):
    global prisma_client, llm_router, proxy_logging_obj
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    if api_key is None or api_key == "undefined":
        api_key = "null"

    if customer is None or customer == "undefined":
        customer = "null"

    startTime = startTime or datetime.now() - timedelta(days=DAYS_IN_A_MONTH)
    endTime = endTime or datetime.now()

    alerting_threshold = (
        proxy_logging_obj.slack_alerting_instance.alerting_threshold or DEFAULT_SLACK_ALERTING_THRESHOLD
    )
    alerting_threshold = int(alerting_threshold)

    sql_query: Final = """
SELECT
    api_base,
    COUNT(*) AS total_count,
    SUM(CASE
        WHEN ("endTime" - "startTime") >= (INTERVAL '1 SECOND' * CAST($1 AS INTEGER)) THEN 1
        ELSE 0
    END) AS slow_count
FROM
    "LiteLLM_SpendLogs"
WHERE
    "model_group" = $2
    AND "cache_hit" != 'True'
    AND "startTime" >= $3::timestamp
    AND "startTime" <= $4::timestamp
    AND (
        CASE
            WHEN $5 != 'null' THEN "api_key" = $5
            ELSE TRUE
        END
    )
    AND (
        CASE
            WHEN $6 != 'null' THEN "end_user" = $6
            ELSE TRUE
        END
    )
GROUP BY
    api_base
ORDER BY
    slow_count DESC;
    """

    db_response: Final = await prisma_client.db.query_raw(
        sql_query,
        alerting_threshold,
        _selected_model_group,
        startTime,
        endTime,
        api_key,
        customer,
    )

    if db_response is not None:
        for row in db_response:
            _api_base = row.get("api_base") or ""
            if "/openai/" in _api_base:
                _api_base = _api_base.split("/openai/")[0]
            row["api_base"] = _api_base
    return db_response


@router.get(
    "/model/metrics/exceptions",
    description="View number of failed requests per model on config.yaml",
    tags=["model management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def model_metrics_exceptions(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    _selected_model_group: str | None = None,
    startTime: datetime | None = None,
    endTime: datetime | None = None,
    api_key: str | None = None,
    customer: str | None = None,
):
    global prisma_client, llm_router
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    startTime = startTime or datetime.now() - timedelta(days=DAYS_IN_A_MONTH)
    endTime = endTime or datetime.now()

    if api_key is None or api_key == "undefined":
        api_key = "null"

    """
    """
    sql_query: Final = """
        WITH cte AS (
            SELECT
                CASE WHEN api_base = '' THEN litellm_model_name ELSE CONCAT(litellm_model_name, '-', api_base) END AS combined_model_api_base,
                exception_type,
                COUNT(*) AS num_rate_limit_exceptions
            FROM "LiteLLM_ErrorLogs"
            WHERE
                "startTime" >= $1::timestamp
                AND "endTime" <= $2::timestamp
                AND model_group = $3
            GROUP BY combined_model_api_base, exception_type
        )
        SELECT
            combined_model_api_base,
            COUNT(*) AS total_exceptions,
            json_object_agg(exception_type, num_rate_limit_exceptions) AS exception_counts
        FROM cte
        GROUP BY combined_model_api_base
        ORDER BY total_exceptions DESC
        LIMIT 200;
    """
    db_response: Final[Sequence[_ExceptionRow] | None] = await prisma_client.db.query_raw(
        sql_query, startTime, endTime, _selected_model_group, api_key
    )
    response: Final[list[dict]] = []
    exception_types: Final = set()

    """
    Return Data
    {
        "combined_model_api_base": "gpt-3.5-turbo-https://api.openai.com/v1/,
        "total_exceptions": 5,
        "BadRequestException": 5,
        "TimeoutException": 2
    }
    """

    if db_response is not None:
        # loop through all models
        for model_data in db_response:
            model = model_data.get("combined_model_api_base", "")
            total_exceptions = model_data.get("total_exceptions", 0)
            exception_counts = model_data.get("exception_counts", {})
            curr_row = {
                "model": model,
                "total_exceptions": total_exceptions,
            }
            curr_row.update(exception_counts)
            response.append(curr_row)
            for k, v in exception_counts.items():
                exception_types.add(k)

    return {"data": response, "exception_types": list(exception_types)}


def _deployment_matches_allowed_model_names(model: dict[str, JsonValue], allowed_model_names: set[str]) -> bool:
    """Match a router deployment against allowed public model names.

    Team-scoped rows store an internal routing key in ``model_name``; callers
    with key/team restrictions still refer to the public name in
    ``model_info.team_public_model_name``.
    """
    if model.get("model_name") in allowed_model_names:
        return True
    model_info: Final = model.get("model_info")
    if not isinstance(model_info, dict):
        return False
    team_public_model_name: Final = model_info.get("team_public_model_name")
    return isinstance(team_public_model_name, str) and team_public_model_name in allowed_model_names


def _get_v1_model_info_allowed_model_names(
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: Router,
) -> set[str] | None:
    """Return key/team allowlisted public model names, or None if unrestricted."""
    model_access_groups: Final = llm_router.get_model_access_groups()
    proxy_model_list: Final = llm_router.get_model_names()
    key_models: Final = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )
    team_models: Final = get_team_models(
        team_models=user_api_key_dict.team_models,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )
    if not key_models and not team_models:
        return None
    return set(
        get_complete_model_list(
            key_models=key_models,
            team_models=team_models,
            proxy_model_list=proxy_model_list,
            user_model=user_model,
            infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
            llm_router=llm_router,
            return_wildcard_routes=False,
        )
    )


def _filter_v1_model_info_deployments(
    all_models: list[dict],
    allowed_model_names: set[str] | None,
) -> list[dict]:
    if allowed_model_names is None:
        return all_models
    return [model for model in all_models if _deployment_matches_allowed_model_names(model, allowed_model_names)]


def _translate_model_name_for_response(model: dict) -> dict:
    """For team-scoped DB rows, replace `model_name` with the public name
    in `model_info.team_public_model_name` before returning. The DB column
    and the in-memory router index keep the internal mangled name
    (`model_name_{team_id}_{uuid}`) as the routing key -- this swap is a
    presentation-layer concern. Returns a shallow copy; never mutates.

    Without this swap the internal name leaks into `/v1/model/info` and
    `/v2/model/info`, the dashboard binds its edit form to it, and a
    non-rename save round-trips the internal name back -- corrupting
    `team_public_model_name` and the team ACL (see issue #28382).
    """
    if not isinstance(model, dict):
        return model
    model_info: Final = model.get("model_info") or {}
    if not isinstance(model_info, dict):
        return model
    team_public: Final = model_info.get("team_public_model_name")
    team_id: Final = model_info.get("team_id")
    if not team_public or not team_id:
        return model
    current: Final = model.get("model_name") or ""
    if not current.startswith(f"model_name_{team_id}_"):
        return model
    return {**model, "model_name": team_public}


def _get_proxy_model_info(model: dict) -> dict:
    # provided model_info in config.yaml
    model_info: Final = model.get("model_info", {})

    # read litellm model_prices_and_context_window.json to get the following:
    # input_cost_per_token, output_cost_per_token, max_tokens
    litellm_model_info = get_litellm_model_info(model=model)

    # 2nd pass on the model, try seeing if we can find model in litellm model_cost map
    if litellm_model_info == {}:
        # use litellm_param model_name to get model_info
        litellm_params = model.get("litellm_params", {})
        litellm_model = litellm_params.get("model", None)
        try:
            litellm_model_info = litellm.get_model_info(model=litellm_model)
        except Exception:
            litellm_model_info = {}
    # 3rd pass on the model, try seeing if we can find model but without the "/" in model cost map
    if litellm_model_info == {}:
        # use litellm_param model_name to get model_info
        litellm_params = model.get("litellm_params", {})
        litellm_model = litellm_params.get("model", None)
        split_model: Final = litellm_model.split("/")
        if len(split_model) > 0:
            litellm_model = split_model[-1]
        try:
            litellm_model_info = litellm.get_model_info(model=litellm_model, custom_llm_provider=split_model[0])
        except Exception:
            litellm_model_info = {}
    for k, v in litellm_model_info.items():
        if k not in model_info:
            model_info[k] = v
    model["model_info"] = model_info
    # don't return the llm credentials
    model = remove_sensitive_info_from_deployment(deployment_dict=model, excluded_keys={"litellm_credential_name"})

    return _translate_model_name_for_response(model)


@router.get(
    "/model/info",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/v1/model/info",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def model_info_v1(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_model_id: str | None = None,
    include_team_models: bool | None = fastapi.Query(
        False,
        description="When true, filter to deployments the caller can use via direct access or team membership.",
    ),
    teamId: str | None = fastapi.Query(
        None,
        description="Filter models by team ID. Returns models with direct_access=True or teamId in access_via_team_ids",
    ),
    healthy_only: bool | None = False,
):
    """
    Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)

    Parameters:
        litellm_model_id: Optional[str] = None (this is the value of `x-litellm-model-id` returned in response headers)

        - When litellm_model_id is passed, it will return the info for that specific model
        - When litellm_model_id is not passed, it will return the info for all models
        - include_team_models: When true, filter to deployments the caller can use (same as /v2/model/info).
        - teamId: Filter to models accessible by the given team.
        - healthy_only: When true, hide models whose backing deployments are all marked
          unhealthy by background health checks, matching `/v1/models?healthy_only=true`.
          Set `general_settings.model_list_healthy_only: true` to apply this to every
          caller without the query parameter. Requires `background_health_checks: true`,
          plus either `model_list_healthy_only` or `enable_health_check_routing` to keep
          deployment health state cached; without health state the listing is returned
          unfiltered (fail open). Ignored when `litellm_model_id` is passed, since that
          is a direct lookup of one deployment rather than a listing. Hiding is
          presentation-only: a hidden model can still be called directly.

    Each model in the list response includes `model_info.access_via_team_ids` and
    `model_info.direct_access` when the proxy database is connected.

    Returns:
        Returns a dictionary containing information about each model.

    Example Response:
    ```json
    {
        "data": [
                    {
                        "model_name": "fake-openai-endpoint",
                        "litellm_params": {
                            "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                            "model": "openai/fake"
                        },
                        "model_info": {
                            "id": "112f74fab24a7a5245d2ced3536dd8f5f9192c57ee6e332af0f0512e08bed5af",
                            "db_model": false
                        }
                    }
                ]
    }

    ```
    """
    global llm_model_list, general_settings, user_config_file_path, proxy_config, llm_router, user_model

    # Unit tests call this handler directly; FastAPI normally resolves Query defaults.
    if not isinstance(include_team_models, bool):
        include_team_models = False
    if not isinstance(teamId, str):
        teamId = None

    if user_model is not None:
        # user is trying to get specific model from litellm router
        try:
            model_info: dict = cast(dict, litellm.get_model_info(model=user_model))
        except Exception:
            model_info = {}
        _deployment_info: Final = Deployment(
            model_name="*",
            litellm_params=LiteLLM_Params(
                model=user_model,
            ),
            model_info=model_info,
        )
        _deployment_info_dict = _deployment_info.model_dump()
        _deployment_info_dict = remove_sensitive_info_from_deployment(
            deployment_dict=_deployment_info_dict,
            excluded_keys={"litellm_credential_name"},
        )
        return {"data": _deployment_info_dict}

    if llm_model_list is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "LLM Model List not loaded in. Make sure you passed models in your config.yaml or on the LiteLLM Admin UI. - https://docs.litellm.ai/docs/proxy/configs"
            },
        )

    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "LLM Router is not loaded in. Make sure you passed models in your config.yaml or on the LiteLLM Admin UI. - https://docs.litellm.ai/docs/proxy/configs"
            },
        )

    if prisma_client is None and (include_team_models or (teamId is not None and teamId.strip())):
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if litellm_model_id is not None:
        # user is trying to get specific model from litellm router
        deployment_info: Final = llm_router.get_deployment(model_id=litellm_model_id)
        if deployment_info is None:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Model id = {litellm_model_id} not found on litellm proxy"},
            )
        _deployment_info_dict = _get_proxy_model_info(model=deployment_info.model_dump(exclude_none=True))
        single_model_list: list[dict] = [_deployment_info_dict]
        if prisma_client is not None:
            single_model_list = await _populate_team_access_on_models(
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
                llm_router=llm_router,
                all_models=single_model_list,
            )
            if include_team_models:
                single_model_list = _filter_models_to_user_accessible(single_model_list)
            if teamId is not None and teamId.strip():
                single_model_list = await _filter_models_by_team_id(
                    all_models=single_model_list,
                    team_id=teamId.strip(),
                    prisma_client=prisma_client,
                    llm_router=llm_router,
                    user_api_key_dict=user_api_key_dict,
                )
        return {"data": single_model_list}

    # Return router deployments (same source as /v2/model/info), not wildcard-
    # expanded model names from get_complete_model_list(). Team-scoped rows
    # use internal routing keys (model_name_{team_id}_{uuid}) and were omitted
    # when v1 resolved models only via public model_name strings.
    all_models: list[dict] = copy.deepcopy(llm_router.model_list)
    alias_models: Final = copy.deepcopy(llm_router.get_model_list_from_model_alias())
    all_models.extend(alias_models)

    all_models = expand_wildcard_deployments_for_model_info(all_models)

    allowed_model_names: Final = _get_v1_model_info_allowed_model_names(
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
    )

    all_models = _filter_v1_model_info_deployments(
        all_models=all_models,
        allowed_model_names=allowed_model_names,
    )

    # Team BYOK deployments carry an internal routing key and other teams'
    # public name/team_id/api_base; drop the ones the caller cannot access so
    # listing the full router model_list does not leak cross-team metadata.
    allowed_team_ids: Final = await _get_caller_byok_team_scope(
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
    )
    all_models = [
        model
        for model in all_models
        if not _byok_row_outside_caller_teams(model.get("model_info") or {}, allowed_team_ids)
    ]

    if prisma_client is not None:
        all_models = await _populate_team_access_on_models(
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            llm_router=llm_router,
            all_models=all_models,
        )

    if include_team_models:
        all_models = _filter_models_to_user_accessible(all_models)

    all_models = [
        _translate_model_name_for_response(_enrich_model_info_with_litellm_data(model=model, llm_router=llm_router))
        for model in all_models
    ]

    if teamId is not None and teamId.strip():
        all_models = await _filter_models_by_team_id(
            all_models=all_models,
            team_id=teamId.strip(),
            prisma_client=cast(PrismaClient, prisma_client),
            llm_router=llm_router,
            user_api_key_dict=user_api_key_dict,
        )

    hidden_names: Final = await get_hidden_unhealthy_model_names(
        healthy_only=healthy_only,
        general_settings=general_settings,
        llm_router=llm_router,
    )
    visible_models: Final = [model for model in all_models if model.get("model_name") not in hidden_names]

    verbose_proxy_logger.debug("all_models: %s", visible_models)
    return {"data": visible_models}


@router.get(
    "/model/deprecations",
    tags=("model management",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ModelDeprecationResponse,
)
@router.get(
    "/v1/model/deprecations",
    tags=("model management",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=ModelDeprecationResponse,
)
async def model_deprecations(
    warn_within_days: int = DEFAULT_DEPRECATION_WARN_DAYS,
) -> ModelDeprecationResponse:
    """List models with known deprecation/sunset dates, bucketed by urgency.

    Reads `deprecation_date` metadata from `model_prices_and_context_window.json`
    (and any per-deployment `model_info.deprecation_date` overrides) for the
    models configured on this proxy.

    Parameters:
        warn_within_days: Window (in days) used to bucket "imminent" models,
            30 by default.

    Returns:
        A payload with three lists of `ModelDeprecationInfo` entries:

        - `deprecated`: deprecation date is in the past, so these requests may
          fail at any time.
        - `imminent`: deprecation date is within `warn_within_days` from today.
        - `upcoming`: deprecation date is further out.

    Example:
    ```shell
    curl -X GET 'http://localhost:4000/model/deprecations' \\
        -H 'Authorization: Bearer sk-1234'
    ```
    """
    return collect_model_deprecations(llm_router=llm_router, warn_within_days=warn_within_days)


def _get_model_group_info(
    llm_router: Router, all_models_str: list[str], model_group: str | None
) -> list[ModelGroupInfoProxy]:
    model_groups: Final[list[ModelGroupInfoProxy]] = []

    unique_models: Final = []
    for model in all_models_str:
        if model not in unique_models:
            unique_models.append(model)

    for model in unique_models:
        if model_group is not None and model_group != model:
            continue

        _model_group_info = llm_router.get_model_group_info(model_group=model)

        if _model_group_info is not None:
            model_groups.append(ModelGroupInfoProxy.model_validate(_model_group_info.model_dump()))
        else:
            model_group_info = ModelGroupInfoProxy(
                model_group=model,
                providers=[],
            )
            model_groups.append(model_group_info)

    ## check for public model groups
    if litellm.public_model_groups is not None:
        for mg in model_groups:
            if mg.model_group in litellm.public_model_groups:
                mg.is_public_model_group = True

    return model_groups


@router.get(
    "/model_group/info",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def model_group_info(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model_group: str | None = None,
):
    """
    Get information about all the deployments on litellm proxy, including config.yaml descriptions (except api key and api base)

    - /model_group/info returns all model groups. End users of proxy should use /model_group/info since those models will be used for /chat/completions, /embeddings, etc.
    - /model_group/info?model_group=rerank-english-v3.0 returns all model groups for a specific model group (`model_name` in config.yaml)



    Example Request (All Models):
    ```shell
    curl -X 'GET' \
    'http://localhost:4000/model_group/info' \
    -H 'accept: application/json' \
    -H 'x-api-key: sk-1234'
    ```

    Example Request (Specific Model Group):
    ```shell
    curl -X 'GET' \
    'http://localhost:4000/model_group/info?model_group=rerank-english-v3.0' \
    -H 'accept: application/json' \
    -H 'Authorization: Bearer sk-1234'
    ```

    Example Request (Specific Wildcard Model Group): (e.g. `model_name: openai/*` on config.yaml)
    ```shell
    curl -X 'GET' \
    'http://localhost:4000/model_group/info?model_group=openai/tts-1'
    -H 'accept: application/json' \
    -H 'Authorization: Bearersk-1234'
    ```

    Learn how to use and set wildcard models [here](https://docs.litellm.ai/docs/wildcard_routing)

    Example Response:
    ```json
        {
            "data": [
                {
                "model_group": "rerank-english-v3.0",
                "providers": [
                    "cohere"
                ],
                "max_input_tokens": null,
                "max_output_tokens": null,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "mode": null,
                "tpm": null,
                "rpm": null,
                "supports_parallel_function_calling": false,
                "supports_vision": false,
                "supports_function_calling": false,
                "supported_openai_params": [
                    "stream",
                    "temperature",
                    "max_tokens",
                    "logit_bias",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                    "n",
                    "extra_headers"
                ]
                },
                {
                "model_group": "gpt-3.5-turbo",
                "providers": [
                    "openai"
                ],
                "max_input_tokens": 16385.0,
                "max_output_tokens": 4096.0,
                "input_cost_per_token": 1.5e-06,
                "output_cost_per_token": 2e-06,
                "mode": "chat",
                "tpm": null,
                "rpm": null,
                "supports_parallel_function_calling": false,
                "supports_vision": false,
                "supports_function_calling": true,
                "supported_openai_params": [
                    "frequency_penalty",
                    "logit_bias",
                    "logprobs",
                    "top_logprobs",
                    "max_tokens",
                    "max_completion_tokens",
                    "n",
                    "presence_penalty",
                    "seed",
                    "stop",
                    "stream",
                    "stream_options",
                    "temperature",
                    "top_p",
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "max_retries",
                    "extra_headers",
                    "parallel_tool_calls",
                    "response_format"
                ]
                },
                {
                "model_group": "llava-hf",
                "providers": [
                    "openai"
                ],
                "max_input_tokens": null,
                "max_output_tokens": null,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "mode": null,
                "tpm": null,
                "rpm": null,
                "supports_parallel_function_calling": false,
                "supports_vision": true,
                "supports_function_calling": false,
                "supported_openai_params": [
                    "frequency_penalty",
                    "logit_bias",
                    "logprobs",
                    "top_logprobs",
                    "max_tokens",
                    "max_completion_tokens",
                    "n",
                    "presence_penalty",
                    "seed",
                    "stop",
                    "stream",
                    "stream_options",
                    "temperature",
                    "top_p",
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "max_retries",
                    "extra_headers",
                    "parallel_tool_calls",
                    "response_format"
                ]
                }
            ]
            }
    ```
    """
    global llm_model_list, general_settings, user_config_file_path, proxy_config, llm_router

    # Return empty data array when no models are configured (graceful handling for fresh installs)
    if llm_model_list is None or llm_router is None or not llm_model_list:
        return {"data": []}

    from litellm.proxy.utils import get_available_models_for_user

    # Get available models for the user
    all_models_str: Final = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=llm_router,
        general_settings=general_settings,
        user_model=user_model,
        prisma_client=prisma_client,
        proxy_logging_obj=proxy_logging_obj,
        team_id=None,
        include_model_access_groups=False,
        only_model_access_groups=False,
        return_wildcard_routes=False,
        user_api_key_cache=user_api_key_cache,
    )
    model_groups: list[ModelGroupInfoProxy] = _get_model_group_info(
        llm_router=llm_router, all_models_str=all_models_str, model_group=model_group
    )

    # Append A2A agents to model groups
    from litellm.proxy.agent_endpoints.model_list_helpers import (
        append_agents_to_model_group,
    )

    model_groups = await append_agents_to_model_group(
        model_groups=model_groups,
        user_api_key_dict=user_api_key_dict,
    )

    return {"data": model_groups}


@router.get(
    "/model/settings",
    description="Returns provider name, description, and required parameters for each provider",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def model_settings():
    """
    Used by UI to generate 'model add' page
    {
        field_name=field_name,
        field_type=allowed_args[field_name]["type"], # string/int
        field_description=field_info.description or "", # human-friendly description
        field_value=general_settings.get(field_name, None), # example value
    }
    """

    returned_list: Final = []
    for provider in litellm.provider_list:
        returned_list.append(
            ProviderInfo(
                name=provider,
                fields=litellm.get_provider_fields(custom_llm_provider=provider),
            )
        )

    return returned_list


#### ALERTING MANAGEMENT ENDPOINTS ####


@router.get(
    "/alerting/settings",
    description="Return the configurable alerting param, description, and current value",
    tags=["alerting"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def alerting_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global proxy_logging_obj, prisma_client
    """
    Used by UI to generate 'alerting settings' page
    {
        field_name=field_name,
        field_type=allowed_args[field_name]["type"], # string/int
        field_description=field_info.description or "", # human-friendly description
        field_value=general_settings.get(field_name, None), # example value
    }
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    ## get general settings from db
    db_general_settings: Final = await _config_param_table(prisma_client).find_first(
        where={"param_name": "general_settings"}
    )

    if db_general_settings is not None and db_general_settings.param_value is not None:
        db_general_settings_dict: Final = dict(db_general_settings.param_value)
        alerting_args_dict: dict = cast(  # cast-ok: ConfigGeneralSettings validates alerting_args as a dict on write
            dict[str, JsonValue], db_general_settings_dict.get("alerting_args", {})
        )
        alerting_values: list | None = cast(  # cast-ok: ConfigGeneralSettings validates alerting as a list on write
            list[JsonValue] | None, db_general_settings_dict.get("alerting")
        )
    else:
        alerting_args_dict = {}
        alerting_values = None

    allowed_args: Final = {
        "slack_alerting": {"type": "Boolean"},
        "daily_report_frequency": {"type": "Integer"},
        "report_check_interval": {"type": "Integer"},
        "budget_alert_ttl": {"type": "Integer"},
        "outage_alert_ttl": {"type": "Integer"},
        "region_outage_alert_ttl": {"type": "Integer"},
        "minor_outage_alert_threshold": {"type": "Integer"},
        "major_outage_alert_threshold": {"type": "Integer"},
        "max_outage_alert_list_size": {"type": "Integer"},
    }

    _slack_alerting: Final[SlackAlerting] = proxy_logging_obj.slack_alerting_instance
    _slack_alerting_args_dict: Final = _slack_alerting.alerting_args.model_dump()

    return_val: Final = []

    is_slack_enabled = False

    if general_settings.get("alerting") and isinstance(general_settings["alerting"], list):
        if "slack" in general_settings["alerting"]:
            is_slack_enabled = True

    _response_obj = ConfigList(
        field_name="slack_alerting",
        field_type=allowed_args["slack_alerting"]["type"],
        field_description="Enable slack alerting for monitoring proxy in production: llm outages, budgets, spend tracking failures.",
        field_value=is_slack_enabled,
        stored_in_db=True if alerting_values is not None else False,
        field_default_value=None,
        premium_field=False,
    )
    return_val.append(_response_obj)

    for field_name, field_info in SlackAlertingArgs.model_fields.items():
        if field_name in allowed_args:
            _stored_in_db: bool | None = None
            if field_name in alerting_args_dict:
                _stored_in_db = True
            else:
                _stored_in_db = False

            _response_obj = ConfigList(
                field_name=field_name,
                field_type=allowed_args[field_name]["type"],
                field_description=field_info.description or "",
                field_value=_slack_alerting_args_dict.get(field_name, None),
                stored_in_db=_stored_in_db,
                field_default_value=field_info.default,
                premium_field=(True if field_name == "region_outage_alert_ttl" else False),
            )
            return_val.append(_response_obj)
    return return_val


#### EXPERIMENTAL QUEUING ####
@router.post(
    "/queue/chat/completions",
    tags=["experimental"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def async_queue_request(
    request: Request,
    fastapi_response: Response,
    model: str | None = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global general_settings, user_debug, proxy_logging_obj
    """
    v2 attempt at a background worker to handle queuing

    Just supports /chat/completion calls currently.

    Now using a FastAPI background task + /chat/completions compatible endpoint
    """
    data = {}
    try:
        data = await request.json()
        data.pop("_litellm_strip_stream_usage", None)

        # Include original request and headers in the data
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": _safe_get_request_headers(request).copy(),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        verbose_proxy_logger.debug("receiving data: %s", data)
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or data.get("model", None)  # default passed in http request
        )

        # users can pass in 'user' param to /chat/completions. Don't override it
        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            # if users are using user_api_key_auth, set `user` in `data`
            data["user"] = user_api_key_dict.user_id

        if not isinstance(data.get("metadata"), dict):
            # Covers both missing and JSON-string metadata (multipart /
            # extra_body); see above for the same guard upstream.
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["user_api_key_metadata"] = strip_callback_config(user_api_key_dict.metadata)
        _headers: Final = _safe_get_request_headers(request).copy()
        _headers.pop("authorization", None)  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["user_api_key_alias"] = getattr(user_api_key_dict, "key_alias", None)
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(user_api_key_dict, "team_id", None)
        data["metadata"]["user_api_key_object_permission_id"] = getattr(user_api_key_dict, "object_permission_id", None)
        data["metadata"]["user_api_key_team_object_permission_id"] = getattr(
            user_api_key_dict, "team_object_permission_id", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        global user_temperature, user_request_timeout, user_max_tokens, user_api_base
        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        if llm_router is None:
            raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value})

        response: Final = await llm_router.schedule_acompletion(**data)

        if "stream" in data and data["stream"] is True:  # use generate_responses to stream responses
            return StreamingResponse(
                async_data_generator(
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                    request_data=data,
                    request=request,
                ),
                media_type="text/event-stream",
            )

        fastapi_response.headers.update({"x-litellm-priority": str(data["priority"])})
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({e})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@app.get("/fallback/login", tags=["experimental"], include_in_schema=False)
async def fallback_login(request: Request):
    """
    Create Proxy API Keys using Google Workspace SSO. Requires setting PROXY_BASE_URL in .env
    PROXY_BASE_URL should be the your deployed proxy endpoint, e.g. PROXY_BASE_URL="https://litellm-production-7002.up.railway.app/"
    Example:
    """
    from litellm.proxy.proxy_server import ui_link

    # get url from request
    redirect_url = get_custom_url(str(request.base_url))
    if redirect_url.endswith("/"):
        redirect_url += "sso/callback"
    else:
        redirect_url += "/sso/callback"

    from fastapi.responses import HTMLResponse

    hide_default_credentials_hint: Final = (
        os.getenv("LITELLM_HIDE_DEFAULT_CREDENTIALS_HINT", "false").lower() == "true"
        or general_settings.get("hide_default_credentials_hint", False) is True
    )
    return HTMLResponse(
        content=build_ui_login_form(
            show_deprecation_banner=False,
            hide_default_credentials_hint=hide_default_credentials_hint,
        ),
        status_code=200,
    )


@router.post("/login", include_in_schema=False)  # hidden since this is a helper for UI sso login
async def login(request: Request):
    global premium_user, general_settings, master_key
    from litellm.proxy.auth.login_utils import authenticate_user, create_ui_token_object, encode_ui_session_jwt
    from litellm.proxy.utils import get_custom_url

    form: Final = await request.form()
    username: Final = str(form.get("username"))
    password: Final = str(form.get("password"))

    # Authenticate user and get login result
    login_result: Final = await authenticate_user(
        username=username,
        password=password,
        master_key=master_key,
        prisma_client=prisma_client,
    )

    # Create UI token object
    returned_ui_token_object: Final = create_ui_token_object(
        login_result=login_result,
        general_settings=general_settings,
        premium_user=premium_user,
    )

    # Generate JWT token
    jwt_token: Final = encode_ui_session_jwt(returned_ui_token_object, cast(str, master_key))

    # Build redirect URL
    litellm_dashboard_ui = get_custom_url(str(request.base_url))
    litellm_dashboard_ui = litellm_dashboard_ui.rstrip("/")
    litellm_dashboard_ui += "/ui?login=success"

    # Honor a same-origin return_to preserved by the sign-in page (e.g. the aggregate DCR connect flow's
    # authorize round-trip), mirroring the SSO callback; otherwise land on the dashboard. Gated by
    # _is_same_origin_return_path (strictly relative path) so it can never be an open redirect, and the
    # one-shot cookie is cleared after use.
    from litellm.proxy.management_endpoints.ui_sso import _sso_return_to_redirect

    # Resume through the SAME resumer the SSO callback uses, rather than a second, narrower arm.
    # _persist_return_to_cookie stores both shapes it accepts (a relative same-origin path AND a
    # control_plane_url-matching absolute URL); honoring only the relative one here silently dropped
    # the control-plane case, landing the user on the dashboard. One function decides how a stored
    # return_to is honored for EVERY sign-in branch, so the write and read sets cannot diverge: it
    # sets the token cookie on the same-origin arm and hands off via a one-time login code on the
    # cross-origin arm, and clears the one-shot cookie in both.
    cp_return_to: Final = request.cookies.get("litellm_cp_return_to")
    if cp_return_to:
        try:
            resumed = await _sso_return_to_redirect(
                return_to=cp_return_to,
                jwt_token=jwt_token,
                redis_usage_cache=redis_usage_cache,
                user_api_key_cache=user_api_key_cache,
            )
        except Exception:  # noqa: BLE001  # resuming must NEVER block a completed sign-in
            # The symmetric half of _persist_return_to_cookie's "never raises" contract. The resumer
            # rejects a return_to that no longer matches control_plane_url (a config change between
            # the cookie's write and this read), and the user has ALREADY authenticated here —
            # failing their login over a stale one-shot cookie is the worst possible outcome. Land
            # on the dashboard instead; the cookie is cleared below either way.
            verbose_proxy_logger.info("Ignoring stale litellm_cp_return_to cookie; landing on dashboard")
            resumed = None
        if resumed is not None:
            return resumed

    # Create redirect response with cookie
    redirect_response: Final = RedirectResponse(url=litellm_dashboard_ui, status_code=303)
    redirect_response.set_cookie(key="token", value=jwt_token)
    if cp_return_to:
        redirect_response.delete_cookie(key="litellm_cp_return_to")
    return redirect_response


@router.post("/v2/login", include_in_schema=False)  # hidden helper for UI logins via API
async def login_v2(request: Request):
    global premium_user, general_settings, master_key
    from litellm.proxy.auth.login_utils import authenticate_user, create_ui_token_object, encode_ui_session_jwt
    from litellm.proxy.utils import get_custom_url

    try:
        body: Final = await request.json()
        username: Final = str(body.get("username"))
        password: Final = str(body.get("password"))

        login_result: Final = await authenticate_user(
            username=username,
            password=password,
            master_key=master_key,
            prisma_client=prisma_client,
        )

        returned_ui_token_object: Final = create_ui_token_object(
            login_result=login_result,
            general_settings=general_settings,
            premium_user=premium_user,
        )

        jwt_token: Final = encode_ui_session_jwt(returned_ui_token_object, cast(str, master_key))

        litellm_dashboard_ui = get_custom_url(str(request.base_url))
        litellm_dashboard_ui = litellm_dashboard_ui.rstrip("/")
        litellm_dashboard_ui += "/ui?login=success"

        # Token is included in the response body so the UI can set a JS-accessible
        # cookie even when a reverse proxy (e.g. nginx-ingress) adds HttpOnly to the
        # server-set cookie, which would otherwise cause an infinite login redirect.
        json_response: Final = JSONResponse(
            content={"redirect_url": litellm_dashboard_ui, "token": jwt_token},
            status_code=status.HTTP_200_OK,
        )
        json_response.set_cookie(key="token", value=jwt_token)
        return json_response
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.login_v2(): Exception occurred - %s", e)
        if isinstance(e, ProxyException):
            raise e
        elif isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=error_msg,
                type=ProxyErrorTypes.auth_error,
                param="None",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@router.post(
    "/v3/login", include_in_schema=False
)  # control-plane login — always returns token in body for cross-origin use
async def login_v3(request: Request):
    global premium_user, general_settings, master_key
    from litellm.proxy.auth.login_utils import authenticate_user, create_ui_token_object, encode_ui_session_jwt
    from litellm.proxy.utils import get_custom_url

    try:
        if not general_settings.get("control_plane_url"):
            raise ProxyException(
                message="/v3/login is only available on workers with control_plane_url configured",
                type=ProxyErrorTypes.not_found_error,
                param="control_plane_url",
                code=status.HTTP_404_NOT_FOUND,
            )

        body: Final = await request.json()
        username: Final = str(body.get("username"))
        password: Final = str(body.get("password"))

        login_result: Final = await authenticate_user(
            username=username,
            password=password,
            master_key=master_key,
            prisma_client=prisma_client,
        )

        returned_ui_token_object: Final = create_ui_token_object(
            login_result=login_result,
            general_settings=general_settings,
            premium_user=premium_user,
        )

        jwt_token: Final = encode_ui_session_jwt(returned_ui_token_object, cast(str, master_key))

        litellm_dashboard_ui = get_custom_url(str(request.base_url))
        litellm_dashboard_ui = litellm_dashboard_ui.rstrip("/")
        litellm_dashboard_ui += "/ui?login=success"

        # Store JWT behind a single-use opaque code (60s TTL)
        code: Final = secrets.token_urlsafe(32)
        cache_key: Final = f"login_code:{code}"
        cache_value: Final = {"token": jwt_token, "redirect_url": litellm_dashboard_ui}
        if redis_usage_cache is not None:
            await redis_usage_cache.async_set_cache(key=cache_key, value=cache_value, ttl=60)
        else:
            await user_api_key_cache.async_set_cache(key=cache_key, value=cache_value, ttl=60)

        return JSONResponse(
            content={"code": code, "expires_in": 60},
            status_code=status.HTTP_200_OK,
        )
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.login_v3(): Exception occurred - %s", e)
        if isinstance(e, ProxyException):
            raise e
        elif isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=error_msg,
                type=ProxyErrorTypes.auth_error,
                param="None",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@router.post("/v3/login/exchange", include_in_schema=False)  # exchange single-use opaque code for JWT
async def login_v3_exchange(request: Request):
    try:
        if not general_settings.get("control_plane_url"):
            raise ProxyException(
                message="/v3/login/exchange is only available on workers with control_plane_url configured",
                type=ProxyErrorTypes.not_found_error,
                param="control_plane_url",
                code=status.HTTP_404_NOT_FOUND,
            )

        body: Final = await request.json()
        code: Final = body.get("code")
        if not code:
            raise ProxyException(
                message="Missing 'code' parameter",
                type=ProxyErrorTypes.auth_error,
                param="code",
                code=status.HTTP_400_BAD_REQUEST,
            )

        cache_key: Final = f"login_code:{code}"
        if redis_usage_cache is not None:
            cached_data = await redis_usage_cache.async_get_cache(key=cache_key)
        else:
            cached_data = await user_api_key_cache.async_get_cache(key=cache_key)

        if not cached_data or not isinstance(cached_data, dict):
            raise ProxyException(
                message="Invalid or expired login code",
                type=ProxyErrorTypes.auth_error,
                param="code",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        # Single-use: delete immediately
        if redis_usage_cache is not None:
            await redis_usage_cache.async_delete_cache(key=cache_key)
        else:
            await user_api_key_cache.async_delete_cache(key=cache_key)

        json_response: Final = JSONResponse(
            content={
                "token": cached_data["token"],
                "redirect_url": cached_data["redirect_url"],
            },
            status_code=status.HTTP_200_OK,
        )
        json_response.set_cookie(key="token", value=cached_data["token"])
        return json_response
    except ProxyException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.login_v3_exchange(): Exception occurred - %s", e)
        raise ProxyException(
            message=str(e),
            type=ProxyErrorTypes.auth_error,
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.get("/onboarding/get_token", include_in_schema=False)
async def onboarding(invite_link: str, request: Request):
    """
    - Get the invite link
    - Validate it's still 'valid'
    - Return a short-lived onboarding token
    - Get user from db
    - Pass in user_email if set
    """
    global prisma_client, master_key, general_settings
    from litellm.types.proxy.ui_sso import ReturnedUITokenObject

    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    ### VALIDATE INVITE LINK ###
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    invite_obj: Final[_InvitationLinkRow | None] = await InvitationLinkRepository(prisma_client).table.find_unique(
        where={"id": invite_link}
    )
    if invite_obj is None:
        raise HTTPException(status_code=401, detail={"error": "Invitation link does not exist in db."})
    #### CHECK IF EXPIRED
    # Extract the date part from both datetime objects
    utc_now_date: Final = litellm.utils.get_utc_datetime().date()
    expires_at_date: Final = invite_obj.expires_at.date()
    if expires_at_date < utc_now_date:
        raise HTTPException(status_code=401, detail={"error": "Invitation link has expired."})

    #### CHECK IF ALREADY USED
    if invite_obj.is_accepted is True or invite_obj.accepted_at is not None:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invitation link has already been used."},
        )

    ### GET USER OBJECT ###
    user_obj: Final[_UserTableRow | None] = await UserRepository(prisma_client).table.find_unique(
        where={"user_id": invite_obj.user_id}
    )

    if user_obj is None:
        raise HTTPException(status_code=401, detail={"error": "User does not exist in db."})

    litellm_dashboard_ui = get_custom_url(str(request.base_url))
    litellm_dashboard_ui = litellm_dashboard_ui.rstrip("/")
    litellm_dashboard_ui += "/ui/onboarding"
    import jwt

    user_email: Final = user_obj.user_email
    onboarding_token: Final = jwt.encode(
        {
            "token_type": "litellm_onboarding",
            "invitation_link": invite_link,
            "user_id": user_obj.user_id,
            "exp": litellm.utils.get_utc_datetime() + timedelta(minutes=15),
        },
        master_key,
        algorithm="HS256",
    )
    disabled_non_admin_personal_key_creation: Final = get_disabled_non_admin_personal_key_creation()

    returned_ui_token_object: Final = ReturnedUITokenObject(
        user_id=user_obj.user_id,
        key=onboarding_token,
        user_email=user_obj.user_email,
        user_role=user_obj.user_role,  # pyright: ignore[reportArgumentType]  # nullable DB column, no unset contract
        login_method="username_password",
        premium_user=premium_user,
        auth_header_name=general_settings.get("litellm_key_header_name", "Authorization"),
        disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
        server_root_path=get_server_root_path(),
    )
    jwt_token: Final = jwt.encode(
        cast(dict, returned_ui_token_object),
        master_key,
        algorithm="HS256",
    )

    litellm_dashboard_ui += f"?token={jwt_token}&user_email={user_email}"
    return {
        "login_url": litellm_dashboard_ui,
        "token": jwt_token,
        "user_email": user_email,
    }


def _get_onboarding_claims_from_request(request: Request) -> dict:
    global master_key, general_settings

    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    auth_header_name: Final = general_settings.get("litellm_key_header_name", "Authorization")
    onboarding_auth_header: Final = request.headers.get(auth_header_name)
    if onboarding_auth_header is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "Missing onboarding session for invitation link."},
        )
    onboarding_token = onboarding_auth_header
    if onboarding_token.lower().startswith("bearer "):
        onboarding_token = onboarding_token.split(" ", 1)[1]

    import jwt

    try:
        return jwt.decode(
            onboarding_token,
            master_key,
            algorithms=["HS256"],
        )
    except Exception:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid onboarding session for invitation link."},
        )


async def _rollback_onboarding_invite_claim(
    invitation_link: str,
    user_id: str,
) -> None:
    global prisma_client

    if prisma_client is None:
        return

    try:
        await InvitationLinkRepository(prisma_client).table.update_many(
            where={"id": invitation_link, "is_accepted": True},
            data={
                "accepted_at": None,
                "is_accepted": False,
                "updated_at": litellm.utils.get_utc_datetime(),
                "updated_by": user_id,
            },
        )
    except Exception:
        verbose_proxy_logger.exception("Failed to roll back onboarding invitation after session key mint failed.")


async def _generate_onboarding_ui_session_token(user_obj: _UserTableRow) -> str:
    global master_key, general_settings

    response: Final = await generate_key_helper_fn(
        request_type="key",
        **{
            "user_role": user_obj.user_role,
            "duration": LITELLM_UI_SESSION_DURATION,
            "key_max_budget": litellm.max_ui_session_budget,
            "models": [],
            "aliases": {},
            "config": {},
            "spend": 0,
            "user_id": user_obj.user_id,
            "team_id": UI_TEAM_ID,
        },
    )
    key: Final = response["token"]

    import jwt

    from litellm.types.proxy.ui_sso import ReturnedUITokenObject

    disabled_non_admin_personal_key_creation: Final = get_disabled_non_admin_personal_key_creation()
    returned_ui_token_object: Final = ReturnedUITokenObject(
        user_id=user_obj.user_id,
        key=key,
        user_email=user_obj.user_email,
        user_role=user_obj.user_role,  # pyright: ignore[reportArgumentType]  # nullable DB column, no unset contract
        login_method="username_password",
        premium_user=premium_user,
        auth_header_name=general_settings.get("litellm_key_header_name", "Authorization"),
        disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
        server_root_path=get_server_root_path(),
    )
    assert master_key is not None
    return jwt.encode(
        cast(dict, returned_ui_token_object),
        master_key,
        algorithm="HS256",
    )


@app.post("/onboarding/claim_token", include_in_schema=False)
async def claim_onboarding_link(data: InvitationClaim, request: Request):
    """
    Special route. Allows UI link share user to update their password.

    - Get the invite link
    - Validate it's still 'valid'
    - Check if user within initial session (prevents abuse)
    - Get user from db
    - Update user password

    This route can only update user password.
    """
    global prisma_client, master_key, general_settings
    ### VALIDATE INVITE LINK ###
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    invite_obj: Final[_InvitationLinkRow | None] = await InvitationLinkRepository(prisma_client).table.find_unique(
        where={"id": data.invitation_link}
    )
    if invite_obj is None:
        raise HTTPException(status_code=401, detail={"error": "Invitation link does not exist in db."})
    #### CHECK IF EXPIRED
    # Extract the date part from both datetime objects
    utc_now_date: Final = litellm.utils.get_utc_datetime().date()
    expires_at_date: Final = invite_obj.expires_at.date()
    if expires_at_date < utc_now_date:
        raise HTTPException(status_code=401, detail={"error": "Invitation link has expired."})

    #### CHECK IF ALREADY USED
    if invite_obj.is_accepted is True or invite_obj.accepted_at is not None:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invitation link has already been used."},
        )

    #### CHECK IF VALID USER ID
    if invite_obj.user_id != data.user_id:
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"Invalid invitation link. The user id submitted does not match the user id this link is attached to. Got={data.user_id}, Expected={invite_obj.user_id}"
            },
        )

    onboarding_claims: Final = _get_onboarding_claims_from_request(request=request)
    if (
        onboarding_claims.get("token_type") != "litellm_onboarding"
        or onboarding_claims.get("invitation_link") != data.invitation_link
        or onboarding_claims.get("user_id") != data.user_id
    ):
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid onboarding session for invitation link."},
        )

    hashed_pw: Final = hash_password(data.password)
    current_time = litellm.utils.get_utc_datetime()
    async with prisma_client.db.tx() as tx:
        updated_count: Final = await tx.litellm_invitationlink.update_many(
            where={"id": data.invitation_link, "is_accepted": False},
            data={
                "is_accepted": True,
                "updated_at": current_time,
                "updated_by": invite_obj.user_id,
            },
        )
        if updated_count == 0:
            raise HTTPException(
                status_code=401,
                detail={"error": "Invitation link has already been used."},
            )

        ### UPDATE USER OBJECT ###
        user_obj: Final[_UserTableRow | None] = await tx.litellm_usertable.update(
            where={"user_id": invite_obj.user_id}, data={"password": hashed_pw}
        )

        if user_obj is None:
            raise HTTPException(status_code=401, detail={"error": "User does not exist in db."})

        #### MARK LINK AS USED
        current_time = litellm.utils.get_utc_datetime()
        await tx.litellm_invitationlink.update(
            where={"id": data.invitation_link},
            data={
                "accepted_at": current_time,
                "updated_at": current_time,
                "updated_by": invite_obj.user_id,
            },
        )

    if user_obj and hasattr(user_obj, "__dict__"):
        user_obj.__dict__.pop("password", None)

    try:
        jwt_token: Final = await _generate_onboarding_ui_session_token(user_obj=user_obj)
    except Exception as e:
        await _rollback_onboarding_invite_claim(
            invitation_link=data.invitation_link,
            user_id=data.user_id,
        )
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to create onboarding session. Please retry the invitation link."},
        ) from e

    litellm_dashboard_ui = get_custom_url(str(request.base_url))
    litellm_dashboard_ui = litellm_dashboard_ui.rstrip("/")
    litellm_dashboard_ui += "/ui?login=success"
    return {
        "login_url": litellm_dashboard_ui,
        "token": jwt_token,
        "user_email": user_obj.user_email,
        "user": user_obj,
    }


@app.get("/get_logo_url", include_in_schema=False)
def get_logo_url():
    """Get the current logo URL from environment.

    Only HTTP(S) URLs are returned — those are intended to be loaded
    directly by the browser from a public/internal CDN. Local file
    paths set via ``UI_LOGO_PATH`` are NOT returned: they are admin-
    only filesystem details, the dashboard falls back to ``/get_image``
    which serves the file only when it is a supported image. Without
    this filter, the unauthenticated endpoint would disclose internal
    hostnames or filesystem paths to any caller.
    """
    logo_path: Final = os.getenv("UI_LOGO_PATH", "")
    if logo_path.startswith(("http://", "https://")):
        return {"logo_url": logo_path}
    return {"logo_url": ""}


def _serve_custom_ui_logo(candidate: str) -> Response | None:
    """Serve one admin-configured logo, or None when it is unusable so the caller falls back."""
    from litellm.proxy.common_utils.static_asset_utils import (
        resolve_validated_local_image_path,
    )

    # Remote logo URLs are loaded by the browser. The proxy should not fetch
    # arbitrary admin-configured URLs server-side.
    if candidate.startswith(("http://", "https://")):
        return RedirectResponse(url=candidate)

    safe_logo: Final = resolve_validated_local_image_path(candidate)
    if safe_logo is None:
        verbose_proxy_logger.warning(
            "Custom UI logo %r is not a supported image file or does not exist, falling back",
            candidate,
        )
        return None

    safe_logo_path, media_type = safe_logo
    return FileResponse(safe_logo_path, media_type=media_type)


@app.get("/get_image", include_in_schema=False)
async def get_image(theme: Literal["light", "dark"] | None = None):
    """Get logo to show on admin UI"""

    # get current_dir
    current_dir: Final = os.path.dirname(os.path.abspath(__file__))
    bundled_light_logo: Final = os.path.join(current_dir, "logo.jpg")
    bundled_dark_logo: Final = os.path.join(current_dir, "logo_dark.png")
    default_site_logo: Final = (
        bundled_dark_logo if theme == "dark" and os.path.isfile(bundled_dark_logo) else bundled_light_logo
    )
    default_logo_filename: Final = os.path.basename(default_site_logo)

    is_non_root: Final = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"

    # Determine assets directory
    # Priority: LITELLM_ASSETS_PATH env var > default based on is_non_root
    default_assets_dir: Final = "/var/lib/litellm/assets" if is_non_root else current_dir
    assets_dir = os.getenv("LITELLM_ASSETS_PATH", default_assets_dir)

    # Try to create assets_dir if it doesn't exist (simple try/except approach)
    if not os.path.exists(assets_dir):
        try:
            os.makedirs(assets_dir, exist_ok=True)
            verbose_proxy_logger.debug("Created assets directory at %s", assets_dir)
        except (PermissionError, OSError) as e:
            verbose_proxy_logger.warning(
                "Cannot create assets directory at %s: %s. Logo caching may not work. Using current directory for assets.",
                assets_dir,
                e,
            )
            assets_dir = current_dir

    # Determine default logo path
    default_logo = os.path.join(assets_dir, default_logo_filename) if assets_dir != current_dir else default_site_logo
    if assets_dir != current_dir and not os.path.exists(default_logo):
        default_logo = default_site_logo

    custom_logo_candidates: Final = tuple(
        candidate.strip()
        for candidate in (
            os.getenv("UI_LOGO_PATH_DARK", "") if theme == "dark" else "",
            os.getenv("UI_LOGO_PATH", ""),
        )
        if candidate.strip()
    )
    verbose_proxy_logger.debug("Custom logo candidates, in fallback order: %s", custom_logo_candidates)

    custom_logo_response: Final = next(
        (
            response
            for response in (_serve_custom_ui_logo(candidate) for candidate in custom_logo_candidates)
            if response is not None
        ),
        None,
    )
    if custom_logo_response is not None:
        return custom_logo_response

    from litellm.proxy.common_utils.static_asset_utils import (
        resolve_validated_local_image_path,
    )

    # Default logo (resolved from the bundled asset, not user-controlled).
    safe_logo: Final = resolve_validated_local_image_path(default_logo)
    if safe_logo is not None:
        safe_logo_path, media_type = safe_logo
        return FileResponse(safe_logo_path, media_type=media_type)
    return FileResponse(bundled_light_logo, media_type="image/jpeg")


@app.get("/get_favicon", include_in_schema=False)
async def get_favicon():
    """Get custom favicon for the admin UI."""
    from litellm.proxy.common_utils.static_asset_utils import (
        resolve_validated_local_image_path,
    )

    current_dir: Final = os.path.dirname(os.path.abspath(__file__))
    default_favicon: Final = os.path.join(current_dir, "_experimental", "out", "favicon.ico")

    favicon_url: Final = os.getenv("LITELLM_FAVICON_URL", "")

    if not favicon_url:
        if os.path.exists(default_favicon):
            return FileResponse(default_favicon, media_type="image/x-icon")
        raise HTTPException(status_code=404, detail="Default favicon not found")

    if favicon_url.startswith(("http://", "https://")):
        return RedirectResponse(url=favicon_url)
    else:
        safe_favicon: Final = resolve_validated_local_image_path(favicon_url)
        if safe_favicon is not None:
            safe_favicon_path, media_type = safe_favicon
            return FileResponse(safe_favicon_path, media_type=media_type)
        verbose_proxy_logger.warning(
            "LITELLM_FAVICON_URL %r is not a supported image file or does not exist, falling back to default favicon",
            favicon_url,
        )
        if os.path.exists(default_favicon):
            return FileResponse(default_favicon, media_type="image/x-icon")
        raise HTTPException(status_code=404, detail="Favicon not found")


#### INVITATION MANAGEMENT ####


@router.post(
    "/invitation/new",
    tags=["Invite Links"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvitationModel,
    include_in_schema=False,
)
async def new_invitation(data: InvitationNew, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    """
    Allow admin to create invite links, to onboard new users to Admin UI.

    ```
    curl -X POST 'http://localhost:4000/invitation/new' \
        -H 'Content-Type: application/json' \
        -d '{
            "user_id": "1234" // 👈 id of user in 'LiteLLM_UserTable'
        }'
    ```
    """
    try:
        from litellm.proxy.management_helpers.user_invitation import (
            create_invitation_for_user,
        )

        global prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=400,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Allow proxy admins and org/team admins (admin status from DB via get_user_object)
        has_access = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN or await _user_has_admin_privileges(
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if not has_access:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"
                },
            )

        # Org/team admins can only invite users within their org/team
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            can_invite: Final = await admin_can_invite_user(
                target_user_id=data.user_id,
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            if not can_invite:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "You can only create invitations for users in your organization or team."},
                )

        response: Final[object] = await create_invitation_for_user(
            data=data,
            user_api_key_dict=user_api_key_dict,
        )
        return response
    except Exception as e:
        raise handle_exception_on_proxy(e)


@router.get(
    "/invitation/info",
    tags=["Invite Links"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvitationModel,
    include_in_schema=False,
)
async def invitation_info(invitation_id: str, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):
    """
    Allow admin to create invite links, to onboard new users to Admin UI.

    ```
    curl -X POST 'http://localhost:4000/invitation/new' \
        -H 'Content-Type: application/json' \
        -d '{
            "user_id": "1234" // 👈 id of user in 'LiteLLM_UserTable'
        }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    response: Final[object] = await InvitationLinkRepository(prisma_client).table.find_unique(
        where={"id": invitation_id}
    )

    if response is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invitation id does not exist in the database."},
        )
    return response


@router.post(
    "/invitation/update",
    tags=["Invite Links"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvitationModel,
    include_in_schema=False,
)
async def invitation_update(
    data: InvitationUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update when invitation is accepted

    ```
    curl -X POST 'http://localhost:4000/invitation/update' \
        -H 'Content-Type: application/json' \
        -d '{
            "invitation_id": "1234" // 👈 id of invitation in 'LiteLLM_InvitationTable'
            "is_accepted": True // when invitation is accepted
        }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_id is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Unable to identify user id. Received={user_api_key_dict.user_id}"},
        )

    current_time: Final = litellm.utils.get_utc_datetime()
    response: Final[object] = await InvitationLinkRepository(prisma_client).table.update(
        where={"id": data.invitation_id},
        data={
            "id": data.invitation_id,
            "is_accepted": data.is_accepted,
            "accepted_at": current_time,
            "updated_at": current_time,
            "updated_by": user_api_key_dict.user_id,
        },
    )

    if response is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invitation id does not exist in the database."},
        )
    return response


@router.post(
    "/invitation/delete",
    tags=["Invite Links"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvitationModel,
    include_in_schema=False,
)
async def invitation_delete(
    data: InvitationDelete,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete invitation link

    ```
    curl -X POST 'http://localhost:4000/invitation/delete' \
        -H 'Content-Type: application/json' \
        -d '{
            "invitation_id": "1234" // 👈 id of invitation in 'LiteLLM_InvitationTable'
        }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Proxy admins can delete any invitation; org admins only their own
    is_proxy_admin: Final = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
    is_other_admin: Final = await _user_has_admin_privileges(
        user_api_key_dict=user_api_key_dict,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    if not is_proxy_admin and not is_other_admin:
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    # Org admins can only delete invitations they created
    if is_other_admin and not is_proxy_admin:
        invitation: Final[_InvitationLinkRow | None] = await InvitationLinkRepository(prisma_client).table.find_unique(
            where={"id": data.invitation_id}
        )
        if invitation is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "Invitation id does not exist in the database."},
            )
        if invitation.created_by != user_api_key_dict.user_id:
            raise HTTPException(
                status_code=403,
                detail={"error": "Organization admins can only delete invitations they created."},
            )

    response: Final[object] = await InvitationLinkRepository(prisma_client).table.delete(
        where={"id": data.invitation_id}
    )

    if response is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invitation id does not exist in the database."},
        )
    return response


#### CONFIG MANAGEMENT ####
@router.post(
    "/config/update",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def update_config(
    config_info: ConfigYAML,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    For Admin UI - allows admin to update config via UI.

    Writes only the sections present in the request body to LiteLLM_Config rows
    (one row per top-level section). Sections the caller did not send are left
    untouched — this endpoint never persists pre-existing YAML values to DB as
    a side effect of an unrelated update.
    """
    global llm_router, llm_model_list, general_settings, proxy_config, proxy_logging_obj, master_key, prisma_client
    try:
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(status_code=403, detail="Only proxy admins can update config")

        if prisma_client is None:
            raise Exception("No DB Connected")

        async def _read_section(param_name: str) -> dict:
            row: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
                where={"param_name": param_name}
            )
            if row is None or row.param_value is None:
                return {}
            return dict(row.param_value)

        async def _upsert_section(param_name: str, value: dict) -> None:
            serialized: Final = json.dumps(value)
            await ConfigRepository(prisma_client).table.upsert(
                where={"param_name": param_name},
                data={
                    "create": {"param_name": param_name, "param_value": serialized},
                    "update": {"param_value": serialized},
                },
            )
            # invalidate the DualCache entry so the next reader (this process
            # or any other proxy in the cluster) goes to DB.
            await invalidate_config_param(param_name)

        # general_settings: merge per-key, with the alert_to_webhook_url side
        # effect of auto-enabling slack alerting.
        if config_info.general_settings is not None:
            existing = await _read_section("general_settings")
            before_general_settings: Final = copy.deepcopy(existing)
            updates: Mapping[str, JsonValue] = config_info.general_settings.dict(exclude_none=True)
            for k, v in updates.items():
                if k == "alert_to_webhook_url":
                    if "alerting" not in existing:
                        existing["alerting"] = ["slack"]
                    elif isinstance(existing["alerting"], list) and "slack" not in existing["alerting"]:
                        existing["alerting"].append("slack")
                existing[k] = v
            await _upsert_section("general_settings", existing)
            asyncio.create_task(
                create_config_audit_log(
                    "general_settings", "updated", before_general_settings, existing, user_api_key_dict
                )
            )

        # environment_variables: idempotently encrypt the request values
        # (plaintext on first write, OR ciphertext the UI read back via
        # /get/config/callbacks and re-submitted on save), then merge into
        # existing. Only the sent keys are re-written; untouched keys keep
        # their stored ciphertext byte-for-byte.
        if config_info.environment_variables is not None:
            existing = await _read_section("environment_variables")
            before_environment_variables: Final = copy.deepcopy(existing)
            existing.update(
                proxy_config._encrypt_env_variables_for_db(environment_variables=config_info.environment_variables)
            )
            await _upsert_section("environment_variables", existing)
            asyncio.create_task(
                create_config_audit_log(
                    "environment_variables", "updated", before_environment_variables, existing, user_api_key_dict
                )
            )

        # litellm_settings: merge existing + request, request wins (matching
        # router_settings semantics — the caller's value for any given key is
        # what gets persisted). success_callback is special-cased: it is
        # always normalized + deduped, and unioned with any existing list,
        # because callbacks are additive (callers send the new entry, not
        # the full set). Normalizing on every write — not only when an
        # existing entry is present — keeps the DB free of mixed-case
        # entries that delete_callback (lowercase lookup) cannot find.
        if config_info.litellm_settings is not None:
            existing = await _read_section("litellm_settings")
            before_litellm_settings: Final = copy.deepcopy(existing)
            updated_litellm_settings: Final = dict(config_info.litellm_settings)

            incoming_cb = updated_litellm_settings.get("success_callback")
            if isinstance(incoming_cb, list):
                updated_litellm_settings["success_callback"] = normalize_callback_names(incoming_cb)

            merged: Final = {**existing, **updated_litellm_settings}

            incoming_cb = updated_litellm_settings.get("success_callback")
            existing_cb: Final = existing.get("success_callback")
            if isinstance(incoming_cb, list):
                if isinstance(existing_cb, list):
                    # Normalize the existing list too — a row written by a
                    # different code path may still hold mixed-case names,
                    # which would otherwise dedup-miss against the lowercase
                    # incoming entries.
                    merged["success_callback"] = list(set(normalize_callback_names(existing_cb) + incoming_cb))
                else:
                    merged["success_callback"] = list(set(incoming_cb))

            await _upsert_section("litellm_settings", merged)
            asyncio.create_task(
                create_config_audit_log(
                    "litellm_settings", "updated", before_litellm_settings, merged, user_api_key_dict
                )
            )

        # router_settings: merge existing + request, request wins.
        if config_info.router_settings is not None:
            existing = await _read_section("router_settings")
            before_router_settings: Final = copy.deepcopy(existing)
            updates = config_info.router_settings.dict(exclude_none=True)
            new_router_settings: Final = {**existing, **updates}
            await _upsert_section("router_settings", new_router_settings)
            asyncio.create_task(
                create_config_audit_log(
                    "router_settings", "updated", before_router_settings, new_router_settings, user_api_key_dict
                )
            )

        await proxy_config.add_deployment(prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj)

        return {"message": "Config updated successfully"}
    except Exception as e:
        verbose_proxy_logger.error("litellm.proxy.proxy_server.update_config(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({e})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


### CONFIG GENERAL SETTINGS
"""
- Update config settings
- Get config settings

Keep it more precise, to prevent overwrite other values unintentially
"""

_PLUGIN_KEY_REDACTED: Final = "***"

_GENERAL_SETTINGS_CONFIG_LIST_FIELD_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max_parallel_requests": "Integer",
        "global_max_parallel_requests": "Integer",
        "max_request_size_mb": "Integer",
        "max_batch_file_size_mb": "Integer",
        "max_response_size_mb": "Integer",
        "proxy_config_reload_interval_seconds": "Integer",
        "pass_through_endpoints": "PydanticModel",
        "store_model_in_db": "Boolean",
        "store_prompts_in_spend_logs": "Boolean",
        "maximum_spend_logs_retention_period": "String",
        "maximum_health_check_retention_period": "String",
        "maximum_spend_logs_cleanup_batch_size": "Integer",
        "maximum_spend_logs_cleanup_max_batches": "Integer",
        "maximum_spend_logs_cleanup_run_budget": "String",
        "maximum_spend_logs_cleanup_batch_timeout": "String",
        "mcp_internal_ip_ranges": "List",
        "mcp_trusted_proxy_ranges": "List",
        "mcp_xff_num_trusted_hops": "Integer",
        "always_include_stream_usage": "Boolean",
        "forward_client_headers_to_llm_api": "Boolean",
        "mcp_required_fields": "List",
        "cancel_on_disconnect": "Boolean",
        "disable_auto_add_proxy_admin_to_teams": "Boolean",
        "apply_user_budget_to_team_keys": "Boolean",
    }
)


def _preserve_redacted_plugin_keys(incoming: object, existing: object) -> object:
    """Restore real plugin_key values the client never sees.

    /config/field/info redacts every plugin_key to ``"***"``, so an admin
    editing a plugin posts that placeholder (or a blank, when the UI clears the
    field) straight back. Treat a blank or redacted plugin_key as "keep the
    stored credential" by sourcing it from the existing config; only a real,
    non-redacted value replaces it, and a blank with no stored key drops the
    field entirely instead of persisting the placeholder.
    """
    if not isinstance(incoming, list):
        return incoming

    stored_keys: Final = {
        p["name"]: p["plugin_key"]
        for p in (existing if isinstance(existing, list) else [])
        if isinstance(p, dict) and p.get("name") and p.get("plugin_key")
    }

    def resolve(plugin: object) -> object:
        if not isinstance(plugin, dict):
            return plugin
        key: Final = plugin.get("plugin_key")
        if key not in (None, "", _PLUGIN_KEY_REDACTED):
            return plugin
        name: Final = plugin.get("name")
        if name in stored_keys:
            return {**plugin, "plugin_key": stored_keys[name]}
        return {k: v for k, v in plugin.items() if k != "plugin_key"}

    return [resolve(p) for p in incoming]


@router.post(
    "/config/field/update",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def update_config_general_settings(
    data: ConfigFieldUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update a specific field in litellm general settings
    """
    global prisma_client
    ## VALIDATION ##
    """
    - Check if prisma_client is None
    - Check if user allowed to call this endpoint (admin-only)
    - Check if param in general settings
    - Check if config value is valid type
    """

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    if data.field_name in _GENERAL_SETTINGS_UI_LITELLM_FIELDS:
        return await _persist_general_settings_ui_litellm_field(data.field_name, data.field_value, user_api_key_dict)

    if data.field_name not in ConfigGeneralSettings.model_fields:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid field={data.field_name} passed in."},
        )

    try:
        ConfigGeneralSettings.model_validate({data.field_name: data.field_value})
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid type of field value={type(data.field_value)} passed in."},
        )

    ## get general settings from db
    db_general_settings: Final = await _config_param_table(prisma_client).find_first(
        where={"param_name": "general_settings"}
    )
    ### update value

    if db_general_settings is None or db_general_settings.param_value is None:
        general_settings = {}
    else:
        general_settings = dict(db_general_settings.param_value)

    before_general_settings: Final = copy.deepcopy(general_settings)

    ## update db

    field_value = data.field_value
    if data.field_name == "plugins":
        field_value = _preserve_redacted_plugin_keys(field_value, general_settings.get("plugins"))

    general_settings[data.field_name] = cast(JsonValue, field_value)  # cast-ok: ConfigGeneralSettings validated it

    response: Final = await ConfigRepository(prisma_client).table.upsert(
        where={"param_name": "general_settings"},
        data={
            "create": {
                "param_name": "general_settings",
                "param_value": json.dumps(general_settings),
            },
            "update": {"param_value": json.dumps(general_settings)},
        },
    )
    await invalidate_config_param("general_settings")
    asyncio.create_task(
        create_config_audit_log(
            "general_settings", "updated", before_general_settings, general_settings, user_api_key_dict
        )
    )

    if data.field_name == "plugins":
        register_plugins_from_config(cast(dict[str, object], general_settings))  # cast-ok: the callee only reads it
    _apply_ssrf_general_settings(general_settings)

    return response


def _is_secret_general_setting_field(field_name: str) -> bool:
    return field_name in _EXTRA_SECRET_GENERAL_SETTINGS_FIELDS or SENSITIVE_DATA_MASKER.is_sensitive_key(field_name)


# Matches the cap on _redact_sensitive_litellm_params (the closest analog in the
# proxy). Past this depth we fail closed by returning "REDACTED" for the whole
# subtree rather than recursing further — better to over-redact a pathological
# config than to silently return a deeply-nested credential verbatim
_REDACT_SECRET_MAX_DEPTH: Final = 10


def _redact_secret_values_in_obj(value: JsonValue, depth: int = 0) -> JsonValue:
    """Recursively redact secret leaves inside a structured field so a nested
    credential (e.g. aws_web_identity_token under database_args) is never
    returned to a non-admin, while non-secret siblings stay visible. At
    _REDACT_SECRET_MAX_DEPTH the whole subtree is replaced with "REDACTED"
    so depth-overrun fails closed."""
    if depth >= _REDACT_SECRET_MAX_DEPTH:
        return "REDACTED"
    if isinstance(value, dict):
        return {
            key: ("REDACTED" if _is_secret_general_setting_field(key) else _redact_secret_values_in_obj(sub, depth + 1))
            for key, sub in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_values_in_obj(item, depth + 1) for item in value]
    return value


def _redact_config_param_value_for_logging(param_name: str | None, param_value: JsonValue) -> JsonValue:
    if param_name == "environment_variables" and isinstance(param_value, dict):
        return {key: "REDACTED" for key in param_value}
    if isinstance(param_value, (dict, list)):
        return _redact_secret_values_in_obj(param_value)
    return param_value


def _redact_general_setting_value(field_name: str, value: JsonValue, is_full_admin: bool) -> JsonValue:
    if is_full_admin:
        return value
    if _is_secret_general_setting_field(field_name):
        return "REDACTED"
    if isinstance(value, (dict, list)):
        return _redact_secret_values_in_obj(value)
    return value


def _dump_redacted_config(value: JsonValue | None, *, redact_all_values: bool = False) -> str | None:
    # `default=str` matches the sibling audit-log serializers in
    # team_endpoints.py and the LiteLLM_AuditLogs validator, so a YAML-loaded
    # value with a non-JSON-native leaf (datetime, custom object) cannot turn
    # an audit write into a 500.
    if value is None:
        return None
    if redact_all_values and isinstance(value, dict):
        return json.dumps({key: "REDACTED" for key in value}, default=str)
    return json.dumps(_redact_secret_values_in_obj(value), default=str)


async def create_config_audit_log(
    param_name: str,
    action: AUDIT_ACTIONS,
    before_value: JsonValue | None,
    after_value: JsonValue | None,
    user_api_key_dict: UserAPIKeyAuth,
    table_name: LitellmTableNames = LitellmTableNames.CONFIG_TABLE_NAME,
) -> None:
    """Record a system-wide settings change in LiteLLM_AuditLog.

    Secret leaves are redacted before the row is written. environment_variables
    hold arbitrary credentials under non-secret-looking uppercase keys (e.g.
    DATABASE_URL), so every value in that section is redacted rather than
    relying on key-name matching; other sections reuse the same matcher
    /config/field/info applies for non-admins.
    """
    redact_all_values: Final = param_name == "environment_variables"
    await create_object_audit_log(
        object_id=param_name,
        action=action,
        table_name=table_name,
        before_value=_dump_redacted_config(before_value, redact_all_values=redact_all_values),
        after_value=_dump_redacted_config(after_value, redact_all_values=redact_all_values),
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=None,
        litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
    )


_EXTRA_SECRET_CALLBACK_ENV_VARS: Final = frozenset(
    {
        "GALILEO_USERNAME",
        "GENERIC_LOGGER_HEADERS",
        "OTEL_HEADERS",
        "SLACK_WEBHOOK_URL",
        "SMTP_USERNAME",
    }
)


def _redact_callback_env_vars(env_vars: dict[str, str | None]) -> dict[str, str | None]:
    """Return a copy of ``env_vars`` with values for keys classified as
    sensitive by ``is_sensitive_callback_key`` replaced with ``"REDACTED"``.
    ``None`` values pass through unchanged.
    """
    return {
        key: (
            "REDACTED"
            if value is not None and is_sensitive_callback_key(key, extra=_EXTRA_SECRET_CALLBACK_ENV_VARS)
            else value
        )
        for key, value in env_vars.items()
    }


def _apply_callback_role_gate(entries: list, is_full_admin: bool) -> list:
    if is_full_admin:
        return entries
    return [{**entry, "variables": _redact_callback_env_vars(entry.get("variables") or {})} for entry in entries]


def _apply_alerting_env_role_gate(env_vars: dict, is_full_admin: bool) -> dict:
    if is_full_admin:
        return mask_sensitive_keys(env_vars, _ALERTING_SENSITIVE_VARS)
    return _redact_callback_env_vars(env_vars)


def _apply_webhook_role_gate(webhook_map, is_full_admin: bool):
    if is_full_admin or not isinstance(webhook_map, dict):
        return webhook_map
    return {alert_type: "REDACTED" for alert_type in webhook_map}


@router.get(
    "/config/field/info",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ConfigFieldInfo,
    include_in_schema=False,
)
async def get_config_general_settings(
    field_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global prisma_client

    ## VALIDATION ##
    """
    - Check if prisma_client is None
    - Check if user allowed to call this endpoint (admin-only)
    - Check if param in general settings
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    if field_name not in ConfigGeneralSettings.model_fields:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid field={field_name} passed in."},
        )

    ## get general settings from db
    db_general_settings: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
        where={"param_name": "general_settings"}
    )
    ### pop the value

    if db_general_settings is None or db_general_settings.param_value is None:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Field name={field_name} not in DB"},
        )
    else:
        general_settings = dict(db_general_settings.param_value)

        if field_name in general_settings:
            field_value = _redact_general_setting_value(
                field_name,
                general_settings[field_name],
                user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN,
            )
            if field_name == "plugins" and isinstance(field_value, list):
                field_value = [
                    ({k: ("***" if k == "plugin_key" else v) for k, v in p.items()} if isinstance(p, dict) else p)
                    for p in field_value
                ]
            return ConfigFieldInfo(field_name=field_name, field_value=field_value)
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Field name={field_name} not in DB"},
            )


GeneralSettingsUILiteLLMValue = float | bool | str | None


class GeneralSettingsUILiteLLMFieldSpec(TypedDict):
    type: Literal["Float", "Dollar", "Boolean", "Select"]
    description: str
    options: NotRequired[tuple[str, ...]]
    tab: NotRequired[str]  # Admin UI sub-tab this field renders under; None groups it with the rest
    default: NotRequired[float]  # reset/clear restores this instead of None; fields whose None means fail-open set it


_GENERAL_SETTINGS_UI_LITELLM_FIELDS: Final[dict[str, GeneralSettingsUILiteLLMFieldSpec]] = {
    "budget_exceeded_throttle_percentage": {
        "type": "Float",
        "description": (
            "Fraction (0, 1] of a key's configured TPM/RPM that an over-budget key with "
            "'Throttle on budget exceeded' enabled keeps serving at. Leave empty to hard-block "
            "over-budget keys."
        ),
    },
    "enable_anthropic_prompt_caching": {
        "type": "Boolean",
        "tab": "prompt_caching",
        "description": (
            "Auto-adds cache_control to the system prompt and trailing turn for supported Anthropic "
            "and Bedrock Claude models. The cache is shared across callers on the same upstream credentials."
        ),
    },
    "anthropic_prompt_caching_ttl": {
        "type": "Select",
        "options": ("5m", "1h"),
        "tab": "prompt_caching",
        "description": "Empty uses Anthropic's 5m default. 1h suits long sessions but doubles the cache write cost.",
    },
    "budget_rollover": {  # mutable-ok: registry literal, frozen with its siblings below
        "type": "Boolean",
        "description": (
            "Carry spend beyond max_budget into the next window when budgets reset, instead of "
            "forgiving it. Applies to key, user, team, team member, org, tag and end-user budgets."
        ),
    },
    "max_ui_session_budget": {
        "type": "Dollar",
        "default": 1.0,
        "description": (
            "USD spend cap for each dashboard login session; covers LLM calls made from the dashboard "
            "such as the playground and auto router Test Connection. Each login starts a fresh session "
            "with this budget. Clearing restores the $1 default."
        ),
    },
}


def _general_settings_ui_litellm_default(
    spec: GeneralSettingsUILiteLLMFieldSpec,
) -> GeneralSettingsUILiteLLMValue:
    """The value a field falls back to when it is cleared or reset."""
    if "default" in spec:
        return spec["default"]
    return False if spec["type"] == "Boolean" else None


def _validate_general_settings_ui_litellm_value(field_name: str, value: object) -> GeneralSettingsUILiteLLMValue:
    spec: Final = _GENERAL_SETTINGS_UI_LITELLM_FIELDS[field_name]
    field_type: Final = spec["type"]
    if value is None or value == "":
        return _general_settings_ui_litellm_default(spec)
    match field_type:
        case "Boolean":
            if not isinstance(value, bool):
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"{field_name} must be true or false"},
                )
            return value
        case "Select":
            options: Final = spec.get("options", ())
            if value not in options:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"{field_name} must be one of: {', '.join(options)}, or empty"},
                )
            return cast(str, value)  # cast-ok: membership in options proves it is one of the option strings
        case "Float":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0 < float(value) <= 1):
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"{field_name} must be a number in (0, 1] or empty"},
                )
            return float(value)
        case "Dollar":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"{field_name} must be a positive dollar amount or empty"},
                )
            return float(value)
        case _:
            assert_never(field_type)


async def _persist_general_settings_ui_litellm_field(
    field_name: str, value: object, user_api_key_dict: UserAPIKeyAuth
) -> dict:
    validated: Final = _validate_general_settings_ui_litellm_value(field_name, value)
    config: Final = await proxy_config.get_config()
    before_value: Final = config.get("litellm_settings", {}).get(field_name)
    setattr(litellm, field_name, validated)
    if "litellm_settings" not in config:
        config["litellm_settings"] = {}
    config["litellm_settings"][field_name] = validated
    await proxy_config.save_config(new_config=config)
    asyncio.create_task(create_config_audit_log(field_name, "updated", before_value, validated, user_api_key_dict))
    return {"message": f"Field {field_name} updated", "status": "success"}


async def _reset_general_settings_ui_litellm_field(field_name: str, user_api_key_dict: UserAPIKeyAuth) -> dict:
    config: Final = await proxy_config.get_config()
    before_value: Final = config.get("litellm_settings", {}).get(field_name)
    default_value: Final = _general_settings_ui_litellm_default(_GENERAL_SETTINGS_UI_LITELLM_FIELDS[field_name])
    setattr(litellm, field_name, default_value)
    if "litellm_settings" in config:
        config["litellm_settings"].pop(field_name, None)
    await proxy_config.save_config(new_config=config)
    asyncio.create_task(create_config_audit_log(field_name, "deleted", before_value, default_value, user_api_key_dict))
    return {"message": f"Field {field_name} reset", "status": "success"}


@router.get(
    "/config/list",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_config_list(
    config_type: Literal["general_settings"],
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list[ConfigList]:
    """
    List the available fields + current values for a given type of setting (currently just 'general_settings'user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),)
    """
    global prisma_client, general_settings

    ## VALIDATION ##
    """
    - Check if prisma_client is None
    - Check if user allowed to call this endpoint (admin-only)
    - Check if param in general settings
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    is_full_admin: Final = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN

    ## get general settings from db
    db_general_settings: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
        where={"param_name": "general_settings"}
    )

    if db_general_settings is not None and db_general_settings.param_value is not None:
        db_general_settings_dict: Mapping[str, JsonValue] = dict(db_general_settings.param_value)
    else:
        db_general_settings_dict = {}

    allowed_args: Final = _GENERAL_SETTINGS_CONFIG_LIST_FIELD_TYPES

    return_val: Final = []

    for field_name, field_info in ConfigGeneralSettings.model_fields.items():
        if field_name in allowed_args:
            ## HANDLE TYPED DICT

            typed_dict_type = allowed_args[field_name]

            if typed_dict_type == "PydanticModel":
                if field_name == "pass_through_endpoints":
                    pydantic_class_list = [PassThroughGenericEndpoint]
                else:
                    pydantic_class_list = []

                for pydantic_class in pydantic_class_list:
                    # Get type hints from the TypedDict to create FieldDetail objects
                    nested_fields = [
                        FieldDetail(
                            field_name=sub_field,
                            field_type=getattr(sub_field_type, "__name__", str(sub_field_type)),
                            field_description="",  # Add custom logic if descriptions are available
                            field_default_value=_redact_general_setting_value(
                                sub_field,
                                general_settings.get(sub_field, None),
                                is_full_admin,
                            ),
                            stored_in_db=None,
                        )
                        for sub_field, sub_field_type in pydantic_class.__annotations__.items()
                    ]

                    idx = 0
                    for (
                        sub_field,
                        sub_field_info,
                    ) in pydantic_class.model_fields.items():
                        if hasattr(sub_field_info, "description") and sub_field_info.description is not None:
                            nested_fields[idx].field_description = sub_field_info.description
                        idx += 1

                    _stored_in_db = None
                    if field_name in db_general_settings_dict:
                        _stored_in_db = True
                    elif field_name in general_settings:
                        _stored_in_db = False

                    _response_obj = ConfigList(
                        field_name=field_name,
                        field_type=allowed_args[field_name],
                        field_description=field_info.description or "",
                        field_value=_redact_general_setting_value(
                            field_name,
                            general_settings.get(field_name, None),
                            is_full_admin,
                        ),
                        stored_in_db=_stored_in_db,
                        field_default_value=field_info.default,
                        nested_fields=nested_fields,
                    )
                    return_val.append(_response_obj)

            else:
                nested_fields = None

                _stored_in_db = None
                if field_name in db_general_settings_dict:
                    _stored_in_db = True
                elif field_name in general_settings:
                    _stored_in_db = False

                _field_value = general_settings.get(field_name, None)
                if _field_value is None and field_name in db_general_settings_dict:
                    _field_value = db_general_settings_dict[field_name]

                _response_obj = ConfigList(
                    field_name=field_name,
                    field_type=allowed_args[field_name],
                    field_description=field_info.description or "",
                    field_value=_redact_general_setting_value(field_name, _field_value, is_full_admin),
                    stored_in_db=_stored_in_db,
                    field_default_value=field_info.default,
                    nested_fields=nested_fields,
                )
                return_val.append(_response_obj)

    db_litellm_settings_row: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
        where={"param_name": "litellm_settings"}
    )
    db_litellm_settings: Final[dict] = (
        dict(db_litellm_settings_row.param_value)
        if db_litellm_settings_row is not None and db_litellm_settings_row.param_value is not None
        else {}
    )
    for litellm_field_name, spec in _GENERAL_SETTINGS_UI_LITELLM_FIELDS.items():
        current_value: GeneralSettingsUILiteLLMValue = getattr(litellm, litellm_field_name, None)
        default_value = _general_settings_ui_litellm_default(spec)
        stored_in_db_litellm: bool | None
        if litellm_field_name in db_litellm_settings:
            stored_in_db_litellm = True
        elif current_value != default_value:
            stored_in_db_litellm = False
        else:
            stored_in_db_litellm = None
        return_val.append(
            ConfigList(
                field_name=litellm_field_name,
                field_type=spec["type"],
                field_description=spec["description"],
                field_value=current_value,
                stored_in_db=stored_in_db_litellm,
                field_default_value=default_value,
                field_options=list(spec.get("options", ())) or None,
                field_tab=spec.get("tab"),
                nested_fields=None,
            )
        )

    return return_val


@router.post(
    "/config/field/delete",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def delete_config_general_settings(
    data: ConfigFieldDelete,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete the db value of this field in litellm general settings. Resets it to it's initial default value on litellm.
    """
    global prisma_client
    ## VALIDATION ##
    """
    - Check if prisma_client is None
    - Check if user allowed to call this endpoint (admin-only)
    - Check if param in general settings
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    if data.field_name in _GENERAL_SETTINGS_UI_LITELLM_FIELDS:
        return await _reset_general_settings_ui_litellm_field(data.field_name, user_api_key_dict)

    if data.field_name not in ConfigGeneralSettings.model_fields:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid field={data.field_name} passed in."},
        )

    ## get general settings from db
    db_general_settings: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_first(
        where={"param_name": "general_settings"}
    )
    ### pop the value

    if db_general_settings is None or db_general_settings.param_value is None:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Field name={data.field_name} not in config"},
        )
    else:
        general_settings = dict(db_general_settings.param_value)

    before_general_settings: Final = copy.deepcopy(general_settings)

    ## update db

    general_settings.pop(data.field_name, None)

    response: Final = await ConfigRepository(prisma_client).table.upsert(
        where={"param_name": "general_settings"},
        data={
            "create": {
                "param_name": "general_settings",
                "param_value": json.dumps(general_settings),
            },
            "update": {"param_value": json.dumps(general_settings)},
        },
    )
    await invalidate_config_param("general_settings")
    asyncio.create_task(
        create_config_audit_log(
            "general_settings", "deleted", before_general_settings, general_settings, user_api_key_dict
        )
    )

    return response


@router.post(
    "/config/callback/delete",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def delete_callback(
    data: CallbackDelete,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete specific logging callback from configuration.
    """
    global prisma_client, proxy_config

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={"error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."},
        )

    try:
        # Get current configuration
        config: Final = await proxy_config.get_config()
        callback_name: Final = data.callback_name.lower()

        # Check if callback exists in current configuration
        litellm_settings: Final = config.get("litellm_settings", {})
        success_callbacks: Final = litellm_settings.get("success_callback", [])

        if callback_name not in success_callbacks:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Callback '{callback_name}' not found in active configuration"},
            )

        before_success_callbacks: Final = list(success_callbacks)

        # Remove callback from success_callback list
        success_callbacks.remove(callback_name)
        config.setdefault("litellm_settings", {})["success_callback"] = success_callbacks

        # Save the updated configuration
        await proxy_config.save_config(new_config=config)

        asyncio.create_task(
            create_config_audit_log(
                "litellm_settings",
                "deleted",
                {"success_callback": before_success_callbacks},
                {"success_callback": success_callbacks},
                user_api_key_dict,
            )
        )

        # Restart the proxy to apply changes
        await proxy_config.add_deployment(prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj)

        return {
            "message": f"Successfully deleted callback: {callback_name}",
            "removed_callback": callback_name,
            "remaining_callbacks": success_callbacks,
            "deleted_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("litellm.proxy.proxy_server.delete_callback(): Exception occurred - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        raise ProxyException(
            message="Error deleting callback: " + str(e),
            type=ProxyErrorTypes.internal_server_error,
            param="callback_name",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/get/config/callbacks",
    tags=["config.yaml"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_config(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    For Admin UI - allows admin to view config via UI
    # return the callbacks and the env variables for the callback

    """
    global llm_router, llm_model_list, general_settings, proxy_config, proxy_logging_obj, master_key
    try:
        all_available_callbacks: Final = AllCallbacks()

        config_data: Final = await proxy_config.get_config()
        _litellm_settings: Final = config_data.get("litellm_settings", {})
        _general_settings: Final = config_data.get("general_settings", {})
        environment_variables: Final = config_data.get("environment_variables", {})

        is_full_admin: Final = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN

        _success_callbacks = _litellm_settings.get("success_callback", [])
        _failure_callbacks = _litellm_settings.get("failure_callback", [])
        _success_and_failure_callbacks = _litellm_settings.get("callbacks", [])

        # Normalize string callbacks to lists
        def normalize_callback(callback):
            if isinstance(callback, str):
                return [callback]
            elif callback is None:
                return []
            return callback

        _success_callbacks = normalize_callback(_success_callbacks)
        _failure_callbacks = normalize_callback(_failure_callbacks)
        _success_and_failure_callbacks = normalize_callback(_success_and_failure_callbacks)

        _data_to_return = []
        """
        [
            {
                "name": "langfuse",
                "variables": {
                    "LANGFUSE_PUB_KEY": "value",
                    "LANGFUSE_SECRET_KEY": "value",
                    "LANGFUSE_HOST": "value"
                },
                "type": "success"
            }
        ]

        """

        for _callback in _success_callbacks:
            _data_to_return.append(process_callback(_callback, "success", environment_variables))

        for _callback in _failure_callbacks:
            _data_to_return.append(process_callback(_callback, "failure", environment_variables))

        for _callback in _success_and_failure_callbacks:
            _data_to_return.append(process_callback(_callback, "success_and_failure", environment_variables))

        _data_to_return = _apply_callback_role_gate(_data_to_return, is_full_admin)

        # Check if slack alerting is on
        _alerting: Final = _general_settings.get("alerting", [])
        alerting_data: Final = []
        if "slack" in _alerting:
            _slack_values, _ = resolve_fields(
                SLACK_DESCRIPTORS, environment_variables, os.environ, empty_db_is_set=True
            )
            _slack_env_vars: Final = _apply_alerting_env_role_gate(_slack_values, is_full_admin)

            _alerting_types: Final = proxy_logging_obj.slack_alerting_instance.alert_types
            _all_alert_types: Final = proxy_logging_obj.slack_alerting_instance._all_possible_alert_types()
            _alerts_to_webhook: Final = _apply_webhook_role_gate(
                proxy_logging_obj.slack_alerting_instance.alert_to_webhook_url, is_full_admin
            )
            alerting_data.append(
                {
                    "name": "slack",
                    "variables": _slack_env_vars,
                    "active_alerts": _alerting_types,
                    "alerts_to_webhook": _alerts_to_webhook,
                }
            )
        # pass email alerting vars
        _email_values, _ = resolve_fields(EMAIL_DESCRIPTORS, environment_variables, os.environ, empty_db_is_set=True)
        _email_env_vars: Final = _apply_alerting_env_role_gate(_email_values, is_full_admin)

        alerting_data.append(
            {
                "name": "email",
                "variables": _email_env_vars,
            }
        )

        if llm_router is None:
            _router_settings = {}
        else:
            _router_settings = llm_router.get_settings()

        return {
            "status": "success",
            "callbacks": _data_to_return,
            "alerts": alerting_data,
            "router_settings": _router_settings,
            "available_callbacks": all_available_callbacks,
        }
    except Exception as e:
        verbose_proxy_logger.exception("litellm.proxy.proxy_server.get_config(): Exception occured - %s", e)
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({e})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.get(
    "/config/yaml",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def config_yaml_endpoint(config_info: ConfigYAML):
    """
    This is a mock endpoint, to show what you can set in config.yaml details in the Swagger UI.

    Parameters:

    The config.yaml object has the following attributes:
    - **model_list**: *Optional[List[ModelParams]]* - A list of supported models on the server, along with model-specific configurations. ModelParams includes "model_name" (name of the model), "litellm_params" (litellm-specific parameters for the model), and "model_info" (additional info about the model such as id, mode, cost per token, etc).

    - **litellm_settings**: *Optional[dict]*: Settings for the litellm module. You can specify multiple properties like "drop_params", "set_verbose", "api_base", "cache".

    - **general_settings**: *Optional[ConfigGeneralSettings]*: General settings for the server like "completion_model" (default model for chat completion calls), "use_azure_key_vault" (option to load keys from azure key vault), "master_key" (key required for all calls to proxy), and others.

    Please, refer to each class's description for a better understanding of the specific attributes within them.

    Note: This is a mock endpoint primarily meant for demonstration purposes, and does not actually provide or change any configurations.
    """
    return {"hello": "world"}


@router.post(
    "/reload/model_cost_map",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def reload_model_cost_map(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Manually reload the model cost map from the remote source.
    This will fetch fresh pricing data from the model_prices_and_context_window.json file.
    """
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")

        # Immediately reload the model cost map in the current pod
        from litellm.litellm_core_utils.get_model_cost_map import (
            ModelCostMapReloadUnavailable,
            refetch_model_cost_map,
        )

        reload_result = await refetch_model_cost_map(url=litellm.model_cost_map_url)
        if isinstance(reload_result, ModelCostMapReloadUnavailable):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reload model cost map: {reload_result.reason}. Current pricing data was kept.",
            )

        models_count = _swap_in_model_cost_map(reload_result.model_cost_map)
        current_time = utc_now()
        proxy_config.model_cost_map_loaded_at = current_time

        # Publish a new revision so every other pod reloads on its next poll; this pod has
        # already served it, so adopt it here rather than reloading again a tick later
        proxy_config.model_cost_map_applied_revision = await record_manual_reload(
            prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME, current_time
        )

        verbose_proxy_logger.info("Model cost map reloaded successfully in current pod. Models count: %s", models_count)

        return {
            "message": f"Price data reloaded successfully! {models_count} models updated.",
            "status": "success",
            "models_count": models_count,
            "timestamp": current_time.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Failed to reload model cost map: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to reload model cost map: {e}")


@router.post(
    "/schedule/model_cost_map_reload",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def schedule_model_cost_map_reload(
    hours: int,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Schedule periodic reload of the model cost map.
    This will create a background job that reloads the model cost map every specified hours.
    """
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    if hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be greater than 0")

    try:
        global prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")

        await write_reload_interval(prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME, hours)

        verbose_proxy_logger.info("Model cost map reload scheduled for every %s hours", hours)

        return {
            "message": f"Model cost map reload scheduled for every {hours} hours",
            "status": "success",
            "interval_hours": hours,
            "timestamp": utc_now().isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to schedule model cost map reload: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule model cost map reload: {e}",
        )


@router.delete(
    "/schedule/model_cost_map_reload",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def cancel_model_cost_map_reload(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Cancel the scheduled periodic reload of the model cost map.
    """
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")

        await clear_reload_interval(prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME)

        verbose_proxy_logger.info("Model cost map reload schedule cancelled")

        return {
            "message": "Model cost map reload schedule cancelled",
            "status": "success",
            "timestamp": utc_now().isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to cancel model cost map reload: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to cancel model cost map reload: {e}")


@router.get(
    "/schedule/model_cost_map_reload/status",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_model_cost_map_reload_status(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Get the status of the scheduled model cost map reload job.
    """
    # Read-only status check — admin viewers can read.
    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client

        if prisma_client is None:
            verbose_proxy_logger.info("No database connection, returning not scheduled")
            return reload_schedule_status(None)

        return reload_schedule_status(await read_reload_schedule(prisma_client, MODEL_COST_MAP_RELOAD_PARAM_NAME))
    except Exception as e:
        verbose_proxy_logger.exception("Failed to get model cost map reload status: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model cost map reload status: {e}",
        )


@router.get(
    "/model/cost_map/source",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_model_cost_map_source(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Returns information about where the current model cost/pricing data was loaded from.

    Response fields:
    - source: "local" (bundled backup) or "remote" (fetched from URL)
    - url: the remote URL that was attempted (null when env-forced local)
    - is_env_forced: true if LITELLM_LOCAL_MODEL_COST_MAP=True forced local usage
    - fallback_reason: human-readable reason why remote failed (null on success)
    - model_count: number of models in the currently loaded cost map
    """
    # Read-only source info — admin viewers can read.
    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        from litellm.litellm_core_utils.get_model_cost_map import (
            get_model_cost_map_source_info,
        )

        source_info: Final = get_model_cost_map_source_info()
        model_count: Final = len(litellm.model_cost) if litellm.model_cost else 0

        return {
            **source_info,
            "model_count": model_count,
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to get model cost map source info: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model cost map source info: {e}",
        )


#### ANTHROPIC BETA HEADERS RELOAD ENDPOINTS ####


@router.post(
    "/reload/anthropic_beta_headers",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def reload_anthropic_beta_headers(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Manually reload the Anthropic beta headers configuration from the remote source.
    This will fetch fresh configuration from the anthropic_beta_headers_config.json file.
    """
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")

        # Immediately reload the beta headers config in the current pod
        from litellm.anthropic_beta_headers_manager import reload_beta_headers_config

        new_config: Final = reload_beta_headers_config()

        # Update pod's in-memory last reload time
        global last_anthropic_beta_headers_reload
        current_time: Final = datetime.utcnow()
        last_anthropic_beta_headers_reload = current_time.isoformat()

        # Set force reload flag in database for other pods, preserving existing interval_hours
        existing_beta_config: Final[_ConfigParamRow | None] = await _config_param_table(prisma_client).find_unique(
            where={"param_name": "anthropic_beta_headers_reload_config"}
        )
        existing_beta_interval = None
        if existing_beta_config and existing_beta_config.param_value:
            existing_beta_interval = existing_beta_config.param_value.get("interval_hours")

        await ConfigRepository(prisma_client).table.upsert(
            where={"param_name": "anthropic_beta_headers_reload_config"},
            data={
                "create": {
                    "param_name": "anthropic_beta_headers_reload_config",
                    "param_value": safe_dumps({"interval_hours": None, "force_reload": True}),
                },
                "update": {"param_value": safe_dumps({"interval_hours": existing_beta_interval, "force_reload": True})},
            },
        )
        await invalidate_config_param("anthropic_beta_headers_reload_config")

        provider_count: Final = sum(1 for k in new_config if k not in ["provider_aliases", "description"])
        verbose_proxy_logger.info(
            "Anthropic beta headers config reloaded successfully in current pod. Providers: %s", provider_count
        )

        return {
            "message": f"Anthropic beta headers configuration reloaded successfully! {provider_count} providers updated.",
            "status": "success",
            "providers_count": provider_count,
            "timestamp": current_time.isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to reload anthropic beta headers: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to reload anthropic beta headers: {e}")


@router.post(
    "/schedule/anthropic_beta_headers_reload",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def schedule_anthropic_beta_headers_reload(
    hours: int,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Schedule periodic reload of the Anthropic beta headers configuration.
    This will create a background job that reloads the configuration every specified hours.
    """
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    if hours <= 0:
        raise HTTPException(status_code=400, detail="Hours must be greater than 0")

    try:
        global prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")

        # Update database with new reload configuration
        await ConfigRepository(prisma_client).table.upsert(
            where={"param_name": "anthropic_beta_headers_reload_config"},
            data={
                "create": {
                    "param_name": "anthropic_beta_headers_reload_config",
                    "param_value": safe_dumps({"interval_hours": hours, "force_reload": False}),
                },
                "update": {"param_value": safe_dumps({"interval_hours": hours, "force_reload": False})},
            },
        )
        await invalidate_config_param("anthropic_beta_headers_reload_config")

        verbose_proxy_logger.info("Anthropic beta headers reload scheduled for every %s hours", hours)

        return {
            "message": f"Anthropic beta headers reload scheduled for every {hours} hours",
            "status": "success",
            "interval_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to schedule anthropic beta headers reload: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule anthropic beta headers reload: {e}",
        )


@router.delete(
    "/schedule/anthropic_beta_headers_reload",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def cancel_anthropic_beta_headers_reload(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Cancel the scheduled periodic reload of the Anthropic beta headers configuration.
    """
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client
        if prisma_client is None:
            raise HTTPException(status_code=500, detail="Database connection not available")

        # Remove reload configuration from database
        await ConfigRepository(prisma_client).table.delete(where={"param_name": "anthropic_beta_headers_reload_config"})
        await invalidate_config_param("anthropic_beta_headers_reload_config")

        verbose_proxy_logger.info("Anthropic beta headers reload schedule cancelled")

        return {
            "message": "Anthropic beta headers reload schedule cancelled",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to cancel anthropic beta headers reload: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel anthropic beta headers reload: {e}",
        )


@router.get(
    "/schedule/anthropic_beta_headers_reload/status",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_anthropic_beta_headers_reload_status(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Get the status of the scheduled Anthropic beta headers reload job.
    """
    # Read-only status — admin viewers can read.
    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client, last_anthropic_beta_headers_reload

        verbose_proxy_logger.info(
            "Checking anthropic beta headers reload status. Last reload: %s", last_anthropic_beta_headers_reload
        )

        if prisma_client is None:
            verbose_proxy_logger.info("No database connection, returning not scheduled")
            return {
                "scheduled": False,
                "interval_hours": None,
                "last_run": None,
                "next_run": None,
            }

        # Get reload configuration from database
        config_record: Final = await _config_param_table(prisma_client).find_unique(
            where={"param_name": "anthropic_beta_headers_reload_config"}
        )

        if config_record is None or config_record.param_value is None:
            verbose_proxy_logger.info("No anthropic beta headers reload configuration found")
            return {
                "scheduled": False,
                "interval_hours": None,
                "last_run": None,
                "next_run": None,
            }

        config: Final = config_record.param_value
        interval_hours: Final = cast(  # cast-ok: every writer of this key stores `hours: int` or an explicit None
            int | None, config.get("interval_hours")
        )

        if interval_hours is None:
            verbose_proxy_logger.info("No interval configured, returning not scheduled")
            return {
                "scheduled": False,
                "interval_hours": None,
                "last_run": None,
                "next_run": None,
            }

        current_time: Final = datetime.utcnow()
        next_run = None

        # Use pod's in-memory last reload time
        if last_anthropic_beta_headers_reload is not None:
            try:
                last_reload_time: Final = datetime.fromisoformat(last_anthropic_beta_headers_reload)
                time_since_last_reload: Final = current_time - last_reload_time
                hours_since_last_reload: Final = time_since_last_reload.total_seconds() / 3600

                if hours_since_last_reload < interval_hours:
                    next_run = (last_reload_time + timedelta(hours=interval_hours)).isoformat()
            except Exception as e:
                verbose_proxy_logger.warning("Error parsing last reload time: %s", e)

        return {
            "scheduled": True,
            "interval_hours": interval_hours,
            "last_run": last_anthropic_beta_headers_reload,
            "next_run": next_run,
        }
    except Exception as e:
        verbose_proxy_logger.exception("Failed to get anthropic beta headers reload status: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get anthropic beta headers reload status: {e}",
        )


@router.get("/", dependencies=[Depends(user_api_key_auth)])
async def home(request: Request):
    return "LiteLLM: RUNNING"


@router.get(
    "/adaptive_router/state",
    tags=["adaptive_router"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_adaptive_router_state(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return live bandit posteriors + queue depth for every configured adaptive router.

    Admin-only. Returns 404 if no adaptive router is configured.

    Response shape: `{"routers": [<snapshot>, ...]}` — one snapshot per
    adaptive-router deployment. Each snapshot's `router_name` field identifies
    which deployment it came from.
    """
    # Read-only state — admin viewers can read.
    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )
    if llm_router is None or not llm_router.adaptive_routers:
        raise HTTPException(
            status_code=404,
            detail={"error": "No adaptive_router is configured on this proxy."},
        )
    snapshots: Final = [
        await tagged.strategy.get_state_snapshot()
        for tagged_routers in llm_router.adaptive_routers.values()
        for tagged in tagged_routers
    ]
    return {"routers": snapshots}


@router.get("/routes", dependencies=[Depends(user_api_key_auth)])
async def get_routes():
    """
    Get a list of available routes in the FastAPI application.
    """
    from litellm.proxy.common_utils.get_routes import GetRoutes

    routes: Final = []
    for route in app.routes:
        endpoint_route = getattr(route, "endpoint", None)
        if endpoint_route is not None:
            routes.extend(
                GetRoutes.get_app_routes(
                    route=route,
                    endpoint_route=endpoint_route,
                )
            )
        # Handle mounted sub-applications (like MCP app)
        elif hasattr(route, "app") and hasattr(route, "path"):
            routes.extend(GetRoutes.get_routes_for_mounted_app(route=route))

    return {"routes": routes}


#### TEST ENDPOINTS ####
# @router.get(
#     "/token/generate",
#     dependencies=[Depends(user_api_key_auth)],
#     include_in_schema=False,
# )
# async def token_generate():
#     """
#     Test endpoint. Admin-only access. Meant for generating admin tokens with specific claims and testing if they work for creating keys, etc.
#     """
#     # Initialize AuthJWTSSO with your OpenID Provider configuration
#     from fastapi_sso import AuthJWTSSO

#     auth_jwt_sso = AuthJWTSSO(
#         issuer=os.getenv("OPENID_BASE_URL"),
#         client_id=os.getenv("OPENID_CLIENT_ID"),
#         client_secret=os.getenv("OPENID_CLIENT_SECRET"),
#         scopes=["litellm_proxy_admin"],
#     )

#     token = auth_jwt_sso.create_access_token()

#     return {"token": token}


app.include_router(router)
app.include_router(response_router)
app.include_router(public_endpoints_router)
app.include_router(rerank_router)
app.include_router(ocr_router)
app.include_router(rag_router)
app.include_router(video_router)
app.include_router(container_router)
app.include_router(search_router)
app.include_router(image_router)
app.include_router(fine_tuning_router)
app.include_router(credential_router)
app.include_router(openai_passthrough_router)
app.include_router(batches_router)
app.include_router(openai_files_router)
app.include_router(llm_passthrough_router)
app.include_router(pass_through_router)
app.include_router(health_router)
app.include_router(key_management_router)
app.include_router(internal_user_router)
app.include_router(team_router)
app.include_router(ui_sso_router)
app.include_router(organization_router)
app.include_router(customer_router)
app.include_router(management_v1_router)
app.include_router(spend_management_router)
app.include_router(caching_router)
app.include_router(analytics_router)
app.include_router(callback_management_endpoints_router)
app.include_router(debugging_endpoints_router)
app.include_router(rust_control_plane_router)
app.include_router(ui_crud_endpoints_router)
app.include_router(user_banner_endpoints_router)
app.include_router(team_callback_router)
app.include_router(budget_management_router)
app.include_router(model_management_router)
app.include_router(model_access_group_management_router)
app.include_router(auto_router_management_router)
app.include_router(tag_management_router)
app.include_router(workflow_management_router)
app.include_router(memory_router)
app.include_router(plugin_router)
app.include_router(cost_tracking_settings_router)
app.include_router(router_settings_router)
app.include_router(fallback_management_router)
app.include_router(cache_settings_router)
app.include_router(coordination_redis_settings_router)
app.include_router(user_agent_analytics_router)
app.include_router(gateway_request_router)
app.include_router(enterprise_router)
app.include_router(ui_discovery_endpoints_router)
# Eager: /models/{name}:method overlaps with the OpenAI /models endpoint.
app.include_router(google_router)

attach_lazy_features(app)
app.add_middleware(
    RequestSizeLimitMiddleware,
    get_max_request_size_mb=lambda: general_settings.get("max_request_size_mb"),
    is_request_size_limit_enabled=lambda: premium_user is True,
)


async def _stream_mcp_asgi_response(handle_fn, scope: dict, receive) -> "StreamingResponse":
    """
    Call an ASGI MCP handler and return a StreamingResponse so SSE/streaming works.

    asyncio.create_task copies the current context, so any ContextVar set before
    this call (e.g. _mcp_active_toolset_id) is visible inside the handler task.
    """
    from starlette.responses import StreamingResponse

    headers_ready: Final[asyncio.Future] = asyncio.get_running_loop().create_future()
    body_queue: Final[asyncio.Queue] = asyncio.Queue(maxsize=1024)

    async def bridging_send(message):
        if message["type"] == "http.response.start":
            if not headers_ready.done():
                headers_ready.set_result((message.get("status", 200), message.get("headers", [])))
        elif message["type"] == "http.response.body":
            chunk: Final = message.get("body", b"")
            if chunk:
                await body_queue.put(chunk)
            if not message.get("more_body", False):
                await body_queue.put(None)  # EOF sentinel

    handler_task: Final = asyncio.create_task(handle_fn(scope, receive, bridging_send))

    # If the handler task dies (exception or cancellation) without sending the EOF
    # sentinel, body_iter() would block forever on body_queue.get().  The callback
    # below guarantees the queue gets unblocked regardless of how the task ends.
    # When this happens before response headers, propagate the original exception
    # instead of waiting for the header timeout.
    def _ensure_eof(task: asyncio.Task) -> None:
        if task.cancelled():
            body_queue.put_nowait(None)
            return

        task_exception: Final = task.exception()
        if task_exception is not None:
            if not headers_ready.done():
                headers_ready.set_exception(task_exception)
            body_queue.put_nowait(None)

    handler_task.add_done_callback(_ensure_eof)

    try:
        status, raw_headers = await asyncio.wait_for(asyncio.shield(headers_ready), timeout=30.0)
    except asyncio.TimeoutError:
        handler_task.cancel()
        raise HTTPException(status_code=504, detail="MCP handler did not respond in time")

    headers_dict: Final = {k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers}

    async def body_iter():
        try:
            while True:
                chunk = await body_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not handler_task.done():
                handler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await handler_task

    return StreamingResponse(
        body_iter(),
        status_code=status,
        headers=headers_dict,
        media_type=headers_dict.get("content-type"),
    )


########################################################
# MCP Server
########################################################


@app.api_route(
    BASE_MCP_ROUTE,
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def aggregate_mcp_route(request: Request):
    """Serve the aggregate MCP endpoint on the bare ``/mcp`` spelling: the
    ``/mcp`` mount cannot match its bare prefix, and the resulting 307 breaks
    MCP clients behind TLS-terminating proxies."""
    from litellm.proxy._experimental.mcp_server.utils import is_mcp_available

    if not is_mcp_available():
        raise HTTPException(status_code=404, detail="Not Found")

    from litellm.proxy._experimental.mcp_server.server import (
        handle_streamable_http_mcp,
    )

    scope = dict(request.scope)
    scope["_original_path"] = scope.get("path", "")
    scope["path"] = BASE_MCP_ROUTE
    return await _stream_mcp_asgi_response(handle_streamable_http_mcp, scope, request.receive)


# Toolset-namespaced MCP routes - handle /toolset/{toolset_name}/mcp
# Must be declared BEFORE /{mcp_server_name}/mcp to avoid being swallowed by the catchall.
@app.api_route(
    "/toolset/{toolset_name}/mcp",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def toolset_mcp_route(toolset_name: str, request: Request):
    """
    Namespace a toolset as its own MCP endpoint.

    Connecting to /toolset/<name>/mcp exposes exactly the tools defined in
    the toolset. Access is enforced: non-admin API keys must have the toolset
    listed in their object_permission.mcp_toolsets grant list, or the request
    will be rejected with a 403.
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.server import (
            _mcp_active_toolset_id,
            handle_streamable_http_mcp,
        )

        if prisma_client is None:
            raise HTTPException(status_code=503, detail="Database not available")

        toolset: Final = await global_mcp_server_manager.get_toolset_by_name_cached(prisma_client, toolset_name)
        if toolset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Toolset '{toolset_name}' not found",
            )

        scope: Final = dict(request.scope)
        scope["path"] = "/mcp"

        token: Final = _mcp_active_toolset_id.set(toolset.toolset_id)
        try:
            return await _stream_mcp_asgi_response(handle_streamable_http_mcp, scope, request.receive)
        finally:
            _mcp_active_toolset_id.reset(token)

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception("Error handling toolset MCP route for %s: %s", toolset_name, str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


async def _mcp_forward_as_path(path_segment: str, request: Request):
    """Rewrite path to /mcp/{path_segment} and stream the response."""
    from litellm.proxy._experimental.mcp_server.server import (
        handle_streamable_http_mcp,
    )

    scope: Final = dict(request.scope)
    # Preserve the public request path for OAuth challenge URL selection.
    scope["_original_path"] = scope.get("path", "")
    scope["path"] = f"/mcp/{path_segment}"
    return await _stream_mcp_asgi_response(handle_streamable_http_mcp, scope, request.receive)


async def _resolve_mcp_csv_tokens(csv_segment: str, client_ip: str | None) -> list[str]:
    """Validate a comma-separated ``/{name1,name2,...}/mcp`` segment.

    For each token, check (in order) whether it is a registered MCP server
    alias / name or an MCP access group tag (cached). Tokens are stripped,
    deduped (exact-match, keeping first occurrence in original order), and
    capped at ``DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS`` to bound the
    per-request DB / cache fan-out an authenticated caller can trigger by
    stuffing the path with tokens. Dedup is case-sensitive on purpose:
    downstream resolvers may treat names case-sensitively, so collapsing
    ``MyGroup`` and ``mygroup`` would risk dropping a valid distinct token.

    Toolset names are intentionally NOT resolved here — toolsets bind a single
    toolset id into request scope and have no defined semantics inside a
    comma-separated server list.

    Returns the subset of resolved tokens in original order. An empty list
    means the segment did not resolve to any known server / group; the caller
    should treat that as a 404 instead of forwarding it downstream (where an
    all-unmatched server filter falls back to the full ``allowed_mcp_servers``
    list and silently broadens the request scope).
    """
    from litellm.constants import DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    seen: Final[set] = set()
    deduped: Final[list[str]] = []
    for raw in csv_segment.split(","):
        token = raw.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
        if len(deduped) >= DEFAULT_MCP_NAMESPACE_CSV_MAX_TOKENS:
            break

    resolved: Final[list[str]] = []
    for token in deduped:
        if global_mcp_server_manager.get_mcp_server_by_name(token, client_ip=client_ip):
            resolved.append(token)
            continue
        if await _is_mcp_access_group_cached(token):
            resolved.append(token)
    return resolved


async def _is_mcp_access_group_cached(name: str) -> bool:
    """Return True if *name* is a known MCP access group tag.

    Positive results are cached for the configured management-object TTL
    (``get_management_object_ttl(user_api_key_cache)``). Negative results are
    cached for a short
    ``DEFAULT_MCP_ACCESS_GROUP_NEGATIVE_CACHE_TTL`` window so unauthenticated
    callers cannot force a fresh DB lookup per request for unknown names, while
    bounding staleness so a transient DB error (which surfaces as an empty
    list) cannot hide a real group for long.
    """
    from litellm.constants import DEFAULT_MCP_ACCESS_GROUP_NEGATIVE_CACHE_TTL
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    cache_key: Final = f"mcp_access_group_exists:{name}"
    cached: Final[object] = await user_api_key_cache.async_get_cache(key=cache_key)
    if cached is not None:
        return bool(cached)
    result: Final = bool(await MCPRequestHandler._get_mcp_servers_from_access_groups([name]))
    await user_api_key_cache.async_set_cache(
        key=cache_key,
        value=result,
        ttl=(get_management_object_ttl(user_api_key_cache) if result else DEFAULT_MCP_ACCESS_GROUP_NEGATIVE_CACHE_TTL),
    )
    return result


# Dynamic MCP server routes - handle /{mcp_server_name}/mcp
@app.api_route(
    "/{mcp_server_name}/mcp",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def dynamic_mcp_route(mcp_server_name: str, request: Request):
    """Handle /{name}/mcp for MCP server aliases, toolsets, MCP access group tags, and comma-separated lists.

    Resolution order:
    1. Registered MCP server alias / name
    2. Comma-separated list (short-circuits before any DB call)
    3. Toolset name (DB lookup, cached)
    4. MCP access group tag (DB lookup, cached)
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.auth.ip_address_utils import IPAddressUtils

        client_ip: Final = IPAddressUtils.get_mcp_client_ip(request)

        # 1. Registered MCP server alias
        if global_mcp_server_manager.get_mcp_server_by_name(mcp_server_name, client_ip=client_ip):
            return await _mcp_forward_as_path(mcp_server_name, request)

        # 2. Comma-separated list — validate every token resolves to a known
        # server alias or access group before forwarding. Bounds DB / cache
        # fan-out and prevents the downstream filter from silently falling back
        # to the full allowed_mcp_servers list when no token matches.
        if "," in mcp_server_name:
            resolved_tokens: Final = await _resolve_mcp_csv_tokens(mcp_server_name, client_ip)
            if not resolved_tokens:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No MCP server, toolset, or access group in '{mcp_server_name}' resolved to a known target"
                    ),
                )
            return await _mcp_forward_as_path(",".join(resolved_tokens), request)

        # 3. Toolset name (cached)
        if prisma_client is not None:
            from litellm.proxy._experimental.mcp_server.server import (
                _mcp_active_toolset_id,
                handle_streamable_http_mcp,
            )

            toolset: Final = await global_mcp_server_manager.get_toolset_by_name_cached(prisma_client, mcp_server_name)
            if toolset is not None:
                scope: Final = dict(request.scope)
                scope["_original_path"] = scope.get("path", "")
                scope["path"] = "/mcp"
                token: Final = _mcp_active_toolset_id.set(toolset.toolset_id)
                try:
                    return await _stream_mcp_asgi_response(handle_streamable_http_mcp, scope, request.receive)
                finally:
                    _mcp_active_toolset_id.reset(token)

        # 4. MCP access group tag (cached)
        if await _is_mcp_access_group_cached(mcp_server_name):
            return await _mcp_forward_as_path(mcp_server_name, request)

        raise HTTPException(
            status_code=404,
            detail=f"MCP server, toolset, or access group '{mcp_server_name}' not found",
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception("Error handling dynamic MCP route for %s: %s", mcp_server_name, str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
