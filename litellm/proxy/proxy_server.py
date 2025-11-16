import asyncio
import copy
import inspect
import io
import os
import random
import secrets
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from litellm._uuid import uuid
from litellm.constants import (
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_KEEPALIVE_TIMEOUT,
    AIOHTTP_TTL_DNS_CACHE,
    BASE_MCP_ROUTE,
    DEFAULT_MAX_RECURSE_DEPTH,
    DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
    DEFAULT_SHARED_HEALTH_CHECK_TTL,
    DEFAULT_SLACK_ALERTING_THRESHOLD,
    LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS,
    LITELLM_SETTINGS_SAFE_DB_OVERRIDES,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.common_utils.callback_utils import normalize_callback_names
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
    TokenCountResponse,
)
from litellm.utils import load_credentials_from_list

if TYPE_CHECKING:
    from aiohttp import ClientSession
    from opentelemetry.trace import Span as _Span

    from litellm.integrations.opentelemetry import OpenTelemetry

    Span = Union[_Span, Any]
else:
    Span = Any
    OpenTelemetry = Any


def showwarning(message, category, filename, lineno, file=None, line=None):
    traceback_info = f"{filename}:{lineno}: {category.__name__}: {message}\n"
    if file is not None:
        file.write(traceback_info)


warnings.showwarning = showwarning
warnings.filterwarnings("default", category=UserWarning)

# Your client code here


messages: list = []
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path - for litellm local dev

try:
    import logging

    import backoff
    import fastapi
    import orjson
    import yaml  # type: ignore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
except ImportError as e:
    raise ImportError(f"Missing dependency {e}. Run `pip install 'litellm[proxy]'`")

list_of_messages = [
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
    box_width = 60

    # Select a random message
    message = random.choice(list_of_messages)

    print()  # noqa
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")  # noqa
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")  # noqa
    print("\033[1;37m" + "# {:^59} #\033[0m".format(message))  # noqa
    print(  # noqa
        "\033[1;37m"
        + "# {:^59} #\033[0m".format("https://github.com/BerriAI/litellm/issues/new")
    )  # noqa
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")  # noqa
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")  # noqa
    print()  # noqa
    print(" Thank you for using LiteLLM! - Krrish & Ishaan")  # noqa
    print()  # noqa
    print()  # noqa
    print()  # noqa
    print(  # noqa
        "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"
    )  # noqa
    print()  # noqa
    print()  # noqa


from collections import defaultdict
from contextlib import asynccontextmanager

import litellm
from litellm import Router
from litellm._logging import verbose_proxy_logger, verbose_router_logger
from litellm.caching.caching import DualCache, RedisCache
from litellm.caching.redis_cluster_cache import RedisClusterCache
from litellm.constants import (
    APSCHEDULER_COALESCE,
    APSCHEDULER_MAX_INSTANCES,
    APSCHEDULER_MISFIRE_GRACE_TIME,
    APSCHEDULER_REPLACE_EXISTING,
    DAYS_IN_A_MONTH,
    DEFAULT_HEALTH_CHECK_INTERVAL,
    DEFAULT_MODEL_CREATED_AT_TIME,
    LITELLM_PROXY_ADMIN_NAME,
    PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS,
    PROXY_BATCH_POLLING_INTERVAL,
    PROXY_BATCH_WRITE_AT,
    PROXY_BUDGET_RESCHEDULER_MAX_TIME,
    PROXY_BUDGET_RESCHEDULER_MIN_TIME,
)
from litellm.exceptions import RejectedRequestError
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,
    get_litellm_metadata_from_kwargs,
)
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    router as mcp_discoverable_endpoints_router,
)
from litellm.proxy._experimental.mcp_server.rest_endpoints import (
    router as mcp_rest_endpoints_router,
)
from litellm.proxy._experimental.mcp_server.server import app as mcp_app
from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)
from litellm.proxy._types import *
from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
from litellm.proxy.agent_endpoints.endpoints import router as agent_endpoints_router
from litellm.proxy.analytics_endpoints.analytics_endpoints import (
    router as analytics_router,
)
from litellm.proxy.anthropic_endpoints.endpoints import router as anthropic_router
from litellm.proxy.auth.auth_checks import (
    ExperimentalUIJWTToken,
    get_team_object,
    log_db_metrics,
)
from litellm.proxy.auth.auth_utils import check_response_size_is_safe
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.auth.model_checks import (
    get_all_fallbacks,
    get_complete_model_list,
    get_key_models,
    get_mcp_server_ids,
    get_team_models,
)
from litellm.proxy.auth.user_api_key_auth import (
    user_api_key_auth,
    user_api_key_auth_websocket,
)
from litellm.proxy.batches_endpoints.endpoints import router as batches_router

## Import All Misc routes here ##
from litellm.proxy.caching_routes import router as caching_router
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_streaming_response,
)
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy
from litellm.proxy.common_utils.debug_utils import init_verbose_loggers
from litellm.proxy.common_utils.debug_utils import router as debugging_endpoints_router
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.html_forms.ui_login import html_form
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    check_file_size_under_limit,
    get_form_data,
)
from litellm.proxy.common_utils.load_config_utils import (
    get_config_file_contents_from_gcs,
    get_file_contents_from_s3,
)
from litellm.proxy.common_utils.openai_endpoint_utils import (
    remove_sensitive_info_from_deployment,
)
from litellm.proxy.common_utils.proxy_state import ProxyState
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.proxy.common_utils.swagger_utils import ERROR_RESPONSES
from litellm.proxy.container_endpoints.endpoints import router as container_router
from litellm.proxy.credential_endpoints.endpoints import router as credential_router
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.discovery_endpoints import ui_discovery_endpoints_router
from litellm.proxy.fine_tuning_endpoints.endpoints import router as fine_tuning_router
from litellm.proxy.fine_tuning_endpoints.endpoints import set_fine_tuning_config
from litellm.proxy.google_endpoints.endpoints import router as google_router
from litellm.proxy.guardrails.guardrail_endpoints import router as guardrails_router
from litellm.proxy.guardrails.init_guardrails import (
    init_guardrails_v2,
    initialize_guardrails,
)
from litellm.proxy.health_check import perform_health_check
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
from litellm.proxy.management_endpoints.budget_management_endpoints import (
    router as budget_management_router,
)
from litellm.proxy.management_endpoints.cache_settings_endpoints import (
    router as cache_settings_router,
)
from litellm.proxy.management_endpoints.callback_management_endpoints import (
    router as callback_management_endpoints_router,
)
from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
from litellm.proxy.management_endpoints.cost_tracking_settings import (
    router as cost_tracking_settings_router,
)
from litellm.proxy.management_endpoints.customer_endpoints import (
    router as customer_router,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    router as internal_user_router,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import user_update
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_verification_tokens,
    duration_in_seconds,
    generate_key_helper_fn,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    router as key_management_router,
)
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    router as mcp_management_router,
)
from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    router as model_access_group_management_router,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _add_model_to_db,
    _add_team_model_to_db,
    _deduplicate_litellm_router_models,
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
from litellm.proxy.management_endpoints.scim.scim_v2 import scim_router
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
from litellm.proxy.management_helpers.audit_logs import create_audit_log_for_update
from litellm.proxy.middleware.prometheus_auth_middleware import PrometheusAuthMiddleware
from litellm.proxy.ocr_endpoints.endpoints import router as ocr_router
from litellm.proxy.openai_files_endpoints.files_endpoints import (
    router as openai_files_router,
)
from litellm.proxy.openai_files_endpoints.files_endpoints import set_files_config
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    passthrough_endpoint_router,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    router as llm_passthrough_router,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    vertex_ai_live_websocket_passthrough,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    initialize_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    router as pass_through_router,
)
from litellm.proxy.prompts.prompt_endpoints import router as prompts_router
from litellm.proxy.public_endpoints import router as public_endpoints_router
from litellm.proxy.rerank_endpoints.endpoints import router as rerank_router
from litellm.proxy.response_api_endpoints.endpoints import router as response_router
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.search_endpoints.endpoints import router as search_router
from litellm.proxy.search_endpoints.search_tool_management import (
    router as search_tool_management_router,
)
from litellm.proxy.spend_tracking.cloudzero_endpoints import router as cloudzero_router
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    router as spend_management_router,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
    router as ui_crud_endpoints_router,
)
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    ProxyUpdateSpend,
    _cache_user_row,
    _get_docs_url,
    _get_projected_spend_over_limit,
    _get_redoc_url,
    _is_projected_spend_over_limit,
    _is_valid_team_configs,
    get_custom_url,
    get_error_message_str,
    get_server_root_path,
    handle_exception_on_proxy,
    hash_token,
    update_spend,
)
from litellm.proxy.vector_store_endpoints.endpoints import router as vector_store_router
from litellm.proxy.vector_store_files_endpoints.endpoints import (
    router as vector_store_files_router,
)
from litellm.proxy.vertex_ai_endpoints.langfuse_endpoints import (
    router as langfuse_router,
)
from litellm.proxy.video_endpoints.endpoints import router as video_router
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
    str_to_bool,
)
from litellm.types.integrations.slack_alerting import SlackAlertingArgs
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    AnthropicResponse,
    AnthropicResponseContentBlockText,
    AnthropicResponseUsageBlock,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)
from litellm.types.proxy.management_endpoints.ui_sso import (
    DefaultTeamSSOParams,
    LiteLLM_UpperboundKeyGenerateParams,
)
from litellm.types.realtime import RealtimeQueryParams
from litellm.types.router import DeploymentTypedDict
from litellm.types.router import ModelInfo as RouterModelInfo
from litellm.types.router import (
    RouterGeneralSettings,
    SearchToolTypedDict,
    updateDeployment,
)
from litellm.types.scheduler import DefaultPriorities
from litellm.types.secret_managers.main import (
    KeyManagementSettings,
    KeyManagementSystem,
)
from litellm.types.utils import CredentialItem, CustomHuggingfaceTokenizer
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.types.utils import RawRequestTypedDict, StandardLoggingPayload
from litellm.utils import _add_custom_logger_callback_to_specific_event

try:
    from litellm._version import version
except Exception:
    version = "0.0.0"
litellm.suppress_debug_info = True
import json
from typing import Union

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

from litellm.types.agents import AgentConfig

# import enterprise folder
enterprise_router = APIRouter()
try:
    # when using litellm cli
    import litellm.proxy.enterprise as enterprise
except Exception:
    # when using litellm docker image
    try:
        import enterprise  # type: ignore
    except Exception:
        pass

###################
# Import enterprise routes
try:
    from litellm_enterprise.proxy.enterprise_routes import router as _enterprise_router
    from litellm_enterprise.proxy.proxy_server import EnterpriseProxyConfig

    enterprise_router = _enterprise_router
    enterprise_proxy_config: Optional[EnterpriseProxyConfig] = EnterpriseProxyConfig()
except ImportError:
    enterprise_proxy_config = None
###################

server_root_path = os.getenv("SERVER_ROOT_PATH", "")
_license_check = LicenseCheck()
premium_user: bool = _license_check.is_premium()
premium_user_data: Optional["EnterpriseLicenseData"] = (
    _license_check.airgapped_license_data
)
global_max_parallel_request_retries_env: Optional[str] = os.getenv(
    "LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRIES"
)
proxy_state = ProxyState()
SENSITIVE_DATA_MASKER = SensitiveDataMasker()
if global_max_parallel_request_retries_env is None:
    global_max_parallel_request_retries: int = 3
else:
    global_max_parallel_request_retries = int(global_max_parallel_request_retries_env)

global_max_parallel_request_retry_timeout_env: Optional[str] = os.getenv(
    "LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRY_TIMEOUT"
)
if global_max_parallel_request_retry_timeout_env is None:
    global_max_parallel_request_retry_timeout: float = 60.0
else:
    global_max_parallel_request_retry_timeout = float(
        global_max_parallel_request_retry_timeout_env
    )

ui_link = f"{server_root_path}/ui/"
model_hub_link = f"{server_root_path}/ui/model_hub_table"
ui_message = (
    f"👉 [```LiteLLM Admin Panel on /ui```]({ui_link}). Create, Edit Keys with SSO"
)
ui_message += "\n\n💸 [```LiteLLM Model Cost Map```](https://models.litellm.ai/)."

ui_message += f"\n\n🔎 [```LiteLLM Model Hub```]({model_hub_link}). See available models on the proxy. [**Docs**](https://docs.litellm.ai/docs/proxy/model_hub)"

custom_swagger_message = "[**Customize Swagger Docs**](https://docs.litellm.ai/docs/proxy/enterprise#swagger-docs---custom-routes--branding)"

### CUSTOM BRANDING [ENTERPRISE FEATURE] ###
_title = os.getenv("DOCS_TITLE", "LiteLLM API") if premium_user else "LiteLLM API"
_description = (
    os.getenv(
        "DOCS_DESCRIPTION",
        f"Enterprise Edition \n\nProxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}",
    )
    if premium_user
    else f"Proxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}"
)


def cleanup_router_config_variables():
    global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, user_custom_key_generate, user_custom_sso, user_custom_ui_sso_sign_in_handler, use_background_health_checks, use_shared_health_check, health_check_interval, prisma_client

    # Set all variables to None
    master_key = None
    user_config_file_path = None
    otel_logging = None
    user_custom_auth = None
    user_custom_auth_path = None
    user_custom_key_generate = None
    user_custom_sso = None
    user_custom_ui_sso_sign_in_handler = None
    use_background_health_checks = None
    use_shared_health_check = None
    health_check_interval = None
    prisma_client = None


async def proxy_shutdown_event():
    global prisma_client, master_key, user_custom_auth, user_custom_key_generate
    verbose_proxy_logger.info("Shutting down LiteLLM Proxy Server")
    if prisma_client:
        verbose_proxy_logger.debug("Disconnecting from Prisma")
        await prisma_client.disconnect()

    if litellm.cache is not None:
        await litellm.cache.disconnect()

    await jwt_handler.close()

    if db_writer_client is not None:
        await db_writer_client.close()

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


async def _initialize_shared_aiohttp_session():
    """Initialize shared aiohttp session for connection reuse."""
    try:
        from aiohttp import ClientSession, TCPConnector

        # Create connector with connection pooling settings optimized for long-lived connections
        connector = TCPConnector(
            limit=AIOHTTP_CONNECTOR_LIMIT,
            keepalive_timeout=AIOHTTP_KEEPALIVE_TIMEOUT,
            ttl_dns_cache=AIOHTTP_TTL_DNS_CACHE,
            enable_cleanup_closed=True,
        )

        session = ClientSession(connector=connector)
        verbose_proxy_logger.info(
            f"SESSION REUSE: Created shared aiohttp session for connection pooling (ID: {id(session)})"
        )
        return session
    except Exception as e:
        verbose_proxy_logger.warning(
            f"Failed to create shared aiohttp session: {e}. Continuing without session reuse."
        )
        return None


@asynccontextmanager
async def proxy_startup_event(app: FastAPI):
    global prisma_client, master_key, use_background_health_checks, llm_router, llm_model_list, general_settings, proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time, litellm_proxy_admin_name, db_writer_client, store_model_in_db, premium_user, _license_check, proxy_batch_polling_interval, shared_aiohttp_session
    import json

    init_verbose_loggers()
    ## CHECK PREMIUM USER
    verbose_proxy_logger.debug(
        "litellm.proxy.proxy_server.py::startup() - CHECKING PREMIUM USER - {}".format(
            premium_user
        )
    )
    if premium_user is False:
        premium_user = _license_check.is_premium()

    ## CHECK MASTER KEY IN ENVIRONMENT ##
    master_key = get_secret_str("LITELLM_MASTER_KEY")
    ### LOAD CONFIG ###
    worker_config: Optional[Union[str, dict]] = get_secret("WORKER_CONFIG")  # type: ignore
    env_config_yaml: Optional[str] = get_secret_str("CONFIG_FILE_PATH")
    verbose_proxy_logger.debug("worker_config: %s", worker_config)
    # check if it's a valid file path
    if env_config_yaml is not None:
        if os.path.isfile(env_config_yaml) and proxy_config.is_yaml(
            config_file_path=env_config_yaml
        ):
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(
                router=llm_router, config_file_path=env_config_yaml
            )
    elif worker_config is not None:
        if (
            isinstance(worker_config, str)
            and os.path.isfile(worker_config)
            and proxy_config.is_yaml(config_file_path=worker_config)
        ):
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(
                router=llm_router, config_file_path=worker_config
            )
        elif os.environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None and isinstance(
            worker_config, str
        ):
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(
                router=llm_router, config_file_path=worker_config
            )
        elif isinstance(worker_config, dict):
            await initialize(**worker_config)
        else:
            # if not, assume it's a json string
            worker_config = json.loads(worker_config)
            if isinstance(worker_config, dict):
                await initialize(**worker_config)

    # check if DATABASE_URL in environment - load from there
    if prisma_client is None:
        _db_url: Optional[str] = get_secret("DATABASE_URL", None)  # type: ignore
        prisma_client = await ProxyStartupEvent._setup_prisma_client(
            database_url=_db_url,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_cache=user_api_key_cache,
        )

    ProxyStartupEvent._initialize_startup_logging(
        llm_router=llm_router,
        proxy_logging_obj=proxy_logging_obj,
        redis_usage_cache=redis_usage_cache,
    )

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
        ProxyStartupEvent._add_proxy_budget_to_db(
            litellm_proxy_budget_name=litellm_proxy_admin_name
        )

    ### START BATCH WRITING DB + CHECKING NEW MODELS###
    if prisma_client is not None:
        await ProxyStartupEvent.initialize_scheduled_background_jobs(
            general_settings=general_settings,
            prisma_client=prisma_client,
            proxy_budget_rescheduler_min_time=proxy_budget_rescheduler_min_time,
            proxy_budget_rescheduler_max_time=proxy_budget_rescheduler_max_time,
            proxy_batch_write_at=proxy_batch_write_at,
            proxy_logging_obj=proxy_logging_obj,
        )

        await ProxyStartupEvent._update_default_team_member_budget()

    # Start background health checks AFTER models are loaded and index is built
    if use_background_health_checks:
        asyncio.create_task(
            _run_background_health_check()
        )  # start the background health check coroutine.

    ## [Optional] Initialize dd tracer
    ProxyStartupEvent._init_dd_tracer()

    ## Initialize shared aiohttp session for connection reuse
    shared_aiohttp_session = await _initialize_shared_aiohttp_session()

    # End of startup event
    yield

    # Shutdown event - close shared aiohttp session
    if shared_aiohttp_session is not None:
        try:
            await shared_aiohttp_session.close()
            verbose_proxy_logger.info("SESSION REUSE: Closed shared aiohttp session")
        except Exception as e:
            verbose_proxy_logger.error(f"Error closing shared aiohttp session: {e}")

    await proxy_shutdown_event()


app = FastAPI(
    docs_url=_get_docs_url(),
    redoc_url=_get_redoc_url(),
    title=_title,
    description=_description,
    version=version,
    root_path=server_root_path,  # check if user passed root path, FastAPI defaults this value to ""
    lifespan=proxy_startup_event,
)

vertex_live_passthrough_vertex_base = VertexBase()


### CUSTOM API DOCS [ENTERPRISE FEATURE] ###
# Custom OpenAPI schema generator to include only selected routes
from fastapi.routing import APIWebSocketRoute


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
    websocket_routes = [
        route for route in app.routes if isinstance(route, APIWebSocketRoute)
    ]

    # Add each WebSocket route to the schema
    for route in websocket_routes:
        # Get the base path without query parameters
        base_path = route.path.split("{")[0].rstrip("?")

        # Extract parameters from the route
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
                                "schema": {
                                    "type": "string"
                                },  # You can make this more specific if needed
                            }
                        )
        except (AttributeError, TypeError):
            # If we can't access query_params, continue without them
            pass

        openapi_schema["paths"][base_path] = {
            "get": {
                "summary": f"WebSocket: {route.name or base_path}",
                "description": "WebSocket connection endpoint",
                "operationId": f"websocket_{route.name or base_path.replace('/', '_')}",
                "parameters": parameters,
                "responses": {"101": {"description": "WebSocket Protocol Switched"}},
                "tags": ["WebSocket"],
            }
        }

    # Add LLM API request schema bodies for documentation
    from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec

    openapi_schema = CustomOpenAPISpec.add_llm_api_request_schema_body(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi_schema()

    # Filter routes to include only specific ones
    openai_routes = LiteLLMRoutes.openai_routes.value
    paths_to_include: dict = {}
    for route in openai_routes:
        if route in openapi_schema["paths"]:
            paths_to_include[route] = openapi_schema["paths"][route]
    openapi_schema["paths"] = paths_to_include

    # Add LLM API request schema bodies for documentation
    from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec

    openapi_schema = CustomOpenAPISpec.add_llm_api_request_schema_body(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema


if os.getenv("DOCS_FILTERED", "False") == "True" and premium_user:
    app.openapi = custom_openapi  # type: ignore
else:
    # For regular users, use get_openapi_schema to include LLM API schemas
    app.openapi = get_openapi_schema  # type: ignore


class UserAPIKeyCacheTTLEnum(enum.Enum):
    in_memory_cache_ttl = 60  # 1 min ttl ## configure via `general_settings::user_api_key_cache_ttl: <your-value>`


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    # NOTE: DO NOT MODIFY THIS, its crucial to map to Openai exceptions
    headers = exc.headers
    error_dict = exc.to_dict()
    return JSONResponse(
        status_code=(
            int(exc.code) if exc.code else status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content={"error": error_dict},
        headers=headers,
    )


router = APIRouter()
origins = ["*"]


# get current directory
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(current_dir, "_experimental", "out")
    litellm_asset_prefix = "/litellm-asset-prefix"

    # For non-root Docker, use the pre-built UI from /tmp/litellm_ui
    # Support both "true" and "True" for case-insensitive comparison
    if os.getenv("LITELLM_NON_ROOT", "").lower() == "true":
        non_root_ui_path = "/tmp/litellm_ui"

        # Check if the UI was built and exists at the expected location
        if os.path.exists(non_root_ui_path) and os.listdir(non_root_ui_path):
            verbose_proxy_logger.info(
                f"Using pre-built UI for non-root Docker: {non_root_ui_path}"
            )
            verbose_proxy_logger.info(
                f"UI files found: {len(os.listdir(non_root_ui_path))} items"
            )
            ui_path = non_root_ui_path
        else:
            verbose_proxy_logger.error(
                f"UI not found at {non_root_ui_path}. UI will not be available."
            )
            verbose_proxy_logger.error(
                f"Path exists: {os.path.exists(non_root_ui_path)}, Has content: {os.path.exists(non_root_ui_path) and bool(os.listdir(non_root_ui_path))}"
            )

    # Only modify files if a custom server root path is set
    if server_root_path and server_root_path != "/":
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
                except UnicodeDecodeError:
                    # Skip binary files that can't be decoded
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

    # Handle HTML file restructuring
    # Skip this for non-root Docker since it's done at build time
    # Support both "true" and "True" for case-insensitive comparison
    if os.getenv("LITELLM_NON_ROOT", "").lower() != "true":
        for filename in os.listdir(ui_path):
            if filename.endswith(".html") and filename != "index.html":
                # Create a folder with the same name as the HTML file
                folder_name = os.path.splitext(filename)[0]
                folder_path = os.path.join(ui_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Move the HTML file into the folder and rename it to 'index.html'
                src = os.path.join(ui_path, filename)
                dst = os.path.join(folder_path, "index.html")
                os.rename(src, dst)
    else:
        verbose_proxy_logger.info(
            "Skipping runtime HTML restructuring for non-root Docker (already done at build time)"
        )

except Exception:
    pass
current_dir = os.path.dirname(os.path.abspath(__file__))
# ui_path = os.path.join(current_dir, "_experimental", "out")
# # Mount this test directory instead
# app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(PrometheusAuthMiddleware)


def mount_swagger_ui():
    swagger_directory = os.path.join(current_dir, "swagger")
    swagger_path = "/" if server_root_path is None else server_root_path
    if not swagger_path.endswith("/"):
        swagger_path = swagger_path + "/"
    custom_root_path_swagger_path = swagger_path + "swagger"

    app.mount("/swagger", StaticFiles(directory=swagger_directory), name="swagger")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_js_url=f"{custom_root_path_swagger_path}/swagger-ui-bundle.js",
            swagger_css_url=f"{custom_root_path_swagger_path}/swagger-ui.css",
            swagger_favicon_url=f"{custom_root_path_swagger_path}/favicon.png",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


mount_swagger_ui()

from typing import Dict

user_api_base = None
user_model = None
user_debug = False
user_max_tokens = None
user_request_timeout = None
user_temperature = None
user_telemetry = True
user_config = None
user_headers = None
user_config_file_path: Optional[str] = None
local_logging = True  # writes logs to a local api_log.json file for debugging
experimental = False
#### GLOBAL VARIABLES ####
llm_router: Optional[Router] = None
llm_model_list: Optional[list] = None
general_settings: dict = {}
config_passthrough_endpoints: Optional[List[Dict[str, Any]]] = None
callback_settings: dict = {}
log_file = "api_log.json"
worker_config = None
master_key: Optional[str] = None
config_agents: Optional[List[AgentConfig]] = None
otel_logging = False
prisma_client: Optional[PrismaClient] = None
shared_aiohttp_session: Optional["ClientSession"] = (
    None  # Global shared session for connection reuse
)
user_api_key_cache = DualCache(
    default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value
)
model_max_budget_limiter = _PROXY_VirtualKeyModelMaxBudgetLimiter(
    dual_cache=user_api_key_cache
)
litellm.logging_callback_manager.add_litellm_callback(model_max_budget_limiter)
redis_usage_cache: Optional[RedisCache] = (
    None  # redis cache used for tracking spend, tpm/rpm limits
)
user_custom_auth = None
user_custom_key_generate = None
user_custom_sso = None
user_custom_ui_sso_sign_in_handler = None
use_background_health_checks = None
use_shared_health_check = None
use_queue = False
health_check_interval = None
health_check_details = None
health_check_results: Dict[str, Union[int, List[Dict[str, Any]]]] = {}
queue: List = []
litellm_proxy_budget_name = "litellm-proxy-budget"
litellm_proxy_admin_name = LITELLM_PROXY_ADMIN_NAME
ui_access_mode: Union[Literal["admin", "all"], Dict] = "all"
proxy_budget_rescheduler_min_time = PROXY_BUDGET_RESCHEDULER_MIN_TIME
proxy_budget_rescheduler_max_time = PROXY_BUDGET_RESCHEDULER_MAX_TIME
proxy_batch_polling_interval = PROXY_BATCH_POLLING_INTERVAL
proxy_batch_write_at = PROXY_BATCH_WRITE_AT
litellm_master_key_hash = None
disable_spend_logs = False
jwt_handler = JWTHandler()
prompt_injection_detection_obj: Optional[_OPTIONAL_PromptInjectionDetection] = None
store_model_in_db: bool = False
open_telemetry_logger: Optional[OpenTelemetry] = None
### INITIALIZE GLOBAL LOGGING OBJECT ###
proxy_logging_obj = ProxyLogging(
    user_api_key_cache=user_api_key_cache, premium_user=premium_user
)
### REDIS QUEUE ###
async_result = None
celery_app_conn = None
celery_fn = None  # Redis Queue for handling requests

# Global variables for model cost map reload scheduling
scheduler = None
last_model_cost_map_reload = None


### DB WRITER ###
db_writer_client: Optional[AsyncHTTPHandler] = None
### logger ###


async def check_request_disconnection(request: Request, llm_api_call_task):
    """
    Asynchronously checks if the request is disconnected at regular intervals.
    If the request is disconnected
    - cancel the litellm.router task
    - raises an HTTPException with status code 499 and detail "Client disconnected the request".

    Parameters:
    - request: Request: The request object to check for disconnection.
    Returns:
    - None
    """

    # only run this function for 10 mins -> if these don't get cancelled -> we don't want the server to have many while loops
    start_time = time.time()
    while time.time() - start_time < 600:
        await asyncio.sleep(1)
        if await request.is_disconnected():
            # cancel the LLM API Call task if any passed - this is passed from individual providers
            # Example OpenAI, Azure, VertexAI etc
            llm_api_call_task.cancel()

            raise HTTPException(
                status_code=499,
                detail="Client disconnected the request",
            )


def _resolve_typed_dict_type(typ):
    """Resolve the actual TypedDict class from a potentially wrapped type."""
    from typing_extensions import _TypedDictMeta  # type: ignore

    origin = get_origin(typ)
    if origin is Union:  # Check if it's a Union (like Optional)
        for arg in get_args(typ):
            if isinstance(arg, _TypedDictMeta):
                return arg
    elif isinstance(typ, type) and isinstance(typ, dict):
        return typ
    return None


def _resolve_pydantic_type(typ) -> List:
    """Resolve the actual TypedDict class from a potentially wrapped type."""
    origin = get_origin(typ)
    typs = []
    if origin is Union:  # Check if it's a Union (like Optional)
        for arg in get_args(typ):
            if (
                arg is not None
                and not isinstance(arg, type(None))
                and "NoneType" not in str(arg)
            ):
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
        KVUri = os.getenv("AZURE_KEY_VAULT_URI", None)

        if KVUri is None:
            raise Exception(
                "Error when loading keys from Azure Key Vault: AZURE_KEY_VAULT_URI is not set."
            )

        credential = DefaultAzureCredential()

        # Create the SecretClient using the credential
        client = SecretClient(vault_url=KVUri, credential=credential)

        litellm.secret_manager_client = client
        litellm._key_management_system = KeyManagementSystem.AZURE_KEY_VAULT
    except Exception as e:
        _error_str = str(e)
        verbose_proxy_logger.exception(
            "Error when loading keys from Azure Key Vault: %s .Ensure you run `pip install azure-identity azure-keyvault-secrets`",
            _error_str,
        )


def cost_tracking():
    global prisma_client
    if prisma_client is not None:
        litellm.logging_callback_manager.add_litellm_callback(_ProxyDBLogger())


async def update_cache(  # noqa: PLR0915
    token: Optional[str],
    user_id: Optional[str],
    end_user_id: Optional[str],
    team_id: Optional[str],
    response_cost: Optional[float],
    parent_otel_span: Optional[Span],  # type: ignore
    tags: Optional[List[str]] = None,
):
    """
    Use this to update the cache with new user spend.

    Put any alerting logic in here.
    """

    values_to_update_in_cache: List[Tuple[Any, Any]] = []

    ### UPDATE KEY SPEND ###
    async def _update_key_cache(token: str, response_cost: float):
        # Fetch the existing cost for the given token
        if isinstance(token, str) and token.startswith("sk-"):
            hashed_token = hash_token(token=token)
        else:
            hashed_token = token
        verbose_proxy_logger.debug("_update_key_cache: hashed_token=%s", hashed_token)
        existing_spend_obj: LiteLLM_VerificationTokenView = await user_api_key_cache.async_get_cache(key=hashed_token)  # type: ignore
        verbose_proxy_logger.debug(
            f"_update_key_cache: existing_spend_obj={existing_spend_obj}"
        )
        if existing_spend_obj is None:
            return
        else:
            existing_spend = existing_spend_obj.spend
        # Calculate the new cost by adding the existing cost and response_cost
        new_spend = existing_spend + response_cost

        ## CHECK IF USER PROJECTED SPEND > SOFT LIMIT
        if (
            existing_spend_obj.soft_budget_cooldown is False
            and existing_spend_obj.litellm_budget_table is not None
            and (
                _is_projected_spend_over_limit(
                    current_spend=new_spend,
                    soft_budget_limit=existing_spend_obj.litellm_budget_table[
                        "soft_budget"
                    ],
                )
                is True
            )
        ):
            projected_spend, projected_exceeded_date = _get_projected_spend_over_limit(
                current_spend=new_spend,
                soft_budget_limit=existing_spend_obj.litellm_budget_table.get(
                    "soft_budget", None
                ),
            )  # type: ignore
            soft_limit = existing_spend_obj.litellm_budget_table.get(
                "soft_budget", float("inf")
            )
            call_info = CallInfo(
                token=existing_spend_obj.token or "",
                spend=new_spend,
                key_alias=existing_spend_obj.key_alias,
                max_budget=soft_limit,
                user_id=existing_spend_obj.user_id,
                projected_spend=projected_spend,
                projected_exceeded_date=projected_exceeded_date,
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

        if (
            existing_spend_obj is not None
            and getattr(existing_spend_obj, "team_spend", None) is not None
        ):
            existing_team_spend = existing_spend_obj.team_spend or 0
            # Calculate the new cost by adding the existing cost and response_cost
            existing_spend_obj.team_spend = existing_team_spend + response_cost

        if (
            existing_spend_obj is not None
            and getattr(existing_spend_obj, "team_member_spend", None) is not None
        ):
            existing_team_member_spend = existing_spend_obj.team_member_spend or 0
            # Calculate the new cost by adding the existing cost and response_cost
            existing_spend_obj.team_member_spend = (
                existing_team_member_spend + response_cost
            )

        # Update the cost column for the given token
        existing_spend_obj.spend = new_spend
        values_to_update_in_cache.append((hashed_token, existing_spend_obj))

    ### UPDATE USER SPEND ###
    async def _update_user_cache():
        ## UPDATE CACHE FOR USER ID + GLOBAL PROXY
        user_ids = [user_id]
        try:
            for _id in user_ids:
                # Fetch the existing cost for the given user
                if _id is None:
                    continue
                existing_spend_obj = await user_api_key_cache.async_get_cache(key=_id)
                if existing_spend_obj is None:
                    # do nothing if there is no cache value
                    return
                verbose_proxy_logger.debug(
                    f"_update_user_db: existing spend: {existing_spend_obj}; response_cost: {response_cost}"
                )

                if isinstance(existing_spend_obj, dict):
                    existing_spend = existing_spend_obj["spend"]
                else:
                    existing_spend = existing_spend_obj.spend
                # Calculate the new cost by adding the existing cost and response_cost
                new_spend = existing_spend + response_cost

                # Update the cost column for the given user
                if isinstance(existing_spend_obj, dict):
                    existing_spend_obj["spend"] = new_spend
                    values_to_update_in_cache.append((_id, existing_spend_obj))
                else:
                    existing_spend_obj.spend = new_spend
                    values_to_update_in_cache.append((_id, existing_spend_obj.json()))
            ## UPDATE GLOBAL PROXY ##
            global_proxy_spend = await user_api_key_cache.async_get_cache(
                key="{}:spend".format(litellm_proxy_admin_name)
            )
            if global_proxy_spend is None:
                # do nothing if not in cache
                return
            elif response_cost is not None and global_proxy_spend is not None:
                increment = global_proxy_spend + response_cost
                values_to_update_in_cache.append(
                    ("{}:spend".format(litellm_proxy_admin_name), increment)
                )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"An error occurred updating user cache: {str(e)}\n\n{traceback.format_exc()}"
            )

    ### UPDATE END-USER SPEND ###
    async def _update_end_user_cache():
        if end_user_id is None or response_cost is None:
            return

        _id = "end_user_id:{}".format(end_user_id)
        try:
            # Fetch the existing cost for the given user
            existing_spend_obj = await user_api_key_cache.async_get_cache(key=_id)
            if existing_spend_obj is None:
                # if user does not exist in LiteLLM_UserTable, create a new user
                # do nothing if end-user not in api key cache
                return
            verbose_proxy_logger.debug(
                f"_update_end_user_db: existing spend: {existing_spend_obj}; response_cost: {response_cost}"
            )
            if existing_spend_obj is None:
                existing_spend = 0
            else:
                if isinstance(existing_spend_obj, dict):
                    existing_spend = existing_spend_obj["spend"]
                else:
                    existing_spend = existing_spend_obj.spend
            # Calculate the new cost by adding the existing cost and response_cost
            new_spend = existing_spend + response_cost

            # Update the cost column for the given user
            if isinstance(existing_spend_obj, dict):
                existing_spend_obj["spend"] = new_spend
                values_to_update_in_cache.append((_id, existing_spend_obj))
            else:
                existing_spend_obj.spend = new_spend
                values_to_update_in_cache.append((_id, existing_spend_obj.json()))
        except Exception as e:
            verbose_proxy_logger.exception(
                f"An error occurred updating end user cache: {str(e)}"
            )

    ### UPDATE TEAM SPEND ###
    async def _update_team_cache():
        if team_id is None or response_cost is None:
            return

        _id = "team_id:{}".format(team_id)
        try:
            # Fetch the existing cost for the given user
            existing_spend_obj: Optional[LiteLLM_TeamTable] = (
                await user_api_key_cache.async_get_cache(key=_id)
            )
            if existing_spend_obj is None:
                # do nothing if team not in api key cache
                return
            verbose_proxy_logger.debug(
                f"_update_team_db: existing spend: {existing_spend_obj}; response_cost: {response_cost}"
            )
            if existing_spend_obj is None:
                existing_spend: Optional[float] = 0.0
            else:
                if isinstance(existing_spend_obj, dict):
                    existing_spend = existing_spend_obj["spend"]
                else:
                    existing_spend = existing_spend_obj.spend

            if existing_spend is None:
                existing_spend = 0.0
            # Calculate the new cost by adding the existing cost and response_cost
            new_spend = existing_spend + response_cost

            # Update the cost column for the given user
            if isinstance(existing_spend_obj, dict):
                existing_spend_obj["spend"] = new_spend
                values_to_update_in_cache.append((_id, existing_spend_obj))
            else:
                existing_spend_obj.spend = new_spend
                values_to_update_in_cache.append((_id, existing_spend_obj))
        except Exception as e:
            verbose_proxy_logger.exception(
                f"An error occurred updating end user cache: {str(e)}"
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

                cache_key = f"tag:{tag_name}"
                # Fetch the existing tag object from cache
                existing_tag_obj = await user_api_key_cache.async_get_cache(
                    key=cache_key
                )
                if existing_tag_obj is None:
                    # do nothing if tag not in api key cache
                    continue

                verbose_proxy_logger.debug(
                    f"_update_tag_cache: existing spend for tag={tag_name}: {existing_tag_obj}; response_cost: {response_cost}"
                )

                if isinstance(existing_tag_obj, dict):
                    existing_spend = existing_tag_obj.get("spend", 0) or 0
                else:
                    existing_spend = getattr(existing_tag_obj, "spend", 0) or 0

                # Calculate the new cost by adding the existing cost and response_cost
                new_spend = existing_spend + response_cost

                # Update the spend column for the given tag
                if isinstance(existing_tag_obj, dict):
                    existing_tag_obj["spend"] = new_spend
                    values_to_update_in_cache.append((cache_key, existing_tag_obj))
                else:
                    existing_tag_obj.spend = new_spend
                    values_to_update_in_cache.append((cache_key, existing_tag_obj))
        except Exception as e:
            verbose_proxy_logger.exception(
                f"An error occurred updating tag cache: {str(e)}"
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

    asyncio.create_task(
        user_api_key_cache.async_set_cache_pipeline(
            cache_list=values_to_update_in_cache,
            ttl=60,
            litellm_parent_otel_span=parent_otel_span,
        )
    )


def run_ollama_serve():
    try:
        command = ["ollama", "serve"]

        with open(os.devnull, "w") as devnull:
            subprocess.Popen(command, stdout=devnull, stderr=devnull)
    except Exception as e:
        verbose_proxy_logger.debug(
            f"""
            LiteLLM Warning: proxy started with `ollama` model\n`ollama serve` failed with Exception{e}. \nEnsure you run `ollama serve`
        """
        )


async def _run_background_health_check():
    """
    Periodically run health checks in the background on the endpoints.

    Update health_check_results, based on this.
    Uses shared health check state when Redis is available to coordinate across pods.
    """
    global health_check_results, llm_model_list, health_check_interval, health_check_details, use_shared_health_check, redis_usage_cache

    if (
        health_check_interval is None
        or not isinstance(health_check_interval, int)
        or health_check_interval <= 0
    ):
        return

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
        # make 1 deep copy of llm_model_list on every health check iteration
        _llm_model_list = copy.deepcopy(llm_model_list) or []

        # filter out models that have disabled background health checks
        _llm_model_list = [
            m
            for m in _llm_model_list
            if not m.get("model_info", {}).get("disable_background_health_check", False)
        ]

        # Use shared health check if available, otherwise fall back to direct health check
        # Convert health_check_details to bool for perform_shared_health_check (defaults to True if None)
        details_bool = (
            health_check_details if health_check_details is not None else True
        )

        if shared_health_manager is not None:
            try:
                (
                    healthy_endpoints,
                    unhealthy_endpoints,
                ) = await shared_health_manager.perform_shared_health_check(
                    model_list=_llm_model_list, details=details_bool
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    "Error in shared health check, falling back to direct health check: %s",
                    str(e),
                )
                healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                    model_list=_llm_model_list, details=health_check_details
                )
        else:
            healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                model_list=_llm_model_list, details=health_check_details
            )

        # Update the global variable with the health check results
        health_check_results["healthy_endpoints"] = healthy_endpoints
        health_check_results["unhealthy_endpoints"] = unhealthy_endpoints
        health_check_results["healthy_count"] = len(healthy_endpoints)
        health_check_results["unhealthy_count"] = len(unhealthy_endpoints)

        await asyncio.sleep(health_check_interval)


class StreamingCallbackError(Exception):
    pass


class ProxyConfig:
    """
    Abstraction class on top of config loading/updating logic. Gives us one place to control all config updating logic.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}

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
            raise Exception(f"Error loading yaml file {file_path}: {str(e)}")

    async def _get_config_from_file(
        self, config_file_path: Optional[str] = None
    ) -> dict:
        """
        Given a config file path, load the config from the file.
        Args:
            config_file_path (str): path to the config file
        Returns:
            dict: config
        """
        global prisma_client, user_config_file_path

        file_path = config_file_path or user_config_file_path
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
        config = self._process_includes(
            config=config, base_dir=os.path.dirname(os.path.abspath(file_path or ""))
        )

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

    async def save_config(self, new_config: dict):
        global prisma_client, general_settings, user_config_file_path, store_model_in_db
        # Load existing config
        ## DB - writes valid config to db
        """
        - Do not write restricted params like 'api_key' to the database
        - if api_key is passed, save that to the local environment or connected secret manage (maybe expose `litellm.save_secret()`)
        """

        if prisma_client is not None and (
            general_settings.get("store_model_in_db", False) is True
            or store_model_in_db
        ):
            # if using - db for config - models are in ModelTable

            # Make a copy to avoid mutating the original config
            config_to_save = new_config.copy()

            # SECURITY: Always encrypt environment_variables before DB write
            if (
                "environment_variables" in config_to_save
                and config_to_save["environment_variables"]
            ):

                # decrypt the environment_variables - in case a caller function has already encrypted the environment_variables
                decrypted_env_vars = self._decrypt_and_set_db_env_variables(
                    environment_variables=config_to_save["environment_variables"],
                    return_original_value=True,
                )

                # encrypt the environment_variables,
                config_to_save["environment_variables"] = self._encrypt_env_variables(
                    environment_variables=decrypted_env_vars
                )

            config_to_save.pop("model_list", None)
            await prisma_client.insert_data(data=config_to_save, table_name="config")
        else:
            # Save the updated config - if user is not using a dB
            ## YAML
            with open(f"{user_config_file_path}", "w") as config_file:
                yaml.dump(new_config, config_file, default_flow_style=False)

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
            verbose_proxy_logger.warning(
                f"Maximum recursion depth ({max_depth}) reached while processing config."
            )
            return config

        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self._check_for_os_environ_vars(
                    config=value, depth=depth + 1, max_depth=max_depth
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item = self._check_for_os_environ_vars(
                            config=item, depth=depth + 1, max_depth=max_depth
                        )
            # if the value is a string and starts with "os.environ/" - then it's an environment variable
            elif isinstance(value, str) and value.startswith("os.environ/"):
                config[key] = get_secret(value)
        return config

    def _get_team_config(self, team_id: str, all_teams_config: List[Dict]) -> Dict:
        team_config: dict = {}
        for team in all_teams_config:
            if "team_id" not in team:
                raise Exception(
                    f"team_id missing from team: {SENSITIVE_DATA_MASKER.mask_dict(team)}"
                )
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
        config = self.get_config_state()

        ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
        litellm_settings = config.get("litellm_settings", {})
        all_teams_config = litellm_settings.get("default_team_settings", None)
        if all_teams_config is None:
            return {}
        team_config = self._get_team_config(
            team_id=team_id, all_teams_config=all_teams_config
        )
        return team_config

    def _init_cache(
        self,
        cache_params: dict,
    ):
        global redis_usage_cache, llm_router
        from litellm import Cache

        if "default_in_memory_ttl" in cache_params:
            litellm.default_in_memory_ttl = cache_params["default_in_memory_ttl"]

        if "default_redis_ttl" in cache_params:
            litellm.default_redis_ttl = cache_params["default_redis_ttl"]

        litellm.cache = Cache(**cache_params)

        if litellm.cache is not None and isinstance(
            litellm.cache.cache, (RedisCache, RedisClusterCache)
        ):
            ## INIT PROXY REDIS USAGE CLIENT ##
            redis_usage_cache = litellm.cache.cache

    def switch_on_llm_response_caching(self):
        """
        Enable caching on the router by setting cache_responses=True.
        This ensures caching works without needing caching=True in request body.
        Router passes caching=self.cache_responses to litellm.completion()
        """
        global llm_router
        import litellm

        if (
            llm_router is not None
            and litellm.cache is not None
            and llm_router.cache_responses is not True
        ):
            llm_router.cache_responses = True
            verbose_proxy_logger.debug(
                "Set router.cache_responses=True after initializing cache"
            )

    async def get_config(self, config_file_path: Optional[str] = None) -> dict:
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
            bucket_name = os.environ.get("LITELLM_CONFIG_BUCKET_NAME")
            object_key = os.environ.get("LITELLM_CONFIG_BUCKET_OBJECT_KEY")
            bucket_type = os.environ.get("LITELLM_CONFIG_BUCKET_TYPE")
            verbose_proxy_logger.debug(
                "bucket_name: %s, object_key: %s", bucket_name, object_key
            )
            if bucket_type == "gcs":
                config = await get_config_file_contents_from_gcs(
                    bucket_name=bucket_name, object_key=object_key
                )
            else:
                config = get_file_contents_from_s3(
                    bucket_name=bucket_name, object_key=object_key
                )

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
        printed_yaml = copy.deepcopy(config)
        printed_yaml.pop("environment_variables", None)

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
                "ProxyConfig:get_config_state(): Error returning copy of config state. self.config={}\nError: {}".format(
                    self.config, e
                )
            )
            return {}

    def load_credential_list(self, config: dict) -> List[CredentialItem]:
        """
        Load the credential list from the database
        """
        credential_list_dict = config.get("credential_list")
        credential_list = []
        if credential_list_dict:
            credential_list = [CredentialItem(**cred) for cred in credential_list_dict]
        return credential_list

    def parse_search_tools(self, config: dict) -> Optional[List[SearchToolTypedDict]]:
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
            return None

        search_tools_parsed: List[SearchToolTypedDict] = []

        print(  # noqa
            "\033[32mLiteLLM: Proxy initialized with Search Tools:\033[0m"
        )  # noqa

        for search_tool in search_tools_raw:
            # Display loaded search tool
            search_tool_name = search_tool.get("search_tool_name", "")
            search_provider = search_tool.get("litellm_params", {}).get(
                "search_provider", ""
            )
            print(f"\033[32m    {search_tool_name} ({search_provider})\033[0m")  # noqa

            # Cast to SearchToolTypedDict for type safety
            try:
                search_tool_typed: SearchToolTypedDict = SearchToolTypedDict(**search_tool)  # type: ignore
                search_tools_parsed.append(search_tool_typed)
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error parsing search tool {search_tool_name}: {str(e)}"
                )
                continue

        return search_tools_parsed if search_tools_parsed else None

    def _load_environment_variables(self, config: dict):
        ## ENVIRONMENT VARIABLES
        global premium_user
        environment_variables = config.get("environment_variables", None)
        if environment_variables:
            for key, value in environment_variables.items():
                #########################################################
                # handles this scenario:
                # ```yaml
                # environment_variables:
                #     ARIZE_ENDPOINT: os.environ/ARIZE_ENDPOINT
                # ```
                #########################################################
                if isinstance(value, str) and value.startswith("os.environ/"):
                    resolved_secret_string: Optional[str] = get_secret_str(
                        secret_name=value
                    )
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
        return

    async def load_config(  # noqa: PLR0915
        self, router: Optional[litellm.Router], config_file_path: str
    ):
        """
        Load config values into proxy global state
        """
        global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, user_custom_key_generate, user_custom_sso, user_custom_ui_sso_sign_in_handler, use_background_health_checks, use_shared_health_check, health_check_interval, use_queue, proxy_budget_rescheduler_max_time, proxy_budget_rescheduler_min_time, ui_access_mode, litellm_master_key_hash, proxy_batch_write_at, disable_spend_logs, prompt_injection_detection_obj, redis_usage_cache, store_model_in_db, premium_user, open_telemetry_logger, health_check_details, callback_settings, proxy_batch_polling_interval, config_passthrough_endpoints

        config: dict = await self.get_config(config_file_path=config_file_path)

        self._load_environment_variables(config=config)

        ## Callback settings
        callback_settings = config.get("callback_settings", {})

        ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
        litellm_settings = config.get("litellm_settings", None)
        if litellm_settings is None:
            litellm_settings = {}
        if litellm_settings:
            # ANSI escape code for blue text
            blue_color_code = "\033[94m"
            reset_color_code = "\033[0m"
            for key, value in litellm_settings.items():
                if key == "cache" and value is True:
                    print(f"{blue_color_code}\nSetting Cache on Proxy")  # noqa
                    from litellm.caching.caching import Cache

                    cache_params = {}
                    if "cache_params" in litellm_settings:
                        cache_params_in_config = litellm_settings["cache_params"]
                        # overwrite cache_params with cache_params_in_config
                        cache_params.update(cache_params_in_config)

                    cache_type = cache_params.get("type", "redis")

                    verbose_proxy_logger.debug("passed cache type=%s", cache_type)

                    if (
                        cache_type == "redis" or cache_type == "redis-semantic"
                    ) and len(cache_params.keys()) == 0:
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
                    self._init_cache(cache_params=cache_params)
                    if litellm.cache is not None:
                        verbose_proxy_logger.debug(
                            f"{blue_color_code}Set Cache on LiteLLM Proxy{reset_color_code}"
                        )
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
                        f"{blue_color_code}Set Global Prompt Directory on LiteLLM Proxy{reset_color_code}"
                    )
                elif key == "global_bitbucket_config":
                    from litellm.integrations.bitbucket import (
                        set_global_bitbucket_config,
                    )

                    set_global_bitbucket_config(value)
                    verbose_proxy_logger.info(
                        f"{blue_color_code}Set Global BitBucket Config on LiteLLM Proxy{reset_color_code}"
                    )
                elif key == "global_gitlab_config":
                    from litellm.integrations.gitlab import set_global_gitlab_config

                    set_global_gitlab_config(value)
                    verbose_proxy_logger.info(
                        f"{blue_color_code}Set Global Gitlab Config on LiteLLM Proxy{reset_color_code}"
                    )
                elif key == "priority_reservation_settings":
                    from litellm.types.utils import PriorityReservationSettings

                    litellm.priority_reservation_settings = PriorityReservationSettings(
                        **value
                    )
                elif key == "callbacks":
                    initialize_callbacks_on_proxy(
                        value=value,
                        premium_user=premium_user,
                        config_file_path=config_file_path,
                        litellm_settings=litellm_settings,
                    )

                elif key == "model_group_settings":
                    from litellm.types.router import ModelGroupSettings

                    litellm.model_group_settings = ModelGroupSettings(**value)

                elif key == "post_call_rules":
                    litellm.post_call_rules = [
                        get_instance_fn(value=value, config_file_path=config_file_path)
                    ]
                    verbose_proxy_logger.debug(
                        f"litellm.post_call_rules: {litellm.post_call_rules}"
                    )
                elif key == "max_internal_user_budget":
                    litellm.max_internal_user_budget = float(value)  # type: ignore
                elif key == "default_max_internal_user_budget":
                    litellm.default_max_internal_user_budget = float(value)
                    if litellm.max_internal_user_budget is None:
                        litellm.max_internal_user_budget = (
                            litellm.default_max_internal_user_budget
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
                                get_instance_fn(value=callback)
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.logging_callback_manager.add_litellm_success_callback(
                                callback
                            )
                            if "prometheus" in callback:
                                from litellm.integrations.prometheus import (
                                    PrometheusLogger,
                                )

                                if PrometheusLogger is not None:
                                    verbose_proxy_logger.debug(
                                        "mounting metrics endpoint"
                                    )
                                    PrometheusLogger._mount_metrics_endpoint()
                    print(  # noqa
                        f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}"
                    )  # noqa
                elif key == "failure_callback":
                    litellm.failure_callback = []

                    # initialize success callbacks
                    for callback in value:
                        # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                        if "." in callback:
                            litellm.logging_callback_manager.add_litellm_failure_callback(
                                get_instance_fn(value=callback)
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.logging_callback_manager.add_litellm_failure_callback(
                                callback
                            )
                    print(  # noqa
                        f"{blue_color_code} Initialized Failure Callbacks - {litellm.failure_callback} {reset_color_code}"
                    )  # noqa
                elif key == "cache_params":
                    # this is set in the cache branch
                    # see usage here: https://docs.litellm.ai/docs/proxy/caching
                    pass
                elif key == "default_team_settings":
                    for idx, team_setting in enumerate(
                        value
                    ):  # run through pydantic validation
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
                        f"{blue_color_code} setting litellm.{key}={value}{reset_color_code}"
                    )
                    setattr(litellm, key, value)
                elif key == "upperbound_key_generate_params":
                    if value is not None and isinstance(value, dict):
                        for _k, _v in value.items():
                            if isinstance(_v, str) and _v.startswith("os.environ/"):
                                value[_k] = get_secret(_v)
                        litellm.upperbound_key_generate_params = (
                            LiteLLM_UpperboundKeyGenerateParams(**value)
                        )
                    else:
                        raise Exception(
                            f"Invalid value set for upperbound_key_generate_params - value={value}"
                        )

                else:
                    verbose_proxy_logger.debug(
                        f"{blue_color_code} setting litellm.{key}={value}{reset_color_code}"
                    )
                    setattr(litellm, key, value)

        ## GENERAL SERVER SETTINGS (e.g. master key,..) # do this after initializing litellm, to ensure sentry logging works for proxylogging
        general_settings = config.get("general_settings", {})
        if general_settings is None:
            general_settings = {}
        if general_settings:
            ### LOAD KEY MANAGEMENT SETTINGS FIRST (needed for custom secret manager) ###
            key_management_settings = general_settings.get(
                "key_management_settings", None
            )
            if key_management_settings is not None:
                litellm._key_management_settings = KeyManagementSettings(
                    **key_management_settings
                )

            ### LOAD SECRET MANAGER ###
            key_management_system = general_settings.get("key_management_system", None)
            self.initialize_secret_manager(
                key_management_system=key_management_system,
                config_file_path=config_file_path,
            )
            ### [DEPRECATED] LOAD FROM GOOGLE KMS ### old way of loading from google kms
            use_google_kms = general_settings.get("use_google_kms", False)
            load_google_kms(use_google_kms=use_google_kms)
            ### [DEPRECATED] LOAD FROM AZURE KEY VAULT ### old way of loading from azure secret manager
            use_azure_key_vault = general_settings.get("use_azure_key_vault", False)
            load_from_azure_key_vault(use_azure_key_vault=use_azure_key_vault)
            ### ALERTING ###
            self._load_alerting_settings(general_settings=general_settings)
            ### CONNECT TO DATABASE ###
            database_url = general_settings.get("database_url", None)
            if database_url and database_url.startswith("os.environ/"):
                verbose_proxy_logger.debug("GOING INTO LITELLM.GET_SECRET!")
                database_url = get_secret(database_url)
                verbose_proxy_logger.debug("RETRIEVED DB URL: %s", database_url)
            ### MASTER KEY ###
            master_key = general_settings.get(
                "master_key", get_secret("LITELLM_MASTER_KEY", None)
            )

            if master_key and master_key.startswith("os.environ/"):
                master_key = get_secret(master_key)  # type: ignore

            if master_key is not None and isinstance(master_key, str):
                litellm_master_key_hash = hash_token(master_key)
            ### USER API KEY CACHE IN-MEMORY TTL ###
            user_api_key_cache_ttl = general_settings.get(
                "user_api_key_cache_ttl", None
            )
            if user_api_key_cache_ttl is not None:
                user_api_key_cache.update_cache_ttl(
                    default_in_memory_ttl=float(user_api_key_cache_ttl),
                    default_redis_ttl=None,  # user_api_key_cache is an in-memory cache
                )
            ### STORE MODEL IN DB ### feature flag for `/model/new`
            store_model_in_db = general_settings.get("store_model_in_db", False)
            if store_model_in_db is None:
                store_model_in_db = False
            ### CUSTOM API KEY AUTH ###
            ## pass filepath
            custom_auth = general_settings.get("custom_auth", None)
            if custom_auth is not None:
                user_custom_auth = get_instance_fn(
                    value=custom_auth, config_file_path=config_file_path
                )

            custom_key_generate = general_settings.get("custom_key_generate", None)
            if custom_key_generate is not None:
                user_custom_key_generate = get_instance_fn(
                    value=custom_key_generate, config_file_path=config_file_path
                )

            custom_sso = general_settings.get("custom_sso", None)
            if custom_sso is not None:
                user_custom_sso = get_instance_fn(
                    value=custom_sso, config_file_path=config_file_path
                )

            custom_ui_sso_sign_in_handler = general_settings.get(
                "custom_ui_sso_sign_in_handler", None
            )
            if custom_ui_sso_sign_in_handler is not None:
                user_custom_ui_sso_sign_in_handler = get_instance_fn(
                    value=custom_ui_sso_sign_in_handler,
                    config_file_path=config_file_path,
                )

            if enterprise_proxy_config is not None:
                await enterprise_proxy_config.load_enterprise_config(general_settings)

            ## pass through endpoints
            if general_settings.get("pass_through_endpoints", None) is not None:
                config_passthrough_endpoints = general_settings[
                    "pass_through_endpoints"
                ]
                await initialize_pass_through_endpoints(
                    pass_through_endpoints=general_settings["pass_through_endpoints"]
                )

            ## ADMIN UI ACCESS ##
            ui_access_mode = general_settings.get(
                "ui_access_mode", "all"
            )  # can be either ["admin_only" or "all"]
            ### ALLOWED IP ###
            allowed_ips = general_settings.get("allowed_ips", None)
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
            proxy_batch_write_at = general_settings.get(
                "proxy_batch_write_at", proxy_batch_write_at
            )
            ## DISABLE SPEND LOGS ## - gives a perf improvement
            disable_spend_logs = general_settings.get(
                "disable_spend_logs", disable_spend_logs
            )
            ### BACKGROUND HEALTH CHECKS ###
            # Enable background health checks
            use_background_health_checks = general_settings.get(
                "background_health_checks", False
            )
            # Enable shared health check state across pods (requires Redis)
            use_shared_health_check = general_settings.get(
                "use_shared_health_check", False
            )
            health_check_interval = general_settings.get(
                "health_check_interval", DEFAULT_HEALTH_CHECK_INTERVAL
            )
            health_check_details = general_settings.get("health_check_details", True)

            ### RBAC ###
            rbac_role_permissions = general_settings.get("role_permissions", None)
            if rbac_role_permissions is not None:
                general_settings["role_permissions"] = [  # validate role permissions
                    RoleBasedPermissions(**role_permission)
                    for role_permission in rbac_role_permissions
                ]

            ## check if user has set a premium feature in general_settings
            if (
                general_settings.get("enforced_params") is not None
                and premium_user is not True
            ):
                raise ValueError(
                    "Trying to use `enforced_params`"
                    + CommonProxyErrors.not_premium_user.value
                )

            # check if litellm_license in general_settings
            if "litellm_license" in general_settings:
                _license_check.license_str = general_settings["litellm_license"]
                premium_user = _license_check.is_premium()

        router_params: dict = {
            "cache_responses": litellm.cache
            is not None,  # cache if user passed in cache values
        }
        ## MODEL LIST
        model_list = config.get("model_list", None)
        if model_list:
            router_params["model_list"] = model_list
            print(  # noqa
                "\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m"
            )  # noqa
            for model in model_list:
                ### LOAD FROM os.environ/ ###
                for k, v in model["litellm_params"].items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        model["litellm_params"][k] = get_secret(v)
                print(f"\033[32m    {model.get('model_name', '')}\033[0m")  # noqa
                litellm_model_name = model["litellm_params"]["model"]
                litellm_model_api_base = model["litellm_params"].get("api_base", None)
                if "ollama" in litellm_model_name and litellm_model_api_base is None:
                    run_ollama_serve()

        ## ASSISTANT SETTINGS
        assistants_config: Optional[AssistantsTypedDict] = None
        assistant_settings = config.get("assistant_settings", None)
        if assistant_settings:
            for k, v in assistant_settings["litellm_params"].items():
                if isinstance(v, str) and v.startswith("os.environ/"):
                    _v = v.replace("os.environ/", "")
                    v = os.getenv(_v)
                    assistant_settings["litellm_params"][k] = v
            assistants_config = AssistantsTypedDict(**assistant_settings)  # type: ignore

        ## SEARCH TOOLS SETTINGS
        search_tools: Optional[List[SearchToolTypedDict]] = self.parse_search_tools(
            config
        )

        ## /fine_tuning/jobs endpoints config
        finetuning_config = config.get("finetune_settings", None)
        set_fine_tuning_config(config=finetuning_config)

        ## /files endpoint config
        files_config = config.get("files_settings", None)
        set_files_config(config=files_config)

        ## default config for vertex ai routes
        default_vertex_config = config.get("default_vertex_config", None)
        passthrough_endpoint_router.set_default_vertex_config(
            config=default_vertex_config
        )

        ## ROUTER SETTINGS (e.g. routing_strategy, ...)
        router_settings = config.get("router_settings", None)

        if router_settings and isinstance(router_settings, dict):
            arg_spec = inspect.getfullargspec(litellm.Router)
            # model list and search_tools already set
            exclude_args = {
                "self",
                "model_list",
                "search_tools",
            }

            available_args = [x for x in arg_spec.args if x not in exclude_args]

            for k, v in router_settings.items():
                if k in available_args:
                    router_params[k] = v
        router = litellm.Router(
            **router_params,
            assistants_config=assistants_config,
            search_tools=search_tools,
            router_general_settings=RouterGeneralSettings(
                async_only_mode=True  # only init async clients
            ),
            ignore_invalid_deployments=True,  # don't raise an error if a deployment is invalid
        )  # type:ignore

        if redis_usage_cache is not None and router.cache.redis_cache is None:
            router._update_redis_cache(cache=redis_usage_cache)

        # Guardrail settings
        guardrails_v2: Optional[List[Dict]] = None

        if config is not None:
            guardrails_v2 = config.get("guardrails", None)
        if guardrails_v2:
            init_guardrails_v2(
                all_guardrails=guardrails_v2, config_file_path=config_file_path
            )

        ## Prompt settings
        prompts: Optional[List[Dict]] = None
        if config is not None:
            prompts = config.get("prompts", None)
        if prompts:
            from litellm.proxy.prompts.init_prompts import init_prompts

            init_prompts(all_prompts=prompts, config_file_path=config_file_path)

        ## CREDENTIALS
        credential_list_dict = self.load_credential_list(config=config)
        litellm.credential_list = credential_list_dict

        ## NON-LLM CONFIGS eg. MCP tools, vector stores, etc.
        await self._init_non_llm_configs(config=config)

        return router, router.get_model_list(), general_settings

    async def _init_non_llm_configs(self, config: dict):
        """
        Initialize non-LLM configs eg. MCP tools, vector stores, etc.
        """
        ## MCP TOOLS
        mcp_tools_config = config.get("mcp_tools", None)
        if mcp_tools_config:
            global_mcp_tool_registry.load_tools_from_config(mcp_tools_config)

        ## AGENTS
        agent_config = config.get("agent_list", None)
        if agent_config:
            global_agent_registry.load_agents_from_config(agent_config)  # type: ignore

        mcp_servers_config = config.get("mcp_servers", None)
        if mcp_servers_config:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            # Get mcp_aliases from litellm_settings if available
            litellm_settings = config.get("litellm_settings", {})
            mcp_aliases = litellm_settings.get("mcp_aliases", None)

            await global_mcp_server_manager.load_servers_from_config(
                mcp_servers_config, mcp_aliases
            )

        ## VECTOR STORES
        vector_store_registry_config = config.get("vector_store_registry", None)
        if vector_store_registry_config:
            from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

            if litellm.vector_store_registry is None:
                litellm.vector_store_registry = VectorStoreRegistry()

            # Load vector stores from config
            litellm.vector_store_registry.load_vector_stores_from_config(
                vector_store_registry_config
            )
        pass

    def _load_alerting_settings(self, general_settings: dict):
        """
        Initialize alerting settings
        """
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        _alerting_callbacks = general_settings.get("alerting", None)
        verbose_proxy_logger.debug(f"_alerting_callbacks: {general_settings}")
        if _alerting_callbacks is None:
            return
        for _alert in _alerting_callbacks:
            if _alert == "slack":
                # [OLD] v0 implementation
                proxy_logging_obj.update_values(
                    alerting=general_settings.get("alerting", None),
                    alerting_threshold=general_settings.get("alerting_threshold", 600),
                    alert_types=general_settings.get("alert_types", None),
                    alert_to_webhook_url=general_settings.get(
                        "alert_to_webhook_url", None
                    ),
                    alerting_args=general_settings.get("alerting_args", None),
                    redis_cache=redis_usage_cache,
                )
            else:
                # [NEW] v1 implementation - init as a custom logger
                if _alert in litellm._known_custom_logger_compatible_callbacks:
                    _logger = _init_custom_logger_compatible_class(
                        logging_integration=_alert,
                        internal_usage_cache=None,
                        llm_router=None,
                        custom_logger_init_args={
                            "alerting_args": general_settings.get("alerting_args", None)
                        },
                    )
                    if _logger is not None:
                        litellm.logging_callback_manager.add_litellm_callback(_logger)
        pass

    def initialize_secret_manager(
        self,
        key_management_system: Optional[str],
        config_file_path: Optional[str] = None,
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
                key_management_system
                == KeyManagementSystem.AWS_SECRET_MANAGER.value  # noqa: F405
            ):
                from litellm.secret_managers.aws_secret_manager_v2 import (
                    AWSSecretsManagerV2,
                )

                AWSSecretsManagerV2.load_aws_secret_manager(use_aws_secret_manager=True)
            elif key_management_system == KeyManagementSystem.AWS_KMS.value:
                load_aws_kms(use_aws_kms=True)
            elif (
                key_management_system == KeyManagementSystem.GOOGLE_SECRET_MANAGER.value
            ):
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
        _id: Optional[str] = getattr(model, "model_id", None)
        if _id is not None:
            model.model_info["id"] = _id
            model.model_info["db_model"] = True

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

    async def _delete_deployment(self, db_models: list) -> int:
        """
        (Helper function of add deployment) -> combined to reduce prisma db calls

        - Create all up list of model id's (db + config)
        - Compare all up list to router model id's
        - Remove any that are missing

        Return:
        - int - returns number of deleted deployments
        """
        global user_config_file_path, llm_router
        combined_id_list = []

        ## BASE CASES ##
        # if llm_router is None or db_models is empty, return 0
        if llm_router is None or len(db_models) == 0:
            return 0

        ## DB MODELS ##
        for m in db_models:
            model_info = self.get_model_info_with_id(model=m)
            if model_info.id is not None:
                combined_id_list.append(model_info.id)

        ## CONFIG MODELS ##
        config = await self.get_config(config_file_path=user_config_file_path)
        model_list = config.get("model_list", None)
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
                    model_id = llm_router._generate_model_id(
                        model_group=model["model_name"],
                        litellm_params=model["litellm_params"],
                    )
                else:
                    model_id = str(model_id)
                combined_id_list.append(model_id)  # ADD CONFIG MODEL TO COMBINED LIST

        router_model_ids = llm_router.get_model_ids()
        # Check for model IDs in llm_router not present in combined_id_list and delete them

        deleted_deployments = 0
        for model_id in router_model_ids:
            if model_id not in combined_id_list:
                is_deleted = llm_router.delete_deployment(id=model_id)
                if is_deleted is not None:
                    deleted_deployments += 1
        return deleted_deployments

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
                    if isinstance(v, str):
                        # decrypt value - returns original value if decryption fails or no key is set
                        _value = decrypt_value_helper(
                            value=v, key=k, return_original_value=True
                        )
                        _litellm_params[k] = _value
                _litellm_params = LiteLLM_Params(**_litellm_params)

            else:
                verbose_proxy_logger.error(
                    f"Invalid model added to proxy db. Invalid litellm params. litellm_params={_litellm_params}"
                )
                continue  # skip to next model
            _model_info = self.get_model_info_with_id(
                model=m, db_model=True
            )  ## 👈 FLAG = True for db_models

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
        _model_list: list = []
        for m in new_models:
            _litellm_params = m.litellm_params
            if isinstance(_litellm_params, BaseModel):
                _litellm_params = _litellm_params.model_dump()
            if isinstance(_litellm_params, dict):
                # decrypt values
                for k, v in _litellm_params.items():
                    decrypted_value = decrypt_value_helper(
                        value=v, key=k, return_original_value=True
                    )
                    _litellm_params[k] = decrypted_value
                _litellm_params = LiteLLM_Params(**_litellm_params)
            else:
                verbose_proxy_logger.error(
                    f"Invalid model added to proxy db. Invalid litellm params. litellm_params={_litellm_params}"
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
        new_models: list,
        proxy_logging_obj: ProxyLogging,
    ):
        global llm_router, llm_model_list, master_key, general_settings

        try:
            if llm_router is None and master_key is not None:
                verbose_proxy_logger.debug(f"len new_models: {len(new_models)}")

                _model_list: list = self.decrypt_model_list_from_db(
                    new_models=new_models
                )
                if len(_model_list) > 0:
                    verbose_proxy_logger.debug(f"_model_list: {_model_list}")
                    llm_router = litellm.Router(
                        model_list=_model_list,
                        router_general_settings=RouterGeneralSettings(
                            async_only_mode=True  # only init async clients
                        ),
                        ignore_invalid_deployments=True,
                    )
                    verbose_proxy_logger.debug(f"updated llm_router: {llm_router}")
            else:
                verbose_proxy_logger.debug(f"len new_models: {len(new_models)}")
                ## DELETE MODEL LOGIC
                await self._delete_deployment(db_models=new_models)

                ## ADD MODEL LOGIC
                self._add_deployment(db_models=new_models)

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error adding/deleting model to llm_router: {str(e)}"
            )

        if llm_router is not None:
            llm_model_list = llm_router.get_model_list()

        # check if user set any callbacks in Config Table
        config_data = await proxy_config.get_config()
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

    def _add_callback_from_db_to_in_memory_litellm_callbacks(
        self,
        callback: str,
        event_types: List[Literal["success", "failure"]],
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
        litellm_settings = config_data.get("litellm_settings", {}) or {}
        success_callbacks = litellm_settings.get("success_callback", None)
        failure_callbacks = litellm_settings.get("failure_callback", None)
        callbacks = litellm_settings.get("callbacks", None)

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

    def _encrypt_env_variables(
        self, environment_variables: dict, new_encryption_key: Optional[str] = None
    ) -> dict:
        """
        Encrypts a dictionary of environment variables and returns them.
        """
        encrypted_env_vars = {}
        for k, v in environment_variables.items():
            encrypted_value = encrypt_value_helper(
                value=v, new_encryption_key=new_encryption_key
            )
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
        decrypted_env_vars = {}
        for k, v in environment_variables.items():
            try:
                decrypted_value = decrypt_value_helper(
                    value=v, key=k, return_original_value=return_original_value
                )
                if decrypted_value is not None:
                    os.environ[k] = decrypted_value
                    decrypted_env_vars[k] = decrypted_value
            except Exception as e:
                verbose_proxy_logger.error(
                    "Error setting env variable: %s - %s", k, str(e)
                )
        return decrypted_env_vars

    def _decrypt_db_variables(self, variables_dict: dict) -> dict:
        """
        Decrypts a dictionary of variables and returns them.
        """
        decrypted_variables = {}
        for k, v in variables_dict.items():
            decrypted_value = decrypt_value_helper(
                value=v, key=k, return_original_value=True
            )
            decrypted_variables[k] = decrypted_value
        return decrypted_variables

    async def _add_router_settings_from_db_config(
        self,
        config_data: dict,
        llm_router: Optional[Router],
        prisma_client: Optional[PrismaClient],
    ) -> None:
        """
        Adds router settings from DB config to litellm proxy

        1. Get router settings from DB
        2. Get router settings from config
        3. Combine both
        4. Update router settings
        """
        if llm_router is not None and prisma_client is not None:
            db_router_settings = await prisma_client.db.litellm_config.find_first(
                where={"param_name": "router_settings"}
            )

            config_router_settings = config_data.get("router_settings", {})

            combined_router_settings = {}
            if (
                config_router_settings is not None
                and isinstance(config_router_settings, dict)
                and db_router_settings is not None
                and isinstance(db_router_settings.param_value, dict)
            ):
                from litellm.utils import _update_dictionary

                combined_router_settings = _update_dictionary(
                    config_router_settings, db_router_settings.param_value
                )
            elif config_router_settings is not None and isinstance(
                config_router_settings, dict
            ):
                combined_router_settings = config_router_settings
            elif db_router_settings is not None and isinstance(
                db_router_settings.param_value, dict
            ):
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
        _general_settings = config_data.get("general_settings", {})
        if _general_settings is not None and "alerting" in _general_settings:
            if (
                general_settings is not None
                and general_settings.get("alerting", None) is not None
                and isinstance(general_settings["alerting"], list)
                and _general_settings.get("alerting", None) is not None
                and isinstance(_general_settings["alerting"], list)
            ):
                verbose_proxy_logger.debug(
                    "Overriding Default 'alerting' values with db 'alerting' values."
                )
                general_settings["alerting"] = _general_settings[
                    "alerting"
                ]  # override yaml values with db
                proxy_logging_obj.alerting = general_settings["alerting"]
                proxy_logging_obj.slack_alerting_instance.alerting = general_settings[
                    "alerting"
                ]
            elif general_settings is None:
                general_settings = {}
                general_settings["alerting"] = _general_settings["alerting"]
                proxy_logging_obj.alerting = general_settings["alerting"]
                proxy_logging_obj.slack_alerting_instance.alerting = general_settings[
                    "alerting"
                ]
            elif isinstance(general_settings, dict):
                general_settings["alerting"] = _general_settings["alerting"]
                proxy_logging_obj.alerting = general_settings["alerting"]
                proxy_logging_obj.slack_alerting_instance.alerting = general_settings[
                    "alerting"
                ]

        if _general_settings is not None and "alert_types" in _general_settings:
            general_settings["alert_types"] = _general_settings["alert_types"]
            proxy_logging_obj.alert_types = general_settings["alert_types"]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alert_types=general_settings["alert_types"], llm_router=llm_router
            )

        if (
            _general_settings is not None
            and "alert_to_webhook_url" in _general_settings
        ):
            general_settings["alert_to_webhook_url"] = _general_settings[
                "alert_to_webhook_url"
            ]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alert_to_webhook_url=general_settings["alert_to_webhook_url"],
                llm_router=llm_router,
            )

    async def _update_general_settings(self, db_general_settings: Optional[Json]):
        """
        Pull from DB, read general settings value
        """
        global general_settings
        if db_general_settings is None:
            return
        _general_settings = dict(db_general_settings)
        ## MAX PARALLEL REQUESTS ##
        if "max_parallel_requests" in _general_settings:
            general_settings["max_parallel_requests"] = _general_settings[
                "max_parallel_requests"
            ]

        if "global_max_parallel_requests" in _general_settings:
            general_settings["global_max_parallel_requests"] = _general_settings[
                "global_max_parallel_requests"
            ]

        ## ALERTING ARGS ##
        if "alerting_args" in _general_settings:
            general_settings["alerting_args"] = _general_settings["alerting_args"]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alerting_args=general_settings["alerting_args"],
            )

        ## PASS-THROUGH ENDPOINTS ##
        if "pass_through_endpoints" in _general_settings:
            general_settings["pass_through_endpoints"] = _general_settings[
                "pass_through_endpoints"
            ]
            await initialize_pass_through_endpoints(
                pass_through_endpoints=general_settings["pass_through_endpoints"]
            )

        ## UI ACCESS MODE ##
        if "ui_access_mode" in _general_settings:
            general_settings["ui_access_mode"] = _general_settings["ui_access_mode"]

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
            Deep-merge src into dst, skipping None values from src.
            On conflicts, src (DB) wins.
            """
            stack = [(dst, src)]
            while stack:
                d, s = stack.pop()
                for k, v in s.items():
                    if v is None:
                        # Preserve existing config when DB value is None (matches prior behavior)
                        continue
                    if isinstance(v, dict) and isinstance(d.get(k), dict):
                        stack.append((d[k], v))
                    else:
                        d[k] = v

        if param_name == "environment_variables":
            decrypted_env_vars = self._decrypt_and_set_db_env_variables(
                db_param_value, return_original_value=True
            )
            current_config.setdefault("environment_variables", {}).update(
                decrypted_env_vars
            )
            return current_config
        elif param_name == "litellm_settings" and isinstance(db_param_value, dict):
            for key, value in db_param_value.items():
                if (
                    key in LITELLM_SETTINGS_SAFE_DB_OVERRIDES
                ):  # params that are safe to override with db values
                    setattr(litellm, key, value)

        # If param doesn't exist in config, add it
        if param_name not in current_config:
            current_config[param_name] = db_param_value

            return current_config

        # For dictionary values, update only non-none values
        if isinstance(current_config[param_name], dict) and isinstance(
            db_param_value, dict
        ):
            _deep_merge_dicts(current_config[param_name], db_param_value)
        else:
            # Non-dict or mismatched types: DB value replaces config (unchanged behavior)
            current_config[param_name] = db_param_value

        return current_config

    async def _update_config_from_db(
        self,
        prisma_client: PrismaClient,
        config: dict,
        store_model_in_db: Optional[bool],
    ):
        if store_model_in_db is not True:
            verbose_proxy_logger.info(
                "'store_model_in_db' is not True, skipping db updates"
            )
            return config

        _tasks = []
        keys = [
            "general_settings",
            "router_settings",
            "litellm_settings",
            "environment_variables",
        ]
        for k in keys:
            response = prisma_client.get_generic_data(
                key="param_name", value=k, table_name="config"
            )
            _tasks.append(response)

        responses = await asyncio.gather(*_tasks)
        for response in responses:
            if response is None:
                continue

            param_name = getattr(response, "param_name", None)
            param_value = getattr(response, "param_value", None)
            verbose_proxy_logger.debug(
                f"param_name={param_name}, param_value={param_value}"
            )

            if param_name is not None and param_value is not None:
                config = self._update_config_fields(
                    current_config=config,
                    param_name=param_name,
                    db_param_value=param_value,
                )

        return config

    def _should_load_db_object(
        self, object_type: Union[str, SupportedDBObjectType]
    ) -> bool:
        """
        Check if an object type should be loaded from the database based on general_settings.supported_db_objects.

        Args:
            object_type: Type of object to check (e.g., SupportedDBObjectType.MODELS, "models", etc.)

        Returns:
            True if the object should be loaded, False otherwise
        """
        global general_settings

        # Get the supported_db_objects configuration
        supported_db_objects = general_settings.get("supported_db_objects", None)

        # If supported_db_objects is not set, load all objects (default behavior)
        if supported_db_objects is None:
            return True

        # If supported_db_objects is set, only load specified objects
        if not isinstance(supported_db_objects, list):
            verbose_proxy_logger.warning(
                f"supported_db_objects is not a list, got {type(supported_db_objects)}. Loading all objects."
            )
            return True

        # Convert object_type to string for comparison (handles both str and enum)
        object_type_str = str(object_type)

        # Check if the object type is in the list (supports both str and enum values)
        return any(str(obj) == object_type_str for obj in supported_db_objects)

    async def _get_models_from_db(self, prisma_client: PrismaClient) -> list:
        try:
            new_models = await prisma_client.db.litellm_proxymodeltable.find_many()
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy_server.py::add_deployment() - Error getting new models from DB - {}".format(
                    str(e)
                )
            )
            new_models = []

        return new_models

    async def add_deployment(
        self,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
    ):
        """
        - Check db for new models
        - Check if model id's in router already
        - If not, add to router
        """
        global llm_router, llm_model_list, master_key, general_settings

        try:
            # Only load models from DB if "models" is in supported_db_objects (or if supported_db_objects is not set)
            if self._should_load_db_object(object_type="models"):
                new_models = await self._get_models_from_db(prisma_client=prisma_client)

                # update llm router
                await self._update_llm_router(
                    new_models=new_models, proxy_logging_obj=proxy_logging_obj
                )

            db_general_settings = await prisma_client.db.litellm_config.find_first(
                where={"param_name": "general_settings"}
            )

            # update general settings
            if db_general_settings is not None:
                await self._update_general_settings(
                    db_general_settings=db_general_settings.param_value,
                )

            # initialize vector stores, guardrails, etc. table in db
            await self._init_non_llm_objects_in_db(prisma_client=prisma_client)

        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:add_deployment - {}".format(
                    str(e)
                )
            )

    async def _init_non_llm_objects_in_db(self, prisma_client: PrismaClient):
        """
        Use this to read non-llm objects from the db and initialize them

        ex. Vector Stores, Guardrails, MCP tools, etc.
        """
        if self._should_load_db_object(object_type="guardrails"):
            await self._init_guardrails_in_db(prisma_client=prisma_client)

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

        if self._should_load_db_object(object_type="model_cost_map"):
            await self._check_and_reload_model_cost_map(prisma_client=prisma_client)
        if self._should_load_db_object(object_type="sso_settings"):
            await self._init_sso_settings_in_db(prisma_client=prisma_client)
        if self._should_load_db_object(object_type="cache_settings"):
            from litellm.proxy.management_endpoints.cache_settings_endpoints import (
                CacheSettingsManager,
            )

            await CacheSettingsManager.init_cache_settings_in_db(
                prisma_client=prisma_client, proxy_config=self
            )

    async def _init_sso_settings_in_db(self, prisma_client: PrismaClient):
        """
        Initialize SSO settings from database into the router on startup.
        """

        try:
            sso_settings = await prisma_client.db.litellm_ssoconfig.find_unique(
                where={"id": "sso_config"}
            )
            if sso_settings is not None:
                # Capitalize all keys in sso_settings dictionary
                uppercase_sso_settings = {
                    key.upper(): value
                    for key, value in sso_settings.sso_settings.items()
                }
                self._decrypt_and_set_db_env_variables(
                    environment_variables=uppercase_sso_settings
                )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_sso_settings_in_db - {}".format(
                    str(e)
                )
            )

    async def _check_and_reload_model_cost_map(self, prisma_client: PrismaClient):
        """
        Check if model cost map needs to be reloaded based on database configuration.
        This function runs every 10 seconds as part of _init_non_llm_objects_in_db.
        """
        try:
            # Get model cost map reload configuration from database
            config_record = await prisma_client.db.litellm_config.find_unique(
                where={"param_name": "model_cost_map_reload_config"}
            )

            if config_record is None or config_record.param_value is None:
                return  # No configuration found, skip reload

            config = config_record.param_value
            interval_hours = config.get("interval_hours")
            force_reload = config.get("force_reload", False)

            if interval_hours is None and force_reload is False:
                return  # No interval configured, skip reload

            current_time = datetime.utcnow()

            # Check if we need to reload based on interval or force reload
            should_reload = False

            if force_reload:
                should_reload = True
                verbose_proxy_logger.info(
                    "Model cost map reload triggered by force reload flag"
                )
            elif interval_hours is not None:
                # Use pod's in-memory last reload time
                global last_model_cost_map_reload
                if last_model_cost_map_reload is not None:
                    try:
                        last_reload_time = datetime.fromisoformat(
                            last_model_cost_map_reload
                        )
                        time_since_last_reload = current_time - last_reload_time
                        hours_since_last_reload = (
                            time_since_last_reload.total_seconds() / 3600
                        )

                        if hours_since_last_reload >= interval_hours:
                            should_reload = True
                            verbose_proxy_logger.info(
                                f"Model cost map reload triggered by interval. Hours since last reload: {hours_since_last_reload:.2f}, Interval: {interval_hours}"
                            )
                    except Exception as e:
                        verbose_proxy_logger.warning(
                            f"Error parsing last reload time: {e}"
                        )
                        # If we can't parse the last reload time, reload anyway
                        should_reload = True
                else:
                    # No last reload time recorded, reload now
                    should_reload = True
                    verbose_proxy_logger.info(
                        "Model cost map reload triggered - no previous reload time recorded"
                    )

            if should_reload:
                # Perform the reload
                from litellm.litellm_core_utils.get_model_cost_map import (
                    get_model_cost_map,
                )

                model_cost_map_url = litellm.model_cost_map_url
                new_model_cost_map = get_model_cost_map(url=model_cost_map_url)
                litellm.model_cost = new_model_cost_map

                # Update pod's in-memory last reload time
                last_model_cost_map_reload = current_time.isoformat()

                # Clear force reload flag in database
                await prisma_client.db.litellm_config.upsert(
                    where={"param_name": "model_cost_map_reload_config"},
                    data={
                        "create": {
                            "param_name": "model_cost_map_reload_config",
                            "param_value": safe_dumps(
                                {
                                    "interval_hours": interval_hours,
                                    "force_reload": False,
                                }
                            ),
                        },
                        "update": {"param_value": safe_dumps({"force_reload": False})},
                    },
                )

                verbose_proxy_logger.info(
                    f"Model cost map reloaded successfully. Models count: {len(new_model_cost_map) if new_model_cost_map else 0}"
                )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in _check_and_reload_model_cost_map: {str(e)}"
            )

    async def _init_prompts_in_db(self, prisma_client: PrismaClient):
        from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

        try:
            prompts_in_db = await prisma_client.db.litellm_prompttable.find_many()
            for prompt in prompts_in_db:
                IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(prompt=prompt)
        except Exception as e:
            verbose_proxy_logger.debug(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_prompts_in_db - {}".format(
                    str(e)
                )
            )

    async def _init_guardrails_in_db(self, prisma_client: PrismaClient):
        from litellm.proxy.guardrails.guardrail_registry import (
            IN_MEMORY_GUARDRAIL_HANDLER,
            Guardrail,
            GuardrailRegistry,
        )

        try:
            guardrails_in_db: List[Guardrail] = (
                await GuardrailRegistry.get_all_guardrails_from_db(
                    prisma_client=prisma_client
                )
            )
            verbose_proxy_logger.debug(
                "guardrails from the DB %s", str(guardrails_in_db)
            )
            for guardrail in guardrails_in_db:
                IN_MEMORY_GUARDRAIL_HANDLER.sync_guardrail_from_db(
                    guardrail=cast(Guardrail, guardrail),
                )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_guardrails_in_db - {}".format(
                    str(e)
                )
            )

    async def _init_vector_stores_in_db(self, prisma_client: PrismaClient):
        from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

        try:
            # read vector stores from db table
            vector_stores = await VectorStoreRegistry._get_vector_stores_from_db(
                prisma_client=prisma_client
            )
            if len(vector_stores) <= 0:
                return

            if litellm.vector_store_registry is None:
                litellm.vector_store_registry = VectorStoreRegistry(
                    vector_stores=vector_stores
                )
            else:
                for vector_store in vector_stores:
                    litellm.vector_store_registry.add_vector_store_to_registry(
                        vector_store=vector_store
                    )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_vector_stores_in_db - {}".format(
                    str(e)
                )
            )

    async def _init_vector_store_indexes_in_db(self, prisma_client: PrismaClient):
        from litellm.vector_stores.vector_store_registry import VectorStoreIndexRegistry

        try:
            # read vector stores from db table
            vector_store_indexes = (
                await VectorStoreIndexRegistry._get_vector_store_indexes_from_db(
                    prisma_client=prisma_client
                )
            )

            if len(vector_store_indexes) <= 0:
                return

            if litellm.vector_store_index_registry is None:
                litellm.vector_store_index_registry = VectorStoreIndexRegistry(
                    vector_store_indexes=vector_store_indexes
                )
            else:
                for vector_store_index in vector_store_indexes:
                    litellm.vector_store_index_registry.upsert_vector_store_index(
                        vector_store_index=vector_store_index
                    )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_vector_stores_in_db - {}".format(
                    str(e)
                )
            )

    async def _init_mcp_servers_in_db(self):
        from litellm.proxy._experimental.mcp_server.utils import is_mcp_available

        if not is_mcp_available():
            verbose_proxy_logger.debug(
                "MCP module not available, skipping MCP server initialization"
            )
            return

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        try:
            await global_mcp_server_manager._add_mcp_servers_from_db_to_in_memory_registry()
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_mcp_servers_in_db - {}".format(
                    str(e)
                )
            )

    async def _init_agents_in_db(self, prisma_client: PrismaClient):
        from litellm.proxy.agent_endpoints.agent_registry import (
            global_agent_registry as AGENT_REGISTRY,
        )

        try:
            db_agents = await AGENT_REGISTRY.get_all_agents_from_db(
                prisma_client=prisma_client
            )
            AGENT_REGISTRY.load_agents_from_db_and_config(
                db_agents=db_agents, agent_config=config_agents
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_agents_in_db - {}".format(
                    str(e)
                )
            )

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
            search_tools = await SearchToolRegistry.get_all_search_tools_from_db(
                prisma_client=prisma_client
            )

            verbose_proxy_logger.info(
                f"Loading {len(search_tools)} search tool(s) from database into router"
            )

            if llm_router is not None:
                # Add search tools to the router
                await SearchAPIRouter.update_router_search_tools(
                    router_instance=llm_router, search_tools=search_tools
                )
                verbose_proxy_logger.info(
                    f"Successfully loaded {len(search_tools)} search tool(s) into router"
                )
            else:
                verbose_proxy_logger.debug(
                    "Router not initialized yet, search tools will be added when router is created"
                )

        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:_init_search_tools_in_db - {}".format(
                    str(e)
                )
            )

    async def _init_pass_through_endpoints_in_db(self):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            initialize_pass_through_endpoints_in_db,
        )

        await initialize_pass_through_endpoints_in_db()

    def decrypt_credentials(self, credential: Union[dict, BaseModel]) -> CredentialItem:
        if isinstance(credential, dict):
            credential_object = CredentialItem(**credential)
        elif isinstance(credential, BaseModel):
            credential_object = CredentialItem(**credential.model_dump())

        decrypted_credential_values = {}
        for k, v in credential_object.credential_values.items():
            decrypted_credential_values[k] = decrypt_value_helper(value=v, key=k) or v

        credential_object.credential_values = decrypted_credential_values
        return credential_object

    async def delete_credentials(self, db_credentials: List[CredentialItem]):
        """
        Create all-up list of db credentials + local credentials
        Compare to the litellm.credential_list
        Delete any from litellm.credential_list that are not in the all-up list
        """
        ## CONFIG credentials ##
        config = await self.get_config(config_file_path=user_config_file_path)
        credential_list = self.load_credential_list(config=config)

        ## COMBINED LIST ##
        combined_list = db_credentials + credential_list

        ## DELETE ##
        idx_to_delete = []
        for idx, credential in enumerate(litellm.credential_list):
            if credential.credential_name not in [
                cred.credential_name for cred in combined_list
            ]:
                idx_to_delete.append(idx)
        for idx in sorted(idx_to_delete, reverse=True):
            litellm.credential_list.pop(idx)

    async def get_credentials(self, prisma_client: PrismaClient):
        try:
            credentials = await prisma_client.db.litellm_credentialstable.find_many()
            credentials = [self.decrypt_credentials(cred) for cred in credentials]
            await self.delete_credentials(
                credentials
            )  # delete credentials that are not in the all-up list
            CredentialAccessor.upsert_credentials(
                credentials
            )  # upsert credentials that are in the all-up list
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy_server.py::get_credentials() - Error getting credentials from DB - {}".format(
                    str(e)
                )
            )
            return []


proxy_config = ProxyConfig()


def save_worker_config(**data):
    import json

    os.environ["WORKER_CONFIG"] = json.dumps(data)


async def initialize(  # noqa: PLR0915
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
    global user_model, user_api_base, user_debug, user_detailed_debug, user_user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers, experimental, llm_model_list, llm_router, general_settings, master_key, user_custom_auth, prisma_client
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
        litellm_log_setting = os.environ.get("LITELLM_LOG", "")
        if litellm_log_setting is not None:
            if litellm_log_setting.upper() == "INFO":
                import logging

                from litellm._logging import verbose_proxy_logger, verbose_router_logger

                # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS

                verbose_router_logger.setLevel(
                    level=logging.INFO
                )  # set router logs to info
                verbose_proxy_logger.setLevel(
                    level=logging.INFO
                )  # set proxy logs to info
            elif litellm_log_setting.upper() == "DEBUG":
                import logging

                from litellm._logging import verbose_proxy_logger, verbose_router_logger

                verbose_router_logger.setLevel(
                    level=logging.DEBUG
                )  # set router logs to info
                verbose_proxy_logger.setLevel(
                    level=logging.DEBUG
                )  # set proxy logs to debug
    dynamic_config = {"general": {}, user_model: {}}
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
        os.environ["AZURE_API_VERSION"] = (
            api_version  # set this for azure - litellm can read this from the env
        )
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
        litellm.max_budget = max_budget
        dynamic_config["general"]["max_budget"] = max_budget
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


async def async_assistants_data_generator(
    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
):
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
            async for c in chunk:  # type: ignore
                c = c.model_dump_json(exclude_none=True)
                try:
                    yield f"data: {c}\n\n"
                except Exception as e:
                    yield f"data: {str(e)}\n\n"

        # Streaming is done, yield the [DONE] chunk
        done_message = "[DONE]"
        yield f"data: {done_message}\n\n"
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.async_assistants_data_generator(): Exception occured - {}".format(
                str(e)
            )
        )
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_data,
        )
        verbose_proxy_logger.debug(
            f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
        )
        if isinstance(e, HTTPException):
            raise e
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"

        proxy_exception = ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )
        error_returned = json.dumps({"error": proxy_exception.to_dict()})
        yield f"data: {error_returned}\n\n"


async def async_data_generator(
    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
):
    verbose_proxy_logger.debug("inside generator")
    try:
        str_so_far = ""
        error_message: Optional[str] = None
        async for chunk in proxy_logging_obj.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_data=request_data,
        ):
            verbose_proxy_logger.debug(
                "async_data_generator: received streaming chunk - {}".format(chunk)
            )

            ### CALL HOOKS ### - modify outgoing data
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict,
                response=chunk,
                data=request_data,
                str_so_far=str_so_far,
            )

            if isinstance(chunk, (ModelResponse, ModelResponseStream)):
                response_str = litellm.get_response_string(response_obj=chunk)
                str_so_far += response_str

            if isinstance(chunk, BaseModel):
                chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            elif isinstance(chunk, str) and chunk.startswith("data: "):
                error_message = chunk
                break

            try:
                yield f"data: {chunk}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        # Streaming is done, yield the [DONE] chunk
        if error_message is not None:
            yield error_message
        done_message = "[DONE]"
        yield f"data: {done_message}\n\n"
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.async_data_generator(): Exception occured - {}".format(
                str(e)
            )
        )
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_data,
        )
        verbose_proxy_logger.debug(
            f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
        )

        if isinstance(e, HTTPException):
            raise e
        elif isinstance(e, StreamingCallbackError):
            error_msg = str(e)
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"

        proxy_exception = ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )
        error_returned = json.dumps({"error": proxy_exception.to_dict()})
        yield f"data: {error_returned}\n\n"


def select_data_generator(
    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
):
    return async_data_generator(
        response=response,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
    )


def get_litellm_model_info(model: dict = {}):
    model_info = model.get("model_info", {})
    model_to_lookup = model.get("litellm_params", {}).get("model", None)
    try:
        if "azure" in model_to_lookup:
            model_to_lookup = model_info.get("base_model", None)
        litellm_model_info = litellm.get_model_info(model_to_lookup)
        return litellm_model_info
    except Exception:
        # this should not block returning on /model/info
        # if litellm does not have info on the model it should return {}
        return {}


def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    verbose_proxy_logger.debug("Backing off... this was attempt # %s", details["tries"])


def giveup(e):
    result = not (
        isinstance(e, ProxyException)
        and getattr(e, "message", None) is not None
        and isinstance(e.message, str)
        and "Max parallel request limit reached" in e.message
    )

    if (
        general_settings.get("disable_retry_on_max_parallel_request_limit_error")
        is True
    ):
        return True  # giveup if queuing max parallel request limits is disabled

    if result:
        verbose_proxy_logger.debug(json.dumps({"event": "giveup", "exception": str(e)}))
    return result


class ProxyStartupEvent:
    @classmethod
    def _initialize_startup_logging(
        cls,
        llm_router: Optional[Router],
        proxy_logging_obj: ProxyLogging,
        redis_usage_cache: Optional[RedisCache],
    ):
        """Initialize logging and alerting on startup"""
        ## COST TRACKING ##
        cost_tracking()

        proxy_logging_obj.startup_event(
            llm_router=llm_router, redis_usage_cache=redis_usage_cache
        )

    @classmethod
    def _initialize_jwt_auth(
        cls,
        general_settings: dict,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
    ):
        """Initialize JWT auth on startup"""
        if general_settings.get("litellm_jwtauth", None) is not None:
            for k, v in general_settings["litellm_jwtauth"].items():
                if isinstance(v, str) and v.startswith("os.environ/"):
                    general_settings["litellm_jwtauth"][k] = get_secret(v)
            litellm_jwtauth = LiteLLM_JWTAuth(**general_settings["litellm_jwtauth"])
        else:
            litellm_jwtauth = LiteLLM_JWTAuth()
        jwt_handler.update_environment(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_jwtauth=litellm_jwtauth,
        )

    @classmethod
    def _add_proxy_budget_to_db(cls, litellm_proxy_budget_name: str):
        """Adds a global proxy budget to db"""
        if litellm.budget_duration is None:
            raise Exception(
                "budget_duration not set on Proxy. budget_duration is required to use max_budget."
            )

        # add proxy budget to db in the user table
        asyncio.create_task(
            generate_key_helper_fn(  # type: ignore
                request_type="user",
                table_name="user",
                user_id=litellm_proxy_budget_name,
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
        )

    @classmethod
    async def _update_default_team_member_budget(cls):
        """Update the default team member budget"""
        if litellm.default_internal_user_params is None:
            return

        _teams = litellm.default_internal_user_params.get("teams") or []
        if _teams and all(isinstance(team, dict) for team in _teams):
            from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
                update_default_team_member_budget,
            )

            teams_pydantic_obj = [NewUserRequestTeam(**team) for team in _teams]
            await update_default_team_member_budget(
                teams=teams_pydantic_obj, user_api_key_dict=UserAPIKeyAuth(token=hash_token(master_key))  # type: ignore
            )

    @classmethod
    async def initialize_scheduled_background_jobs(
        cls,
        general_settings: dict,
        prisma_client: PrismaClient,
        proxy_budget_rescheduler_min_time: int,
        proxy_budget_rescheduler_max_time: int,
        proxy_batch_write_at: int,
        proxy_logging_obj: ProxyLogging,
    ):
        """Initializes scheduled background jobs"""
        global store_model_in_db

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
        budget_interval = proxy_budget_rescheduler_min_time + random.randint(
            0,
            min(
                30,
                proxy_budget_rescheduler_max_time - proxy_budget_rescheduler_min_time,
            ),
        )

        # Ensure minimum interval of 30 seconds for batch writing to prevent memory issues
        batch_writing_interval = proxy_batch_write_at + random.randint(0, 5)

        ### RESET BUDGET ###
        if general_settings.get("disable_reset_budget", False) is False:
            budget_reset_job = ResetBudgetJob(
                proxy_logging_obj=proxy_logging_obj,
                prisma_client=prisma_client,
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

        ### ADD NEW MODELS ###
        store_model_in_db = (
            get_secret_bool("STORE_MODEL_IN_DB", store_model_in_db) or store_model_in_db
        )

        if store_model_in_db is True:
            # MEMORY LEAK FIX: Increase interval from 10s to 30s minimum
            # Frequent polling was causing excessive memory allocations
            scheduler.add_job(
                proxy_config.add_deployment,
                "interval",
                seconds=30,  # increased from 10s to reduce memory pressure
                # REMOVED jitter parameter - major cause of memory leak
                args=[prisma_client, proxy_logging_obj],
                id="add_deployment_job",
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )

            # this will load all existing models on proxy startup
            await proxy_config.add_deployment(
                prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
            )

            ### GET STORED CREDENTIALS ###
            scheduler.add_job(
                proxy_config.get_credentials,
                "interval",
                seconds=30,  # increased from 10s to reduce memory pressure
                # REMOVED jitter parameter - major cause of memory leak
                args=[prisma_client],
                id="get_credentials_job",
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )
            await proxy_config.get_credentials(prisma_client=prisma_client)
        if (
            proxy_logging_obj is not None
            and proxy_logging_obj.slack_alerting_instance.alerting is not None
            and prisma_client is not None
        ):
            print("Alerting: Initializing Weekly/Monthly Spend Reports")  # noqa
            ### Schedule weekly/monthly spend reports ###
            ### Schedule spend reports ###
            spend_report_frequency: str = (
                general_settings.get("spend_report_frequency", "7d") or "7d"
            )

            # Parse the frequency
            days = int(spend_report_frequency[:-1])
            if spend_report_frequency[-1].lower() != "d":
                raise ValueError(
                    "spend_report_frequency must be specified in days, e.g., '1d', '7d'"
                )

            scheduler.add_job(
                proxy_logging_obj.slack_alerting_instance.send_weekly_spend_report,
                "interval",
                days=days,
                # REMOVED jitter parameter - major cause of memory leak
                # Use random start time instead for distribution
                next_run_time=datetime.now()
                + timedelta(
                    seconds=10 + random.randint(0, 300)
                ),  # Random 0-5 min offset
                args=[spend_report_frequency],
                id="weekly_spend_report_job",
                replace_existing=True,
                misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
            )

            scheduler.add_job(
                proxy_logging_obj.slack_alerting_instance.send_monthly_spend_report,
                "cron",
                day=1,
                id="monthly_spend_report_job",
                replace_existing=True,
            )

            # Beta Feature - only used when prometheus api is in .env
            if os.getenv("PROMETHEUS_URL"):
                from zoneinfo import ZoneInfo

                scheduler.add_job(
                    proxy_logging_obj.slack_alerting_instance.send_fallback_stats_from_prometheus,
                    "cron",
                    hour=PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS,
                    minute=0,
                    timezone=ZoneInfo("America/Los_Angeles"),  # Pacific Time
                    id="prometheus_fallback_stats_job",
                    replace_existing=True,
                )
                await proxy_logging_obj.slack_alerting_instance.send_fallback_stats_from_prometheus()

        await cls._initialize_spend_tracking_background_jobs(scheduler=scheduler)

        ### SPEND LOG CLEANUP ###
        if general_settings.get("maximum_spend_logs_retention_period") is not None:
            spend_log_cleanup = SpendLogCleanup()
            # Get the interval from config or default to 1 day
            retention_interval = general_settings.get(
                "maximum_spend_logs_retention_interval", "1d"
            )
            try:
                interval_seconds = duration_in_seconds(retention_interval)
                scheduler.add_job(
                    spend_log_cleanup.cleanup_old_spend_logs,
                    "interval",
                    seconds=interval_seconds
                    + random.randint(0, 60),  # Add small random offset
                    # REMOVED jitter parameter - major cause of memory leak
                    args=[prisma_client],
                    id="spend_log_cleanup_job",
                    replace_existing=True,
                    misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                )
            except ValueError:
                verbose_proxy_logger.error(
                    "Invalid maximum_spend_logs_retention_interval value"
                )
        ### CHECK BATCH COST ###
        if llm_router is not None:
            try:
                from litellm_enterprise.proxy.common_utils.check_batch_cost import (
                    CheckBatchCost,
                )

                check_batch_cost_job = CheckBatchCost(
                    proxy_logging_obj=proxy_logging_obj,
                    prisma_client=prisma_client,
                    llm_router=llm_router,
                )
                scheduler.add_job(
                    check_batch_cost_job.check_batch_cost,
                    "interval",
                    seconds=proxy_batch_polling_interval
                    + random.randint(0, 30),  # Add small random offset
                    # REMOVED jitter parameter - major cause of memory leak
                    id="check_batch_cost_job",
                    replace_existing=True,
                    misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
                )
                verbose_proxy_logger.info("Batch cost check job scheduled successfully")

            except Exception as e:
                verbose_proxy_logger.error(f"Failed to setup batch cost checking: {e}")
                verbose_proxy_logger.debug(
                    "Checking batch cost for LiteLLM Managed Files is an Enterprise Feature. Skipping..."
                )
                pass

        # MEMORY LEAK FIX: Start scheduler with paused=False to avoid backlog processing
        # Do NOT reset job times to "now" as this can trigger the memory leak
        # The misfire_grace_time and coalesce settings will handle any missed runs properly

        # Start the scheduler immediately without processing backlogs
        scheduler.start(paused=False)
        verbose_proxy_logger.info(
            f"APScheduler started with memory leak prevention settings: "
            f"removed jitter, increased intervals, misfire_grace_time={APSCHEDULER_MISFIRE_GRACE_TIME}"
        )

    @classmethod
    async def _initialize_spend_tracking_background_jobs(
        cls, scheduler: AsyncIOScheduler
    ):
        """
        Initialize the spend tracking and other background jobs
        1. CloudZero Background Job
        2. Prometheus Background Job
        3. Key Rotation Background Job

        Args:
            scheduler: The scheduler to add the background jobs to
        """
        ########################################################
        # CloudZero Background Job
        ########################################################
        from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger
        from litellm.proxy.spend_tracking.cloudzero_endpoints import is_cloudzero_setup

        if await is_cloudzero_setup():
            await CloudZeroLogger.init_cloudzero_background_job(scheduler=scheduler)

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

        key_rotation_enabled: Optional[bool] = str_to_bool(LITELLM_KEY_ROTATION_ENABLED)
        verbose_proxy_logger.debug(f"key_rotation_enabled: {key_rotation_enabled}")

        if key_rotation_enabled is True:
            try:
                from litellm.proxy.common_utils.key_rotation_manager import (
                    KeyRotationManager,
                )

                # Get prisma_client from global scope
                global prisma_client
                if prisma_client is not None:
                    key_rotation_manager = KeyRotationManager(prisma_client)
                    verbose_proxy_logger.debug(
                        f"Key rotation background job scheduled every {LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS} seconds (LITELLM_KEY_ROTATION_ENABLED=true)"
                    )
                    scheduler.add_job(
                        key_rotation_manager.process_rotations,
                        "interval",
                        seconds=LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS,
                        id="key_rotation_job",
                    )
                else:
                    verbose_proxy_logger.warning(
                        "Key rotation enabled but prisma_client not available"
                    )
            except Exception as e:
                verbose_proxy_logger.warning(f"Failed to setup key rotation job: {e}")
        else:
            verbose_proxy_logger.debug(
                "Key rotation disabled (set LITELLM_KEY_ROTATION_ENABLED=true to enable)"
            )

    @classmethod
    async def _setup_prisma_client(
        cls,
        database_url: Optional[str],
        proxy_logging_obj: ProxyLogging,
        user_api_key_cache: DualCache,
    ) -> Optional[PrismaClient]:
        """
        - Sets up prisma client
        - Adds necessary views to proxy
        """
        try:
            prisma_client: Optional[PrismaClient] = None
            if database_url is not None:
                try:
                    prisma_client = PrismaClient(
                        database_url=database_url, proxy_logging_obj=proxy_logging_obj
                    )
                except Exception as e:
                    raise e

                await prisma_client.connect()

                ## Add necessary views to proxy ##
                asyncio.create_task(
                    prisma_client.check_view_exists()
                )  # check if all necessary views exist. Don't block execution

                asyncio.create_task(
                    prisma_client._set_spend_logs_row_count_in_proxy_state()
                )  # set the spend logs row count in proxy state. Don't block execution

                # run a health check to ensure the DB is ready
                if (
                    get_secret_bool("DISABLE_PRISMA_HEALTH_CHECK_ON_STARTUP", False)
                    is not True
                ):
                    await prisma_client.health_check()
            return prisma_client
        except Exception as e:
            PrismaDBExceptionHandler.handle_db_exception(e)
            return None

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

            prof = Profiler()
            prof.start()
            verbose_proxy_logger.debug("Datadog Profiler started......")


#### API ENDPOINTS ####
@router.get(
    "/v1/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)
@router.get(
    "/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)  # if project requires model list
async def model_list(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    return_wildcard_routes: Optional[bool] = False,
    team_id: Optional[str] = None,
    include_model_access_groups: Optional[bool] = False,
    only_model_access_groups: Optional[bool] = False,
    include_metadata: Optional[bool] = False,
    fallback_type: Optional[str] = None,
):
    """
    Use `/model/info` - to get detailed model information, example - pricing, mode, etc.

    This is just for compatibility with openai projects like aider.

    Query Parameters:
    - include_metadata: Include additional metadata in the response with fallback information
    - fallback_type: Type of fallbacks to include ("general", "context_window", "content_policy")
                    Defaults to "general" when include_metadata=true
    """
    global llm_model_list, general_settings, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj

    from litellm.proxy.utils import (
        create_model_info_response,
        get_available_models_for_user,
    )

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

    # Build response data
    model_data = []
    for model in all_models:
        model_info = create_model_info_response(
            model_id=model,
            provider="openai",
            include_metadata=include_metadata or False,
            fallback_type=fallback_type,
            llm_router=llm_router,
        )
        model_data.append(model_info)

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
):
    """
    Retrieve information about a specific model accessible to your API key.

    Returns model details only if the model is available to your API key/team.
    Returns 404 if the model doesn't exist or is not accessible.

    Follows OpenAI API specification for individual model retrieval.
    https://platform.openai.com/docs/api-reference/models/retrieve
    """
    global llm_model_list, general_settings, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj

    from litellm.proxy.utils import (
        create_model_info_response,
        get_available_models_for_user,
        validate_model_access,
    )

    # Get available models for the user
    all_models = await get_available_models_for_user(
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

    # Validate that the requested model is accessible
    validate_model_access(model_id=model_id, available_models=all_models)

    # Get provider information from the router deployment
    if llm_router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")

    deployment = llm_router.get_deployment_by_model_group_name(model_id)
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found in router configuration",
        )

    # Use the actual litellm model from the deployment to get provider info
    _, provider, _, _ = litellm.get_llm_provider(model=deployment.litellm_params.model)

    # Return the model information in the same format as the list endpoint
    return create_model_info_response(
        model_id=model_id,
        provider=provider,
        include_metadata=False,
        fallback_type=None,
        llm_router=llm_router,
    )


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
async def chat_completion(  # noqa: PLR0915
    request: Request,
    fastapi_response: Response,
    model: Optional[str] = None,
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
    data = await _read_request_body(request=request)
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result = await base_llm_response_processor.base_process_llm_request(
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
            return result.model_dump(exclude_none=True, exclude_unset=True)
        else:
            return result
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        _chat_response = litellm.ModelResponse()
        _chat_response.choices[0].message.content = e.message  # type: ignore

        if data.get("stream", None) is not None and data["stream"] is True:
            _iterator = litellm.utils.ModelResponseIterator(
                model_response=_chat_response, convert_to_delta=True
            )
            _streaming_response = litellm.CustomStreamWrapper(
                completion_stream=_iterator,
                model=data.get("model", ""),
                custom_llm_provider="cached_response",
                logging_obj=data.get("litellm_logging_obj", None),
            )
            selected_data_generator = select_data_generator(
                response=_streaming_response,
                user_api_key_dict=user_api_key_dict,
                request_data=_data,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                status_code=(
                    e.status_code
                    if hasattr(e, "status_code")
                    else status.HTTP_400_BAD_REQUEST
                ),
            )
        _usage = litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        _chat_response.usage = _usage  # type: ignore
        return _chat_response
    except Exception as e:
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


@router.post(
    "/v1/completions", dependencies=[Depends(user_api_key_auth)], tags=["completions"]
)
@router.post(
    "/completions", dependencies=[Depends(user_api_key_auth)], tags=["completions"]
)
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
async def completion(  # noqa: PLR0915
    request: Request,
    fastapi_response: Response,
    model: Optional[str] = None,
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
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
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
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        if _data.get("stream", None) is not None and _data["stream"] is True:
            _chat_response = litellm.ModelResponse()
            _usage = litellm.Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
            _chat_response.usage = _usage  # type: ignore
            _chat_response.choices[0].message.content = e.message  # type: ignore
            _iterator = litellm.utils.ModelResponseIterator(
                model_response=_chat_response, convert_to_delta=True
            )
            _streaming_response = litellm.TextCompletionStreamWrapper(
                completion_stream=_iterator,
                model=_data.get("model", ""),
            )

            selected_data_generator = select_data_generator(
                response=_streaming_response,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                headers={},
                status_code=(
                    e.status_code
                    if hasattr(e, "status_code")
                    else status.HTTP_400_BAD_REQUEST
                ),
            )
        else:
            _response = litellm.TextCompletionResponse()
            _response.choices[0].text = e.message
            return _response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.completion(): Exception occured - {}".format(
                str(e)
            )
        )
        error_msg = f"{str(e)}"
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
async def embeddings(  # noqa: PLR0915
    request: Request,
    fastapi_response: Response,
    model: Optional[str] = None,
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
    data: Any = {}
    try:
        # Use shared request body reading helper (same as chat/completions)
        data = await _read_request_body(request=request)

        ### HANDLE TOKEN ARRAY INPUT DECODING ###
        # This must happen BEFORE base_process_llm_request() since it modifies the input
        router_model_names = llm_router.model_names if llm_router is not None else []
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
                deployment = llm_router.get_deployment(model_id=data["model"])
                if deployment is not None:
                    litellm_params = deployment.get("litellm_params", {}) or {}
                    litellm_model = litellm_params.get("model", "")
                    # Check if this provider supports token arrays
                    supports_token_arrays = litellm_model in litellm.open_ai_embedding_models or any(
                        litellm_model.startswith(provider)
                        for provider in LITELLM_EMBEDDING_PROVIDERS_SUPPORTING_INPUT_ARRAY_OF_TOKENS
                    )
                    if not supports_token_arrays:
                        # non-openai/azure embedding model called with token input - decode tokens
                        input_list = []
                        for i in data["input"]:
                            input_list.append(
                                litellm.decode(model="gpt-3.5-turbo", tokens=i)
                            )
                        data["input"] = input_list

        # Use unified request processor (same as chat/completions and responses)
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)

        # Process the request with all optimizations (shared sessions, network tuning, etc.)
        response = await base_llm_response_processor.base_process_llm_request(
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
        # Use unified error handler
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
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
    data: Dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
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
        llm_call = await route_request(
            data=data,
            route_type="amoderation",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.moderations(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


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
    data: Dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
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
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call = await route_request(
            data=data,
            route_type="aspeech",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        litellm_call_id = hidden_params.get("litellm_call_id", None) or ""

        # Printing each chunk size
        async def generate(_response: HttpxBinaryResponseContent):
            _generator = await _response.aiter_bytes(chunk_size=1024)
            async for chunk in _generator:
                yield chunk

        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
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

        select_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=data,
        )
        # Determine media type based on model type
        media_type = "audio/mpeg"  # Default for OpenAI TTS
        request_model = data.get("model", "")
        if "gemini" in request_model.lower() and (
            "tts" in request_model.lower() or "preview-tts" in request_model.lower()
        ):
            media_type = "audio/wav"  # Gemini TTS returns WAV format after conversion

        return StreamingResponse(
            generate(response), media_type=media_type, headers=custom_headers  # type: ignore
        )

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.audio_speech(): Exception occured - {}".format(
                str(e)
            )
        )
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
    data: Dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        form_data = await get_form_data(request)
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

        router_model_names = llm_router.model_names if llm_router is not None else []

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

        file_content = await file.read()
        file_object = io.BytesIO(file_content)
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
            llm_call = await route_request(
                data=data,
                route_type="atranscription",
                llm_router=llm_router,
                user_model=user_model,
            )
            response = await llm_call
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            file_object.close()  # close the file read in by io library

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        litellm_call_id = hidden_params.get("litellm_call_id", None) or ""
        additional_headers: dict = hidden_params.get("additional_headers", {}) or {}

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

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.audio_transcription(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    model: Optional[str] = fastapi.Query(
        None,
        description="Optional model name, used to determine Vertex region for global models.",
    ),
    vertex_project: Optional[str] = fastapi.Query(
        None,
        description="Override the Vertex AI project id used for the upstream connection.",
    ),
    vertex_location: Optional[str] = fastapi.Query(
        None,
        description="Override the Vertex AI region (for example, 'us-central1').",
    ),
    user_api_key_dict=Depends(user_api_key_auth_websocket),
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
from litellm import _arealtime


@app.websocket("/v1/realtime")
@app.websocket("/realtime")
async def websocket_endpoint(
    websocket: WebSocket,
    model: str,
    intent: str = fastapi.Query(
        None, description="The intent of the websocket connection."
    ),
    user_api_key_dict=Depends(user_api_key_auth_websocket),
):
    import websockets

    await websocket.accept()

    # Only use explicit parameters, not all query params
    query_params: RealtimeQueryParams = {"model": model}
    if intent is not None:
        query_params["intent"] = intent

    data = {
        "model": model,
        "websocket": websocket,
        "query_params": query_params,  # Only explicit params
    }

    headers = dict(websocket.headers.items())  # Convert headers to dict first

    request = Request(
        scope={
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "method": "POST",
            "path": "/v1/realtime",
        }
    )

    request._url = websocket.url

    async def return_body():
        return_string = f'{{"model": "{model}"}}'
        # return string as bytes
        return return_string.encode()

    request.body = return_body  # type: ignore

    ### ROUTE THE REQUEST ###
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
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
            model=model,
            route_type="_arealtime",
        )
        llm_call = await route_request(
            data=data,
            route_type="_arealtime",
            llm_router=llm_router,
            user_model=user_model,
        )

        await llm_call
    except websockets.exceptions.InvalidStatusCode as e:  # type: ignore
        verbose_proxy_logger.exception("Invalid status code")
        await websocket.close(code=e.status_code, reason="Invalid status code")
    except Exception:
        verbose_proxy_logger.exception("Internal server error")
        await websocket.close(code=1011, reason="Internal server error")


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
    data: Dict = {}
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.aget_assistants(**data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.get_assistants(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
        body = await request.body()
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.acreate_assistants(**data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.create_assistant(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    data: Dict = {}
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.adelete_assistant(assistant_id=assistant_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.delete_assistant(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    data: Dict = {}
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.acreate_thread(**data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.create_threads(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    data: Dict = {}
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.aget_thread(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.get_thread(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    data: Dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.a_add_message(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.add_messages(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    data: Dict = {}
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.aget_messages(thread_id=thread_id, **data)

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.get_messages(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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
    data: Dict = {}
    try:
        body = await request.body()
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.arun_thread(thread_id=thread_id, **data)

        if (
            "stream" in data and data["stream"] is True
        ):  # use generate_responses to stream responses
            return await create_streaming_response(
                generator=async_assistants_data_generator(
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                    request_data=data,
                ),
                media_type="text/event-stream",
                headers={},  # Added empty headers dict, original call missed this argument
            )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.run_thread(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
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


def _get_provider_token_counter(
    deployment: dict, model_to_use: str
) -> Tuple[Optional[BaseTokenCounter], Optional[str], Optional[str]]:
    """
    Auto-route to the correct provider's token counter based on model/deployment.
    Uses the existing get_provider_model_info infrastructure with switch-case pattern.
    """
    if deployment is None:
        return None

    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    full_model = deployment.get("litellm_params", {}).get("model", "")
    model: Optional[str] = None
    custom_llm_provider: Optional[str] = None

    try:
        # Use existing LiteLLM logic to determine provider
        model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
            model=full_model,
            custom_llm_provider=deployment.get("litellm_params", {}).get(
                "custom_llm_provider"
            ),
            api_base=deployment.get("litellm_params", {}).get("api_base"),
            api_key=deployment.get("litellm_params", {}).get("api_key"),
        )

        # Switch case pattern using existing get_provider_model_info
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        # Convert string provider to LlmProviders enum
        llm_provider_enum = LlmProviders(custom_llm_provider)
        # Add more provider mappings as needed

        if llm_provider_enum:
            provider_model_info = ProviderConfigManager.get_provider_model_info(
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

            anthropic_model_info = AnthropicModelInfo()
            return anthropic_model_info.get_token_counter(), model, custom_llm_provider

    return None, None, None


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
    from litellm import token_counter

    global llm_router

    prompt = request.prompt
    messages = request.messages
    contents = request.contents

    #########################################################
    # Validate request
    #########################################################
    if prompt is None and messages is None and contents is None:
        raise HTTPException(
            status_code=400, detail="prompt or messages or contents must be provided"
        )

    deployment: Optional[Dict[str, Any]] = None
    litellm_model_name = None
    model_info: Optional[ModelMapInfo] = None
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
            pass
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
    provider_counter: Optional[BaseTokenCounter] = None
    custom_llm_provider: Optional[str] = None
    if call_endpoint is True and deployment is not None:
        # Auto-route to the correct provider based on model
        provider_counter, _model, custom_llm_provider = _get_provider_token_counter(
            deployment, model_to_use
        )
        if _model is not None:
            model_to_use = _model

    if provider_counter is not None:
        if (
            provider_counter.should_use_token_counting_api(
                custom_llm_provider=custom_llm_provider
            )
            is True
        ):
            result = await provider_counter.count_tokens(
                model_to_use=model_to_use or "",
                messages=messages,  # type: ignore
                contents=contents,
                deployment=deployment,
                request_model=request.model,
            )
            #########################################################
            # Transfrom the Response to the well known format
            #########################################################
            if result is not None:
                return result

    # Default LiteLLM token counting
    custom_tokenizer: Optional[CustomHuggingfaceTokenizer] = None
    if model_info is not None:
        custom_tokenizer = cast(
            Optional[CustomHuggingfaceTokenizer],
            model_info.get("custom_tokenizer", None),
        )
    _tokenizer_used = litellm.utils._select_tokenizer(
        model=model_to_use, custom_tokenizer=custom_tokenizer
    )

    tokenizer_used = str(_tokenizer_used["type"])
    total_tokens = token_counter(
        model=model_to_use,
        text=prompt,
        messages=messages,
        custom_tokenizer=_tokenizer_used,  # type: ignore
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
        raise HTTPException(
            status_code=400, detail={"error": "Could not map model={}".format(model)}
        )


@router.post(
    "/utils/transform_request",
    tags=["llm utils"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RawRequestTypedDict,
)
async def transform_request(request: TransformRequestBody):
    from litellm.utils import return_raw_request

    return return_raw_request(endpoint=request.call_type, kwargs=request.request_body)


async def _check_if_model_is_user_added(
    models: List[Dict],
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Optional[PrismaClient],
) -> List[Dict]:
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
    filtered_models = []
    for model in models:
        id = model.get("model_info", {}).get("id", None)
        if id is None:
            continue
        db_model = await prisma_client.db.litellm_proxymodeltable.find_unique(
            where={"model_id": id}
        )
        if db_model is not None:
            if db_model.created_by == user_api_key_dict.user_id:
                filtered_models.append(model)
    return filtered_models


def _check_if_model_is_team_model(
    models: List[DeploymentTypedDict], user_row: LiteLLM_UserTable
) -> List[Dict]:
    """
    Check if model is a team model

    Check if user is a member of the team that the model belongs to
    """

    user_team_models: List[Dict] = []
    for model in models:
        model_team_id = model.get("model_info", {}).get("team_id", None)

        if model_team_id is not None:
            if model_team_id in user_row.teams:
                user_team_models.append(cast(Dict, model))

    return user_team_models


async def non_admin_all_models(
    all_models: List[Dict],
    llm_router: Router,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Optional[PrismaClient],
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
            user_row = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_api_key_dict.user_id}
            )
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "User not found"})

        # Get all models that are team models, when model team_id == user_row.teams
        all_models += _check_if_model_is_team_model(
            models=llm_router.get_model_list() or [],
            user_row=user_row,
        )

    # de-duplicate models. Only return unique model ids
    unique_models = _deduplicate_litellm_router_models(models=all_models)
    return unique_models


def _add_team_models_to_all_models(
    team_db_objects_typed: List[LiteLLM_TeamTable],
    llm_router: Router,
) -> Dict[str, Set[str]]:
    """
    Add team models to all models
    """
    team_models: Dict[str, Set[str]] = {}

    for team_object in team_db_objects_typed:
        if (
            len(team_object.models) == 0  # empty list = all model access
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
                    if team_model_id is None:
                        can_add_model = True
                    elif team_model_id in team_object.team_id:
                        can_add_model = True

                    if can_add_model:
                        team_models.setdefault(model_id, set()).add(team_object.team_id)
        else:
            for model_name in team_object.models:
                _models = llm_router.get_model_list(
                    model_name=model_name, team_id=team_object.team_id
                )
                if _models is not None:
                    for model in _models:
                        model_id = model.get("model_info", {}).get("id", None)
                        if model_id is not None:
                            team_models.setdefault(model_id, set()).add(
                                team_object.team_id
                            )
    return team_models


async def get_all_team_models(
    user_teams: Union[List[str], Literal["*"]],
    prisma_client: PrismaClient,
    llm_router: Router,
) -> Dict[str, List[str]]:
    """
    Get all models across all teams user is in.

    1. Get all teams user is in
    2. Get all models across all teams
    3. Return {"model_id": ["team_id1", "team_id2"]}
    """

    team_db_objects_typed: List[LiteLLM_TeamTable] = []

    if user_teams == "*":
        team_db_objects = await prisma_client.db.litellm_teamtable.find_many()
        team_db_objects_typed = [
            LiteLLM_TeamTable(**team_db_object.model_dump())
            for team_db_object in team_db_objects
        ]
    else:
        team_db_objects = await prisma_client.db.litellm_teamtable.find_many(
            where={"team_id": {"in": user_teams}}
        )

        team_db_objects_typed = [
            LiteLLM_TeamTable(**team_db_object.model_dump())
            for team_db_object in team_db_objects
        ]

    team_models = _add_team_models_to_all_models(
        team_db_objects_typed=team_db_objects_typed,
        llm_router=llm_router,
    )

    # convert set to list
    returned_team_models: Dict[str, List[str]] = {}
    for model_id, team_ids in team_models.items():
        returned_team_models[model_id] = list(team_ids)

    return returned_team_models


def get_direct_access_models(
    user_db_object: LiteLLM_UserTable,
    llm_router: Router,
) -> List[str]:
    """
    Get all models that user has direct access to
    """

    direct_access_models: List[str] = []
    for model in user_db_object.models:
        deployments = llm_router.get_model_list(model_name=model)
        if deployments is not None:
            for deployment in deployments:
                model_id = deployment.get("model_info", {}).get("id", None)
                if model_id is not None:
                    direct_access_models.append(model_id)
    return direct_access_models


async def get_all_team_and_direct_access_models(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    llm_router: Router,
    all_models: List[Dict],
) -> List[Dict]:
    """
    Get all models across all teams user is in.
    """

    user_teams: Optional[Union[List[str], Literal["*"]]] = None
    direct_access_models: List[str] = []
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        user_teams = "*"
        direct_access_models = llm_router.get_model_ids(
            exclude_team_models=True
        )  # has access to all models
    elif user_api_key_dict.user_id is not None:
        user_db_object = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_api_key_dict.user_id}
        )
        if user_db_object is not None:
            user_object = LiteLLM_UserTable(**user_db_object.model_dump())
            user_teams = user_object.teams or []
            direct_access_models = get_direct_access_models(
                user_db_object=user_object,
                llm_router=llm_router,
            )
    ## ADD ACCESS_VIA_TEAM_IDS TO ALL MODELS
    if user_teams is not None:
        team_models = await get_all_team_models(
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
                    _model["model_info"]["access_via_team_ids"] = team_models.get(
                        model_id, []
                    )

    ## ADD DIRECT_ACCESS TO RELEVANT MODELS

    for _model in all_models:
        model_id = _model.get("model_info", {}).get("id", None)
        if model_id is not None and model_id in direct_access_models:
            _model["model_info"]["direct_access"] = True

    ## FILTER OUT MODELS THAT ARE NOT IN DIRECT_ACCESS_MODELS OR ACCESS_VIA_TEAM_IDS - only show user models they can call
    all_models = [
        _model
        for _model in all_models
        if _model.get("model_info", {}).get("direct_access", False)
        or _model.get("model_info", {}).get("access_via_team_ids", [])
    ]
    return all_models


@router.get(
    "/v2/model/info",
    description="v2 - returns models available to the user based on their API key permissions. Shows model info from config.yaml (except api key and api base). Filter to just user-added models with ?user_models_only=true",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def model_info_v2(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Specify the model name (optional)"
    ),
    user_models_only: Optional[bool] = fastapi.Query(
        False, description="Only return models added by this user"
    ),
    include_team_models: Optional[bool] = fastapi.Query(
        False, description="Return all models across all teams user is in."
    ),
    debug: Optional[bool] = False,
):
    """
    BETA ENDPOINT. Might change unexpectedly. Use `/v1/model/info` for now.
    """
    global llm_model_list, general_settings, user_config_file_path, proxy_config, llm_router

    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"No model list passed, models router={llm_router}. You can add a model through the config.yaml or on the LiteLLM Admin UI."
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Load existing config
    await proxy_config.get_config()
    all_models = copy.deepcopy(llm_router.model_list)

    if user_model is not None:
        # if user does not use a config.yaml, https://github.com/BerriAI/litellm/issues/2061
        all_models += [user_model]

    if model is not None:
        all_models = [m for m in all_models if m["model_name"] == model]

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
    # fill in model info based on config.yaml and litellm model_prices_and_context_window.json
    for _model in all_models:
        # provided model_info in config.yaml
        model_info = _model.get("model_info", {})
        if debug is True:
            _openai_client = "None"
            if llm_router is not None:
                _openai_client = (
                    llm_router._get_client(
                        deployment=_model, kwargs={}, client_type="async"
                    )
                    or "None"
                )
            else:
                _openai_client = "llm_router_is_None"
            openai_client = str(_openai_client)
            _model["openai_client"] = openai_client

        # read litellm model_prices_and_context_window.json to get the following:
        # input_cost_per_token, output_cost_per_token, max_tokens
        litellm_model_info = get_litellm_model_info(model=_model)

        # 2nd pass on the model, try seeing if we can find model in litellm model_cost map
        if litellm_model_info == {}:
            # use litellm_param model_name to get model_info
            litellm_params = _model.get("litellm_params", {})
            litellm_model = litellm_params.get("model", None)
            try:
                litellm_model_info = litellm.get_model_info(model=litellm_model)
            except Exception:
                litellm_model_info = {}
        # 3rd pass on the model, try seeing if we can find model but without the "/" in model cost map
        if litellm_model_info == {}:
            # use litellm_param model_name to get model_info
            litellm_params = _model.get("litellm_params", {})
            litellm_model = litellm_params.get("model", None)
            split_model = litellm_model.split("/")
            if len(split_model) > 0:
                litellm_model = split_model[-1]
            try:
                litellm_model_info = litellm.get_model_info(
                    model=litellm_model, custom_llm_provider=split_model[0]
                )
            except Exception:
                litellm_model_info = {}
        for k, v in litellm_model_info.items():
            if k not in model_info:
                model_info[k] = v
        _model["model_info"] = model_info
        # don't return the api key / vertex credentials
        # don't return the llm credentials
        _model = remove_sensitive_info_from_deployment(_model)

    verbose_proxy_logger.debug("all_models: %s", all_models)
    return {"data": all_models}


@router.get(
    "/model/streaming_metrics",
    description="View time to first token for models in spend logs",
    tags=["model management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def model_streaming_metrics(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    _selected_model_group: Optional[str] = None,
    startTime: Optional[datetime] = None,
    endTime: Optional[datetime] = None,
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

    is_same_day = startTime.date() == endTime.date()
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

    _all_api_bases = set()
    db_response = await prisma_client.db.query_raw(
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

        response: List[dict] = []

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
    _selected_model_group: Optional[str] = "gpt-4-32k",
    startTime: Optional[datetime] = None,
    endTime: Optional[datetime] = None,
    api_key: Optional[str] = None,
    customer: Optional[str] = None,
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

    sql_query = """
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
    _all_api_bases = set()
    db_response = await prisma_client.db.query_raw(
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

        response: List[dict] = []

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
    _selected_model_group: Optional[str] = "gpt-4-32k",
    startTime: Optional[datetime] = None,
    endTime: Optional[datetime] = None,
    api_key: Optional[str] = None,
    customer: Optional[str] = None,
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
        proxy_logging_obj.slack_alerting_instance.alerting_threshold
        or DEFAULT_SLACK_ALERTING_THRESHOLD
    )
    alerting_threshold = int(alerting_threshold)

    sql_query = """
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

    db_response = await prisma_client.db.query_raw(
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
    _selected_model_group: Optional[str] = None,
    startTime: Optional[datetime] = None,
    endTime: Optional[datetime] = None,
    api_key: Optional[str] = None,
    customer: Optional[str] = None,
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
    sql_query = """
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
    db_response = await prisma_client.db.query_raw(
        sql_query, startTime, endTime, _selected_model_group, api_key
    )
    response: List[dict] = []
    exception_types = set()

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


def _get_proxy_model_info(model: dict) -> dict:
    # provided model_info in config.yaml
    model_info = model.get("model_info", {})

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
        split_model = litellm_model.split("/")
        if len(split_model) > 0:
            litellm_model = split_model[-1]
        try:
            litellm_model_info = litellm.get_model_info(
                model=litellm_model, custom_llm_provider=split_model[0]
            )
        except Exception:
            litellm_model_info = {}
    for k, v in litellm_model_info.items():
        if k not in model_info:
            model_info[k] = v
    model["model_info"] = model_info
    # don't return the llm credentials
    model = remove_sensitive_info_from_deployment(deployment_dict=model)

    return model


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
async def model_info_v1(  # noqa: PLR0915
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    litellm_model_id: Optional[str] = None,
):
    """
    Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)

    Parameters:
        litellm_model_id: Optional[str] = None (this is the value of `x-litellm-model-id` returned in response headers)

        - When litellm_model_id is passed, it will return the info for that specific model
        - When litellm_model_id is not passed, it will return the info for all models

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

    if user_model is not None:
        # user is trying to get specific model from litellm router
        try:
            model_info: Dict = cast(Dict, litellm.get_model_info(model=user_model))
        except Exception:
            model_info = {}
        _deployment_info = Deployment(
            model_name="*",
            litellm_params=LiteLLM_Params(
                model=user_model,
            ),
            model_info=model_info,
        )
        _deployment_info_dict = _deployment_info.model_dump()
        _deployment_info_dict = remove_sensitive_info_from_deployment(
            deployment_dict=_deployment_info_dict
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

    if litellm_model_id is not None:
        # user is trying to get specific model from litellm router
        deployment_info = llm_router.get_deployment(model_id=litellm_model_id)
        if deployment_info is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Model id = {litellm_model_id} not found on litellm proxy"
                },
            )
        _deployment_info_dict = _get_proxy_model_info(
            model=deployment_info.model_dump(exclude_none=True)
        )
        return {"data": [_deployment_info_dict]}

    all_models: List[dict] = []
    model_access_groups: Dict[str, List[str]] = defaultdict(list)
    ## CHECK IF MODEL RESTRICTIONS ARE SET AT KEY/TEAM LEVEL ##
    if llm_router is None:
        proxy_model_list = []
    else:
        proxy_model_list = llm_router.get_model_names()
        model_access_groups = llm_router.get_model_access_groups()
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )
    team_models = get_team_models(
        team_models=user_api_key_dict.team_models,
        proxy_model_list=proxy_model_list,
        model_access_groups=model_access_groups,
    )
    all_models_str = get_complete_model_list(
        key_models=key_models,
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        user_model=user_model,
        infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
        llm_router=llm_router,
    )

    if len(all_models_str) > 0:
        _relevant_models = []
        for model in all_models_str:
            router_models = llm_router.get_model_list(model_name=model)
            if router_models is not None:
                _relevant_models.extend(router_models)
        if llm_model_list is not None:
            all_models = copy.deepcopy(_relevant_models)  # type: ignore
        else:
            all_models = []

    for in_place_model in all_models:
        in_place_model = _get_proxy_model_info(model=in_place_model)

    verbose_proxy_logger.debug("all_models: %s", all_models)
    return {"data": all_models}


def _get_model_group_info(
    llm_router: Router, all_models_str: List[str], model_group: Optional[str]
) -> List[ModelGroupInfoProxy]:
    model_groups: List[ModelGroupInfoProxy] = []

    unique_models = []
    for model in all_models_str:
        if model not in unique_models:
            unique_models.append(model)

    for model in unique_models:
        if model_group is not None and model_group != model:
            continue

        _model_group_info = llm_router.get_model_group_info(model_group=model)

        if _model_group_info is not None:
            model_groups.append(ModelGroupInfoProxy(**_model_group_info.model_dump()))
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
    model_group: Optional[str] = None,
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

    if llm_model_list is None:
        raise HTTPException(
            status_code=500, detail={"error": "LLM Model List not loaded in"}
        )
    if llm_router is None:
        raise HTTPException(
            status_code=500, detail={"error": "LLM Router is not loaded in"}
        )

    from litellm.proxy.utils import get_available_models_for_user

    # Get available models for the user
    all_models_str = await get_available_models_for_user(
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
    model_groups: List[ModelGroupInfoProxy] = _get_model_group_info(
        llm_router=llm_router, all_models_str=all_models_str, model_group=model_group
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

    returned_list = []
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

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    ## get general settings from db
    db_general_settings = await prisma_client.db.litellm_config.find_first(
        where={"param_name": "general_settings"}
    )

    if db_general_settings is not None and db_general_settings.param_value is not None:
        db_general_settings_dict = dict(db_general_settings.param_value)
        alerting_args_dict: dict = db_general_settings_dict.get("alerting_args", {})  # type: ignore
        alerting_values: Optional[list] = db_general_settings_dict.get("alerting")  # type: ignore
    else:
        alerting_args_dict = {}
        alerting_values = None

    allowed_args = {
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

    _slack_alerting: SlackAlerting = proxy_logging_obj.slack_alerting_instance
    _slack_alerting_args_dict = _slack_alerting.alerting_args.model_dump()

    return_val = []

    is_slack_enabled = False

    if general_settings.get("alerting") and isinstance(
        general_settings["alerting"], list
    ):
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
            _stored_in_db: Optional[bool] = None
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
                premium_field=(
                    True if field_name == "region_outage_alert_ttl" else False
                ),
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
    model: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global general_settings, user_debug, proxy_logging_obj
    """
    v2 attempt at a background worker to handle queuing.

    Just supports /chat/completion calls currently.

    Now using a FastAPI background task + /chat/completions compatible endpoint
    """
    data = {}
    try:
        data = await request.json()  # type: ignore

        # Include original request and headers in the data
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
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

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
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
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )

        response = await llm_router.schedule_acompletion(**data)

        if (
            "stream" in data and data["stream"] is True
        ):  # use generate_responses to stream responses
            return StreamingResponse(
                async_data_generator(
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                    request_data=data,
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
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
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
    ui_username = os.getenv("UI_USERNAME")
    if redirect_url.endswith("/"):
        redirect_url += "sso/callback"
    else:
        redirect_url += "/sso/callback"

    if ui_username is not None:
        # No Google, Microsoft SSO
        # Use UI Credentials set in .env
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)
    else:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)


@router.post(
    "/login", include_in_schema=False
)  # hidden since this is a helper for UI sso login
async def login(request: Request):  # noqa: PLR0915
    global premium_user, general_settings, master_key
    from litellm.types.proxy.ui_sso import ReturnedUITokenObject

    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    form = await request.form()
    username = str(form.get("username"))
    password = str(form.get("password"))
    ui_username = os.getenv("UI_USERNAME", "admin")
    ui_password = os.getenv("UI_PASSWORD", None)
    if ui_password is None:
        ui_password = str(master_key) if master_key is not None else None
    if ui_password is None:
        raise ProxyException(
            message="set Proxy master key to use UI. https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="UI_PASSWORD",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # check if we can find the `username` in the db. on the ui, users can enter username=their email
    _user_row: Optional[LiteLLM_UserTable] = None
    user_role: Optional[
        Literal[
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]
    ] = None
    if prisma_client is not None:
        _user_row = cast(
            Optional[LiteLLM_UserTable],
            await prisma_client.db.litellm_usertable.find_first(
                where={"user_email": {"equals": username}}
            ),
        )
    disabled_non_admin_personal_key_creation = (
        get_disabled_non_admin_personal_key_creation()
    )
    """
    To login to Admin UI, we support the following
    - Login with UI_USERNAME and UI_PASSWORD
    - Login with Invite Link `user_email` and `password` combination
    """
    if secrets.compare_digest(username, ui_username) and secrets.compare_digest(
        password, ui_password
    ):
        # Non SSO -> If user is using UI_USERNAME and UI_PASSWORD they are Proxy admin
        user_role = LitellmUserRoles.PROXY_ADMIN
        user_id = litellm_proxy_admin_name

        # we want the key created to have PROXY_ADMIN_PERMISSIONS
        key_user_id = litellm_proxy_admin_name
        if (
            os.getenv("PROXY_ADMIN_ID", None) is not None
            and os.environ["PROXY_ADMIN_ID"] == user_id
        ) or user_id == litellm_proxy_admin_name:
            # checks if user is admin
            key_user_id = os.getenv("PROXY_ADMIN_ID", litellm_proxy_admin_name)

        # Admin is Authe'd in - generate key for the UI to access Proxy

        # ensure this user is set as the proxy admin, in this route there is no sso, we can assume this user is only the admin
        await user_update(
            data=UpdateUserRequest(
                user_id=key_user_id,
                user_role=user_role,
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
            ),
        )
        if os.getenv("DATABASE_URL") is not None:
            response = await generate_key_helper_fn(
                request_type="key",
                **{
                    "user_role": LitellmUserRoles.PROXY_ADMIN,
                    "duration": "24hr",
                    "key_max_budget": litellm.max_ui_session_budget,
                    "models": [],
                    "aliases": {},
                    "config": {},
                    "spend": 0,
                    "user_id": key_user_id,
                    "team_id": "litellm-dashboard",
                },  # type: ignore
            )
        else:
            raise ProxyException(
                message="No Database connected. Set DATABASE_URL in .env. If set, use `--detailed_debug` to debug issue.",
                type=ProxyErrorTypes.auth_error,
                param="DATABASE_URL",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        key = response["token"]  # type: ignore
        litellm_dashboard_ui = get_custom_url(str(request.base_url))
        if litellm_dashboard_ui.endswith("/"):
            litellm_dashboard_ui += "ui/"
        else:
            litellm_dashboard_ui += "/ui/"
        import jwt

        if get_secret_bool("EXPERIMENTAL_UI_LOGIN"):
            user_info: Optional[LiteLLM_UserTable] = None
            if _user_row is not None:
                user_info = _user_row
            elif (
                user_id is not None
            ):  # if user_id is not None, we are using the UI_USERNAME and UI_PASSWORD
                user_info = LiteLLM_UserTable(
                    user_id=user_id,
                    user_role=user_role,
                    models=[],
                    max_budget=litellm.max_ui_session_budget,
                )
            if user_info is None:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "User Information is required for experimental UI login"
                    },
                )

            key = ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(
                user_info
            )

        returned_ui_token_object = ReturnedUITokenObject(
            user_id=user_id,
            key=key,
            user_email=None,
            user_role=user_role,
            login_method="username_password",
            premium_user=premium_user,
            auth_header_name=general_settings.get(
                "litellm_key_header_name", "Authorization"
            ),
            disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
            server_root_path=get_server_root_path(),
        )

        jwt_token = jwt.encode(  # type: ignore
            cast(dict, returned_ui_token_object),
            master_key,
            algorithm="HS256",
        )
        litellm_dashboard_ui += "?login=success"
        redirect_response = RedirectResponse(url=litellm_dashboard_ui, status_code=303)
        redirect_response.set_cookie(key="token", value=jwt_token)
        return redirect_response
    elif _user_row is not None:
        """
        When sharing invite links

        -> if the user has no role in the DB assume they are only a viewer
        """
        user_id = getattr(_user_row, "user_id", "unknown")
        user_role = getattr(
            _user_row, "user_role", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        )
        user_email = getattr(_user_row, "user_email", "unknown")
        _password = getattr(_user_row, "password", "unknown")

        if _password is None:
            raise ProxyException(
                message="User has no password set. Please set a password for the user via `/user/update`.",
                type=ProxyErrorTypes.auth_error,
                param="password",
                code=status.HTTP_401_UNAUTHORIZED,
            )

        # check if password == _user_row.password
        hash_password = hash_token(token=password)
        if secrets.compare_digest(password, _password) or secrets.compare_digest(
            hash_password, _password
        ):
            if os.getenv("DATABASE_URL") is not None:
                response = await generate_key_helper_fn(
                    request_type="key",
                    **{  # type: ignore
                        "user_role": user_role,
                        "duration": "24hr",
                        "key_max_budget": litellm.max_ui_session_budget,
                        "models": [],
                        "aliases": {},
                        "config": {},
                        "spend": 0,
                        "user_id": user_id,
                        "team_id": "litellm-dashboard",
                    },
                )
            else:
                raise ProxyException(
                    message="No Database connected. Set DATABASE_URL in .env. If set, use `--detailed_debug` to debug issue.",
                    type=ProxyErrorTypes.auth_error,
                    param="DATABASE_URL",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            key = response["token"]  # type: ignore
            litellm_dashboard_ui = get_custom_url(str(request.base_url))
            if litellm_dashboard_ui.endswith("/"):
                litellm_dashboard_ui += "ui/"
            else:
                litellm_dashboard_ui += "/ui/"
            import jwt

            returned_ui_token_object = ReturnedUITokenObject(
                user_id=user_id,
                key=key,
                user_email=user_email,
                user_role=cast(str, user_role),
                login_method="username_password",
                premium_user=premium_user,
                auth_header_name=general_settings.get(
                    "litellm_key_header_name", "Authorization"
                ),
                disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
                server_root_path=get_server_root_path(),
            )

            jwt_token = jwt.encode(  # type: ignore
                cast(dict, returned_ui_token_object),
                master_key,
                algorithm="HS256",
            )
            litellm_dashboard_ui += "?login=success"
            redirect_response = RedirectResponse(
                url=litellm_dashboard_ui, status_code=303
            )
            redirect_response.set_cookie(key="token", value=jwt_token)
            return redirect_response
        else:
            raise ProxyException(
                message=f"Invalid credentials used to access UI.\nNot valid credentials for {username}",
                type=ProxyErrorTypes.auth_error,
                param="invalid_credentials",
                code=status.HTTP_401_UNAUTHORIZED,
            )
    else:
        raise ProxyException(
            message="Invalid credentials used to access UI.\nCheck 'UI_USERNAME', 'UI_PASSWORD' in .env file",
            type=ProxyErrorTypes.auth_error,
            param="invalid_credentials",
            code=status.HTTP_401_UNAUTHORIZED,
        )


@app.get("/onboarding/get_token", include_in_schema=False)
async def onboarding(invite_link: str, request: Request):
    """
    - Get the invite link
    - Validate it's still 'valid'
    - Invalidate the link (prevents abuse)
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

    invite_obj = await prisma_client.db.litellm_invitationlink.find_unique(
        where={"id": invite_link}
    )
    if invite_obj is None:
        raise HTTPException(
            status_code=401, detail={"error": "Invitation link does not exist in db."}
        )
    #### CHECK IF EXPIRED
    # Extract the date part from both datetime objects
    utc_now_date = litellm.utils.get_utc_datetime().date()
    expires_at_date = invite_obj.expires_at.date()
    if expires_at_date < utc_now_date:
        raise HTTPException(
            status_code=401, detail={"error": "Invitation link has expired."}
        )

    #### INVALIDATE LINK
    current_time = litellm.utils.get_utc_datetime()

    _ = await prisma_client.db.litellm_invitationlink.update(
        where={"id": invite_link},
        data={
            "accepted_at": current_time,
            "updated_at": current_time,
            "is_accepted": True,
            "updated_by": invite_obj.user_id,  # type: ignore
        },
    )

    ### GET USER OBJECT ###
    user_obj = await prisma_client.db.litellm_usertable.find_unique(
        where={"user_id": invite_obj.user_id}
    )

    if user_obj is None:
        raise HTTPException(
            status_code=401, detail={"error": "User does not exist in db."}
        )

    user_email = user_obj.user_email

    response = await generate_key_helper_fn(
        request_type="key",
        **{
            "user_role": user_obj.user_role,
            "duration": "24hr",
            "key_max_budget": litellm.max_ui_session_budget,
            "models": [],
            "aliases": {},
            "config": {},
            "spend": 0,
            "user_id": user_obj.user_id,
            "team_id": "litellm-dashboard",
        },  # type: ignore
    )
    key = response["token"]  # type: ignore

    litellm_dashboard_ui = get_custom_url(str(request.base_url))
    if litellm_dashboard_ui.endswith("/"):
        litellm_dashboard_ui += "ui/onboarding"
    else:
        litellm_dashboard_ui += "/ui/onboarding"
    import jwt

    disabled_non_admin_personal_key_creation = (
        get_disabled_non_admin_personal_key_creation()
    )

    returned_ui_token_object = ReturnedUITokenObject(
        user_id=user_obj.user_id,
        key=key,
        user_email=user_obj.user_email,
        user_role=user_obj.user_role,
        login_method="username_password",
        premium_user=premium_user,
        auth_header_name=general_settings.get(
            "litellm_key_header_name", "Authorization"
        ),
        disabled_non_admin_personal_key_creation=disabled_non_admin_personal_key_creation,
        server_root_path=get_server_root_path(),
    )
    jwt_token = jwt.encode(  # type: ignore
        cast(dict, returned_ui_token_object),
        master_key,
        algorithm="HS256",
    )

    litellm_dashboard_ui += "?token={}&user_email={}".format(jwt_token, user_email)
    return {
        "login_url": litellm_dashboard_ui,
        "token": jwt_token,
        "user_email": user_email,
    }


@app.post("/onboarding/claim_token", include_in_schema=False)
async def claim_onboarding_link(data: InvitationClaim):
    """
    Special route. Allows UI link share user to update their password.

    - Get the invite link
    - Validate it's still 'valid'
    - Check if user within initial session (prevents abuse)
    - Get user from db
    - Update user password

    This route can only update user password.
    """
    global prisma_client
    ### VALIDATE INVITE LINK ###
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    invite_obj = await prisma_client.db.litellm_invitationlink.find_unique(
        where={"id": data.invitation_link}
    )
    if invite_obj is None:
        raise HTTPException(
            status_code=401, detail={"error": "Invitation link does not exist in db."}
        )
    #### CHECK IF EXPIRED
    # Extract the date part from both datetime objects
    utc_now_date = litellm.utils.get_utc_datetime().date()
    expires_at_date = invite_obj.expires_at.date()
    if expires_at_date < utc_now_date:
        raise HTTPException(
            status_code=401, detail={"error": "Invitation link has expired."}
        )

    #### CHECK IF CLAIMED
    ##### if claimed - accept
    ##### if unclaimed - reject

    if invite_obj.is_accepted is True:
        # this is a valid invite that was accepted
        pass
    else:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "The invitation link was never validated. Please file an issue, if this is not intended - https://github.com/BerriAI/litellm/issues."
            },
        )

    #### CHECK IF VALID USER ID
    if invite_obj.user_id != data.user_id:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid invitation link. The user id submitted does not match the user id this link is attached to. Got={}, Expected={}".format(
                    data.user_id, invite_obj.user_id
                )
            },
        )
    ### UPDATE USER OBJECT ###
    hash_password = hash_token(token=data.password)
    user_obj = await prisma_client.db.litellm_usertable.update(
        where={"user_id": invite_obj.user_id}, data={"password": hash_password}
    )

    if user_obj is None:
        raise HTTPException(
            status_code=401, detail={"error": "User does not exist in db."}
        )

    return user_obj


@app.get("/get_logo_url", include_in_schema=False)
def get_logo_url():
    """Get the current logo URL from environment"""
    logo_path = os.getenv("UI_LOGO_PATH", "")
    return {"logo_url": logo_path}


@app.get("/get_image", include_in_schema=False)
def get_image():
    """Get logo to show on admin UI"""

    # get current_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_logo = os.path.join(current_dir, "logo.jpg")

    logo_path = os.getenv("UI_LOGO_PATH", default_logo)
    verbose_proxy_logger.debug("Reading logo from path: %s", logo_path)

    # Check if the logo path is an HTTP/HTTPS URL
    if logo_path.startswith(("http://", "https://")):
        # Download the image and cache it
        client = HTTPHandler()
        response = client.get(logo_path)
        if response.status_code == 200:
            # Save the image to a local file
            cache_path = os.path.join(current_dir, "cached_logo.jpg")
            with open(cache_path, "wb") as f:
                f.write(response.content)

            # Return the cached image as a FileResponse
            return FileResponse(cache_path, media_type="image/jpeg")
        else:
            # Handle the case when the image cannot be downloaded
            return FileResponse(default_logo, media_type="image/jpeg")
    else:
        # Return the local image file if the logo path is not an HTTP/HTTPS URL
        return FileResponse(logo_path, media_type="image/jpeg")


#### INVITATION MANAGEMENT ####


@router.post(
    "/invitation/new",
    tags=["Invite Links"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvitationModel,
    include_in_schema=False,
)
async def new_invitation(
    data: InvitationNew, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)
):
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

        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "{}, your role={}".format(
                        CommonProxyErrors.not_allowed_access.value,
                        user_api_key_dict.user_role,
                    )
                },
            )

        response = await create_invitation_for_user(
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
async def invitation_info(
    invitation_id: str, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)
):
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

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    response = await prisma_client.db.litellm_invitationlink.find_unique(
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
            detail={
                "error": "Unable to identify user id. Received={}".format(
                    user_api_key_dict.user_id
                )
            },
        )

    current_time = litellm.utils.get_utc_datetime()
    response = await prisma_client.db.litellm_invitationlink.update(
        where={"id": data.invitation_id},
        data={
            "id": data.invitation_id,
            "is_accepted": data.is_accepted,
            "accepted_at": current_time,
            "updated_at": current_time,
            "updated_by": user_api_key_dict.user_id,  # type: ignore
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

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    response = await prisma_client.db.litellm_invitationlink.delete(
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
async def update_config(config_info: ConfigYAML):  # noqa: PLR0915
    """
    For Admin UI - allows admin to update config via UI

    Currently supports modifying General Settings + LiteLLM settings
    """
    global llm_router, llm_model_list, general_settings, proxy_config, proxy_logging_obj, master_key, prisma_client
    try:
        import base64

        """
        - Update the ConfigTable DB
        - Run 'add_deployment'
        """
        if prisma_client is None:
            raise Exception("No DB Connected")

        if store_model_in_db is not True:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

        updated_settings = config_info.json(exclude_none=True)
        updated_settings = prisma_client.jsonify_object(updated_settings)
        for k, v in updated_settings.items():
            if k == "router_settings":
                await prisma_client.db.litellm_config.upsert(
                    where={"param_name": k},
                    data={
                        "create": {"param_name": k, "param_value": v},
                        "update": {"param_value": v},
                    },
                )

        ### OLD LOGIC [TODO] MOVE TO DB ###

        # Load existing config
        config = await proxy_config.get_config()
        verbose_proxy_logger.debug("Loaded config: %s", config)

        # update the general settings
        if config_info.general_settings is not None:
            config.setdefault("general_settings", {})
            updated_general_settings = config_info.general_settings.dict(
                exclude_none=True
            )

            _existing_settings = config["general_settings"]
            for k, v in updated_general_settings.items():
                # overwrite existing settings with updated values
                if k == "alert_to_webhook_url":
                    # check if slack is already enabled. if not, enable it
                    if "alerting" not in _existing_settings:
                        _existing_settings = {"alerting": ["slack"]}
                    elif isinstance(_existing_settings["alerting"], list):
                        if "slack" not in _existing_settings["alerting"]:
                            _existing_settings["alerting"].append("slack")
                _existing_settings[k] = v
            config["general_settings"] = _existing_settings

        if config_info.environment_variables is not None:
            config.setdefault("environment_variables", {})
            _updated_environment_variables = config_info.environment_variables

            # encrypt updated_environment_variables #
            for k, v in _updated_environment_variables.items():
                encrypted_value = encrypt_value_helper(value=v)
                _updated_environment_variables[k] = encrypted_value

            _existing_env_variables = config["environment_variables"]

            for k, v in _updated_environment_variables.items():
                # overwrite existing env variables with updated values
                _existing_env_variables[k] = _updated_environment_variables[k]

        # update the litellm settings
        if config_info.litellm_settings is not None:
            config.setdefault("litellm_settings", {})
            updated_litellm_settings = config_info.litellm_settings
            config["litellm_settings"] = {
                **updated_litellm_settings,
                **config["litellm_settings"],
            }

            # if litellm.success_callback in updated_litellm_settings and config["litellm_settings"]
            if (
                "success_callback" in updated_litellm_settings
                and "success_callback" in config["litellm_settings"]
            ):
                # check both success callback are lists
                if isinstance(
                    config["litellm_settings"]["success_callback"], list
                ) and isinstance(updated_litellm_settings["success_callback"], list):
                    updated_success_callbacks_normalized = normalize_callback_names(
                        updated_litellm_settings["success_callback"]
                    )
                    combined_success_callback = (
                        config["litellm_settings"]["success_callback"]
                        + updated_success_callbacks_normalized
                    )
                    combined_success_callback = list(set(combined_success_callback))
                    config["litellm_settings"][
                        "success_callback"
                    ] = combined_success_callback

        # Save the updated config
        await proxy_config.save_config(new_config=config)

        await proxy_config.add_deployment(
            prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
        )

        return {"message": "Config updated successfully"}
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.update_config(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
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

    if data.field_name not in ConfigGeneralSettings.model_fields:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid field={} passed in.".format(data.field_name)},
        )

    try:
        ConfigGeneralSettings(**{data.field_name: data.field_value})
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid type of field value={} passed in.".format(
                    type(data.field_value),
                )
            },
        )

    ## get general settings from db
    db_general_settings = await prisma_client.db.litellm_config.find_first(
        where={"param_name": "general_settings"}
    )
    ### update value

    if db_general_settings is None or db_general_settings.param_value is None:
        general_settings = {}
    else:
        general_settings = dict(db_general_settings.param_value)

    ## update db

    general_settings[data.field_name] = data.field_value

    response = await prisma_client.db.litellm_config.upsert(
        where={"param_name": "general_settings"},
        data={
            "create": {"param_name": "general_settings", "param_value": json.dumps(general_settings)},  # type: ignore
            "update": {"param_value": json.dumps(general_settings)},  # type: ignore
        },
    )

    return response


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

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    if field_name not in ConfigGeneralSettings.model_fields:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid field={} passed in.".format(field_name)},
        )

    ## get general settings from db
    db_general_settings = await prisma_client.db.litellm_config.find_first(
        where={"param_name": "general_settings"}
    )
    ### pop the value

    if db_general_settings is None or db_general_settings.param_value is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Field name={} not in DB".format(field_name)},
        )
    else:
        general_settings = dict(db_general_settings.param_value)

        if field_name in general_settings:
            return ConfigFieldInfo(
                field_name=field_name, field_value=general_settings[field_name]
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": "Field name={} not in DB".format(field_name)},
            )


@router.get(
    "/config/list",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_config_list(
    config_type: Literal["general_settings"],
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[ConfigList]:
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

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    ## get general settings from db
    db_general_settings = await prisma_client.db.litellm_config.find_first(
        where={"param_name": "general_settings"}
    )

    if db_general_settings is not None and db_general_settings.param_value is not None:
        db_general_settings_dict = dict(db_general_settings.param_value)
    else:
        db_general_settings_dict = {}

    allowed_args = {
        "max_parallel_requests": {"type": "Integer"},
        "global_max_parallel_requests": {"type": "Integer"},
        "max_request_size_mb": {"type": "Integer"},
        "max_response_size_mb": {"type": "Integer"},
        "pass_through_endpoints": {"type": "PydanticModel"},
    }

    return_val = []

    for field_name, field_info in ConfigGeneralSettings.model_fields.items():
        if field_name in allowed_args:
            ## HANDLE TYPED DICT

            typed_dict_type = allowed_args[field_name]["type"]

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
                            field_type=sub_field_type.__name__,
                            field_description="",  # Add custom logic if descriptions are available
                            field_default_value=general_settings.get(sub_field, None),
                            stored_in_db=None,
                        )
                        for sub_field, sub_field_type in pydantic_class.__annotations__.items()
                    ]

                    idx = 0
                    for (
                        sub_field,
                        sub_field_info,
                    ) in pydantic_class.model_fields.items():
                        if (
                            hasattr(sub_field_info, "description")
                            and sub_field_info.description is not None
                        ):
                            nested_fields[idx].field_description = (
                                sub_field_info.description
                            )
                        idx += 1

                    _stored_in_db = None
                    if field_name in db_general_settings_dict:
                        _stored_in_db = True
                    elif field_name in general_settings:
                        _stored_in_db = False

                    _response_obj = ConfigList(
                        field_name=field_name,
                        field_type=allowed_args[field_name]["type"],
                        field_description=field_info.description or "",
                        field_value=general_settings.get(field_name, None),
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

                _response_obj = ConfigList(
                    field_name=field_name,
                    field_type=allowed_args[field_name]["type"],
                    field_description=field_info.description or "",
                    field_value=general_settings.get(field_name, None),
                    stored_in_db=_stored_in_db,
                    field_default_value=field_info.default,
                    nested_fields=nested_fields,
                )
                return_val.append(_response_obj)

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
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    if data.field_name not in ConfigGeneralSettings.model_fields:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid field={} passed in.".format(data.field_name)},
        )

    ## get general settings from db
    db_general_settings = await prisma_client.db.litellm_config.find_first(
        where={"param_name": "general_settings"}
    )
    ### pop the value

    if db_general_settings is None or db_general_settings.param_value is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Field name={} not in config".format(data.field_name)},
        )
    else:
        general_settings = dict(db_general_settings.param_value)

    ## update db

    general_settings.pop(data.field_name, None)

    response = await prisma_client.db.litellm_config.upsert(
        where={"param_name": "general_settings"},
        data={
            "create": {"param_name": "general_settings", "param_value": json.dumps(general_settings)},  # type: ignore
            "update": {"param_value": json.dumps(general_settings)},  # type: ignore
        },
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
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )

    try:
        # Get current configuration
        config = await proxy_config.get_config()
        callback_name = data.callback_name.lower()

        # Check if callback exists in current configuration
        litellm_settings = config.get("litellm_settings", {})
        success_callbacks = litellm_settings.get("success_callback", [])

        if callback_name not in success_callbacks:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Callback '{callback_name}' not found in active configuration"
                },
            )

        # Remove callback from success_callback list
        success_callbacks.remove(callback_name)
        config.setdefault("litellm_settings", {})[
            "success_callback"
        ] = success_callbacks

        # Save the updated configuration
        await proxy_config.save_config(new_config=config)

        # Restart the proxy to apply changes
        await proxy_config.add_deployment(
            prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
        )

        return {
            "message": f"Successfully deleted callback: {callback_name}",
            "removed_callback": callback_name,
            "remaining_callbacks": success_callbacks,
            "deleted_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(
            f"litellm.proxy.proxy_server.delete_callback(): Exception occurred - {str(e)}"
        )
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
async def get_config():  # noqa: PLR0915
    """
    For Admin UI - allows admin to view config via UI
    # return the callbacks and the env variables for the callback

    """
    global llm_router, llm_model_list, general_settings, proxy_config, proxy_logging_obj, master_key
    try:
        import base64

        all_available_callbacks = AllCallbacks()

        config_data = await proxy_config.get_config()
        _litellm_settings = config_data.get("litellm_settings", {})
        _general_settings = config_data.get("general_settings", {})
        environment_variables = config_data.get("environment_variables", {})

        # check if "langfuse" in litellm_settings
        _success_callbacks = _litellm_settings.get("success_callback", [])
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
            }
        ]

        """
        for _callback in _success_callbacks:
            if _callback != "langfuse":
                if _callback == "openmeter":
                    env_vars = [
                        "OPENMETER_API_KEY",
                    ]
                elif _callback == "braintrust":
                    env_vars = [
                        "BRAINTRUST_API_KEY",
                        "BRAINTRUST_API_BASE",
                    ]
                elif _callback == "traceloop":
                    env_vars = ["TRACELOOP_API_KEY"]
                elif _callback == "custom_callback_api":
                    env_vars = ["GENERIC_LOGGER_ENDPOINT"]
                elif _callback == "otel":
                    env_vars = ["OTEL_EXPORTER", "OTEL_ENDPOINT", "OTEL_HEADERS"]
                elif _callback == "langsmith":
                    env_vars = [
                        "LANGSMITH_API_KEY",
                        "LANGSMITH_PROJECT",
                        "LANGSMITH_DEFAULT_RUN_NAME",
                    ]
                else:
                    env_vars = []

                env_vars_dict = {}
                for _var in env_vars:
                    env_variable = environment_variables.get(_var, None)
                    if env_variable is None:
                        env_vars_dict[_var] = None
                    else:
                        # decode + decrypt the value
                        decrypted_value = decrypt_value_helper(
                            value=env_variable, key=_var
                        )
                        env_vars_dict[_var] = decrypted_value

                _data_to_return.append({"name": _callback, "variables": env_vars_dict})
            elif _callback == "langfuse":
                _langfuse_vars = [
                    "LANGFUSE_PUBLIC_KEY",
                    "LANGFUSE_SECRET_KEY",
                    "LANGFUSE_HOST",
                ]
                _langfuse_env_vars = {}
                for _var in _langfuse_vars:
                    env_variable = environment_variables.get(_var, None)
                    if env_variable is None:
                        _langfuse_env_vars[_var] = None
                    else:
                        # decode + decrypt the value
                        decrypted_value = decrypt_value_helper(
                            value=env_variable, key=_var
                        )
                        _langfuse_env_vars[_var] = decrypted_value

                _data_to_return.append(
                    {"name": _callback, "variables": _langfuse_env_vars}
                )

        # Check if slack alerting is on
        _alerting = _general_settings.get("alerting", [])
        alerting_data = []
        if "slack" in _alerting:
            _slack_vars = [
                "SLACK_WEBHOOK_URL",
            ]
            _slack_env_vars = {}
            for _var in _slack_vars:
                env_variable = environment_variables.get(_var, None)
                if env_variable is None:
                    _value = os.getenv("SLACK_WEBHOOK_URL", None)
                    _slack_env_vars[_var] = _value
                else:
                    # decode + decrypt the value
                    _decrypted_value = decrypt_value_helper(
                        value=env_variable, key=_var
                    )
                    _slack_env_vars[_var] = _decrypted_value

            _alerting_types = proxy_logging_obj.slack_alerting_instance.alert_types
            _all_alert_types = (
                proxy_logging_obj.slack_alerting_instance._all_possible_alert_types()
            )
            _alerts_to_webhook = (
                proxy_logging_obj.slack_alerting_instance.alert_to_webhook_url
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
        _email_vars = [
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SMTP_SENDER_EMAIL",
            "TEST_EMAIL_ADDRESS",
            "EMAIL_LOGO_URL",
            "EMAIL_SUPPORT_CONTACT",
        ]
        _email_env_vars = {}
        for _var in _email_vars:
            env_variable = environment_variables.get(_var, None)
            if env_variable is None:
                _email_env_vars[_var] = None
            else:
                # decode + decrypt the value
                _decrypted_value = decrypt_value_helper(value=env_variable, key=_var)
                _email_env_vars[_var] = _decrypted_value

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
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.get_config(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
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


@router.get(
    "/get/litellm_model_cost_map",
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_litellm_model_cost_map(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        _model_cost_map = litellm.model_cost
        return _model_cost_map
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error ({str(e)})",
        )


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
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Immediately reload the model cost map in the current pod
        from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

        model_cost_map_url = litellm.model_cost_map_url
        new_model_cost_map = get_model_cost_map(url=model_cost_map_url)
        litellm.model_cost = new_model_cost_map

        # Update pod's in-memory last reload time
        global last_model_cost_map_reload
        current_time = datetime.utcnow()
        last_model_cost_map_reload = current_time.isoformat()

        # Set force reload flag in database for other pods
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "model_cost_map_reload_config"},
            data={
                "create": {
                    "param_name": "model_cost_map_reload_config",
                    "param_value": safe_dumps(
                        {"interval_hours": None, "force_reload": True}
                    ),
                },
                "update": {"param_value": safe_dumps({"force_reload": True})},
            },
        )

        models_count = len(new_model_cost_map) if new_model_cost_map else 0
        verbose_proxy_logger.info(
            f"Model cost map reloaded successfully in current pod. Models count: {models_count}"
        )

        return {
            "message": f"Price data reloaded successfully! {models_count} models updated.",
            "status": "success",
            "models_count": models_count,
            "timestamp": current_time.isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception(f"Failed to reload model cost map: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reload model cost map: {str(e)}"
        )


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
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Update database with new reload configuration
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "model_cost_map_reload_config"},
            data={
                "create": {
                    "param_name": "model_cost_map_reload_config",
                    "param_value": safe_dumps(
                        {"interval_hours": hours, "force_reload": False}
                    ),
                },
                "update": {
                    "param_value": safe_dumps(
                        {"interval_hours": hours, "force_reload": False}
                    )
                },
            },
        )

        verbose_proxy_logger.info(
            f"Model cost map reload scheduled for every {hours} hours"
        )

        return {
            "message": f"Model cost map reload scheduled for every {hours} hours",
            "status": "success",
            "interval_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Failed to schedule model cost map reload: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule model cost map reload: {str(e)}",
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
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Remove reload configuration from database
        await prisma_client.db.litellm_config.delete(
            where={"param_name": "model_cost_map_reload_config"}
        )

        verbose_proxy_logger.info("Model cost map reload schedule cancelled")

        return {
            "message": "Model cost map reload schedule cancelled",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Failed to cancel model cost map reload: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel model cost map reload: {str(e)}"
        )


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
    # Check if user is admin
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Admin role required. Current role: {user_api_key_dict.user_role}",
        )

    try:
        global prisma_client, last_model_cost_map_reload

        verbose_proxy_logger.info(
            f"Checking model cost map reload status. Last reload: {last_model_cost_map_reload}"
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
        config_record = await prisma_client.db.litellm_config.find_unique(
            where={"param_name": "model_cost_map_reload_config"}
        )

        if config_record is None or config_record.param_value is None:
            verbose_proxy_logger.info("No model cost map reload configuration found")
            return {
                "scheduled": False,
                "interval_hours": None,
                "last_run": None,
                "next_run": None,
            }

        config = config_record.param_value
        interval_hours = config.get("interval_hours")

        if interval_hours is None:
            verbose_proxy_logger.info("No interval configured, returning not scheduled")
            return {
                "scheduled": False,
                "interval_hours": None,
                "last_run": None,
                "next_run": None,
            }

        current_time = datetime.utcnow()
        next_run = None

        # Use pod's in-memory last reload time
        if last_model_cost_map_reload is not None:
            try:
                last_reload_time = datetime.fromisoformat(last_model_cost_map_reload)
                time_since_last_reload = current_time - last_reload_time
                hours_since_last_reload = time_since_last_reload.total_seconds() / 3600

                if hours_since_last_reload < interval_hours:
                    next_run = (
                        last_reload_time + timedelta(hours=interval_hours)
                    ).isoformat()
            except Exception as e:
                verbose_proxy_logger.warning(f"Error parsing last reload time: {e}")

        return {
            "scheduled": True,
            "interval_hours": interval_hours,
            "last_run": last_model_cost_map_reload,
            "next_run": next_run,
        }
    except Exception as e:
        verbose_proxy_logger.exception(
            f"Failed to get model cost map reload status: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model cost map reload status: {str(e)}",
        )


@router.get("/", dependencies=[Depends(user_api_key_auth)])
async def home(request: Request):
    return "LiteLLM: RUNNING"


@router.get("/routes", dependencies=[Depends(user_api_key_auth)])
async def get_routes():
    """
    Get a list of available routes in the FastAPI application.
    """
    from litellm.proxy.common_utils.get_routes import GetRoutes

    routes = []
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
app.include_router(batches_router)
app.include_router(public_endpoints_router)
app.include_router(rerank_router)
app.include_router(ocr_router)
app.include_router(video_router)
app.include_router(container_router)
app.include_router(search_router)
app.include_router(image_router)
app.include_router(fine_tuning_router)
app.include_router(vector_store_router)
app.include_router(vector_store_files_router)
app.include_router(credential_router)
app.include_router(llm_passthrough_router)
app.include_router(mcp_management_router)
app.include_router(anthropic_router)
app.include_router(google_router)
app.include_router(langfuse_router)
app.include_router(pass_through_router)
app.include_router(health_router)
app.include_router(key_management_router)
app.include_router(internal_user_router)
app.include_router(team_router)
app.include_router(ui_sso_router)
app.include_router(scim_router)
app.include_router(organization_router)
app.include_router(customer_router)
app.include_router(spend_management_router)
app.include_router(cloudzero_router)
app.include_router(caching_router)
app.include_router(analytics_router)
app.include_router(guardrails_router)
app.include_router(search_tool_management_router)
app.include_router(prompts_router)
app.include_router(callback_management_endpoints_router)
app.include_router(debugging_endpoints_router)
app.include_router(ui_crud_endpoints_router)
app.include_router(openai_files_router)
app.include_router(team_callback_router)
app.include_router(budget_management_router)
app.include_router(model_management_router)
app.include_router(model_access_group_management_router)
app.include_router(tag_management_router)
app.include_router(cost_tracking_settings_router)
app.include_router(router_settings_router)
app.include_router(cache_settings_router)
app.include_router(user_agent_analytics_router)
app.include_router(enterprise_router)
app.include_router(ui_discovery_endpoints_router)
app.include_router(agent_endpoints_router)
########################################################
# MCP Server
########################################################


# Dynamic MCP server routes - handle /{mcp_server_name}/mcp
@app.api_route(
    "/{mcp_server_name}/mcp",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def dynamic_mcp_route(mcp_server_name: str, request: Request):
    """Handle dynamic MCP server routes like /github_mcp/mcp"""
    try:
        # Validate that the MCP server exists
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp import MCPAuth

        mcp_server = global_mcp_server_manager.get_mcp_server_by_name(mcp_server_name)
        if mcp_server is None:
            raise HTTPException(
                status_code=404, detail=f"MCP server '{mcp_server_name}' not found"
            )

        # Create a new scope with the correct path format that the MCP handler expects
        # Transform /{mcp_server_name}/mcp to /mcp/{mcp_server_name}
        scope = dict(request.scope)
        scope["path"] = f"/mcp/{mcp_server_name}"

        # Import the MCP handler
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
        )

        # Create a custom send function to capture the response
        response_started = False
        response_body = b""
        response_status = 200
        response_headers = []

        async def custom_send(message):
            nonlocal response_started, response_body, response_status, response_headers
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                response_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")

        # Call the existing MCP handler
        await handle_streamable_http_mcp(
            scope, receive=request.receive, send=custom_send
        )

        # Return the response
        from starlette.responses import Response

        headers_dict = {k.decode(): v.decode() for k, v in response_headers}
        return Response(
            content=response_body,
            status_code=response_status,
            headers=headers_dict,
            media_type=headers_dict.get("content-type", "application/json"),
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error handling dynamic MCP route for {mcp_server_name}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


app.mount(path=BASE_MCP_ROUTE, app=mcp_app)
app.include_router(mcp_rest_endpoints_router)
app.include_router(mcp_discoverable_endpoints_router)
