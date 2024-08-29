import ast
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
import uuid
import warnings
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

import requests

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


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

import random

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


import pydantic

import litellm
from litellm import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchRequest,
    RetrieveBatchRequest,
)
from litellm._logging import verbose_proxy_logger, verbose_router_logger
from litellm.caching import DualCache, RedisCache
from litellm.exceptions import RejectedRequestError
from litellm.integrations.slack_alerting import SlackAlerting, SlackAlertingArgs
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.proxy._types import *
from litellm.proxy.analytics_endpoints.analytics_endpoints import (
    router as analytics_router,
)
from litellm.proxy.auth.auth_checks import (
    allowed_routes_check,
    common_checks,
    get_actual_routes,
    get_end_user_object,
    get_org_object,
    get_team_object,
    get_user_object,
    log_to_opentelemetry,
)
from litellm.proxy.auth.auth_utils import check_response_size_is_safe
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.auth.model_checks import (
    get_complete_model_list,
    get_key_models,
    get_team_models,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

## Import All Misc routes here ##
from litellm.proxy.caching_routes import router as caching_router
from litellm.proxy.common_utils.admin_ui_utils import (
    html_form,
    show_missing_vars_in_env,
)
from litellm.proxy.common_utils.callback_utils import (
    get_logging_caching_headers,
    get_remaining_tokens_and_requests_from_request_data,
    initialize_callbacks_on_proxy,
)
from litellm.proxy.common_utils.debug_utils import init_verbose_loggers
from litellm.proxy.common_utils.debug_utils import router as debugging_endpoints_router
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    check_file_size_under_limit,
)
from litellm.proxy.common_utils.load_config_utils import get_file_contents_from_s3
from litellm.proxy.common_utils.openai_endpoint_utils import (
    remove_sensitive_info_from_deployment,
)
from litellm.proxy.fine_tuning_endpoints.endpoints import router as fine_tuning_router
from litellm.proxy.fine_tuning_endpoints.endpoints import set_fine_tuning_config
from litellm.proxy.guardrails.init_guardrails import (
    init_guardrails_v2,
    initialize_guardrails,
)
from litellm.proxy.health_check import perform_health_check
from litellm.proxy.health_endpoints._health_endpoints import router as health_router
from litellm.proxy.hooks.prompt_injection_detection import (
    _OPTIONAL_PromptInjectionDetection,
)
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    router as internal_user_router,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import user_update
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _duration_in_seconds,
    delete_verification_token,
    generate_key_helper_fn,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    router as key_management_router,
)
from litellm.proxy.management_endpoints.team_callback_endpoints import (
    router as team_callback_router,
)
from litellm.proxy.management_endpoints.team_endpoints import router as team_router
from litellm.proxy.openai_files_endpoints.files_endpoints import (
    router as openai_files_router,
)
from litellm.proxy.openai_files_endpoints.files_endpoints import set_files_config
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    initialize_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    router as pass_through_router,
)
from litellm.proxy.rerank_endpoints.endpoints import router as rerank_router
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.secret_managers.aws_secret_manager import (
    load_aws_kms,
    load_aws_secret_manager,
)
from litellm.proxy.secret_managers.google_kms import load_google_kms
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    router as spend_management_router,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints import (
    router as ui_crud_endpoints_router,
)
from litellm.proxy.utils import (
    DBClient,
    PrismaClient,
    ProxyLogging,
    _cache_user_row,
    _get_projected_spend_over_limit,
    _is_projected_spend_over_limit,
    _is_valid_team_configs,
    get_error_message_str,
    get_instance_fn,
    hash_token,
    log_to_opentelemetry,
    reset_budget,
    send_email,
    update_spend,
)
from litellm.proxy.vertex_ai_endpoints.google_ai_studio_endpoints import (
    router as gemini_router,
)
from litellm.proxy.vertex_ai_endpoints.langfuse_endpoints import (
    router as langfuse_router,
)
from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import router as vertex_router
from litellm.proxy.vertex_ai_endpoints.vertex_endpoints import set_default_vertex_config
from litellm.router import (
    AssistantsTypedDict,
    Deployment,
    LiteLLM_Params,
    ModelGroupInfo,
)
from litellm.router import ModelInfo as RouterModelInfo
from litellm.router import updateDeployment
from litellm.scheduler import DefaultPriorities, FlowItem, Scheduler
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    AnthropicResponse,
    AnthropicResponseContentBlockText,
    AnthropicResponseUsageBlock,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import RouterGeneralSettings

try:
    from litellm._version import version
except:
    version = "0.0.0"
litellm.suppress_debug_info = True
import json
import logging
from typing import Union

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
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
try:
    # when using litellm cli
    import litellm.proxy.enterprise as enterprise
except Exception as e:
    # when using litellm docker image
    try:
        import enterprise  # type: ignore
    except Exception as e:
        pass

server_root_path = os.getenv("SERVER_ROOT_PATH", "")
_license_check = LicenseCheck()
premium_user: bool = _license_check.is_premium()
global_max_parallel_request_retries_env: Optional[str] = os.getenv(
    "LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRIES"
)
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
ui_message = (
    f"👉 [```LiteLLM Admin Panel on /ui```]({ui_link}). Create, Edit Keys with SSO"
)
ui_message += f"\n\n💸 [```LiteLLM Model Cost Map```](https://models.litellm.ai/)."

custom_swagger_message = f"[**Customize Swagger Docs**](https://docs.litellm.ai/docs/proxy/enterprise#swagger-docs---custom-routes--branding)"

### CUSTOM BRANDING [ENTERPRISE FEATURE] ###
_docs_url = None if os.getenv("NO_DOCS", "False") == "True" else "/"
_title = os.getenv("DOCS_TITLE", "LiteLLM API") if premium_user else "LiteLLM API"
_description = (
    os.getenv(
        "DOCS_DESCRIPTION",
        f"Enterprise Edition \n\nProxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}",
    )
    if premium_user
    else f"Proxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}"
)


app = FastAPI(
    docs_url=_docs_url,
    title=_title,
    description=_description,
    version=version,
    root_path=server_root_path,  # check if user passed root path, FastAPI defaults this value to ""
)


### CUSTOM API DOCS [ENTERPRISE FEATURE] ###
# Custom OpenAPI schema generator to include only selected routes
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Filter routes to include only specific ones
    openai_routes = LiteLLMRoutes.openai_routes.value
    paths_to_include: dict = {}
    for route in openai_routes:
        paths_to_include[route] = openapi_schema["paths"][route]
    openapi_schema["paths"] = paths_to_include
    app.openapi_schema = openapi_schema
    return app.openapi_schema


if os.getenv("DOCS_FILTERED", "False") == "True" and premium_user:
    app.openapi = custom_openapi  # type: ignore


class UserAPIKeyCacheTTLEnum(enum.Enum):
    in_memory_cache_ttl = 60  # 1 min ttl ## configure via `general_settings::user_api_key_cache_ttl: <your-value>`


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    # NOTE: DO NOT MODIFY THIS, its crucial to map to Openai exceptions
    headers = exc.headers
    return JSONResponse(
        status_code=(
            int(exc.code) if exc.code else status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content={
            "error": {
                "message": exc.message,
                "type": exc.type,
                "param": exc.param,
                "code": exc.code,
            }
        },
        headers=headers,
    )


router = APIRouter()
origins = ["*"]

# get current directory
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(current_dir, "_experimental", "out")
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")
    # Iterate through files in the UI directory
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

    if server_root_path != "":
        print(  # noqa
            f"server_root_path is set, forwarding any /ui requests to {server_root_path}/ui"
        )  # noqa
        if os.getenv("PROXY_BASE_URL") is None:
            os.environ["PROXY_BASE_URL"] = server_root_path

        @app.middleware("http")
        async def redirect_ui_middleware(request: Request, call_next):
            if request.url.path.startswith("/ui"):
                new_path = request.url.path.replace("/ui", f"{server_root_path}/ui", 1)
                return RedirectResponse(new_path)
            return await call_next(request)

except:
    pass
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
user_config_file_path = f"config_{int(time.time())}.yaml"
local_logging = True  # writes logs to a local api_log.json file for debugging
experimental = False
#### GLOBAL VARIABLES ####
llm_router: Optional[litellm.Router] = None
llm_model_list: Optional[list] = None
general_settings: dict = {}
log_file = "api_log.json"
worker_config = None
master_key = None
otel_logging = False
prisma_client: Optional[PrismaClient] = None
custom_db_client: Optional[DBClient] = None
user_api_key_cache = DualCache(
    default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value
)
redis_usage_cache: Optional[RedisCache] = (
    None  # redis cache used for tracking spend, tpm/rpm limits
)
user_custom_auth = None
user_custom_key_generate = None
use_background_health_checks = None
use_queue = False
health_check_interval = None
health_check_details = None
health_check_results = {}
queue: List = []
litellm_proxy_budget_name = "litellm-proxy-budget"
litellm_proxy_admin_name = "default_user_id"
ui_access_mode: Literal["admin", "all"] = "all"
proxy_budget_rescheduler_min_time = 597
proxy_budget_rescheduler_max_time = 605
proxy_batch_write_at = 10  # in seconds
litellm_master_key_hash = None
disable_spend_logs = False
jwt_handler = JWTHandler()
prompt_injection_detection_obj: Optional[_OPTIONAL_PromptInjectionDetection] = None
store_model_in_db: bool = False
open_telemetry_logger = None
### INITIALIZE GLOBAL LOGGING OBJECT ###
proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
### REDIS QUEUE ###
async_result = None
celery_app_conn = None
celery_fn = None  # Redis Queue for handling requests
### DB WRITER ###
db_writer_client: Optional[HTTPHandler] = None
### logger ###


def _get_pydantic_json_dict(pydantic_obj: BaseModel) -> dict:
    try:
        return pydantic_obj.model_dump()  # type: ignore
    except:
        # if using pydantic v1
        return pydantic_obj.dict()


def get_custom_headers(
    *,
    user_api_key_dict: UserAPIKeyAuth,
    call_id: Optional[str] = None,
    model_id: Optional[str] = None,
    cache_key: Optional[str] = None,
    api_base: Optional[str] = None,
    version: Optional[str] = None,
    model_region: Optional[str] = None,
    response_cost: Optional[Union[float, str]] = None,
    fastest_response_batch_completion: Optional[bool] = None,
    request_data: Optional[dict] = {},
    **kwargs,
) -> dict:
    exclude_values = {"", None}
    headers = {
        "x-litellm-call-id": call_id,
        "x-litellm-model-id": model_id,
        "x-litellm-cache-key": cache_key,
        "x-litellm-model-api-base": api_base,
        "x-litellm-version": version,
        "x-litellm-model-region": model_region,
        "x-litellm-response-cost": str(response_cost),
        "x-litellm-key-tpm-limit": str(user_api_key_dict.tpm_limit),
        "x-litellm-key-rpm-limit": str(user_api_key_dict.rpm_limit),
        "x-litellm-fastest_response_batch_completion": (
            str(fastest_response_batch_completion)
            if fastest_response_batch_completion is not None
            else None
        ),
        **{k: str(v) for k, v in kwargs.items()},
    }
    if request_data:
        remaining_tokens_header = get_remaining_tokens_and_requests_from_request_data(
            request_data
        )
        headers.update(remaining_tokens_header)

        logging_caching_headers = get_logging_caching_headers(request_data)
        if logging_caching_headers:
            headers.update(logging_caching_headers)

    try:
        return {
            key: value for key, value in headers.items() if value not in exclude_values
        }
    except Exception as e:
        verbose_proxy_logger.error(f"Error setting custom headers: {e}")
        return {}


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


def prisma_setup(database_url: Optional[str]):
    global prisma_client, proxy_logging_obj, user_api_key_cache

    if database_url is not None:
        try:
            prisma_client = PrismaClient(
                database_url=database_url, proxy_logging_obj=proxy_logging_obj
            )
        except Exception as e:
            raise e


def load_from_azure_key_vault(use_azure_key_vault: bool = False):
    if use_azure_key_vault is False:
        return

    try:
        from azure.identity import ClientSecretCredential
        from azure.keyvault.secrets import SecretClient

        # Set your Azure Key Vault URI
        KVUri = os.getenv("AZURE_KEY_VAULT_URI", None)

        # Set your Azure AD application/client ID, client secret, and tenant ID
        client_id = os.getenv("AZURE_CLIENT_ID", None)
        client_secret = os.getenv("AZURE_CLIENT_SECRET", None)
        tenant_id = os.getenv("AZURE_TENANT_ID", None)

        if (
            KVUri is not None
            and client_id is not None
            and client_secret is not None
            and tenant_id is not None
        ):
            # Initialize the ClientSecretCredential
            credential = ClientSecretCredential(
                client_id=client_id, client_secret=client_secret, tenant_id=tenant_id
            )

            # Create the SecretClient using the credential
            client = SecretClient(vault_url=KVUri, credential=credential)

            litellm.secret_manager_client = client
            litellm._key_management_system = KeyManagementSystem.AZURE_KEY_VAULT
        else:
            raise Exception(
                f"Missing KVUri or client_id or client_secret or tenant_id from environment"
            )
    except Exception as e:
        verbose_proxy_logger.debug(
            "Error when loading keys from Azure Key Vault. Ensure you run `pip install azure-identity azure-keyvault-secrets`"
        )


def cost_tracking():
    global prisma_client, custom_db_client
    if prisma_client is not None or custom_db_client is not None:
        if isinstance(litellm.success_callback, list):
            verbose_proxy_logger.debug("setting litellm success callback to track cost")
            if (_PROXY_track_cost_callback) not in litellm.success_callback:  # type: ignore
                litellm.success_callback.append(_PROXY_track_cost_callback)  # type: ignore


async def _PROXY_failure_handler(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    global prisma_client
    if prisma_client is not None:
        verbose_proxy_logger.debug(
            "inside _PROXY_failure_handler kwargs=", extra=kwargs
        )

        _exception = kwargs.get("exception")
        _exception_type = _exception.__class__.__name__
        _model = kwargs.get("model", None)

        _optional_params = kwargs.get("optional_params", {})
        _optional_params = copy.deepcopy(_optional_params)

        for k, v in _optional_params.items():
            v = str(v)
            v = v[:100]

        _status_code = "500"
        try:
            _status_code = str(_exception.status_code)
        except:
            # Don't let this fail logging the exception to the dB
            pass

        _litellm_params = kwargs.get("litellm_params", {}) or {}
        _metadata = _litellm_params.get("metadata", {}) or {}
        _model_id = _metadata.get("model_info", {}).get("id", "")
        _model_group = _metadata.get("model_group", "")
        api_base = litellm.get_api_base(model=_model, optional_params=_litellm_params)
        _exception_string = str(_exception)

        error_log = LiteLLM_ErrorLogs(
            request_id=str(uuid.uuid4()),
            model_group=_model_group,
            model_id=_model_id,
            litellm_model_name=kwargs.get("model"),
            request_kwargs=_optional_params,
            api_base=api_base,
            exception_type=_exception_type,
            status_code=_status_code,
            exception_string=_exception_string,
            startTime=kwargs.get("start_time"),
            endTime=kwargs.get("end_time"),
        )

        # helper function to convert to dict on pydantic v2 & v1
        error_log_dict = _get_pydantic_json_dict(error_log)
        error_log_dict["request_kwargs"] = json.dumps(error_log_dict["request_kwargs"])

        await prisma_client.db.litellm_errorlogs.create(
            data=error_log_dict  # type: ignore
        )

    pass


@log_to_opentelemetry
async def _PROXY_track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    verbose_proxy_logger.debug("INSIDE _PROXY_track_cost_callback")
    global prisma_client, custom_db_client
    try:
        # check if it has collected an entire stream response
        verbose_proxy_logger.debug(
            "Proxy: In track_cost_callback for: kwargs=%s and completion_response: %s",
            kwargs,
            completion_response,
        )
        verbose_proxy_logger.debug(
            f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}"
        )
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
        user_id = metadata.get("user_api_key_user_id", None)
        team_id = metadata.get("user_api_key_team_id", None)
        org_id = metadata.get("user_api_key_org_id", None)
        key_alias = metadata.get("user_api_key_alias", None)
        end_user_max_budget = metadata.get("user_api_end_user_max_budget", None)
        if kwargs.get("response_cost", None) is not None:
            response_cost = kwargs["response_cost"]
            user_api_key = metadata.get("user_api_key", None)

            if kwargs.get("cache_hit", False) == True:
                response_cost = 0.0
                verbose_proxy_logger.info(
                    f"Cache Hit: response_cost {response_cost}, for user_id {user_id}"
                )

            verbose_proxy_logger.debug(
                f"user_api_key {user_api_key}, prisma_client: {prisma_client}, custom_db_client: {custom_db_client}"
            )
            if user_api_key is not None or user_id is not None or team_id is not None:
                ## UPDATE DATABASE
                await update_database(
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
                )

                await update_cache(
                    token=user_api_key,
                    user_id=user_id,
                    end_user_id=end_user_id,
                    response_cost=response_cost,
                    team_id=team_id,
                )

                await proxy_logging_obj.slack_alerting_instance.customer_spend_alert(
                    token=user_api_key,
                    key_alias=key_alias,
                    end_user_id=end_user_id,
                    response_cost=response_cost,
                    max_budget=end_user_max_budget,
                )
            else:
                raise Exception(
                    "User API key and team id and user id missing from custom callback."
                )
        else:
            if kwargs["stream"] != True or (
                kwargs["stream"] == True and "complete_streaming_response" in kwargs
            ):
                raise Exception(
                    f"Model not in litellm model cost map. Add custom pricing - https://docs.litellm.ai/docs/proxy/custom_pricing"
                )
    except Exception as e:
        error_msg = f"error in tracking cost callback - {traceback.format_exc()}"
        model = kwargs.get("model", "")
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        error_msg += f"\n Args to _PROXY_track_cost_callback\n model: {model}\n metadata: {metadata}\n"
        asyncio.create_task(
            proxy_logging_obj.failed_tracking_alert(
                error_message=error_msg,
            )
        )
        verbose_proxy_logger.debug("error in tracking cost callback - %s", e)


def error_tracking():
    global prisma_client, custom_db_client
    if prisma_client is not None or custom_db_client is not None:
        if isinstance(litellm.failure_callback, list):
            verbose_proxy_logger.debug("setting litellm failure callback to track cost")
            if (_PROXY_failure_handler) not in litellm.failure_callback:  # type: ignore
                litellm.failure_callback.append(_PROXY_failure_handler)  # type: ignore


def _set_spend_logs_payload(
    payload: dict, prisma_client: PrismaClient, spend_logs_url: Optional[str] = None
):
    if prisma_client is not None and spend_logs_url is not None:
        if isinstance(payload["startTime"], datetime):
            payload["startTime"] = payload["startTime"].isoformat()
        if isinstance(payload["endTime"], datetime):
            payload["endTime"] = payload["endTime"].isoformat()
        prisma_client.spend_log_transactions.append(payload)
    elif prisma_client is not None:
        prisma_client.spend_log_transactions.append(payload)
    return prisma_client


async def update_database(
    token,
    response_cost,
    user_id=None,
    end_user_id=None,
    team_id=None,
    kwargs=None,
    completion_response=None,
    start_time=None,
    end_time=None,
    org_id=None,
):
    try:
        global prisma_client
        verbose_proxy_logger.info(
            f"Enters prisma db call, response_cost: {response_cost}, token: {token}; user_id: {user_id}; team_id: {team_id}"
        )
        if token is not None and isinstance(token, str) and token.startswith("sk-"):
            hashed_token = hash_token(token=token)
        else:
            hashed_token = token

        ### UPDATE USER SPEND ###
        async def _update_user_db():
            """
            - Update that user's row
            - Update litellm-proxy-budget row (global proxy spend)
            """
            ## if an end-user is passed in, do an upsert - we can't guarantee they already exist in db
            existing_token_obj = await user_api_key_cache.async_get_cache(
                key=hashed_token
            )
            existing_user_obj = await user_api_key_cache.async_get_cache(key=user_id)
            if existing_user_obj is not None and isinstance(existing_user_obj, dict):
                existing_user_obj = LiteLLM_UserTable(**existing_user_obj)
            data_list = []
            try:
                if prisma_client is not None:  # update
                    user_ids = [user_id]
                    if (
                        litellm.max_budget > 0
                    ):  # track global proxy budget, if user set max budget
                        user_ids.append(litellm_proxy_budget_name)
                    ### KEY CHANGE ###
                    for _id in user_ids:
                        if _id is not None:
                            prisma_client.user_list_transactons[_id] = (
                                response_cost
                                + prisma_client.user_list_transactons.get(_id, 0)
                            )
                    if end_user_id is not None:
                        prisma_client.end_user_list_transactons[end_user_id] = (
                            response_cost
                            + prisma_client.end_user_list_transactons.get(
                                end_user_id, 0
                            )
                        )
            except Exception as e:
                verbose_proxy_logger.info(
                    "\033[91m"
                    + f"Update User DB call failed to execute {str(e)}\n{traceback.format_exc()}"
                )

        ### UPDATE KEY SPEND ###
        async def _update_key_db():
            try:
                verbose_proxy_logger.debug(
                    f"adding spend to key db. Response cost: {response_cost}. Token: {hashed_token}."
                )
                if hashed_token is None:
                    return
                if prisma_client is not None:
                    prisma_client.key_list_transactons[hashed_token] = (
                        response_cost
                        + prisma_client.key_list_transactons.get(hashed_token, 0)
                    )
            except Exception as e:
                verbose_proxy_logger.exception(
                    f"Update Key DB Call failed to execute - {str(e)}"
                )
                raise e

        ### UPDATE SPEND LOGS ###
        async def _insert_spend_log_to_db():
            try:
                global prisma_client
                if prisma_client is not None:
                    # Helper to generate payload to log
                    payload = get_logging_payload(
                        kwargs=kwargs,
                        response_obj=completion_response,
                        start_time=start_time,
                        end_time=end_time,
                        end_user_id=end_user_id,
                    )

                    payload["spend"] = response_cost
                    prisma_client = _set_spend_logs_payload(
                        payload=payload,
                        spend_logs_url=os.getenv("SPEND_LOGS_URL"),
                        prisma_client=prisma_client,
                    )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Update Spend Logs DB failed to execute - {str(e)}\n{traceback.format_exc()}"
                )
                raise e

        ### UPDATE TEAM SPEND ###
        async def _update_team_db():
            try:
                verbose_proxy_logger.debug(
                    f"adding spend to team db. Response cost: {response_cost}. team_id: {team_id}."
                )
                if team_id is None:
                    verbose_proxy_logger.debug(
                        "track_cost_callback: team_id is None. Not tracking spend for team"
                    )
                    return
                if prisma_client is not None:
                    prisma_client.team_list_transactons[team_id] = (
                        response_cost
                        + prisma_client.team_list_transactons.get(team_id, 0)
                    )

                    try:
                        # Track spend of the team member within this team
                        # key is "team_id::<value>::user_id::<value>"
                        team_member_key = f"team_id::{team_id}::user_id::{user_id}"
                        prisma_client.team_member_list_transactons[team_member_key] = (
                            response_cost
                            + prisma_client.team_member_list_transactons.get(
                                team_member_key, 0
                            )
                        )
                    except:
                        pass
            except Exception as e:
                verbose_proxy_logger.info(
                    f"Update Team DB failed to execute - {str(e)}\n{traceback.format_exc()}"
                )
                raise e

        ### UPDATE ORG SPEND ###
        async def _update_org_db():
            try:
                verbose_proxy_logger.debug(
                    "adding spend to org db. Response cost: {}. org_id: {}.".format(
                        response_cost, org_id
                    )
                )
                if org_id is None:
                    verbose_proxy_logger.debug(
                        "track_cost_callback: org_id is None. Not tracking spend for org"
                    )
                    return
                if prisma_client is not None:
                    prisma_client.org_list_transactons[org_id] = (
                        response_cost
                        + prisma_client.org_list_transactons.get(org_id, 0)
                    )
            except Exception as e:
                verbose_proxy_logger.info(
                    f"Update Org DB failed to execute - {str(e)}\n{traceback.format_exc()}"
                )
                raise e

        asyncio.create_task(_update_user_db())
        asyncio.create_task(_update_key_db())
        asyncio.create_task(_update_team_db())
        asyncio.create_task(_update_org_db())
        # asyncio.create_task(_insert_spend_log_to_db())
        if disable_spend_logs is False:
            await _insert_spend_log_to_db()
        else:
            verbose_proxy_logger.info(
                "disable_spend_logs=True. Skipping writing spend logs to db. Other spend updates - Key/User/Team table will still occur."
            )

        verbose_proxy_logger.debug("Runs spend update on all tables")
    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error updating Prisma database: {traceback.format_exc()}"
        )


async def update_cache(
    token: Optional[str],
    user_id: Optional[str],
    end_user_id: Optional[str],
    team_id: Optional[str],
    response_cost: Optional[float],
):
    """
    Use this to update the cache with new user spend.

    Put any alerting logic in here.
    """

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
        verbose_proxy_logger.debug(
            f"_update_key_cache: existing spend: {existing_spend_obj}"
        )
        if existing_spend_obj is None:
            return
        else:
            existing_spend = existing_spend_obj.spend
        # Calculate the new cost by adding the existing cost and response_cost
        new_spend = existing_spend + response_cost

        ## CHECK IF USER PROJECTED SPEND > SOFT LIMIT
        soft_budget_cooldown = existing_spend_obj.soft_budget_cooldown
        if (
            existing_spend_obj.soft_budget_cooldown == False
            and existing_spend_obj.litellm_budget_table is not None
            and (
                _is_projected_spend_over_limit(
                    current_spend=new_spend,
                    soft_budget_limit=existing_spend_obj.litellm_budget_table[
                        "soft_budget"
                    ],
                )
                == True
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
            )
            # alert user
            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="projected_limit_exceeded",
                    user_info=call_info,
                )
            )
            # set cooldown on alert
            soft_budget_cooldown = True

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
        user_api_key_cache.set_cache(key=hashed_token, value=existing_spend_obj)

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
                    user_api_key_cache.set_cache(key=_id, value=existing_spend_obj)
                else:
                    existing_spend_obj.spend = new_spend
                    user_api_key_cache.set_cache(
                        key=_id, value=existing_spend_obj.json()
                    )
            ## UPDATE GLOBAL PROXY ##
            global_proxy_spend = await user_api_key_cache.async_get_cache(
                key="{}:spend".format(litellm_proxy_admin_name)
            )
            if global_proxy_spend is None:
                # do nothing if not in cache
                return
            elif response_cost is not None and global_proxy_spend is not None:
                increment = global_proxy_spend + response_cost
                await user_api_key_cache.async_set_cache(
                    key="{}:spend".format(litellm_proxy_admin_name), value=increment
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
                user_api_key_cache.set_cache(key=_id, value=existing_spend_obj)
            else:
                existing_spend_obj.spend = new_spend
                user_api_key_cache.set_cache(key=_id, value=existing_spend_obj.json())
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
                user_api_key_cache.set_cache(key=_id, value=existing_spend_obj)
            else:
                existing_spend_obj.spend = new_spend
                user_api_key_cache.set_cache(key=_id, value=existing_spend_obj)
        except Exception as e:
            verbose_proxy_logger.exception(
                f"An error occurred updating end user cache: {str(e)}"
            )

    if token is not None and response_cost is not None:
        asyncio.create_task(_update_key_cache(token=token, response_cost=response_cost))

    if user_id is not None:
        asyncio.create_task(_update_user_cache())

    if end_user_id is not None:
        asyncio.create_task(_update_end_user_cache())

    if team_id is not None:
        asyncio.create_task(_update_team_cache())


def run_ollama_serve():
    try:
        command = ["ollama", "serve"]

        with open(os.devnull, "w") as devnull:
            process = subprocess.Popen(command, stdout=devnull, stderr=devnull)
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
    """
    global health_check_results, llm_model_list, health_check_interval, health_check_details

    # make 1 deep copy of llm_model_list -> use this for all background health checks
    _llm_model_list = copy.deepcopy(llm_model_list)

    while True:
        healthy_endpoints, unhealthy_endpoints = await perform_health_check(
            model_list=_llm_model_list, details=health_check_details
        )

        # Update the global variable with the health check results
        health_check_results["healthy_endpoints"] = healthy_endpoints
        health_check_results["unhealthy_endpoints"] = unhealthy_endpoints
        health_check_results["healthy_count"] = len(healthy_endpoints)
        health_check_results["unhealthy_count"] = len(unhealthy_endpoints)

        await asyncio.sleep(health_check_interval)


class ProxyConfig:
    """
    Abstraction class on top of config loading/updating logic. Gives us one place to control all config updating logic.
    """

    def __init__(self) -> None:
        pass

    def is_yaml(self, config_file_path: str) -> bool:
        if not os.path.isfile(config_file_path):
            return False

        _, file_extension = os.path.splitext(config_file_path)
        return file_extension.lower() == ".yaml" or file_extension.lower() == ".yml"

    async def get_config(self, config_file_path: Optional[str] = None) -> dict:
        global prisma_client, user_config_file_path

        file_path = config_file_path or user_config_file_path
        if config_file_path is not None:
            user_config_file_path = config_file_path
        # Load existing config
        ## Yaml
        if os.path.exists(f"{file_path}"):
            with open(f"{file_path}", "r") as config_file:
                config = yaml.safe_load(config_file)
        else:
            config = {
                "model_list": [],
                "general_settings": {},
                "router_settings": {},
                "litellm_settings": {},
            }

        ## DB
        if prisma_client is not None and (
            general_settings.get("store_model_in_db", False) == True
            or store_model_in_db is True
        ):
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
                if response is not None:
                    param_name = getattr(response, "param_name", None)
                    param_value = getattr(response, "param_value", None)
                    if param_name is not None and param_value is not None:
                        # check if param_name is already in the config
                        if param_name in config:
                            if isinstance(config[param_name], dict):
                                config[param_name].update(param_value)
                            else:
                                config[param_name] = param_value
                        else:
                            # if it's not in the config - then add it
                            config[param_name] = param_value

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
            general_settings.get("store_model_in_db", False) == True
            or store_model_in_db
        ):
            # if using - db for config - models are in ModelTable
            new_config.pop("model_list", None)
            await prisma_client.insert_data(data=new_config, table_name="config")
        else:
            # Save the updated config - if user is not using a dB
            ## YAML
            with open(f"{user_config_file_path}", "w") as config_file:
                yaml.dump(new_config, config_file, default_flow_style=False)

    async def load_team_config(self, team_id: str):
        """
        - for a given team id
        - return the relevant completion() call params
        """
        # load existing config
        config = await self.get_config()
        ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
        litellm_settings = config.get("litellm_settings", {})
        all_teams_config = litellm_settings.get("default_team_settings", None)
        team_config: dict = {}
        if all_teams_config is None:
            return team_config
        for team in all_teams_config:
            if "team_id" not in team:
                raise Exception(f"team_id missing from team: {team}")
            if team_id == team["team_id"]:
                team_config = team
                break
        for k, v in team_config.items():
            if isinstance(v, str) and v.startswith("os.environ/"):
                team_config[k] = litellm.get_secret(v)
        return team_config

    def _init_cache(
        self,
        cache_params: dict,
    ):
        global redis_usage_cache
        from litellm import Cache

        if "default_in_memory_ttl" in cache_params:
            litellm.default_in_memory_ttl = cache_params["default_in_memory_ttl"]

        if "default_redis_ttl" in cache_params:
            litellm.default_redis_ttl = cache_params["default_in_redis_ttl"]

        litellm.cache = Cache(**cache_params)

        if litellm.cache is not None and isinstance(litellm.cache.cache, RedisCache):
            ## INIT PROXY REDIS USAGE CLIENT ##
            redis_usage_cache = litellm.cache.cache

    async def load_config(
        self, router: Optional[litellm.Router], config_file_path: str
    ):
        """
        Load config values into proxy global state
        """
        global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, user_custom_key_generate, use_background_health_checks, health_check_interval, use_queue, custom_db_client, proxy_budget_rescheduler_max_time, proxy_budget_rescheduler_min_time, ui_access_mode, litellm_master_key_hash, proxy_batch_write_at, disable_spend_logs, prompt_injection_detection_obj, redis_usage_cache, store_model_in_db, premium_user, open_telemetry_logger, health_check_details

        # Load existing config
        if os.environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None:
            bucket_name = os.environ.get("LITELLM_CONFIG_BUCKET_NAME")
            object_key = os.environ.get("LITELLM_CONFIG_BUCKET_OBJECT_KEY")
            verbose_proxy_logger.debug(
                "bucket_name: %s, object_key: %s", bucket_name, object_key
            )
            config = get_file_contents_from_s3(
                bucket_name=bucket_name, object_key=object_key
            )
        else:
            # default to file
            config = await self.get_config(config_file_path=config_file_path)
        ## PRINT YAML FOR CONFIRMING IT WORKS
        printed_yaml = copy.deepcopy(config)
        printed_yaml.pop("environment_variables", None)

        verbose_proxy_logger.debug(
            f"Loaded config YAML (api_key and environment_variables are not shown):\n{json.dumps(printed_yaml, indent=2)}"
        )

        ## ENVIRONMENT VARIABLES
        environment_variables = config.get("environment_variables", None)
        if environment_variables:
            for key, value in environment_variables.items():
                os.environ[key] = str(
                    litellm.get_secret(secret_name=key, default_value=value)
                )

            # check if litellm_license in general_settings
            if "LITELLM_LICENSE" in environment_variables:
                _license_check.license_str = os.getenv("LITELLM_LICENSE", None)
                premium_user = _license_check.is_premium()

        ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
        litellm_settings = config.get("litellm_settings", None)
        if litellm_settings is None:
            litellm_settings = {}
        if litellm_settings:
            # ANSI escape code for blue text
            blue_color_code = "\033[94m"
            reset_color_code = "\033[0m"
            for key, value in litellm_settings.items():
                if key == "cache" and value == True:
                    print(f"{blue_color_code}\nSetting Cache on Proxy")  # noqa
                    from litellm.caching import Cache

                    cache_params = {}
                    if "cache_params" in litellm_settings:
                        cache_params_in_config = litellm_settings["cache_params"]
                        # overwrie cache_params with cache_params_in_config
                        cache_params.update(cache_params_in_config)

                    cache_type = cache_params.get("type", "redis")

                    verbose_proxy_logger.debug("passed cache type=%s", cache_type)

                    if (
                        cache_type == "redis" or cache_type == "redis-semantic"
                    ) and len(cache_params.keys()) == 0:
                        cache_host = litellm.get_secret("REDIS_HOST", None)
                        cache_port = litellm.get_secret("REDIS_PORT", None)
                        cache_password = None
                        cache_params.update(
                            {
                                "type": cache_type,
                                "host": cache_host,
                                "port": cache_port,
                            }
                        )

                        if litellm.get_secret("REDIS_PASSWORD", None) is not None:
                            cache_password = litellm.get_secret("REDIS_PASSWORD", None)
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
                    if cache_type == "redis-semantic":
                        # by default this should always be async
                        cache_params.update({"redis_semantic_cache_use_async": True})

                    # users can pass os.environ/ variables on the proxy - we should read them from the env
                    for key, value in cache_params.items():
                        if type(value) is str and value.startswith("os.environ/"):
                            cache_params[key] = litellm.get_secret(value)

                    ## to pass a complete url, or set ssl=True, etc. just set it as `os.environ[REDIS_URL] = <your-redis-url>`, _redis.py checks for REDIS specific environment variables
                    self._init_cache(cache_params=cache_params)
                    if litellm.cache is not None:
                        verbose_proxy_logger.debug(  # noqa
                            f"{blue_color_code}Set Cache on LiteLLM Proxy= {vars(litellm.cache.cache)}{vars(litellm.cache)}{reset_color_code}"
                        )
                elif key == "cache" and value is False:
                    pass
                elif key == "guardrails":
                    if premium_user is not True:
                        raise ValueError(
                            "Trying to use `guardrails` on config.yaml "
                            + CommonProxyErrors.not_premium_user.value
                        )

                    guardrail_name_config_map = initialize_guardrails(
                        guardrails_config=value,
                        premium_user=premium_user,
                        config_file_path=config_file_path,
                        litellm_settings=litellm_settings,
                    )

                    litellm.guardrail_name_config_map = guardrail_name_config_map
                elif key == "callbacks":

                    initialize_callbacks_on_proxy(
                        value=value,
                        premium_user=premium_user,
                        config_file_path=config_file_path,
                        litellm_settings=litellm_settings,
                    )

                elif key == "post_call_rules":
                    litellm.post_call_rules = [
                        get_instance_fn(value=value, config_file_path=config_file_path)
                    ]
                    verbose_proxy_logger.debug(
                        f"litellm.post_call_rules: {litellm.post_call_rules}"
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
                            litellm.success_callback.append(
                                get_instance_fn(value=callback)
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.success_callback.append(callback)
                            if "prometheus" in callback:
                                verbose_proxy_logger.debug(
                                    "Starting Prometheus Metrics on /metrics"
                                )
                                from prometheus_client import make_asgi_app

                                # Add prometheus asgi middleware to route /metrics requests
                                metrics_app = make_asgi_app()
                                app.mount("/metrics", metrics_app)
                    print(  # noqa
                        f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}"
                    )  # noqa
                elif key == "failure_callback":
                    litellm.failure_callback = []

                    # initialize success callbacks
                    for callback in value:
                        # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                        if "." in callback:
                            litellm.failure_callback.append(
                                get_instance_fn(value=callback)
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.failure_callback.append(callback)
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
                        except:
                            raise Exception(
                                f"team_id missing from default_team_settings at index={idx}\npassed in value={team_setting}"
                            )
                    verbose_proxy_logger.debug(
                        f"{blue_color_code} setting litellm.{key}={value}{reset_color_code}"
                    )
                    setattr(litellm, key, value)
                elif key == "upperbound_key_generate_params":
                    if value is not None and isinstance(value, dict):
                        for _k, _v in value.items():
                            if isinstance(_v, str) and _v.startswith("os.environ/"):
                                value[_k] = litellm.get_secret(_v)
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
            ### LOAD SECRET MANAGER ###
            key_management_system = general_settings.get("key_management_system", None)
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
                    ### LOAD FROM AWS SECRET MANAGER ###
                    load_aws_secret_manager(use_aws_secret_manager=True)
                elif key_management_system == KeyManagementSystem.AWS_KMS.value:
                    load_aws_kms(use_aws_kms=True)
                else:
                    raise ValueError("Invalid Key Management System selected")
            key_management_settings = general_settings.get(
                "key_management_settings", None
            )
            if key_management_settings is not None:
                litellm._key_management_settings = KeyManagementSettings(
                    **key_management_settings
                )
            ### [DEPRECATED] LOAD FROM GOOGLE KMS ### old way of loading from google kms
            use_google_kms = general_settings.get("use_google_kms", False)
            load_google_kms(use_google_kms=use_google_kms)
            ### [DEPRECATED] LOAD FROM AZURE KEY VAULT ### old way of loading from azure secret manager
            use_azure_key_vault = general_settings.get("use_azure_key_vault", False)
            load_from_azure_key_vault(use_azure_key_vault=use_azure_key_vault)
            ### ALERTING ###

            proxy_logging_obj.update_values(
                alerting=general_settings.get("alerting", None),
                alerting_threshold=general_settings.get("alerting_threshold", 600),
                alert_types=general_settings.get("alert_types", None),
                alert_to_webhook_url=general_settings.get("alert_to_webhook_url", None),
                alerting_args=general_settings.get("alerting_args", None),
                redis_cache=redis_usage_cache,
            )
            ### CONNECT TO DATABASE ###
            database_url = general_settings.get("database_url", None)
            if database_url and database_url.startswith("os.environ/"):
                verbose_proxy_logger.debug("GOING INTO LITELLM.GET_SECRET!")
                database_url = litellm.get_secret(database_url)
                verbose_proxy_logger.debug("RETRIEVED DB URL: %s", database_url)
            ### MASTER KEY ###
            master_key = general_settings.get(
                "master_key", litellm.get_secret("LITELLM_MASTER_KEY", None)
            )

            if master_key and master_key.startswith("os.environ/"):
                master_key = litellm.get_secret(master_key)
                if not isinstance(master_key, str):
                    raise Exception(
                        "Master key must be a string. Current type - {}".format(
                            type(master_key)
                        )
                    )

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
            ## pass through endpoints
            if general_settings.get("pass_through_endpoints", None) is not None:
                await initialize_pass_through_endpoints(
                    pass_through_endpoints=general_settings["pass_through_endpoints"]
                )
            ## dynamodb
            database_type = general_settings.get("database_type", None)
            if database_type is not None and (
                database_type == "dynamo_db" or database_type == "dynamodb"
            ):
                database_args = general_settings.get("database_args", None)
                ### LOAD FROM os.environ/ ###
                for k, v in database_args.items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        database_args[k] = litellm.get_secret(v)
                    if isinstance(k, str) and k == "aws_web_identity_token":
                        value = database_args[k]
                        verbose_proxy_logger.debug(
                            f"Loading AWS Web Identity Token from file: {value}"
                        )
                        if os.path.exists(value):
                            with open(value, "r") as file:
                                token_content = file.read()
                                database_args[k] = token_content
                        else:
                            verbose_proxy_logger.info(
                                f"DynamoDB Loading - {value} is not a valid file path"
                            )
                verbose_proxy_logger.debug("database_args: %s", database_args)
                custom_db_client = DBClient(
                    custom_db_args=database_args, custom_db_type=database_type
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
            health_check_interval = general_settings.get("health_check_interval", 300)
            health_check_details = general_settings.get("health_check_details", True)

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
            != None,  # cache if user passed in cache values
        }
        ## MODEL LIST
        model_list = config.get("model_list", None)
        if model_list:
            router_params["model_list"] = model_list
            print(  # noqa
                f"\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m"
            )  # noqa
            for model in model_list:
                ### LOAD FROM os.environ/ ###
                for k, v in model["litellm_params"].items():
                    if isinstance(v, str) and v.startswith("os.environ/"):
                        model["litellm_params"][k] = litellm.get_secret(v)
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

        ## /fine_tuning/jobs endpoints config
        finetuning_config = config.get("finetune_settings", None)
        set_fine_tuning_config(config=finetuning_config)

        ## /files endpoint config
        files_config = config.get("files_settings", None)
        set_files_config(config=files_config)

        ## default config for vertex ai routes
        default_vertex_config = config.get("default_vertex_config", None)
        set_default_vertex_config(config=default_vertex_config)

        ## ROUTER SETTINGS (e.g. routing_strategy, ...)
        router_settings = config.get("router_settings", None)
        if router_settings and isinstance(router_settings, dict):
            arg_spec = inspect.getfullargspec(litellm.Router)
            # model list already set
            exclude_args = {
                "self",
                "model_list",
            }

            available_args = [x for x in arg_spec.args if x not in exclude_args]

            for k, v in router_settings.items():
                if k in available_args:
                    router_params[k] = v
        router = litellm.Router(
            **router_params,
            assistants_config=assistants_config,
            router_general_settings=RouterGeneralSettings(
                async_only_mode=True  # only init async clients
            ),
        )  # type:ignore

        # Guardrail settings
        guardrails_v2 = config.get("guardrails", None)
        if guardrails_v2:
            init_guardrails_v2(
                all_guardrails=guardrails_v2, config_file_path=config_file_path
            )
        return router, router.get_model_list(), general_settings

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
            if "db_model" in model.model_info and model.model_info["db_model"] == False:
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
        if llm_router is None:
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
                        model["litellm_params"][k] = litellm.get_secret(v)

                ## check if they have model-id's ##
                model_id = model.get("model_info", {}).get("id", None)
                if model_id is None:
                    ## else - generate stable id's ##
                    model_id = llm_router._generate_model_id(
                        model_group=model["model_name"],
                        litellm_params=model["litellm_params"],
                    )
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

        if master_key is None or not isinstance(master_key, str):
            raise Exception(
                f"Master key is not initialized or formatted. master_key={master_key}"
            )

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
                        # decrypt value
                        _value = decrypt_value_helper(value=v)
                        # sanity check if string > size 0
                        if len(_value) > 0:
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

    async def _update_llm_router(
        self,
        new_models: list,
        proxy_logging_obj: ProxyLogging,
    ):
        global llm_router, llm_model_list, master_key, general_settings
        import base64

        try:
            if llm_router is None and master_key is not None:
                verbose_proxy_logger.debug(f"len new_models: {len(new_models)}")

                _model_list: list = []
                for m in new_models:
                    _litellm_params = m.litellm_params
                    if isinstance(_litellm_params, dict):
                        # decrypt values
                        for k, v in _litellm_params.items():
                            decrypted_value = decrypt_value_helper(value=v)
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
                if len(_model_list) > 0:
                    verbose_proxy_logger.debug(f"_model_list: {_model_list}")
                    llm_router = litellm.Router(
                        model_list=_model_list,
                        router_general_settings=RouterGeneralSettings(
                            async_only_mode=True  # only init async clients
                        ),
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
        litellm_settings = config_data.get("litellm_settings", {}) or {}
        success_callbacks = litellm_settings.get("success_callback", None)
        failure_callbacks = litellm_settings.get("failure_callback", None)

        if success_callbacks is not None and isinstance(success_callbacks, list):
            for success_callback in success_callbacks:
                if success_callback not in litellm.success_callback:
                    litellm.success_callback.append(success_callback)

        # Add failure callbacks from DB to litellm
        if failure_callbacks is not None and isinstance(failure_callbacks, list):
            for failure_callback in failure_callbacks:
                if failure_callback not in litellm.failure_callback:
                    litellm.failure_callback.append(failure_callback)
        # we need to set env variables too
        environment_variables = config_data.get("environment_variables", {})
        for k, v in environment_variables.items():
            try:
                decrypted_value = decrypt_value_helper(value=v)
                os.environ[k] = decrypted_value
            except Exception as e:
                verbose_proxy_logger.error(
                    "Error setting env variable: %s - %s", k, str(e)
                )

        # router settings
        if llm_router is not None and prisma_client is not None:
            db_router_settings = await prisma_client.db.litellm_config.find_first(
                where={"param_name": "router_settings"}
            )
            if (
                db_router_settings is not None
                and db_router_settings.param_value is not None
            ):
                _router_settings = db_router_settings.param_value
                llm_router.update_settings(**_router_settings)

        ## ALERTING ## [TODO] move this to the _update_general_settings() block
        _general_settings = config_data.get("general_settings", {})
        if "alerting" in _general_settings:
            if (
                general_settings is not None
                and general_settings.get("alerting", None) is not None
                and isinstance(general_settings["alerting"], list)
                and _general_settings.get("alerting", None) is not None
                and isinstance(_general_settings["alerting"], list)
            ):
                for alert in _general_settings["alerting"]:
                    if alert not in general_settings["alerting"]:
                        general_settings["alerting"].append(alert)
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

        if "alert_types" in _general_settings:
            general_settings["alert_types"] = _general_settings["alert_types"]
            proxy_logging_obj.alert_types = general_settings["alert_types"]
            proxy_logging_obj.slack_alerting_instance.update_values(
                alert_types=general_settings["alert_types"], llm_router=llm_router
            )

        if "alert_to_webhook_url" in _general_settings:
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
            if master_key is None or not isinstance(master_key, str):
                raise ValueError(
                    f"Master key is not initialized or formatted. master_key={master_key}"
                )
            try:
                new_models = await prisma_client.db.litellm_proxymodeltable.find_many()
            except Exception as e:
                verbose_proxy_logger.exception(
                    "litellm.proxy_server.py::add_deployment() - Error getting new models from DB - {}".format(
                        str(e)
                    )
                )
                new_models = []
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

        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.py::ProxyConfig:add_deployment - {}".format(
                    str(e)
                )
            )


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
    global user_model, user_api_base, user_debug, user_detailed_debug, user_user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers, experimental, llm_model_list, llm_router, general_settings, master_key, user_custom_auth, prisma_client
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
    if detailed_debug == True:
        import logging

        from litellm._logging import (
            verbose_logger,
            verbose_proxy_logger,
            verbose_router_logger,
        )

        verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
        verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
        verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug
    elif debug == False and detailed_debug == False:
        # users can control proxy debugging using env variable = 'LITELLM_LOG'
        litellm_log_setting = os.environ.get("LITELLM_LOG", "")
        if litellm_log_setting != None:
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
        user_max_tokens = max_tokens
        dynamic_config[user_model]["max_tokens"] = max_tokens
    if temperature:  # model-specific param
        user_temperature = temperature
        dynamic_config[user_model]["temperature"] = temperature
    if request_timeout:
        user_request_timeout = request_timeout
        dynamic_config[user_model]["request_timeout"] = request_timeout
    if alias:  # model-specific param
        dynamic_config[user_model]["alias"] = alias
    if drop_params == True:  # litellm-specific param
        litellm.drop_params = True
        dynamic_config["general"]["drop_params"] = True
    if add_function_to_prompt == True:  # litellm-specific param
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
        except:
            yield f"data: {json.dumps(chunk)}\n\n"


async def async_assistants_data_generator(
    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
):
    verbose_proxy_logger.debug("inside generator")
    try:
        start_time = time.time()
        async with response as chunk:

            ### CALL HOOKS ### - modify outgoing data
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict, response=chunk
            )

            # chunk = chunk.model_dump_json(exclude_none=True)
            async for c in chunk:
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
        router_model_names = llm_router.model_names if llm_router is not None else []
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
        start_time = time.time()
        async for chunk in response:
            verbose_proxy_logger.debug(
                "async_data_generator: received streaming chunk - {}".format(chunk)
            )
            ### CALL HOOKS ### - modify outgoing data
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict, response=chunk
            )

            if isinstance(chunk, BaseModel):
                chunk = chunk.model_dump_json(exclude_none=True, exclude_unset=True)

            try:
                yield f"data: {chunk}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        # Streaming is done, yield the [DONE] chunk
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
        router_model_names = llm_router.model_names if llm_router is not None else []

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


async def async_data_generator_anthropic(
    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
):
    verbose_proxy_logger.debug("inside generator")
    try:
        start_time = time.time()
        async for chunk in response:
            verbose_proxy_logger.debug(
                "async_data_generator: received streaming chunk - {}".format(chunk)
            )
            ### CALL HOOKS ### - modify outgoing data
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict, response=chunk
            )

            event_type = chunk.get("type")

            try:
                yield f"event: {event_type}\ndata:{json.dumps(chunk)}\n\n"
            except Exception as e:
                yield f"event: {event_type}\ndata:{str(e)}\n\n"
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
        router_model_names = llm_router.model_names if llm_router is not None else []

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
    except:
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
        verbose_proxy_logger.info(json.dumps({"event": "giveup", "exception": str(e)}))
    return result


@router.on_event("startup")
async def startup_event():
    global prisma_client, master_key, use_background_health_checks, llm_router, llm_model_list, general_settings, proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time, litellm_proxy_admin_name, db_writer_client, store_model_in_db, premium_user, _license_check
    import json

    init_verbose_loggers()

    ### LOAD MASTER KEY ###
    # check if master key set in environment - load from there
    master_key = litellm.get_secret("LITELLM_MASTER_KEY", None)
    # check if DATABASE_URL in environment - load from there
    if prisma_client is None:
        prisma_setup(database_url=litellm.get_secret("DATABASE_URL", None))

    ### LOAD CONFIG ###
    worker_config = litellm.get_secret("WORKER_CONFIG")
    verbose_proxy_logger.debug("worker_config: %s", worker_config)
    # check if it's a valid file path
    if os.path.isfile(worker_config):
        if proxy_config.is_yaml(config_file_path=worker_config):
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(
                router=llm_router, config_file_path=worker_config
            )
        else:
            await initialize(**worker_config)
    elif os.environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None:
        (
            llm_router,
            llm_model_list,
            general_settings,
        ) = await proxy_config.load_config(
            router=llm_router, config_file_path=worker_config
        )

    else:
        # if not, assume it's a json string
        worker_config = json.loads(os.getenv("WORKER_CONFIG"))
        await initialize(**worker_config)

    ## CHECK PREMIUM USER
    verbose_proxy_logger.debug(
        "litellm.proxy.proxy_server.py::startup() - CHECKING PREMIUM USER - {}".format(
            premium_user
        )
    )
    if premium_user is False:
        premium_user = _license_check.is_premium()

    verbose_proxy_logger.debug(
        "litellm.proxy.proxy_server.py::startup() - PREMIUM USER value - {}".format(
            premium_user
        )
    )

    ## COST TRACKING ##
    cost_tracking()

    ## Error Tracking ##
    error_tracking()

    ## UPDATE SLACK ALERTING ##
    proxy_logging_obj.slack_alerting_instance.update_values(llm_router=llm_router)

    db_writer_client = HTTPHandler()

    ## UPDATE INTERNAL USAGE CACHE ##
    proxy_logging_obj.update_values(
        redis_cache=redis_usage_cache
    )  # used by parallel request limiter for rate limiting keys across instances

    proxy_logging_obj._init_litellm_callbacks(
        llm_router=llm_router
    )  # INITIALIZE LITELLM CALLBACKS ON SERVER STARTUP <- do this to catch any logging errors on startup, not when calls are being made

    if "daily_reports" in proxy_logging_obj.slack_alerting_instance.alert_types:
        asyncio.create_task(
            proxy_logging_obj.slack_alerting_instance._run_scheduled_daily_report(
                llm_router=llm_router
            )
        )  # RUN DAILY REPORT (if scheduled)

    ## JWT AUTH ##
    if general_settings.get("litellm_jwtauth", None) is not None:
        for k, v in general_settings["litellm_jwtauth"].items():
            if isinstance(v, str) and v.startswith("os.environ/"):
                general_settings["litellm_jwtauth"][k] = litellm.get_secret(v)
        litellm_jwtauth = LiteLLM_JWTAuth(**general_settings["litellm_jwtauth"])
    else:
        litellm_jwtauth = LiteLLM_JWTAuth()
    jwt_handler.update_environment(
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        litellm_jwtauth=litellm_jwtauth,
    )

    if use_background_health_checks:
        asyncio.create_task(
            _run_background_health_check()
        )  # start the background health check coroutine.

    if prompt_injection_detection_obj is not None:
        prompt_injection_detection_obj.update_environment(router=llm_router)

    verbose_proxy_logger.debug("prisma_client: %s", prisma_client)
    if prisma_client is not None:
        await prisma_client.connect()

    verbose_proxy_logger.debug("custom_db_client client - %s", custom_db_client)
    if custom_db_client is not None:
        verbose_proxy_logger.debug("custom_db_client: connecting %s", custom_db_client)
        await custom_db_client.connect()

    if prisma_client is not None and master_key is not None:
        if os.getenv("PROXY_ADMIN_ID", None) is not None:
            litellm_proxy_admin_name = os.getenv(
                "PROXY_ADMIN_ID", litellm_proxy_admin_name
            )
        if general_settings.get("disable_adding_master_key_hash_to_db") is True:
            verbose_proxy_logger.info("Skipping writing master key hash to db")
        else:
            # add master key to db
            asyncio.create_task(
                generate_key_helper_fn(
                    request_type="user",
                    duration=None,
                    models=[],
                    aliases={},
                    config={},
                    spend=0,
                    token=master_key,
                    user_id=litellm_proxy_admin_name,
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    query_type="update_data",
                    update_key_values={"user_role": LitellmUserRoles.PROXY_ADMIN},
                )
            )

    if prisma_client is not None and litellm.max_budget > 0:
        if litellm.budget_duration is None:
            raise Exception(
                "budget_duration not set on Proxy. budget_duration is required to use max_budget."
            )

        # add proxy budget to db in the user table
        asyncio.create_task(
            generate_key_helper_fn(
                request_type="user",
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

    if custom_db_client is not None and master_key is not None:
        # add master key to db
        await generate_key_helper_fn(
            request_type="key",
            duration=None,
            models=[],
            aliases={},
            config={},
            spend=0,
            token=master_key,
        )

    ### CHECK IF VIEW EXISTS ###
    if prisma_client is not None:
        create_view_response = await prisma_client.check_view_exists()
        # Apply misc fixes on DB
        # [non-blocking] helper to apply fixes from older litellm versions
        asyncio.create_task(prisma_client.apply_db_fixes())

    ### START BATCH WRITING DB + CHECKING NEW MODELS###
    if prisma_client is not None:
        scheduler = AsyncIOScheduler()
        interval = random.randint(
            proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time
        )  # random interval, so multiple workers avoid resetting budget at the same time
        batch_writing_interval = random.randint(
            proxy_batch_write_at - 3, proxy_batch_write_at + 3
        )  # random interval, so multiple workers avoid batch writing at the same time

        ### RESET BUDGET ###
        if general_settings.get("disable_reset_budget", False) == False:
            scheduler.add_job(
                reset_budget, "interval", seconds=interval, args=[prisma_client]
            )

        ### UPDATE SPEND ###
        scheduler.add_job(
            update_spend,
            "interval",
            seconds=batch_writing_interval,
            args=[prisma_client, db_writer_client, proxy_logging_obj],
        )

        ### ADD NEW MODELS ###
        store_model_in_db = (
            litellm.get_secret("STORE_MODEL_IN_DB", store_model_in_db)
            or store_model_in_db
        )  # type: ignore
        if store_model_in_db == True:
            scheduler.add_job(
                proxy_config.add_deployment,
                "interval",
                seconds=10,
                args=[prisma_client, proxy_logging_obj],
            )

            # this will load all existing models on proxy startup
            await proxy_config.add_deployment(
                prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
            )

        if (
            proxy_logging_obj is not None
            and proxy_logging_obj.slack_alerting_instance is not None
            and prisma_client is not None
        ):
            print("Alerting: Initializing Weekly/Monthly Spend Reports")  # noqa
            ### Schedule weekly/monhtly spend reports ###
            scheduler.add_job(
                proxy_logging_obj.slack_alerting_instance.send_weekly_spend_report,
                "cron",
                day_of_week="mon",
            )

            scheduler.add_job(
                proxy_logging_obj.slack_alerting_instance.send_monthly_spend_report,
                "cron",
                day=1,
            )

            # Beta Feature - only used when prometheus api is in .env
            if os.getenv("PROMETHEUS_URL"):
                from zoneinfo import ZoneInfo

                scheduler.add_job(
                    proxy_logging_obj.slack_alerting_instance.send_fallback_stats_from_prometheus,
                    "cron",
                    hour=9,
                    minute=0,
                    timezone=ZoneInfo("America/Los_Angeles"),  # Pacific Time
                )
                await proxy_logging_obj.slack_alerting_instance.send_fallback_stats_from_prometheus()

        scheduler.start()


#### API ENDPOINTS ####
@router.get(
    "/v1/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)
@router.get(
    "/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)  # if project requires model list
def model_list(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use `/model/info` - to get detailed model information, example - pricing, mode, etc.

    This is just for compatibility with openai projects like aider.
    """
    global llm_model_list, general_settings
    all_models = []
    ## CHECK IF MODEL RESTRICTIONS ARE SET AT KEY/TEAM LEVEL ##
    if llm_model_list is None:
        proxy_model_list = []
    else:
        proxy_model_list = [m["model_name"] for m in llm_model_list]
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict, proxy_model_list=proxy_model_list
    )
    team_models = get_team_models(
        user_api_key_dict=user_api_key_dict, proxy_model_list=proxy_model_list
    )
    all_models = get_complete_model_list(
        key_models=key_models,
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        user_model=user_model,
        infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
    )
    return dict(
        data=[
            {
                "id": model,
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
            }
            for model in all_models
        ],
        object="list",
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
)  # azure compatible endpoint
@backoff.on_exception(
    backoff.expo,
    Exception,  # base exception to catch for the backoff
    max_tries=global_max_parallel_request_retries,  # maximum number of retries
    max_time=global_max_parallel_request_retry_timeout,  # maximum total time to retry for
    on_backoff=on_backoff,  # specifying the function to call on backoff
    giveup=giveup,
    logger=verbose_proxy_logger,
)
async def chat_completion(
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

    data = {}
    try:
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except:
            data = json.loads(body_str)

        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
        )

        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or data["model"]  # default passed in http request
        )

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

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if isinstance(data["model"], str) and data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify/reject incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="completion"
        )

        ## LOGGING OBJECT ## - initialize logging object for logging success/failure events for call
        ## IMPORTANT Note: - initialize this before running pre-call checks. Ensures we log rejected requests to langfuse.
        data["litellm_call_id"] = request.headers.get(
            "x-litellm-call-id", str(uuid.uuid4())
        )
        logging_obj, data = litellm.utils.function_setup(
            original_function="acompletion",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.now(),
            **data,
        )

        data["litellm_logging_obj"] = logging_obj

        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type="completion",
            )
        )

        ### ROUTE THE REQUEST ###
        # Do not change this - it should be a constant time fetch - ALWAYS
        llm_call = await route_request(
            data=data,
            route_type="acompletion",
            llm_router=llm_router,
            user_model=user_model,
        )
        tasks.append(llm_call)

        # wait for call to end
        llm_responses = asyncio.gather(
            *tasks
        )  # run the moderation check in parallel to the actual llm api call

        responses = await llm_responses

        response = responses[1]

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        fastest_response_batch_completion = hidden_params.get(
            "fastest_response_batch_completion", None
        )
        additional_headers: dict = hidden_params.get("additional_headers", {}) or {}

        # Post Call Processing
        if llm_router is not None:
            data["deployment"] = llm_router.get_deployment(model_id=model_id)
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )
        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
                request_data=data,
                **additional_headers,
            )
            selected_data_generator = select_data_generator(
                response=response,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
            )
            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                headers=custom_headers,
            )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )

        hidden_params = (
            getattr(response, "_hidden_params", {}) or {}
        )  # get any updated response headers
        additional_headers = hidden_params.get("additional_headers", {}) or {}

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
                request_data=data,
                **additional_headers,
            )
        )
        await check_response_size_is_safe(response=response)

        return response
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        _chat_response = litellm.ModelResponse()
        _chat_response.choices[0].message.content = e.message  # type: ignore

        if data.get("stream", None) is not None and data["stream"] == True:
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
            )
        _usage = litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        _chat_response.usage = _usage  # type: ignore
        return _chat_response
    except Exception as e:
        verbose_proxy_logger.exception(
            f"litellm.proxy.proxy_server.chat_completion(): Exception occured - {str(e)}"
        )
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        litellm_debug_info = getattr(e, "litellm_debug_info", "")
        verbose_proxy_logger.debug(
            "\033[1;31mAn error occurred: %s %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
            litellm_debug_info,
        )
        router_model_names = llm_router.model_names if llm_router is not None else []

        if isinstance(e, HTTPException):
            # print("e.headers={}".format(e.headers))
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
                headers=getattr(e, "headers", {}),
            )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
            headers=getattr(e, "headers", {}),
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
async def completion(
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
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except:
            data = json.loads(body_str)

        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="text_completion"
        )

        ### ROUTE THE REQUESTs ###
        llm_call = await route_request(
            data=data,
            route_type="atext_completion",
            llm_router=llm_router,
            user_model=user_model,
        )

        # Await the llm_response task
        response = await llm_call

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        litellm_call_id = hidden_params.get("litellm_call_id", None) or ""

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        verbose_proxy_logger.debug("final response: %s", response)
        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
            )
            selected_data_generator = select_data_generator(
                response=response,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                headers=custom_headers,
            )
        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
            )
        )
        await check_response_size_is_safe(response=response)
        return response
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        if _data.get("stream", None) is not None and _data["stream"] == True:
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
            )
        else:
            _response = litellm.TextCompletionResponse()
            _response.choices[0].text = e.message
            return _response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.completion(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
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
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
        data = orjson.loads(body)

        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n%s",
            json.dumps(data, indent=4),
        )

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
            general_settings.get("embedding_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        router_model_names = llm_router.model_names if llm_router is not None else []
        if (
            "input" in data
            and isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
            and isinstance(data["input"][0][0], int)
        ):  # check if array of tokens passed in
            # check if non-openai/azure model called - e.g. for langchain integration
            if llm_model_list is not None and data["model"] in router_model_names:
                for m in llm_model_list:
                    if m["model_name"] == data["model"] and (
                        m["litellm_params"]["model"] in litellm.open_ai_embedding_models
                        or m["litellm_params"]["model"].startswith("azure/")
                    ):
                        pass
                    else:
                        # non-openai/azure embedding model called with token input
                        input_list = []
                        for i in data["input"]:
                            input_list.append(
                                litellm.decode(model="gpt-3.5-turbo", tokens=i)
                            )
                        data["input"] = input_list
                        break

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="embeddings"
        )

        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type="embeddings",
            )
        )

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call = await route_request(
            data=data,
            route_type="aembedding",
            llm_router=llm_router,
            user_model=user_model,
        )
        tasks.append(llm_call)

        # wait for call to end
        llm_responses = asyncio.gather(
            *tasks
        )  # run the moderation check in parallel to the actual llm api call

        responses = await llm_responses

        response = responses[1]

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

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
            )
        )
        await check_response_size_is_safe(response=response)

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        litellm_debug_info = getattr(e, "litellm_debug_info", "")
        verbose_proxy_logger.debug(
            "\033[1;31mAn error occurred: %s %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
            litellm_debug_info,
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.embeddings(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            message = get_error_message_str(e)
            raise ProxyException(
                message=message,
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
    "/v1/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
async def image_generation(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global proxy_logging_obj
    data = {}
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
            general_settings.get("image_generation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        router_model_names = llm_router.model_names if llm_router is not None else []

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call = await route_request(
            data=data,
            route_type="aimage_generation",
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

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.image_generation(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
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

        router_model_names = llm_router.model_names if llm_router is not None else []

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

        custom_headers = get_custom_headers(
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
        )

        selected_data_generator = select_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=data,
        )
        return StreamingResponse(
            generate(response), media_type="audio/mpeg", headers=custom_headers
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
        form_data = await request.form()
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
            or data["model"]  # default passed in http request
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
                call_type="audio_transcription",
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

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.audio_transcription(): Exception occured - {}".format(
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
                code=getattr(e, "status_code", 500),
            )


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
        body = await request.body()

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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
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
        body = await request.body()

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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
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
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            return StreamingResponse(
                async_assistants_data_generator(
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                    request_data=data,
                ),
                media_type="text/event-stream",
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
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
                code=getattr(e, "status_code", 500),
            )


######################################################################

#                          /v1/batches Endpoints


######################################################################
@router.post(
    "/{provider}/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.post(
    "/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.post(
    "/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def create_batch(
    request: Request,
    fastapi_response: Response,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create large batches of API requests for asynchronous processing.
    This is the equivalent of POST https://api.openai.com/v1/batch
    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch

    Example Curl
    ```
    curl http://localhost:4000/v1/batches \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "input_file_id": "file-abc123",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
    }'
    ```
    """
    global proxy_logging_obj
    data: Dict = {}

    try:
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except:
            data = json.loads(body_str)

        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
        )

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        _create_batch_data = CreateBatchRequest(**data)

        if provider is None:
            provider = "openai"
        response = await litellm.acreate_batch(
            custom_llm_provider=provider, **_create_batch_data  # type: ignore
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.create_batch(): Exception occured - {}".format(
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
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/batches/{batch_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/v1/batches/{batch_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/batches/{batch_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def retrieve_batch(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    provider: Optional[str] = None,
    batch_id: str = Path(
        title="Batch ID to retrieve", description="The ID of the batch to retrieve"
    ),
):
    """
    Retrieves a batch.
    This is the equivalent of GET https://api.openai.com/v1/batches/{batch_id}
    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch/retrieve

    Example Curl
    ```
    curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \

    ```
    """
    global proxy_logging_obj
    data: Dict = {}
    try:
        _retrieve_batch_request = RetrieveBatchRequest(
            batch_id=batch_id,
        )

        if provider is None:
            provider = "openai"
        response = await litellm.aretrieve_batch(
            custom_llm_provider=provider, **_retrieve_batch_request  # type: ignore
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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.retrieve_batch(): Exception occured - {}".format(
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
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/v1/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/batches",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def list_batches(
    fastapi_response: Response,
    provider: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Lists 
    This is the equivalent of GET https://api.openai.com/v1/batches/
    Supports Identical Params as: https://platform.openai.com/docs/api-reference/batch/list

    Example Curl
    ```
    curl http://localhost:4000/v1/batches?limit=2 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \

    ```
    """
    global proxy_logging_obj
    verbose_proxy_logger.debug("GET /v1/batches after={} limit={}".format(after, limit))
    try:
        if provider is None:
            provider = "openai"
        response = await litellm.alist_batches(
            custom_llm_provider=provider,  # type: ignore
            after=after,
            limit=limit,
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data={"after": after, "limit": limit},
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.retrieve_batch(): Exception occured - {}".format(
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
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


######################################################################

#            END OF  /v1/batches Endpoints Implementation

######################################################################


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

        router_model_names = llm_router.model_names if llm_router is not None else []

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="moderation"
        )

        start_time = time.time()

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
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                request_data=data,
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.moderations(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
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


#### ANTHROPIC ENDPOINTS ####


@router.post(
    "/v1/messages",
    tags=["[beta] Anthropic `/v1/messages`"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AnthropicResponse,
)
async def anthropic_response(
    anthropic_data: AnthropicMessagesRequest,
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm import adapter_completion
    from litellm.adapters.anthropic_adapter import anthropic_adapter

    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
    body = await request.body()
    body_str = body.decode()
    try:
        request_data: dict = ast.literal_eval(body_str)
    except Exception:
        request_data = json.loads(body_str)
    data: dict = {**request_data, "adapter_id": "anthropic"}
    try:
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        data = await add_litellm_data_to_request(
            data=data,  # type: ignore
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="text_completion"
        )

        ### ROUTE THE REQUESTs ###
        router_model_names = llm_router.model_names if llm_router is not None else []
        # skip router if user passed their key
        if "api_key" in data:
            llm_response = asyncio.create_task(litellm.aadapter_completion(**data))
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            llm_response = asyncio.create_task(
                llm_router.aadapter_completion(**data, specific_deployment=True)
            )
        elif (
            llm_router is not None and data["model"] in llm_router.get_model_ids()
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and (
                llm_router.default_deployment is not None
                or len(llm_router.provider_default_deployments) > 0
            )
        ):  # model in router deployments, calling a specific deployment on the router
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif user_model is not None:  # `litellm --model <your-model-name>`
            llm_response = asyncio.create_task(litellm.aadapter_completion(**data))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "completion: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        # Await the llm_response task
        response = await llm_response

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        verbose_proxy_logger.debug("final response: %s", response)

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
            )
        )

        if (
            "stream" in data and data["stream"] is True
        ):  # use generate_responses to stream responses
            selected_data_generator = async_data_generator_anthropic(
                response=response,
                user_api_key_dict=user_api_key_dict,
                request_data=data,
            )
            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
            )

        verbose_proxy_logger.info("\nResponse from Litellm:\n{}".format(response))
        return response
    except RejectedRequestError as e:
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )
        if _data.get("stream", None) is not None and _data["stream"] == True:
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
            "litellm.proxy.proxy_server.anthropic_response(): Exception occured - {}".format(
                str(e)
            )
        )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


#### DEV UTILS ####

# @router.get(
#     "/utils/available_routes",
#     tags=["llm utils"],
#     dependencies=[Depends(user_api_key_auth)],
# )
# async def get_available_routes(user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth)):


@router.post(
    "/utils/token_counter",
    tags=["llm utils"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountResponse,
)
async def token_counter(request: TokenCountRequest):
    """ """
    from litellm import token_counter

    global llm_router

    prompt = request.prompt
    messages = request.messages
    if prompt is None and messages is None:
        raise HTTPException(
            status_code=400, detail="prompt or messages must be provided"
        )

    deployment = None
    litellm_model_name = None
    if llm_router is not None:
        # get 1 deployment corresponding to the model
        for _model in llm_router.model_list:
            if _model["model_name"] == request.model:
                deployment = _model
                break
    if deployment is not None:
        litellm_model_name = deployment.get("litellm_params", {}).get("model")
        # remove the custom_llm_provider_prefix in the litellm_model_name
        if "/" in litellm_model_name:
            litellm_model_name = litellm_model_name.split("/", 1)[1]

    model_to_use = (
        litellm_model_name or request.model
    )  # use litellm model name, if it's not avalable then fallback to request.model
    _tokenizer_used = litellm.utils._select_tokenizer(model=model_to_use)
    tokenizer_used = str(_tokenizer_used["type"])
    total_tokens = token_counter(
        model=model_to_use,
        text=prompt,
        messages=messages,
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
    except Exception as e:
        raise HTTPException(
            status_code=400, detail={"error": "Could not map model={}".format(model)}
        )


#### END-USER MANAGEMENT ####


@router.post(
    "/end_user/block",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
@router.post(
    "/customer/block",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def block_user(data: BlockUsers):
    """
    [BETA] Reject calls with this end-user id

        (any /chat/completion call with this user={end-user-id} param, will be rejected.)

        ```
        curl -X POST "http://0.0.0.0:8000/user/block"
        -H "Authorization: Bearer sk-1234"
        -D '{
        "user_ids": [<user_id>, ...]
        }'
        ```
    """
    try:
        records = []
        if prisma_client is not None:
            for id in data.user_ids:
                record = await prisma_client.db.litellm_endusertable.upsert(
                    where={"user_id": id},  # type: ignore
                    data={
                        "create": {"user_id": id, "blocked": True},  # type: ignore
                        "update": {"blocked": True},
                    },
                )
                records.append(record)
        else:
            raise HTTPException(
                status_code=500,
                detail={"error": "Postgres DB Not connected"},
            )

        return {"blocked_users": records}
    except Exception as e:
        verbose_proxy_logger.error(f"An error occurred - {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/end_user/unblock",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
@router.post(
    "/customer/unblock",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def unblock_user(data: BlockUsers):
    """
    [BETA] Unblock calls with this user id

    Example
    ```
    curl -X POST "http://0.0.0.0:8000/user/unblock"
    -H "Authorization: Bearer sk-1234"
    -D '{
    "user_ids": [<user_id>, ...]
    }'
    ```
    """
    from enterprise.enterprise_hooks.blocked_user_list import (
        _ENTERPRISE_BlockedUserList,
    )

    if (
        not any(isinstance(x, _ENTERPRISE_BlockedUserList) for x in litellm.callbacks)
        or litellm.blocked_user_list is None
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Blocked user check was never set. This call has no effect."
            },
        )

    if isinstance(litellm.blocked_user_list, list):
        for id in data.user_ids:
            litellm.blocked_user_list.remove(id)
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "`blocked_user_list` must be set as a list. Filepaths can't be updated."
            },
        )

    return {"blocked_users": litellm.blocked_user_list}


@router.post(
    "/end_user/new",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/customer/new",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_end_user(
    data: NewCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Allow creating a new Customer 


    Parameters:
    - user_id: str - The unique identifier for the user.
    - alias: Optional[str] - A human-friendly alias for the user.
    - blocked: bool - Flag to allow or disallow requests for this end-user. Default is False.
    - max_budget: Optional[float] - The maximum budget allocated to the user. Either 'max_budget' or 'budget_id' should be provided, not both.
    - budget_id: Optional[str] - The identifier for an existing budget allocated to the user. Either 'max_budget' or 'budget_id' should be provided, not both.
    - allowed_model_region: Optional[Literal["eu"]] - Require all user requests to use models in this specific region.
    - default_model: Optional[str] - If no equivalent model in the allowed region, default all requests to this model.
    - metadata: Optional[dict] = Metadata for customer, store information for customer. Example metadata = {"data_training_opt_out": True}
    
    
    - Allow specifying allowed regions 
    - Allow specifying default model

    Example curl:
    ```
    curl --location 'http://0.0.0.0:4000/customer/new' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "user_id" : "ishaan-jaff-3",
            "allowed_region": "eu",
            "budget_id": "free_tier",
            "default_model": "azure/gpt-3.5-turbo-eu" <- all calls from this user, use this model? 
        }'

        # return end-user object
    ```

    NOTE: This used to be called `/end_user/new`, we will still be maintaining compatibility for /end_user/XXX for these endpoints
    """
    global prisma_client, llm_router
    """
    Validation:
        - check if default model exists 
        - create budget object if not already created
    
    - Add user to end user table 

    Return 
    - end-user object
    - currently allowed models 
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    try:

        ## VALIDATION ##
        if data.default_model is not None:
            if llm_router is None:
                raise HTTPException(
                    status_code=422,
                    detail={"error": CommonProxyErrors.no_llm_router.value},
                )
            elif data.default_model not in llm_router.get_model_names():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Default Model not on proxy. Configure via `/model/new` or config.yaml. Default_model={}, proxy_model_names={}".format(
                            data.default_model, set(llm_router.get_model_names())
                        )
                    },
                )

        new_end_user_obj: Dict = {}

        ## CREATE BUDGET ## if set
        if data.max_budget is not None:
            budget_record = await prisma_client.db.litellm_budgettable.create(
                data={
                    "max_budget": data.max_budget,
                    "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,  # type: ignore
                    "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                }
            )

            new_end_user_obj["budget_id"] = budget_record.budget_id
        elif data.budget_id is not None:
            new_end_user_obj["budget_id"] = data.budget_id

        _user_data = data.dict(exclude_none=True)

        for k, v in _user_data.items():
            if k != "max_budget" and k != "budget_id":
                new_end_user_obj[k] = v

        ## WRITE TO DB ##
        end_user_record = await prisma_client.db.litellm_endusertable.create(
            data=new_end_user_obj  # type: ignore
        )

        return end_user_record
    except Exception as e:
        if "Unique constraint failed on the fields: (`user_id`)" in str(e):
            raise ProxyException(
                message=f"Customer already exists, passed user_id={data.user_id}. Please pass a new user_id.",
                type="bad_request",
                code=400,
                param="user_id",
            )

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/customer/info",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_EndUserTable,
)
@router.get(
    "/end_user/info",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def end_user_info(
    end_user_id: str = fastapi.Query(
        description="End User ID in the request parameters"
    ),
):
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    user_info = await prisma_client.db.litellm_endusertable.find_first(
        where={"user_id": end_user_id}
    )

    if user_info is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "End User Id={} does not exist in db".format(end_user_id)},
        )
    return user_info.model_dump(exclude_none=True)


@router.post(
    "/customer/update",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/end_user/update",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def update_end_user(
    data: UpdateCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Example curl 

    ```
    curl --location 'http://0.0.0.0:4000/customer/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "test-litellm-user-4",
        "budget_id": "paid_tier"
    }'

    See below for all params 
    ```
    """

    global prisma_client
    try:
        data_json: dict = data.json()
        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        # get non default values for key
        non_default_values = {}
        for k, v in data_json.items():
            if v is not None and v not in (
                [],
                {},
                0,
            ):  # models default to [], spend defaults to 0, we should not reset these values
                non_default_values[k] = v

        ## ADD USER, IF NEW ##
        verbose_proxy_logger.debug("/customer/update: Received data = %s", data)
        if data.user_id is not None and len(data.user_id) > 0:
            non_default_values["user_id"] = data.user_id  # type: ignore
            verbose_proxy_logger.debug("In update customer, user_id condition block.")
            response = await prisma_client.db.litellm_endusertable.update(
                where={"user_id": data.user_id}, data=non_default_values  # type: ignore
            )
            if response is None:
                raise ValueError(
                    f"Failed updating customer data. User ID does not exist passed user_id={data.user_id}"
                )
            verbose_proxy_logger.debug(
                f"received response from updating prisma client. response={response}"
            )
            return response
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_id}")

        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.update_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    pass


@router.post(
    "/customer/delete",
    tags=["Customer Management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/end_user/delete",
    tags=["Customer Management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_end_user(
    data: DeleteCustomerRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Example curl 

    ```
    curl --location 'http://0.0.0.0:4000/customer/delete' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "user_ids" :["ishaan-jaff-5"]
    }'

    See below for all params 
    ```
    """
    global prisma_client

    try:
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        verbose_proxy_logger.debug("/customer/delete: Received data = %s", data)
        if (
            data.user_ids is not None
            and isinstance(data.user_ids, list)
            and len(data.user_ids) > 0
        ):
            response = await prisma_client.db.litellm_endusertable.delete_many(
                where={"user_id": {"in": data.user_ids}}
            )
            if response is None:
                raise ValueError(
                    f"Failed deleting customer data. User ID does not exist passed user_id={data.user_ids}"
                )
            if response != len(data.user_ids):
                raise ValueError(
                    f"Failed deleting all customer data. User ID does not exist passed user_id={data.user_ids}. Deleted {response} customers, passed {len(data.user_ids)} customers"
                )
            verbose_proxy_logger.debug(
                f"received response from updating prisma client. response={response}"
            )
            return {
                "deleted_customers": response,
                "message": "Successfully deleted customers with ids: "
                + str(data.user_ids),
            }
        else:
            raise ValueError(f"user_id is required, passed user_id = {data.user_ids}")

        # update based on remaining passed in values
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.delete_end_user(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Internal Server Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Internal Server Error, " + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    pass


async def create_audit_log_for_update(request_data: LiteLLM_AuditLogs):
    if premium_user is not True:
        return

    if litellm.store_audit_logs is not True:
        return
    if prisma_client is None:
        raise Exception("prisma_client is None, no DB connected")

    verbose_proxy_logger.debug("creating audit log for %s", request_data)

    if isinstance(request_data.updated_values, dict):
        request_data.updated_values = json.dumps(request_data.updated_values)

    if isinstance(request_data.before_value, dict):
        request_data.before_value = json.dumps(request_data.before_value)

    _request_data = request_data.dict(exclude_none=True)

    try:
        await prisma_client.db.litellm_auditlog.create(
            data={
                **_request_data,  # type: ignore
            }
        )
    except Exception as e:
        # [Non-Blocking Exception. Do not allow blocking LLM API call]
        verbose_proxy_logger.error(f"Failed Creating audit log {e}")

    return


#### ORGANIZATION MANAGEMENT ####


@router.post(
    "/organization/new",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewOrganizationResponse,
)
async def new_organization(
    data: NewOrganizationRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Allow orgs to own teams

    Set org level budgets + model access.

    Only admins can create orgs.

    # Parameters

    - `organization_alias`: *str* = The name of the organization.
    - `models`: *List* = The models the organization has access to.
    - `budget_id`: *Optional[str]* = The id for a budget (tpm/rpm/max budget) for the organization.
    ### IF NO BUDGET ID - CREATE ONE WITH THESE PARAMS ###
    - `max_budget`: *Optional[float]* = Max budget for org
    - `tpm_limit`: *Optional[int]* = Max tpm limit for org
    - `rpm_limit`: *Optional[int]* = Max rpm limit for org
    - `model_max_budget`: *Optional[dict]* = Max budget for a specific model
    - `budget_duration`: *Optional[str]* = Frequency of reseting org budget

    Case 1: Create new org **without** a budget_id

    ```bash
    curl --location 'http://0.0.0.0:4000/organization/new' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data '{
        "organization_alias": "my-secret-org",
        "models": ["model1", "model2"],
        "max_budget": 100
    }'


    ```

    Case 2: Create new org **with** a budget_id

    ```bash
    curl --location 'http://0.0.0.0:4000/organization/new' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data '{
        "organization_alias": "my-secret-org",
        "models": ["model1", "model2"],
        "budget_id": "428eeaa8-f3ac-4e85-a8fb-7dc8d7aa8689"
    }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if (
        user_api_key_dict.user_role is None
        or user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"Only admins can create orgs. Your role is = {user_api_key_dict.user_role}"
            },
        )

    if data.budget_id is None:
        """
        Every organization needs a budget attached.

        If none provided, create one based on provided values
        """
        budget_params = LiteLLM_BudgetTable.model_fields.keys()

        # Only include Budget Params when creating an entry in litellm_budgettable
        _json_data = data.json(exclude_none=True)
        _budget_data = {k: v for k, v in _json_data.items() if k in budget_params}
        budget_row = LiteLLM_BudgetTable(**_budget_data)

        new_budget = prisma_client.jsonify_object(budget_row.json(exclude_none=True))

        _budget = await prisma_client.db.litellm_budgettable.create(
            data={
                **new_budget,  # type: ignore
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )  # type: ignore

        data.budget_id = _budget.budget_id

    """
    Ensure only models that user has access to, are given to org
    """
    if len(user_api_key_dict.models) == 0:  # user has access to all models
        pass
    else:
        if len(data.models) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"User not allowed to give access to all models. Select models you want org to have access to."
                },
            )
        for m in data.models:
            if m not in user_api_key_dict.models:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"User not allowed to give access to model={m}. Models you have access to = {user_api_key_dict.models}"
                    },
                )
    organization_row = LiteLLM_OrganizationTable(
        **data.json(exclude_none=True),
        created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
        updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
    )
    new_organization_row = prisma_client.jsonify_object(
        organization_row.json(exclude_none=True)
    )
    response = await prisma_client.db.litellm_organizationtable.create(
        data={
            **new_organization_row,  # type: ignore
        }
    )

    return response


@router.post(
    "/organization/update",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_organization():
    """[TODO] Not Implemented yet. Let us know if you need this - https://github.com/BerriAI/litellm/issues"""
    pass


@router.post(
    "/organization/delete",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_organization():
    """[TODO] Not Implemented yet. Let us know if you need this - https://github.com/BerriAI/litellm/issues"""
    pass


@router.post(
    "/organization/info",
    tags=["organization management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_organization(data: OrganizationRequest):
    """
    Get the org specific information
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if len(data.organizations) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Specify list of organization id's to query. Passed in={data.organizations}"
            },
        )
    response = await prisma_client.db.litellm_organizationtable.find_many(
        where={"organization_id": {"in": data.organizations}},
        include={"litellm_budget_table": True},
    )

    return response


#### BUDGET TABLE MANAGEMENT ####


@router.post(
    "/budget/new",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_budget(
    budget_obj: BudgetNew,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new budget object. Can apply this to teams, orgs, end-users, keys.
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    response = await prisma_client.db.litellm_budgettable.create(
        data={
            **budget_obj.model_dump(exclude_none=True),  # type: ignore
            "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        }  # type: ignore
    )

    return response


@router.post(
    "/budget/info",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_budget(data: BudgetRequest):
    """
    Get the budget id specific information
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if len(data.budgets) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Specify list of budget id's to query. Passed in={data.budgets}"
            },
        )
    response = await prisma_client.db.litellm_budgettable.find_many(
        where={"budget_id": {"in": data.budgets}},
    )

    return response


@router.get(
    "/budget/settings",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def budget_settings(
    budget_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get list of configurable params + current value for a budget item + description of each field

    Used on Admin UI.
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

    ## get budget item from db
    db_budget_row = await prisma_client.db.litellm_budgettable.find_first(
        where={"budget_id": budget_id}
    )

    if db_budget_row is not None:
        db_budget_row_dict = db_budget_row.model_dump(exclude_none=True)
    else:
        db_budget_row_dict = {}

    allowed_args = {
        "max_parallel_requests": {"type": "Integer"},
        "tpm_limit": {"type": "Integer"},
        "rpm_limit": {"type": "Integer"},
        "budget_duration": {"type": "String"},
        "max_budget": {"type": "Float"},
        "soft_budget": {"type": "Float"},
    }

    return_val = []

    for field_name, field_info in BudgetNew.model_fields.items():
        if field_name in allowed_args:

            _stored_in_db = True

            _response_obj = ConfigList(
                field_name=field_name,
                field_type=allowed_args[field_name]["type"],
                field_description=field_info.description or "",
                field_value=db_budget_row_dict.get(field_name, None),
                stored_in_db=_stored_in_db,
                field_default_value=field_info.default,
            )
            return_val.append(_response_obj)

    return return_val


@router.get(
    "/budget/list",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_budget(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all the created budgets in proxy db. Used on Admin UI."""
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

    response = await prisma_client.db.litellm_budgettable.find_many()

    return response


@router.post(
    "/budget/delete",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_budget(
    data: BudgetDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Delete budget"""
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
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

    response = await prisma_client.db.litellm_budgettable.delete(
        where={"budget_id": data.id}
    )

    return response


#### MODEL MANAGEMENT ####


#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post(
    "/model/new",
    description="Allows adding new models to the model list in the config.yaml",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def add_new_model(
    model_params: Deployment,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global llm_router, llm_model_list, general_settings, user_config_file_path, proxy_config, prisma_client, master_key, store_model_in_db, proxy_logging_obj
    try:
        import base64

        global prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        model_response = None
        # update DB
        if store_model_in_db == True:
            """
            - store model_list in db
            - store keys separately
            """
            # encrypt litellm params #
            _litellm_params_dict = model_params.litellm_params.dict(exclude_none=True)
            _orignal_litellm_model_name = model_params.litellm_params.model
            for k, v in _litellm_params_dict.items():
                encrypted_value = encrypt_value_helper(value=v)
                model_params.litellm_params[k] = encrypted_value
            _data: dict = {
                "model_id": model_params.model_info.id,
                "model_name": model_params.model_name,
                "litellm_params": model_params.litellm_params.model_dump_json(exclude_none=True),  # type: ignore
                "model_info": model_params.model_info.model_dump_json(  # type: ignore
                    exclude_none=True
                ),
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
            if model_params.model_info.id is not None:
                _data["model_id"] = model_params.model_info.id
            model_response = await prisma_client.db.litellm_proxymodeltable.create(
                data=_data  # type: ignore
            )

            await proxy_config.add_deployment(
                prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
            )
            try:
                # don't let failed slack alert block the /model/new response
                _alerting = general_settings.get("alerting", []) or []
                if "slack" in _alerting:
                    # send notification - new model added
                    await proxy_logging_obj.slack_alerting_instance.model_added_alert(
                        model_name=model_params.model_name,
                        litellm_model_name=_orignal_litellm_model_name,
                        passed_model_info=model_params.model_info,
                    )
            except:
                pass

        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

        return model_response

    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.add_new_model(): Exception occured - {}".format(
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


#### MODEL MANAGEMENT ####
@router.post(
    "/model/update",
    description="Edit existing model params",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_model(
    model_params: updateDeployment,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global llm_router, llm_model_list, general_settings, user_config_file_path, proxy_config, prisma_client, master_key, store_model_in_db, proxy_logging_obj
    try:
        import base64

        global prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )
        # update DB
        if store_model_in_db == True:
            _model_id = None
            _model_info = getattr(model_params, "model_info", None)
            if _model_info is None:
                raise Exception("model_info not provided")

            _model_id = _model_info.id
            if _model_id is None:
                raise Exception("model_info.id not provided")
            _existing_litellm_params = (
                await prisma_client.db.litellm_proxymodeltable.find_unique(
                    where={"model_id": _model_id}
                )
            )
            if _existing_litellm_params is None:
                if (
                    llm_router is not None
                    and llm_router.get_deployment(model_id=_model_id) is not None
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Can't edit model. Model in config. Store model in db via `/model/new`. to edit."
                        },
                    )
                raise Exception("model not found")
            _existing_litellm_params_dict = dict(
                _existing_litellm_params.litellm_params
            )

            if model_params.litellm_params is None:
                raise Exception("litellm_params not provided")

            _new_litellm_params_dict = model_params.litellm_params.dict(
                exclude_none=True
            )

            ### ENCRYPT PARAMS ###
            for k, v in _new_litellm_params_dict.items():
                encrypted_value = encrypt_value_helper(value=v)
                model_params.litellm_params[k] = encrypted_value

            ### MERGE WITH EXISTING DATA ###
            merged_dictionary = {}
            _mp = model_params.litellm_params.dict()

            for key, value in _mp.items():
                if value is not None:
                    merged_dictionary[key] = value
                elif (
                    key in _existing_litellm_params_dict
                    and _existing_litellm_params_dict[key] is not None
                ):
                    merged_dictionary[key] = _existing_litellm_params_dict[key]
                else:
                    pass

            _data: dict = {
                "litellm_params": json.dumps(merged_dictionary),  # type: ignore
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
            model_response = await prisma_client.db.litellm_proxymodeltable.update(
                where={"model_id": _model_id},
                data=_data,  # type: ignore
            )

            return model_response
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.update_model(): Exception occured - {}".format(
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


@router.get(
    "/v2/model/info",
    description="v2 - returns all the models set on the config.yaml, shows 'user_access' = True if the user has access to the model. Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def model_info_v2(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Specify the model name (optional)"
    ),
    debug: Optional[bool] = False,
):
    """
    BETA ENDPOINT. Might change unexpectedly. Use `/v1/model/info` for now.
    """
    global llm_model_list, general_settings, user_config_file_path, proxy_config, llm_router

    if llm_model_list is None or not isinstance(llm_model_list, list):
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"No model list passed, models={llm_model_list}. You can add a model through the config.yaml or on the LiteLLM Admin UI."
            },
        )

    # Load existing config
    config = await proxy_config.get_config()

    all_models = copy.deepcopy(llm_model_list)
    if user_model is not None:
        # if user does not use a config.yaml, https://github.com/BerriAI/litellm/issues/2061
        all_models += [user_model]

    # check all models user has access to in user_api_key_dict
    user_models = []
    if len(user_api_key_dict.models) > 0:
        user_models = user_api_key_dict.models

    if model is not None:
        all_models = [m for m in all_models if m["model_name"] == model]

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
        _model["litellm_params"].pop("api_key", None)
        _model["litellm_params"].pop("vertex_credentials", None)
        _model["litellm_params"].pop("aws_access_key_id", None)
        _model["litellm_params"].pop("aws_secret_access_key", None)

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
    startTime = startTime or datetime.now() - timedelta(days=30)
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
            if "https://" in _api_base:
                _combined_model_name = str(_api_base)
            if "/openai/" in _combined_model_name:
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

    startTime = startTime or datetime.now() - timedelta(days=30)
    endTime = endTime or datetime.now()

    alerting_threshold = (
        proxy_logging_obj.slack_alerting_instance.alerting_threshold or 300
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

    startTime = startTime or datetime.now() - timedelta(days=30)
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
    global llm_model_list, general_settings, user_config_file_path, proxy_config, llm_router

    if llm_model_list is None:
        raise HTTPException(
            status_code=500, detail={"error": "LLM Model List not loaded in"}
        )

    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "LLM Router is not loaded in. Make sure you passed models in your config.yaml or on the LiteLLM Admin UI."
            },
        )

    if litellm_model_id is not None:
        # user is trying to get specific model from litellm router
        deployment_info = llm_router.get_deployment(model_id=litellm_model_id)
        if deployment_info is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Model id = {litellm_model_id} not found on litellm proxy"
                },
            )
        _deployment_info_dict = deployment_info.model_dump()
        _deployment_info_dict = remove_sensitive_info_from_deployment(
            deployment_dict=_deployment_info_dict
        )
        return {"data": _deployment_info_dict}

    all_models: List[dict] = []
    ## CHECK IF MODEL RESTRICTIONS ARE SET AT KEY/TEAM LEVEL ##
    if llm_model_list is None:
        proxy_model_list = []
    else:
        proxy_model_list = [m["model_name"] for m in llm_model_list]
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict, proxy_model_list=proxy_model_list
    )
    team_models = get_team_models(
        user_api_key_dict=user_api_key_dict, proxy_model_list=proxy_model_list
    )
    all_models_str = get_complete_model_list(
        key_models=key_models,
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        user_model=user_model,
        infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
    )

    if len(all_models_str) > 0:
        model_names = all_models_str
        _relevant_models = [m for m in llm_model_list if m["model_name"] in model_names]
        all_models = copy.deepcopy(_relevant_models)

    for model in all_models:
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
            except:
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
            except:
                litellm_model_info = {}
        for k, v in litellm_model_info.items():
            if k not in model_info:
                model_info[k] = v
        model["model_info"] = model_info
        # don't return the llm credentials
        model = remove_sensitive_info_from_deployment(deployment_dict=model)

    verbose_proxy_logger.debug("all_models: %s", all_models)
    return {"data": all_models}


@router.get(
    "/model_group/info",
    description="Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def model_group_info(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Returns model info at the model group level.
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
    all_models: List[dict] = []
    ## CHECK IF MODEL RESTRICTIONS ARE SET AT KEY/TEAM LEVEL ##
    if llm_model_list is None:
        proxy_model_list = []
    else:
        proxy_model_list = [m["model_name"] for m in llm_model_list]
    key_models = get_key_models(
        user_api_key_dict=user_api_key_dict, proxy_model_list=proxy_model_list
    )
    team_models = get_team_models(
        user_api_key_dict=user_api_key_dict, proxy_model_list=proxy_model_list
    )
    all_models_str = get_complete_model_list(
        key_models=key_models,
        team_models=team_models,
        proxy_model_list=proxy_model_list,
        user_model=user_model,
        infer_model_from_keys=general_settings.get("infer_model_from_keys", False),
    )

    model_groups: List[ModelGroupInfo] = []
    for model in all_models_str:

        _model_group_info = llm_router.get_model_group_info(model_group=model)
        if _model_group_info is not None:
            model_groups.append(_model_group_info)

    return {"data": model_groups}


#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post(
    "/model/delete",
    description="Allows deleting models in the model list in the config.yaml",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_model(model_info: ModelInfoDelete):
    global llm_router, llm_model_list, general_settings, user_config_file_path, proxy_config
    try:
        """
        [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964

        - Check if id in db
        - Delete
        """

        global prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        # update DB
        if store_model_in_db == True:
            """
            - store model_list in db
            - store keys separately
            """
            # encrypt litellm params #
            result = await prisma_client.db.litellm_proxymodeltable.delete(
                where={"model_id": model_info.id}
            )

            if result is None:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Model with id={model_info.id} not found in db"},
                )

            ## DELETE FROM ROUTER ##
            if llm_router is not None:
                llm_router.delete_deployment(id=model_info.id)

            return {"message": f"Model: {result.model_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

    except Exception as e:
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
    else:
        alerting_args_dict = {}

    allowed_args = {
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
            or data["model"]  # default passed in http request
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
            "stream" in data and data["stream"] == True
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


#### LOGIN ENDPOINTS ####


@app.get("/sso/key/generate", tags=["experimental"], include_in_schema=False)
async def google_login(request: Request):
    """
    Create Proxy API Keys using Google Workspace SSO. Requires setting PROXY_BASE_URL in .env
    PROXY_BASE_URL should be the your deployed proxy endpoint, e.g. PROXY_BASE_URL="https://litellm-production-7002.up.railway.app/"
    Example:
    """
    global premium_user, prisma_client, master_key

    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)

    ####### Check if user is a Enterprise / Premium User #######
    if (
        microsoft_client_id is not None
        or google_client_id is not None
        or generic_client_id is not None
    ):
        if premium_user != True:
            raise ProxyException(
                message="You must be a LiteLLM Enterprise user to use SSO. If you have a license please set `LITELLM_LICENSE` in your env. If you want to obtain a license meet with us here: https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat You are seeing this error message because You set one of `MICROSOFT_CLIENT_ID`, `GOOGLE_CLIENT_ID`, or `GENERIC_CLIENT_ID` in your env. Please unset this",
                type=ProxyErrorTypes.auth_error,
                param="premium_user",
                code=status.HTTP_403_FORBIDDEN,
            )

    ####### Detect DB + MASTER KEY in .env #######
    missing_env_vars = show_missing_vars_in_env()
    if missing_env_vars is not None:
        return missing_env_vars

    # get url from request
    redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
    ui_username = os.getenv("UI_USERNAME")
    if redirect_url.endswith("/"):
        redirect_url += "sso/callback"
    else:
        redirect_url += "/sso/callback"
    # Google SSO Auth
    if google_client_id is not None:
        from fastapi_sso.sso.google import GoogleSSO

        google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", None)
        if google_client_secret is None:
            raise ProxyException(
                message="GOOGLE_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GOOGLE_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        google_sso = GoogleSSO(
            client_id=google_client_id,
            client_secret=google_client_secret,
            redirect_uri=redirect_url,
        )
        verbose_proxy_logger.info(
            f"In /google-login/key/generate, \nGOOGLE_REDIRECT_URI: {redirect_url}\nGOOGLE_CLIENT_ID: {google_client_id}"
        )
        with google_sso:
            return await google_sso.get_login_redirect()
    # Microsoft SSO Auth
    elif microsoft_client_id is not None:
        from fastapi_sso.sso.microsoft import MicrosoftSSO

        microsoft_client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", None)
        microsoft_tenant = os.getenv("MICROSOFT_TENANT", None)
        if microsoft_client_secret is None:
            raise ProxyException(
                message="MICROSOFT_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="MICROSOFT_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        microsoft_sso = MicrosoftSSO(
            client_id=microsoft_client_id,
            client_secret=microsoft_client_secret,
            tenant=microsoft_tenant,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
        )
        with microsoft_sso:
            return await microsoft_sso.get_login_redirect()
    elif generic_client_id is not None:
        from fastapi_sso.sso.generic import DiscoveryDocument, create_provider

        generic_client_secret = os.getenv("GENERIC_CLIENT_SECRET", None)
        generic_scope = os.getenv("GENERIC_SCOPE", "openid email profile").split(" ")
        generic_authorization_endpoint = os.getenv(
            "GENERIC_AUTHORIZATION_ENDPOINT", None
        )
        generic_token_endpoint = os.getenv("GENERIC_TOKEN_ENDPOINT", None)
        generic_userinfo_endpoint = os.getenv("GENERIC_USERINFO_ENDPOINT", None)
        if generic_client_secret is None:
            raise ProxyException(
                message="GENERIC_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_authorization_endpoint is None:
            raise ProxyException(
                message="GENERIC_AUTHORIZATION_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_AUTHORIZATION_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_token_endpoint is None:
            raise ProxyException(
                message="GENERIC_TOKEN_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_TOKEN_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_userinfo_endpoint is None:
            raise ProxyException(
                message="GENERIC_USERINFO_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_USERINFO_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        verbose_proxy_logger.debug(
            f"authorization_endpoint: {generic_authorization_endpoint}\ntoken_endpoint: {generic_token_endpoint}\nuserinfo_endpoint: {generic_userinfo_endpoint}"
        )
        verbose_proxy_logger.debug(
            f"GENERIC_REDIRECT_URI: {redirect_url}\nGENERIC_CLIENT_ID: {generic_client_id}\n"
        )
        discovery = DiscoveryDocument(
            authorization_endpoint=generic_authorization_endpoint,
            token_endpoint=generic_token_endpoint,
            userinfo_endpoint=generic_userinfo_endpoint,
        )
        SSOProvider = create_provider(name="oidc", discovery_document=discovery)
        generic_sso = SSOProvider(
            client_id=generic_client_id,
            client_secret=generic_client_secret,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
            scope=generic_scope,
        )
        with generic_sso:
            # TODO: state should be a random string and added to the user session with cookie
            # or a cryptographicly signed state that we can verify stateless
            # For simplification we are using a static state, this is not perfect but some
            # SSO providers do not allow stateless verification
            redirect_params = {}
            state = os.getenv("GENERIC_CLIENT_STATE", None)
            if state:
                redirect_params["state"] = state
            return await generic_sso.get_login_redirect(**redirect_params)  # type: ignore
    elif ui_username is not None:
        # No Google, Microsoft SSO
        # Use UI Credentials set in .env
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)
    else:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=html_form, status_code=200)


@app.get("/fallback/login", tags=["experimental"], include_in_schema=False)
async def fallback_login(request: Request):
    """
    Create Proxy API Keys using Google Workspace SSO. Requires setting PROXY_BASE_URL in .env
    PROXY_BASE_URL should be the your deployed proxy endpoint, e.g. PROXY_BASE_URL="https://litellm-production-7002.up.railway.app/"
    Example:
    """
    # get url from request
    redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
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
async def login(request: Request):
    global premium_user, general_settings
    try:
        import multipart
    except ImportError:
        subprocess.run(["pip", "install", "python-multipart"])
    global master_key
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
    _user_row = None
    if prisma_client is not None:
        _user_row = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": {"equals": username}}
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
        user_id = username

        # we want the key created to have PROXY_ADMIN_PERMISSIONS
        key_user_id = litellm_proxy_admin_name
        if (
            os.getenv("PROXY_ADMIN_ID", None) is not None
            and os.environ["PROXY_ADMIN_ID"] == user_id
        ) or user_id == "admin":
            # checks if user is admin
            key_user_id = os.getenv("PROXY_ADMIN_ID", "default_user_id")

        # Admin is Authe'd in - generate key for the UI to access Proxy

        # ensure this user is set as the proxy admin, in this route there is no sso, we can assume this user is only the admin
        await user_update(
            data=UpdateUserRequest(
                user_id=key_user_id,
                user_role=user_role,
            )
        )
        if os.getenv("DATABASE_URL") is not None:
            response = await generate_key_helper_fn(
                request_type="key",
                **{
                    "user_role": LitellmUserRoles.PROXY_ADMIN,
                    "duration": "24hr",
                    "key_max_budget": 5,
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
        litellm_dashboard_ui = os.getenv("PROXY_BASE_URL", "")
        if litellm_dashboard_ui.endswith("/"):
            litellm_dashboard_ui += "ui/"
        else:
            litellm_dashboard_ui += "/ui/"
        import jwt

        jwt_token = jwt.encode(  # type: ignore
            {
                "user_id": user_id,
                "key": key,
                "user_email": user_id,
                "user_role": user_role,  # this is the path without sso - we can assume only admins will use this
                "login_method": "username_password",
                "premium_user": premium_user,
                "auth_header_name": general_settings.get(
                    "litellm_key_header_name", "Authorization"
                ),
            },
            master_key,
            algorithm="HS256",
        )
        litellm_dashboard_ui += "?userID=" + user_id
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
                        "key_max_budget": 5,
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
            litellm_dashboard_ui = os.getenv("PROXY_BASE_URL", "")
            if litellm_dashboard_ui.endswith("/"):
                litellm_dashboard_ui += "ui/"
            else:
                litellm_dashboard_ui += "/ui/"
            import jwt

            jwt_token = jwt.encode(  # type: ignore
                {
                    "user_id": user_id,
                    "key": key,
                    "user_email": user_email,
                    "user_role": user_role,
                    "login_method": "username_password",
                    "premium_user": premium_user,
                    "auth_header_name": general_settings.get(
                        "litellm_key_header_name", "Authorization"
                    ),
                },
                master_key,
                algorithm="HS256",
            )
            litellm_dashboard_ui += "?userID=" + user_id
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


@app.get(
    "/sso/get/logout_url",
    tags=["experimental"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_logout_url(request: Request):
    _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
    _logout_url = os.getenv("PROXY_LOGOUT_URL", None)

    return {"PROXY_BASE_URL": _proxy_base_url, "PROXY_LOGOUT_URL": _logout_url}


@app.get("/onboarding/get_token", include_in_schema=False)
async def onboarding(invite_link: str):
    """
    - Get the invite link
    - Validate it's still 'valid'
    - Invalidate the link (prevents abuse)
    - Get user from db
    - Pass in user_email if set
    """
    global prisma_client, master_key, general_settings
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
            "key_max_budget": 5,
            "models": [],
            "aliases": {},
            "config": {},
            "spend": 0,
            "user_id": user_obj.user_id,
            "team_id": "litellm-dashboard",
        },  # type: ignore
    )
    key = response["token"]  # type: ignore

    litellm_dashboard_ui = os.getenv("PROXY_BASE_URL", "")
    if litellm_dashboard_ui.endswith("/"):
        litellm_dashboard_ui += "ui/onboarding"
    else:
        litellm_dashboard_ui += "/ui/onboarding"
    import jwt

    jwt_token = jwt.encode(  # type: ignore
        {
            "user_id": user_obj.user_id,
            "key": key,
            "user_email": user_obj.user_email,
            "user_role": user_obj.user_role,
            "login_method": "username_password",
            "premium_user": premium_user,
            "auth_header_name": general_settings.get(
                "litellm_key_header_name", "Authorization"
            ),
        },
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


@app.get("/get_image", include_in_schema=False)
def get_image():
    """Get logo to show on admin UI"""
    from fastapi.responses import FileResponse

    # get current_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_logo = os.path.join(current_dir, "logo.jpg")

    logo_path = os.getenv("UI_LOGO_PATH", default_logo)
    verbose_proxy_logger.debug("Reading logo from path: %s", logo_path)

    # Check if the logo path is an HTTP/HTTPS URL
    if logo_path.startswith(("http://", "https://")):
        # Download the image and cache it
        response = requests.get(logo_path)
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


@app.get("/sso/callback", tags=["experimental"], include_in_schema=False)
async def auth_callback(request: Request):
    """Verify login"""
    global general_settings, ui_access_mode, premium_user, master_key
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)
    # get url from request
    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type=ProxyErrorTypes.auth_error,
            param="master_key",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
    if redirect_url.endswith("/"):
        redirect_url += "sso/callback"
    else:
        redirect_url += "/sso/callback"
    if google_client_id is not None:
        from fastapi_sso.sso.google import GoogleSSO

        google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", None)
        if google_client_secret is None:
            raise ProxyException(
                message="GOOGLE_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GOOGLE_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        google_sso = GoogleSSO(
            client_id=google_client_id,
            redirect_uri=redirect_url,
            client_secret=google_client_secret,
        )
        result = await google_sso.verify_and_process(request)
    elif microsoft_client_id is not None:
        from fastapi_sso.sso.microsoft import MicrosoftSSO

        microsoft_client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", None)
        microsoft_tenant = os.getenv("MICROSOFT_TENANT", None)
        if microsoft_client_secret is None:
            raise ProxyException(
                message="MICROSOFT_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="MICROSOFT_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if microsoft_tenant is None:
            raise ProxyException(
                message="MICROSOFT_TENANT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="MICROSOFT_TENANT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        microsoft_sso = MicrosoftSSO(
            client_id=microsoft_client_id,
            client_secret=microsoft_client_secret,
            tenant=microsoft_tenant,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
        )
        result = await microsoft_sso.verify_and_process(request)
    elif generic_client_id is not None:
        # make generic sso provider
        from fastapi_sso.sso.generic import DiscoveryDocument, OpenID, create_provider

        generic_client_secret = os.getenv("GENERIC_CLIENT_SECRET", None)
        generic_scope = os.getenv("GENERIC_SCOPE", "openid email profile").split(" ")
        generic_authorization_endpoint = os.getenv(
            "GENERIC_AUTHORIZATION_ENDPOINT", None
        )
        generic_token_endpoint = os.getenv("GENERIC_TOKEN_ENDPOINT", None)
        generic_userinfo_endpoint = os.getenv("GENERIC_USERINFO_ENDPOINT", None)
        generic_include_client_id = (
            os.getenv("GENERIC_INCLUDE_CLIENT_ID", "false").lower() == "true"
        )
        if generic_client_secret is None:
            raise ProxyException(
                message="GENERIC_CLIENT_SECRET not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_authorization_endpoint is None:
            raise ProxyException(
                message="GENERIC_AUTHORIZATION_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_AUTHORIZATION_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_token_endpoint is None:
            raise ProxyException(
                message="GENERIC_TOKEN_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_TOKEN_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_userinfo_endpoint is None:
            raise ProxyException(
                message="GENERIC_USERINFO_ENDPOINT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="GENERIC_USERINFO_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        verbose_proxy_logger.debug(
            f"authorization_endpoint: {generic_authorization_endpoint}\ntoken_endpoint: {generic_token_endpoint}\nuserinfo_endpoint: {generic_userinfo_endpoint}"
        )
        verbose_proxy_logger.debug(
            f"GENERIC_REDIRECT_URI: {redirect_url}\nGENERIC_CLIENT_ID: {generic_client_id}\n"
        )

        generic_user_id_attribute_name = os.getenv(
            "GENERIC_USER_ID_ATTRIBUTE", "preferred_username"
        )
        generic_user_display_name_attribute_name = os.getenv(
            "GENERIC_USER_DISPLAY_NAME_ATTRIBUTE", "sub"
        )
        generic_user_email_attribute_name = os.getenv(
            "GENERIC_USER_EMAIL_ATTRIBUTE", "email"
        )
        generic_user_role_attribute_name = os.getenv(
            "GENERIC_USER_ROLE_ATTRIBUTE", "role"
        )
        generic_user_first_name_attribute_name = os.getenv(
            "GENERIC_USER_FIRST_NAME_ATTRIBUTE", "first_name"
        )
        generic_user_last_name_attribute_name = os.getenv(
            "GENERIC_USER_LAST_NAME_ATTRIBUTE", "last_name"
        )

        verbose_proxy_logger.debug(
            f" generic_user_id_attribute_name: {generic_user_id_attribute_name}\n generic_user_email_attribute_name: {generic_user_email_attribute_name}\n generic_user_role_attribute_name: {generic_user_role_attribute_name}"
        )

        discovery = DiscoveryDocument(
            authorization_endpoint=generic_authorization_endpoint,
            token_endpoint=generic_token_endpoint,
            userinfo_endpoint=generic_userinfo_endpoint,
        )

        def response_convertor(response, client):
            return OpenID(
                id=response.get(generic_user_id_attribute_name),
                display_name=response.get(generic_user_display_name_attribute_name),
                email=response.get(generic_user_email_attribute_name),
                first_name=response.get(generic_user_first_name_attribute_name),
                last_name=response.get(generic_user_last_name_attribute_name),
            )

        SSOProvider = create_provider(
            name="oidc",
            discovery_document=discovery,
            response_convertor=response_convertor,
        )
        generic_sso = SSOProvider(
            client_id=generic_client_id,
            client_secret=generic_client_secret,
            redirect_uri=redirect_url,
            allow_insecure_http=True,
            scope=generic_scope,
        )
        verbose_proxy_logger.debug("calling generic_sso.verify_and_process")
        result = await generic_sso.verify_and_process(
            request, params={"include_client_id": generic_include_client_id}
        )
        verbose_proxy_logger.debug("generic result: %s", result)

    # User is Authe'd in - generate key for the UI to access Proxy
    user_email = getattr(result, "email", None)
    user_id = getattr(result, "id", None)

    if user_email is not None and os.getenv("ALLOWED_EMAIL_DOMAINS") is not None:
        email_domain = user_email.split("@")[1]
        allowed_domains = os.getenv("ALLOWED_EMAIL_DOMAINS").split(",")  # type: ignore
        if email_domain not in allowed_domains:
            raise HTTPException(
                status_code=401,
                detail={
                    "message": "The email domain={}, is not an allowed email domain={}. Contact your admin to change this.".format(
                        email_domain, allowed_domains
                    )
                },
            )

    # generic client id
    if generic_client_id is not None:
        user_id = getattr(result, "id", None)
        user_email = getattr(result, "email", None)
        user_role = getattr(result, generic_user_role_attribute_name, None)

    if user_id is None:
        _first_name = getattr(result, "first_name", "") or ""
        _last_name = getattr(result, "last_name", "") or ""
        user_id = _first_name + _last_name

    if user_email is not None and (user_id is None or len(user_id) == 0):
        user_id = user_email

    user_info = None
    user_id_models: List = []
    max_internal_user_budget = litellm.max_internal_user_budget
    internal_user_budget_duration = litellm.internal_user_budget_duration

    # User might not be already created on first generation of key
    # But if it is, we want their models preferences
    default_ui_key_values = {
        "duration": "24hr",
        "key_max_budget": 0.01,
        "aliases": {},
        "config": {},
        "spend": 0,
        "team_id": "litellm-dashboard",
    }
    user_defined_values: SSOUserDefinedValues = {
        "models": user_id_models,
        "user_id": user_id,
        "user_email": user_email,
        "max_budget": max_internal_user_budget,
        "user_role": None,
        "budget_duration": internal_user_budget_duration,
    }
    _user_id_from_sso = user_id
    try:
        user_role = None
        if prisma_client is not None:
            user_info = await prisma_client.get_data(user_id=user_id, table_name="user")
            verbose_proxy_logger.debug(
                f"user_info: {user_info}; litellm.default_user_params: {litellm.default_user_params}"
            )
            if user_info is not None:
                user_defined_values = {
                    "models": getattr(user_info, "models", user_id_models),
                    "user_id": getattr(user_info, "user_id", user_id),
                    "user_email": getattr(user_info, "user_id", user_email),
                    "user_role": getattr(user_info, "user_role", None),
                    "max_budget": getattr(
                        user_info, "max_budget", max_internal_user_budget
                    ),
                    "budget_duration": getattr(
                        user_info, "budget_duration", internal_user_budget_duration
                    ),
                }
                user_role = getattr(user_info, "user_role", None)

            ## check if user-email in db ##
            user_info = await prisma_client.db.litellm_usertable.find_first(
                where={"user_email": user_email}
            )
            if user_info is not None:
                user_defined_values = {
                    "models": getattr(user_info, "models", user_id_models),
                    "user_id": user_id,
                    "user_email": getattr(user_info, "user_id", user_email),
                    "user_role": getattr(user_info, "user_role", None),
                    "max_budget": getattr(
                        user_info, "max_budget", max_internal_user_budget
                    ),
                    "budget_duration": getattr(
                        user_info, "budget_duration", internal_user_budget_duration
                    ),
                }
                user_role = getattr(user_info, "user_role", None)

                # update id
                await prisma_client.db.litellm_usertable.update_many(
                    where={"user_email": user_email}, data={"user_id": user_id}  # type: ignore
                )
            elif litellm.default_user_params is not None and isinstance(
                litellm.default_user_params, dict
            ):
                user_defined_values = {
                    "models": litellm.default_user_params.get("models", user_id_models),
                    "user_id": litellm.default_user_params.get("user_id", user_id),
                    "user_email": litellm.default_user_params.get(
                        "user_email", user_email
                    ),
                    "user_role": litellm.default_user_params.get("user_role", None),
                    "max_budget": litellm.default_user_params.get(
                        "max_budget", max_internal_user_budget
                    ),
                    "budget_duration": litellm.default_user_params.get(
                        "budget_duration", internal_user_budget_duration
                    ),
                }

    except Exception as e:
        pass

    is_internal_user = False
    if (
        user_defined_values["user_role"] is not None
        and user_defined_values["user_role"] == LitellmUserRoles.INTERNAL_USER.value
    ):
        is_internal_user = True
    if (
        is_internal_user is True
        and user_defined_values["max_budget"] is None
        and litellm.max_internal_user_budget is not None
    ):
        user_defined_values["max_budget"] = litellm.max_internal_user_budget

    if (
        is_internal_user is True
        and user_defined_values["budget_duration"] is None
        and litellm.internal_user_budget_duration is not None
    ):
        user_defined_values["budget_duration"] = litellm.internal_user_budget_duration

    verbose_proxy_logger.info(
        f"user_defined_values for creating ui key: {user_defined_values}"
    )

    default_ui_key_values.update(user_defined_values)
    default_ui_key_values["request_type"] = "key"
    response = await generate_key_helper_fn(
        **default_ui_key_values,  # type: ignore
    )
    key = response["token"]  # type: ignore
    user_id = response["user_id"]  # type: ignore

    # This should always be true
    # User_id on SSO == user_id in the LiteLLM_VerificationToken Table
    assert user_id == _user_id_from_sso
    litellm_dashboard_ui = "/ui/"
    user_role = user_role or "app_owner"
    if (
        os.getenv("PROXY_ADMIN_ID", None) is not None
        and os.environ["PROXY_ADMIN_ID"] == user_id
    ):
        # checks if user is admin
        user_role = "app_admin"

    verbose_proxy_logger.debug(
        f"user_role: {user_role}; ui_access_mode: {ui_access_mode}"
    )
    ## CHECK IF ROLE ALLOWED TO USE PROXY ##
    if ui_access_mode == "admin_only" and "admin" not in user_role:
        verbose_proxy_logger.debug("EXCEPTION RAISED")
        raise HTTPException(
            status_code=401,
            detail={
                "error": f"User not allowed to access proxy. User role={user_role}, proxy mode={ui_access_mode}"
            },
        )

    import jwt

    jwt_token = jwt.encode(  # type: ignore
        {
            "user_id": user_id,
            "key": key,
            "user_email": user_email,
            "user_role": user_role,
            "login_method": "sso",
            "premium_user": premium_user,
            "auth_header_name": general_settings.get(
                "litellm_key_header_name", "Authorization"
            ),
        },
        master_key,
        algorithm="HS256",
    )
    litellm_dashboard_ui += "?userID=" + user_id
    redirect_response = RedirectResponse(url=litellm_dashboard_ui, status_code=303)
    redirect_response.set_cookie(key="token", value=jwt_token)
    return redirect_response


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
        -D '{
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

    current_time = litellm.utils.get_utc_datetime()
    expires_at = current_time + timedelta(days=7)

    try:
        response = await prisma_client.db.litellm_invitationlink.create(
            data={
                "user_id": data.user_id,
                "created_at": current_time,
                "expires_at": expires_at,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_at": current_time,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }  # type: ignore
        )
    except Exception as e:
        if "Foreign key constraint failed on the field" in str(e):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "User id does not exist in 'LiteLLM_UserTable'. Fix this by creating user via `/user/new`."
                },
            )
    return response


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
        -D '{
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
        -D '{
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
        -D '{
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
async def update_config(config_info: ConfigYAML):
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

        import base64

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
                        _existing_settings["alerting"].append("slack")
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
                    combined_success_callback = (
                        config["litellm_settings"]["success_callback"]
                        + updated_litellm_settings["success_callback"]
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
        cgs = ConfigGeneralSettings(**{data.field_name: data.field_value})
    except:
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


@router.get(
    "/get/config/callbacks",
    tags=["config.yaml"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def get_config():
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
                        decrypted_value = decrypt_value_helper(value=env_variable)
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
                        decrypted_value = decrypt_value_helper(value=env_variable)
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
                    _decrypted_value = decrypt_value_helper(value=env_variable)
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
                _decrypted_value = decrypt_value_helper(value=env_variable)
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
async def get_litellm_model_cost_map():
    try:
        _model_cost_map = litellm.model_cost
        return _model_cost_map
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error ({str(e)})",
        )


@router.get("/", dependencies=[Depends(user_api_key_auth)])
async def home(request: Request):
    return "LiteLLM: RUNNING"


@router.get("/routes", dependencies=[Depends(user_api_key_auth)])
async def get_routes():
    """
    Get a list of available routes in the FastAPI application.
    """
    routes = []
    for route in app.routes:
        route_info = {
            "path": getattr(route, "path", None),
            "methods": getattr(route, "methods", None),
            "name": getattr(route, "name", None),
            "endpoint": (
                getattr(route, "endpoint", None).__name__
                if getattr(route, "endpoint", None)
                else None
            ),
        }
        routes.append(route_info)

    return {"routes": routes}


#### TEST ENDPOINTS ####
@router.get(
    "/token/generate",
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def token_generate():
    """
    Test endpoint. Admin-only access. Meant for generating admin tokens with specific claims and testing if they work for creating keys, etc.
    """
    # Initialize AuthJWTSSO with your OpenID Provider configuration
    from fastapi_sso import AuthJWTSSO

    auth_jwt_sso = AuthJWTSSO(
        issuer=os.getenv("OPENID_BASE_URL"),
        client_id=os.getenv("OPENID_CLIENT_ID"),
        client_secret=os.getenv("OPENID_CLIENT_SECRET"),
        scopes=["litellm_proxy_admin"],
    )

    token = auth_jwt_sso.create_access_token()

    return {"token": token}


@router.on_event("shutdown")
async def shutdown_event():
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

            langFuseLogger.Langfuse.flush()
        except:
            # [DO NOT BLOCK shutdown events for this]
            pass

    ## RESET CUSTOM VARIABLES ##
    cleanup_router_config_variables()


def cleanup_router_config_variables():
    global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, user_custom_key_generate, use_background_health_checks, health_check_interval, prisma_client, custom_db_client

    # Set all variables to None
    master_key = None
    user_config_file_path = None
    otel_logging = None
    user_custom_auth = None
    user_custom_auth_path = None
    user_custom_key_generate = None
    use_background_health_checks = None
    health_check_interval = None
    health_check_details = None
    prisma_client = None
    custom_db_client = None


app.include_router(router)
app.include_router(rerank_router)
app.include_router(fine_tuning_router)
app.include_router(vertex_router)
app.include_router(gemini_router)
app.include_router(langfuse_router)
app.include_router(pass_through_router)
app.include_router(health_router)
app.include_router(key_management_router)
app.include_router(internal_user_router)
app.include_router(team_router)
app.include_router(spend_management_router)
app.include_router(caching_router)
app.include_router(analytics_router)
app.include_router(debugging_endpoints_router)
app.include_router(ui_crud_endpoints_router)
app.include_router(openai_files_router)
app.include_router(team_callback_router)
