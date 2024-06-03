import sys, os, platform, time, copy, re, asyncio, inspect
import threading, ast
import shutil, random, traceback, requests
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Callable, get_args, Set
import secrets, subprocess
import hashlib, uuid
import warnings
import importlib


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
    import fastapi
    import backoff
    import yaml  # type: ignore
    import orjson
    import logging
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


import litellm
from litellm.types.llms.openai import (
    HttpxBinaryResponseContent,
)
from litellm.proxy.utils import (
    PrismaClient,
    DBClient,
    get_instance_fn,
    ProxyLogging,
    _cache_user_row,
    send_email,
    get_logging_payload,
    reset_budget,
    hash_token,
    html_form,
    missing_keys_html_form,
    _read_request_body,
    _is_valid_team_configs,
    _is_user_proxy_admin,
    _get_user_role,
    _is_projected_spend_over_limit,
    _get_projected_spend_over_limit,
    update_spend,
    encrypt_value,
    decrypt_value,
)
from litellm import (
    CreateBatchRequest,
    RetrieveBatchRequest,
    ListBatchRequest,
    CancelBatchRequest,
    CreateFileRequest,
)
from litellm.proxy.secret_managers.google_kms import load_google_kms
from litellm.proxy.secret_managers.aws_secret_manager import load_aws_secret_manager
import pydantic
from litellm.proxy._types import *
from litellm.caching import DualCache, RedisCache
from litellm.proxy.health_check import perform_health_check
from litellm.router import LiteLLM_Params, Deployment, updateDeployment, ModelGroupInfo
from litellm.router import ModelInfo as RouterModelInfo
from litellm._logging import verbose_router_logger, verbose_proxy_logger
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.auth.model_checks import (
    get_complete_model_list,
    get_key_models,
    get_team_models,
)
from litellm.proxy.hooks.prompt_injection_detection import (
    _OPTIONAL_PromptInjectionDetection,
)
from litellm.proxy.auth.auth_checks import (
    common_checks,
    get_end_user_object,
    get_org_object,
    get_team_object,
    get_user_object,
    allowed_routes_check,
    get_actual_routes,
)
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.exceptions import RejectedRequestError
from litellm.integrations.slack_alerting import SlackAlertingArgs, SlackAlerting
from litellm.scheduler import Scheduler, FlowItem, DefaultPriorities

try:
    from litellm._version import version
except:
    version = "0.0.0"
litellm.suppress_debug_info = True
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    status,
    Path,
    Depends,
    Header,
    Response,
    Form,
    UploadFile,
    File,
)
from fastapi.routing import APIRouter
from fastapi.security import OAuth2PasswordBearer
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    StreamingResponse,
    FileResponse,
    ORJSONResponse,
    JSONResponse,
)
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
import json
import logging
from typing import Union

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

_license_check = LicenseCheck()
premium_user: bool = _license_check.is_premium()
ui_link = f"/ui/"
ui_message = (
    f"👉 [```LiteLLM Admin Panel on /ui```]({ui_link}). Create, Edit Keys with SSO"
)
custom_swagger_message = f"[**Customize Swagger Docs**](https://docs.litellm.ai/docs/proxy/enterprise#swagger-docs---custom-routes--branding)"

### CUSTOM BRANDING [ENTERPRISE FEATURE] ###
_docs_url = None if os.getenv("NO_DOCS", "False") == "True" else "/"
_title = os.getenv("DOCS_TITLE", "LiteLLM API") if premium_user else "LiteLLM API"
_description = (
    os.getenv(
        "DOCS_DESCRIPTION",
        f"Proxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}",
    )
    if premium_user
    else f"Proxy Server to call 100+ LLMs in the OpenAI format. {custom_swagger_message}\n\n{ui_message}"
)

app = FastAPI(
    docs_url=_docs_url,
    title=_title,
    description=_description,
    version=version,
    root_path=os.environ.get(
        "SERVER_ROOT_PATH", ""
    ),  # check if user passed root path, FastAPI defaults this value to ""
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


class ProxyException(Exception):
    # NOTE: DO NOT MODIFY THIS
    # This is used to map exactly to OPENAI Exceptions
    def __init__(
        self,
        message: str,
        type: str,
        param: Optional[str],
        code: Optional[int],
    ):
        self.message = message
        self.type = type
        self.param = param
        self.code = code

        # rules for proxyExceptions
        # Litellm router.py returns "No healthy deployment available" when there are no deployments available
        # Should map to 429 errors https://github.com/BerriAI/litellm/issues/2487
        if (
            "No healthy deployment available" in self.message
            or "No deployments available" in self.message
        ):
            self.code = 429

    def to_dict(self) -> dict:
        """Converts the ProxyException instance to a dictionary."""
        return {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }


class UserAPIKeyCacheTTLEnum(enum.Enum):
    in_memory_cache_ttl = 60  # 1 min ttl ## configure via `general_settings::user_api_key_cache_ttl: <your-value>`


class CommonProxyErrors(enum.Enum):
    db_not_connected_error = "DB not connected"
    no_llm_router = "No models configured on proxy"
    not_allowed_access = "Admin-only endpoint. Not allowed to access this."
    not_premium_user = "You must be a LiteLLM Enterprise user to use this feature. If you have a license please set `LITELLM_LICENSE` in your env. If you want to obtain a license meet with us here: https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    # NOTE: DO NOT MODIFY THIS, its crucial to map to Openai exceptions
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

api_key_header = APIKeyHeader(
    name="Authorization", auto_error=False, description="Bearer token"
)
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
### INITIALIZE GLOBAL LOGGING OBJECT ###
proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
### REDIS QUEUE ###
async_result = None
celery_app_conn = None
celery_fn = None  # Redis Queue for handling requests
### DB WRITER ###
db_writer_client: Optional[HTTPHandler] = None
### logger ###


def _get_bearer_token(
    api_key: str,
):
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    else:
        api_key = ""
    return api_key


def _get_pydantic_json_dict(pydantic_obj: BaseModel) -> dict:
    try:
        return pydantic_obj.model_dump()  # type: ignore
    except:
        # if using pydantic v1
        return pydantic_obj.dict()


def get_custom_headers(
    *,
    user_api_key_dict: UserAPIKeyAuth,
    model_id: Optional[str] = None,
    cache_key: Optional[str] = None,
    api_base: Optional[str] = None,
    version: Optional[str] = None,
    model_region: Optional[str] = None,
    fastest_response_batch_completion: Optional[bool] = None,
) -> dict:
    exclude_values = {"", None}
    headers = {
        "x-litellm-model-id": model_id,
        "x-litellm-cache-key": cache_key,
        "x-litellm-model-api-base": api_base,
        "x-litellm-version": version,
        "x-litellm-model-region": model_region,
        "x-litellm-key-tpm-limit": str(user_api_key_dict.tpm_limit),
        "x-litellm-key-rpm-limit": str(user_api_key_dict.rpm_limit),
        "x-litellm-fastest_response_batch_completion": (
            str(fastest_response_batch_completion)
            if fastest_response_batch_completion is not None
            else None
        ),
    }
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


async def user_api_key_auth(
    request: Request, api_key: str = fastapi.Security(api_key_header)
) -> UserAPIKeyAuth:
    global master_key, prisma_client, llm_model_list, user_custom_auth, custom_db_client, general_settings
    try:
        if isinstance(api_key, str):
            passed_in_key = api_key
            api_key = _get_bearer_token(api_key=api_key)

        ### USER-DEFINED AUTH FUNCTION ###
        if user_custom_auth is not None:
            response = await user_custom_auth(request=request, api_key=api_key)
            return UserAPIKeyAuth.model_validate(response)

        ### LITELLM-DEFINED AUTH FUNCTION ###
        #### IF JWT ####
        """
        LiteLLM supports using JWTs.

        Enable this in proxy config, by setting
        ```
        general_settings:
            enable_jwt_auth: true
        ```
        """
        route: str = request.url.path

        if route in LiteLLMRoutes.public_routes.value:
            # check if public endpoint
            return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)

        if general_settings.get("enable_jwt_auth", False) == True:
            is_jwt = jwt_handler.is_jwt(token=api_key)
            verbose_proxy_logger.debug("is_jwt: %s", is_jwt)
            if is_jwt:
                # check if valid token
                jwt_valid_token: dict = await jwt_handler.auth_jwt(token=api_key)
                # get scopes
                scopes = jwt_handler.get_scopes(token=jwt_valid_token)

                # check if admin
                is_admin = jwt_handler.is_admin(scopes=scopes)
                # if admin return
                if is_admin:
                    # check allowed admin routes
                    is_allowed = allowed_routes_check(
                        user_role=LitellmUserRoles.PROXY_ADMIN,
                        user_route=route,
                        litellm_proxy_roles=jwt_handler.litellm_jwtauth,
                    )
                    if is_allowed:
                        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
                    else:
                        allowed_routes = (
                            jwt_handler.litellm_jwtauth.admin_allowed_routes
                        )
                        actual_routes = get_actual_routes(allowed_routes=allowed_routes)
                        raise Exception(
                            f"Admin not allowed to access this route. Route={route}, Allowed Routes={actual_routes}"
                        )
                # get team id
                team_id = jwt_handler.get_team_id(
                    token=jwt_valid_token, default_value=None
                )

                if team_id is None and jwt_handler.is_required_team_id() == True:
                    raise Exception(
                        f"No team id passed in. Field checked in jwt token - '{jwt_handler.litellm_jwtauth.team_id_jwt_field}'"
                    )

                team_object: Optional[LiteLLM_TeamTable] = None
                if team_id is not None:
                    # check allowed team routes
                    is_allowed = allowed_routes_check(
                        user_role=LitellmUserRoles.TEAM,
                        user_route=route,
                        litellm_proxy_roles=jwt_handler.litellm_jwtauth,
                    )
                    if is_allowed == False:
                        allowed_routes = jwt_handler.litellm_jwtauth.team_allowed_routes  # type: ignore
                        actual_routes = get_actual_routes(allowed_routes=allowed_routes)
                        raise Exception(
                            f"Team not allowed to access this route. Route={route}, Allowed Routes={actual_routes}"
                        )

                    # check if team in db
                    team_object = await get_team_object(
                        team_id=team_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                    )

                # [OPTIONAL] track spend for an org id - `LiteLLM_OrganizationTable`
                org_id = jwt_handler.get_org_id(
                    token=jwt_valid_token, default_value=None
                )
                if org_id is not None:
                    _ = await get_org_object(
                        org_id=org_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                    )
                # [OPTIONAL] track spend against an internal employee - `LiteLLM_UserTable`
                user_object = None
                user_id = jwt_handler.get_user_id(
                    token=jwt_valid_token, default_value=None
                )
                if user_id is not None:
                    # get the user object
                    user_object = await get_user_object(
                        user_id=user_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        user_id_upsert=jwt_handler.is_upsert_user_id(),
                    )

                # [OPTIONAL] track spend against an external user - `LiteLLM_EndUserTable`
                end_user_object = None
                end_user_id = jwt_handler.get_end_user_id(
                    token=jwt_valid_token, default_value=None
                )
                if end_user_id is not None:
                    # get the end-user object
                    end_user_object = await get_end_user_object(
                        end_user_id=end_user_id,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                    )

                global_proxy_spend = None
                if litellm.max_budget > 0:  # user set proxy max budget
                    # check cache
                    global_proxy_spend = await user_api_key_cache.async_get_cache(
                        key="{}:spend".format(litellm_proxy_admin_name)
                    )
                    if global_proxy_spend is None and prisma_client is not None:
                        # get from db
                        sql_query = """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""

                        response = await prisma_client.db.query_raw(query=sql_query)

                        global_proxy_spend = response[0]["total_spend"]

                        await user_api_key_cache.async_set_cache(
                            key="{}:spend".format(litellm_proxy_admin_name),
                            value=global_proxy_spend,
                        )
                    if global_proxy_spend is not None:
                        user_info = CallInfo(
                            user_id=litellm_proxy_admin_name,
                            max_budget=litellm.max_budget,
                            spend=global_proxy_spend,
                            token=jwt_valid_token["token"],
                        )
                        asyncio.create_task(
                            proxy_logging_obj.budget_alerts(
                                type="proxy_budget",
                                user_info=user_info,
                            )
                        )

                # get the request body
                request_data = await _read_request_body(request=request)

                # run through common checks
                _ = common_checks(
                    request_body=request_data,
                    team_object=team_object,
                    user_object=user_object,
                    end_user_object=end_user_object,
                    general_settings=general_settings,
                    global_proxy_spend=global_proxy_spend,
                    route=route,
                )

                # return UserAPIKeyAuth object
                return UserAPIKeyAuth(
                    api_key=None,
                    team_id=team_object.team_id if team_object is not None else None,
                    team_tpm_limit=(
                        team_object.tpm_limit if team_object is not None else None
                    ),
                    team_rpm_limit=(
                        team_object.rpm_limit if team_object is not None else None
                    ),
                    team_models=team_object.models if team_object is not None else [],
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    user_id=user_id,
                    org_id=org_id,
                )
        #### ELSE ####
        if master_key is None:
            if isinstance(api_key, str):
                return UserAPIKeyAuth(
                    api_key=api_key, user_role=LitellmUserRoles.PROXY_ADMIN
                )
            else:
                return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        elif api_key is None:  # only require api key if master key is set
            raise Exception("No api key passed in.")
        elif api_key == "":
            # missing 'Bearer ' prefix
            raise Exception(
                f"Malformed API Key passed in. Ensure Key has `Bearer ` prefix. Passed in: {passed_in_key}"
            )

        if route == "/user/auth":
            if general_settings.get("allow_user_auth", False) == True:
                return UserAPIKeyAuth()
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="'allow_user_auth' not set or set to False",
                )

        ## Check END-USER OBJECT
        request_data = await _read_request_body(request=request)
        _end_user_object = None
        end_user_params = {}
        if "user" in request_data:
            try:
                _end_user_object = await get_end_user_object(
                    end_user_id=request_data["user"],
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                )
                if _end_user_object is not None:
                    end_user_params["allowed_model_region"] = (
                        _end_user_object.allowed_model_region
                    )
                    if _end_user_object.litellm_budget_table is not None:
                        budget_info = _end_user_object.litellm_budget_table
                        end_user_params["end_user_id"] = _end_user_object.user_id
                        if budget_info.tpm_limit is not None:
                            end_user_params["end_user_tpm_limit"] = (
                                budget_info.tpm_limit
                            )
                        if budget_info.rpm_limit is not None:
                            end_user_params["end_user_rpm_limit"] = (
                                budget_info.rpm_limit
                            )
                        if budget_info.max_budget is not None:
                            end_user_params["end_user_max_budget"] = (
                                budget_info.max_budget
                            )
            except Exception as e:
                if isinstance(e, litellm.BudgetExceededError):
                    raise e
                verbose_proxy_logger.debug(
                    "Unable to find user in db. Error - {}".format(str(e))
                )
                pass

        ### CHECK IF ADMIN ###
        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        ### CHECK IF ADMIN ###
        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        ## Check CACHE
        valid_token: Optional[UserAPIKeyAuth] = user_api_key_cache.get_cache(
            key=hash_token(api_key)
        )
        if (
            valid_token is not None
            and isinstance(valid_token, UserAPIKeyAuth)
            and valid_token.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            # update end-user params on valid token
            valid_token.end_user_id = end_user_params.get("end_user_id")
            valid_token.end_user_tpm_limit = end_user_params.get("end_user_tpm_limit")
            valid_token.end_user_rpm_limit = end_user_params.get("end_user_rpm_limit")
            valid_token.allowed_model_region = end_user_params.get(
                "allowed_model_region"
            )

            return valid_token

        try:
            is_master_key_valid = secrets.compare_digest(api_key, master_key)  # type: ignore
        except Exception as e:
            is_master_key_valid = False

        ## VALIDATE MASTER KEY ##
        try:
            assert isinstance(master_key, str)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "Master key must be a valid string. Current type={}".format(
                        type(master_key)
                    )
                },
            )

        if is_master_key_valid:
            _user_api_key_obj = UserAPIKeyAuth(
                api_key=master_key,
                user_role=LitellmUserRoles.PROXY_ADMIN,
                user_id=litellm_proxy_admin_name,
                **end_user_params,
            )
            await user_api_key_cache.async_set_cache(
                key=hash_token(master_key), value=_user_api_key_obj
            )

            return _user_api_key_obj

        ## IF it's not a master key
        ## Route should not be in master_key_only_routes
        if route in LiteLLMRoutes.master_key_only_routes.value:
            raise Exception(
                f"Tried to access route={route}, which is only for MASTER KEY"
            )

        ## Check DB
        if isinstance(
            api_key, str
        ):  # if generated token, make sure it starts with sk-.
            assert api_key.startswith("sk-")  # prevent token hashes from being used

        if (
            prisma_client is None and custom_db_client is None
        ):  # if both master key + user key submitted, and user key != master key, and no db connected, raise an error
            raise Exception("No connected db.")

        ## check for cache hit (In-Memory Cache)
        original_api_key = api_key  # (Patch: For DynamoDB Backwards Compatibility)
        if api_key.startswith("sk-"):
            api_key = hash_token(token=api_key)
        valid_token: Optional[UserAPIKeyAuth] = user_api_key_cache.get_cache(  # type: ignore
            key=api_key
        )
        if valid_token is None:
            ## check db
            verbose_proxy_logger.debug("api key: %s", api_key)
            if prisma_client is not None:
                _valid_token: Optional[BaseModel] = await prisma_client.get_data(
                    token=api_key, table_name="combined_view"
                )
                if _valid_token is not None:
                    valid_token = UserAPIKeyAuth(
                        **_valid_token.model_dump(exclude_none=True)
                    )
            verbose_proxy_logger.debug("Token from db: %s", valid_token)
        elif valid_token is not None and isinstance(valid_token, UserAPIKeyAuth):
            verbose_proxy_logger.debug("API Key Cache Hit!")

            # update end-user params on valid token
            # These can change per request - it's important to update them here
            valid_token.end_user_id = end_user_params.get("end_user_id")
            valid_token.end_user_tpm_limit = end_user_params.get("end_user_tpm_limit")
            valid_token.end_user_rpm_limit = end_user_params.get("end_user_rpm_limit")
            valid_token.allowed_model_region = end_user_params.get(
                "allowed_model_region"
            )

        user_id_information: Optional[List] = None
        if valid_token is not None:
            # Got Valid Token from Cache, DB
            # Run checks for
            # 1. If token can call model
            # 2. If user_id for this token is in budget
            # 3. If the user spend within their own team is within budget
            # 4. If 'user' passed to /chat/completions, /embeddings endpoint is in budget
            # 5. If token is expired
            # 6. If token spend is under Budget for the token
            # 7. If token spend per model is under budget per model
            # 8. If token spend is under team budget
            # 9. If team spend is under team budget

            # Check 1. If token can call model
            _model_alias_map = {}
            if (
                hasattr(valid_token, "team_model_aliases")
                and valid_token.team_model_aliases is not None
            ):
                _model_alias_map = {
                    **valid_token.aliases,
                    **valid_token.team_model_aliases,
                }
            else:
                _model_alias_map = {**valid_token.aliases}
            litellm.model_alias_map = _model_alias_map
            config = valid_token.config
            if config != {}:
                model_list = config.get("model_list", [])
                llm_model_list = model_list
                verbose_proxy_logger.debug(
                    f"\n new llm router model list {llm_model_list}"
                )
            if (
                len(valid_token.models) == 0
            ):  # assume an empty model list means all models are allowed to be called
                pass
            elif (
                isinstance(valid_token.models, list)
                and "all-team-models" in valid_token.models
            ):
                # Do not do any validation at this step
                # the validation will occur when checking the team has access to this model
                pass
            else:
                try:
                    data = await request.json()
                except json.JSONDecodeError:
                    data = {}  # Provide a default value, such as an empty dictionary
                model = data.get("model", None)
                if model in litellm.model_alias_map:
                    model = litellm.model_alias_map[model]

                ## check if model in allowed model names
                verbose_proxy_logger.debug(
                    f"LLM Model List pre access group check: {llm_model_list}"
                )
                from collections import defaultdict

                access_groups = defaultdict(list)
                if llm_model_list is not None:
                    for m in llm_model_list:
                        for group in m.get("model_info", {}).get("access_groups", []):
                            model_name = m["model_name"]
                            access_groups[group].append(model_name)

                models_in_current_access_groups = []
                if (
                    len(access_groups) > 0
                ):  # check if token contains any model access groups
                    for idx, m in enumerate(
                        valid_token.models
                    ):  # loop token models, if any of them are an access group add the access group
                        if m in access_groups:
                            # if it is an access group we need to remove it from valid_token.models
                            models_in_group = access_groups[m]
                            models_in_current_access_groups.extend(models_in_group)

                # Filter out models that are access_groups
                filtered_models = [
                    m for m in valid_token.models if m not in access_groups
                ]

                filtered_models += models_in_current_access_groups
                verbose_proxy_logger.debug(
                    f"model: {model}; allowed_models: {filtered_models}"
                )
                if (
                    model is not None
                    and model not in filtered_models
                    and "*" not in filtered_models
                ):
                    raise ValueError(
                        f"API Key not allowed to access model. This token can only access models={valid_token.models}. Tried to access {model}"
                    )
                valid_token.models = filtered_models
                verbose_proxy_logger.debug(
                    f"filtered allowed_models: {filtered_models}; valid_token.models: {valid_token.models}"
                )

            # Check 2. If user_id for this token is in budget
            if valid_token.user_id is not None:
                user_id_list = [valid_token.user_id]
                for id in user_id_list:
                    value = user_api_key_cache.get_cache(key=id)
                    if value is not None:
                        if user_id_information is None:
                            user_id_information = []
                        user_id_information.append(value)
                if user_id_information is None or (
                    isinstance(user_id_information, list)
                    and len(user_id_information) < 1
                ):
                    if prisma_client is not None:
                        user_id_information = await prisma_client.get_data(
                            user_id_list=[
                                valid_token.user_id,
                            ],
                            table_name="user",
                            query_type="find_all",
                        )
                        if user_id_information is not None:
                            for _id in user_id_information:
                                await user_api_key_cache.async_set_cache(
                                    key=_id["user_id"],
                                    value=_id,
                                )

                verbose_proxy_logger.debug(
                    f"user_id_information: {user_id_information}"
                )

                if user_id_information is not None:
                    if isinstance(user_id_information, list):
                        ## Check if user in budget
                        for _user in user_id_information:
                            if _user is None:
                                continue
                            assert isinstance(_user, dict)
                            # check if user is admin #

                            # Token exists, not expired now check if its in budget for the user
                            user_max_budget = _user.get("max_budget", None)
                            user_current_spend = _user.get("spend", None)

                            verbose_proxy_logger.debug(
                                f"user_id: {_user.get('user_id', None)}; user_max_budget: {user_max_budget}; user_current_spend: {user_current_spend}"
                            )

                            if (
                                user_max_budget is not None
                                and user_current_spend is not None
                            ):
                                call_info = CallInfo(
                                    token=valid_token.token,
                                    spend=user_current_spend,
                                    max_budget=user_max_budget,
                                    user_id=_user.get("user_id", None),
                                    user_email=_user.get("user_email", None),
                                    key_alias=valid_token.key_alias,
                                )
                                asyncio.create_task(
                                    proxy_logging_obj.budget_alerts(
                                        type="user_budget",
                                        user_info=call_info,
                                    )
                                )

                                _user_id = _user.get("user_id", None)
                                if user_current_spend > user_max_budget:
                                    raise Exception(
                                        f"ExceededBudget: User {_user_id} has exceeded their budget. Current spend: {user_current_spend}; Max Budget: {user_max_budget}"
                                    )
                    else:
                        # Token exists, not expired now check if its in budget for the user
                        user_max_budget = getattr(
                            user_id_information, "max_budget", None
                        )
                        user_current_spend = getattr(user_id_information, "spend", None)

                        if (
                            user_max_budget is not None
                            and user_current_spend is not None
                        ):
                            call_info = CallInfo(
                                token=valid_token.token,
                                spend=user_current_spend,
                                max_budget=user_max_budget,
                                user_id=getattr(user_id_information, "user_id", None),
                                user_email=getattr(
                                    user_id_information, "user_email", None
                                ),
                                key_alias=valid_token.key_alias,
                            )
                            asyncio.create_task(
                                proxy_logging_obj.budget_alerts(
                                    type="user_budget",
                                    user_info=call_info,
                                )
                            )

                            if user_current_spend > user_max_budget:
                                raise Exception(
                                    f"ExceededBudget: User {valid_token.user_id} has exceeded their budget. Current spend: {user_current_spend}; Max Budget: {user_max_budget}"
                                )
            # Check 3. Check if user is in their team budget
            if valid_token.team_member_spend is not None:
                if prisma_client is not None:

                    _cache_key = f"{valid_token.team_id}_{valid_token.user_id}"

                    team_member_info = await user_api_key_cache.async_get_cache(
                        key=_cache_key
                    )
                    if team_member_info is None:
                        team_member_info = (
                            await prisma_client.db.litellm_teammembership.find_first(
                                where={
                                    "user_id": valid_token.user_id,
                                    "team_id": valid_token.team_id,
                                },  # type: ignore
                                include={"litellm_budget_table": True},
                            )
                        )
                        await user_api_key_cache.async_set_cache(
                            key=_cache_key,
                            value=team_member_info,
                        )

                    if (
                        team_member_info is not None
                        and team_member_info.litellm_budget_table is not None
                    ):
                        team_member_budget = (
                            team_member_info.litellm_budget_table.max_budget
                        )
                        if team_member_budget is not None and team_member_budget > 0:
                            if valid_token.team_member_spend > team_member_budget:
                                raise Exception(
                                    f"ExceededBudget: Crossed spend within team. UserID: {valid_token.user_id}, in team {valid_token.team_id} has exceeded their budget. Current spend: {valid_token.team_member_spend}; Max Budget: {team_member_budget}"
                                )

            # Check 3. If token is expired
            if valid_token.expires is not None:
                current_time = datetime.now(timezone.utc)
                expiry_time = datetime.fromisoformat(valid_token.expires)
                if (
                    expiry_time.tzinfo is None
                    or expiry_time.tzinfo.utcoffset(expiry_time) is None
                ):
                    expiry_time = expiry_time.replace(tzinfo=timezone.utc)
                verbose_proxy_logger.debug(
                    f"Checking if token expired, expiry time {expiry_time} and current time {current_time}"
                )
                if expiry_time < current_time:
                    # Token exists but is expired.
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                    )

            # Check 4. Token Spend is under budget
            if valid_token.spend is not None and valid_token.max_budget is not None:

                ####################################
                # collect information for alerting #
                ####################################

                user_email: Optional[str] = None
                # Check if the token has any user id information
                if user_id_information is not None and len(user_id_information) > 0:
                    specific_user_id_information = user_id_information[0]
                    _user_email = specific_user_id_information.get("user_email", None)
                    if _user_email is not None:
                        user_email = str(_user_email)

                call_info = CallInfo(
                    token=valid_token.token,
                    spend=valid_token.spend,
                    max_budget=valid_token.max_budget,
                    user_id=valid_token.user_id,
                    team_id=valid_token.team_id,
                    user_email=user_email,
                    key_alias=valid_token.key_alias,
                )
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        type="token_budget",
                        user_info=call_info,
                    )
                )

                ####################################
                # collect information for alerting #
                ####################################

                if valid_token.spend >= valid_token.max_budget:
                    raise Exception(
                        f"ExceededTokenBudget: Current spend for token: {valid_token.spend}; Max Budget for Token: {valid_token.max_budget}"
                    )

            # Check 5. Token Model Spend is under Model budget
            max_budget_per_model = valid_token.model_max_budget

            if (
                max_budget_per_model is not None
                and isinstance(max_budget_per_model, dict)
                and len(max_budget_per_model) > 0
            ):
                current_model = request_data.get("model")
                ## GET THE SPEND FOR THIS MODEL
                twenty_eight_days_ago = datetime.now() - timedelta(days=28)
                model_spend = await prisma_client.db.litellm_spendlogs.group_by(
                    by=["model"],
                    sum={"spend": True},
                    where={
                        "AND": [
                            {"api_key": valid_token.token},
                            {"startTime": {"gt": twenty_eight_days_ago}},
                            {"model": current_model},
                        ]
                    },  # type: ignore
                )
                if (
                    len(model_spend) > 0
                    and max_budget_per_model.get(current_model, None) is not None
                ):
                    if (
                        model_spend[0]["model"] == current_model
                        and model_spend[0]["_sum"]["spend"]
                        >= max_budget_per_model[current_model]
                    ):
                        current_model_spend = model_spend[0]["_sum"]["spend"]
                        current_model_budget = max_budget_per_model[current_model]
                        raise Exception(
                            f"ExceededModelBudget: Current spend for model: {current_model_spend}; Max Budget for Model: {current_model_budget}"
                        )

            # Check 6. Team spend is under Team budget
            if (
                hasattr(valid_token, "team_spend")
                and valid_token.team_spend is not None
                and hasattr(valid_token, "team_max_budget")
                and valid_token.team_max_budget is not None
            ):
                call_info = CallInfo(
                    token=valid_token.token,
                    spend=valid_token.team_spend,
                    max_budget=valid_token.team_max_budget,
                    user_id=valid_token.user_id,
                )
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        type="team_budget",
                        user_info=call_info,
                    )
                )

                if valid_token.team_spend >= valid_token.team_max_budget:
                    raise Exception(
                        f"ExceededTokenBudget: Current Team Spend: {valid_token.team_spend}; Max Budget for Team: {valid_token.team_max_budget}"
                    )

            # Check 8: Additional Common Checks across jwt + key auth
            _team_obj = LiteLLM_TeamTable(
                team_id=valid_token.team_id,
                max_budget=valid_token.team_max_budget,
                spend=valid_token.team_spend,
                tpm_limit=valid_token.team_tpm_limit,
                rpm_limit=valid_token.team_rpm_limit,
                blocked=valid_token.team_blocked,
                models=valid_token.team_models,
            )

            user_api_key_cache.set_cache(
                key=valid_token.team_id, value=_team_obj
            )  # save team table in cache - used for tpm/rpm limiting - tpm_rpm_limiter.py

            global_proxy_spend = None
            if (
                litellm.max_budget > 0 and prisma_client is not None
            ):  # user set proxy max budget
                # check cache
                global_proxy_spend = await user_api_key_cache.async_get_cache(
                    key="{}:spend".format(litellm_proxy_admin_name)
                )
                if global_proxy_spend is None:
                    # get from db
                    sql_query = """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""

                    response = await prisma_client.db.query_raw(query=sql_query)

                    global_proxy_spend = response[0]["total_spend"]
                    await user_api_key_cache.async_set_cache(
                        key="{}:spend".format(litellm_proxy_admin_name),
                        value=global_proxy_spend,
                    )

                if global_proxy_spend is not None:
                    call_info = CallInfo(
                        token=valid_token.token,
                        spend=global_proxy_spend,
                        max_budget=litellm.max_budget,
                        user_id=litellm_proxy_admin_name,
                        team_id=valid_token.team_id,
                    )
                    asyncio.create_task(
                        proxy_logging_obj.budget_alerts(
                            type="proxy_budget",
                            user_info=call_info,
                        )
                    )
            _ = common_checks(
                request_body=request_data,
                team_object=_team_obj,
                user_object=None,
                end_user_object=_end_user_object,
                general_settings=general_settings,
                global_proxy_spend=global_proxy_spend,
                route=route,
            )
            # Token passed all checks
            api_key = valid_token.token

            # Add hashed token to cache
            await user_api_key_cache.async_set_cache(
                key=api_key,
                value=valid_token,
            )
            valid_token_dict = valid_token.model_dump(exclude_none=True)
            valid_token_dict.pop("token", None)

            if _end_user_object is not None:
                valid_token_dict.update(end_user_params)

            _user_role = _get_user_role(user_id_information=user_id_information)

            if not _is_user_proxy_admin(user_id_information):  # if non-admin
                if route in LiteLLMRoutes.openai_routes.value:
                    pass
                elif request["route"].name in LiteLLMRoutes.openai_route_names.value:
                    pass
                elif (
                    route in LiteLLMRoutes.info_routes.value
                ):  # check if user allowed to call an info route
                    if route == "/key/info":
                        # check if user can access this route
                        query_params = request.query_params
                        key = query_params.get("key")
                        if key is not None and hash_token(token=key) != api_key:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="user not allowed to access this key's info",
                            )
                    elif route == "/user/info":
                        # check if user can access this route
                        query_params = request.query_params
                        user_id = query_params.get("user_id")
                        verbose_proxy_logger.debug(
                            f"user_id: {user_id} & valid_token.user_id: {valid_token.user_id}"
                        )
                        if user_id != valid_token.user_id:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="key not allowed to access this user's info",
                            )
                    elif route == "/model/info":
                        # /model/info just shows models user has access to
                        pass
                    elif route == "/team/info":
                        # check if key can access this team's info
                        query_params = request.query_params
                        team_id = query_params.get("team_id")
                        if team_id != valid_token.team_id:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="key not allowed to access this team's info",
                            )
                elif (
                    _has_user_setup_sso()
                    and route in LiteLLMRoutes.sso_only_routes.value
                ):
                    pass
                elif (
                    route in LiteLLMRoutes.global_spend_tracking_routes.value
                    and getattr(valid_token, "permissions", None) is not None
                    and "get_spend_routes" in getattr(valid_token, "permissions", None)
                ):

                    pass
                elif _user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
                    if route in LiteLLMRoutes.openai_routes.value:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"user not allowed to access this OpenAI routes, role= {_user_role}",
                        )
                    if route in LiteLLMRoutes.management_routes.value:
                        # the Admin Viewer is only allowed to call /user/update for their own user_id and can only update
                        if route == "/user/update":

                            # Check the Request params are valid for PROXY_ADMIN_VIEW_ONLY
                            if request_data is not None and isinstance(
                                request_data, dict
                            ):
                                _params_updated = request_data.keys()
                                for param in _params_updated:
                                    if param not in ["user_email", "password"]:
                                        raise HTTPException(
                                            status_code=status.HTTP_403_FORBIDDEN,
                                            detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route} and updating invalid param: {param}. only user_email and password can be updated",
                                        )
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"user not allowed to access this route, role= {_user_role}. Trying to access: {route}",
                            )
                else:
                    user_role = "unknown"
                    user_id = "unknown"
                    if user_id_information is not None and isinstance(
                        user_id_information, list and len(user_id_information) > 0
                    ):
                        _user = user_id_information[0]
                        user_role = _user.get("user_role", "unknown")
                        user_id = _user.get("user_id", "unknown")
                    raise Exception(
                        f"Only proxy admin can be used to generate, delete, update info for new keys/users/teams. Route={route}. Your role={user_role}. Your user_id={user_id}"
                    )

        # check if token is from litellm-ui, litellm ui makes keys to allow users to login with sso. These keys can only be used for LiteLLM UI functions
        # sso/login, ui/login, /key functions and /user functions
        # this will never be allowed to call /chat/completions
        token_team = getattr(valid_token, "team_id", None)

        if token_team is not None and token_team == "litellm-dashboard":
            # this token is only used for managing the ui
            allowed_routes = [
                "/sso",
                "/login",
                "/key/generate",
                "/key/update",
                "/key/info",
                "/config",
                "/spend",
                "/user",
                "/model/info",
                "/v2/model/info",
                "/v2/key/info",
                "/models",
                "/v1/models",
                "/global/spend",
                "/global/spend/logs",
                "/global/spend/keys",
                "/global/spend/models",
                "/global/predict/spend/logs",
                "/global/activity",
                "/health/services",
            ]
            # check if the current route startswith any of the allowed routes
            if (
                route is not None
                and isinstance(route, str)
                and any(
                    route.startswith(allowed_route) for allowed_route in allowed_routes
                )
            ):
                # Do something if the current route starts with any of the allowed routes
                pass
            else:
                if user_id_information is not None and _is_user_proxy_admin(
                    user_id_information
                ):
                    return UserAPIKeyAuth(
                        api_key=api_key,
                        user_role=LitellmUserRoles.PROXY_ADMIN,
                        **valid_token_dict,
                    )
                elif (
                    _has_user_setup_sso()
                    and route in LiteLLMRoutes.sso_only_routes.value
                ):
                    return UserAPIKeyAuth(
                        api_key=api_key, user_role="app_owner", **valid_token_dict
                    )
                else:
                    raise Exception(
                        f"This key is made for LiteLLM UI, Tried to access route: {route}. Not allowed"
                    )
        if valid_token is None:
            # No token was found when looking up in the DB
            raise Exception("Invalid token passed")
        if valid_token_dict is not None:
            if user_id_information is not None and _is_user_proxy_admin(
                user_id_information
            ):
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    **valid_token_dict,
                )
            elif _has_user_setup_sso() and route in LiteLLMRoutes.sso_only_routes.value:
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    **valid_token_dict,
                )
            else:
                return UserAPIKeyAuth(
                    api_key=api_key,
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    **valid_token_dict,
                )
        else:
            raise Exception()
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, litellm.BudgetExceededError):
            raise ProxyException(
                message=e.message, type="auth_error", param=None, code=400
            )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_401_UNAUTHORIZED,
        )


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
        from azure.keyvault.secrets import SecretClient
        from azure.identity import ClientSecretCredential

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
        _exception_string = str(_exception)[:500]

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
        verbose_proxy_logger.debug("Proxy: In track_cost_callback for: %s", kwargs)
        verbose_proxy_logger.debug(
            f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}"
        )
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        end_user_id = proxy_server_request.get("body", {}).get("user", None)
        user_id = kwargs["litellm_params"]["metadata"].get("user_api_key_user_id", None)
        team_id = kwargs["litellm_params"]["metadata"].get("user_api_key_team_id", None)
        org_id = kwargs["litellm_params"]["metadata"].get("user_api_key_org_id", None)
        key_alias = kwargs["litellm_params"]["metadata"].get("user_api_key_alias", None)
        end_user_max_budget = kwargs["litellm_params"]["metadata"].get(
            "user_api_end_user_max_budget", None
        )
        if kwargs.get("response_cost", None) is not None:
            response_cost = kwargs["response_cost"]
            user_api_key = kwargs["litellm_params"]["metadata"].get(
                "user_api_key", None
            )

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
                verbose_proxy_logger.info(
                    f"Update Key DB Call failed to execute - {str(e)}\n{traceback.format_exc()}"
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
        if disable_spend_logs == False:
            await _insert_spend_log_to_db()

        verbose_proxy_logger.debug("Runs spend update on all tables")
    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error updating Prisma database: {traceback.format_exc()}"
        )


async def update_cache(
    token: Optional[str],
    user_id: Optional[str],
    end_user_id: Optional[str],
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
            existing_spend = 0
            existing_spend_obj = LiteLLM_VerificationTokenView(token=token)
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
                    # if user does not exist in LiteLLM_UserTable, create a new user
                    existing_spend = 0
                    max_user_budget = None
                    if litellm.max_user_budget is not None:
                        max_user_budget = litellm.max_user_budget
                    existing_spend_obj = LiteLLM_UserTable(
                        user_id=_id,
                        spend=0,
                        max_budget=max_user_budget,
                        user_email=None,
                    )
                verbose_proxy_logger.debug(
                    f"_update_user_db: existing spend: {existing_spend_obj}; response_cost: {response_cost}"
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
                    user_api_key_cache.set_cache(
                        key=_id, value=existing_spend_obj.json()
                    )
            ## UPDATE GLOBAL PROXY ##
            global_proxy_spend = await user_api_key_cache.async_get_cache(
                key="{}:spend".format(litellm_proxy_admin_name)
            )
            if global_proxy_spend is None:
                await user_api_key_cache.async_set_cache(
                    key="{}:spend".format(litellm_proxy_admin_name), value=response_cost
                )
            elif response_cost is not None and global_proxy_spend is not None:
                increment = global_proxy_spend + response_cost
                await user_api_key_cache.async_set_cache(
                    key="{}:spend".format(litellm_proxy_admin_name), value=increment
                )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"An error occurred updating user cache: {str(e)}\n\n{traceback.format_exc()}"
            )

    async def _update_end_user_cache():
        if end_user_id is None or response_cost is None:
            return

        _id = "end_user_id:{}".format(end_user_id)
        try:
            # Fetch the existing cost for the given user
            existing_spend_obj = await user_api_key_cache.async_get_cache(key=_id)
            if existing_spend_obj is None:
                # if user does not exist in LiteLLM_UserTable, create a new user
                existing_spend = 0
                max_user_budget = None
                max_end_user_budget = None
                if litellm.max_end_user_budget is not None:
                    max_end_user_budget = litellm.max_end_user_budget
                existing_spend_obj = LiteLLM_EndUserTable(
                    user_id=end_user_id,
                    spend=0,
                    blocked=False,
                    litellm_budget_table=LiteLLM_BudgetTable(
                        max_budget=max_end_user_budget
                    ),
                )
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
            verbose_proxy_logger.error(
                f"An error occurred updating end user cache: {str(e)}\n\n{traceback.format_exc()}"
            )

    if token is not None and response_cost is not None:
        asyncio.create_task(_update_key_cache(token=token, response_cost=response_cost))

    asyncio.create_task(_update_user_cache())

    if end_user_id is not None:
        asyncio.create_task(_update_end_user_cache())


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
    global health_check_results, llm_model_list, health_check_interval
    while True:
        healthy_endpoints, unhealthy_endpoints = await perform_health_check(
            model_list=llm_model_list
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
        global prisma_client, general_settings, user_config_file_path
        # Load existing config
        ## DB - writes valid config to db
        """
        - Do not write restricted params like 'api_key' to the database
        - if api_key is passed, save that to the local environment or connected secret manage (maybe expose `litellm.save_secret()`)
        """
        if prisma_client is not None and (
            general_settings.get("store_model_in_db", False) == True
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
        global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, user_custom_key_generate, use_background_health_checks, health_check_interval, use_queue, custom_db_client, proxy_budget_rescheduler_max_time, proxy_budget_rescheduler_min_time, ui_access_mode, litellm_master_key_hash, proxy_batch_write_at, disable_spend_logs, prompt_injection_detection_obj, redis_usage_cache, store_model_in_db, premium_user

        # Load existing config
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
                os.environ[key] = value

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
                        print(  # noqa
                            f"{blue_color_code}Cache Type:{reset_color_code} {cache_type}"
                        )  # noqa
                        print(  # noqa
                            f"{blue_color_code}Cache Host:{reset_color_code} {cache_host}"
                        )  # noqa
                        print(  # noqa
                            f"{blue_color_code}Cache Port:{reset_color_code} {cache_port}"
                        )  # noqa
                        print(  # noqa
                            f"{blue_color_code}Cache Password:{reset_color_code} {cache_password}"
                        )
                        print()  # noqa
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
                            f"{blue_color_code}Set Cache on LiteLLM Proxy: {vars(litellm.cache.cache)}{reset_color_code}"
                        )
                elif key == "cache" and value == False:
                    pass
                elif key == "callbacks":
                    if isinstance(value, list):
                        imported_list: List[Any] = []
                        known_compatible_callbacks = list(
                            get_args(
                                litellm._custom_logger_compatible_callbacks_literal
                            )
                        )
                        for callback in value:  # ["presidio", <my-custom-callback>]
                            if (
                                isinstance(callback, str)
                                and callback in known_compatible_callbacks
                            ):
                                imported_list.append(callback)
                            elif isinstance(callback, str) and callback == "presidio":
                                from litellm.proxy.hooks.presidio_pii_masking import (
                                    _OPTIONAL_PresidioPIIMasking,
                                )

                                pii_masking_object = _OPTIONAL_PresidioPIIMasking()
                                imported_list.append(pii_masking_object)
                            elif (
                                isinstance(callback, str)
                                and callback == "llamaguard_moderations"
                            ):
                                from enterprise.enterprise_hooks.llama_guard import (
                                    _ENTERPRISE_LlamaGuard,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use Llama Guard"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                llama_guard_object = _ENTERPRISE_LlamaGuard()
                                imported_list.append(llama_guard_object)
                            elif (
                                isinstance(callback, str)
                                and callback == "openai_moderations"
                            ):
                                from enterprise.enterprise_hooks.openai_moderation import (
                                    _ENTERPRISE_OpenAI_Moderation,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use OpenAI Moderations Check"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                openai_moderations_object = (
                                    _ENTERPRISE_OpenAI_Moderation()
                                )
                                imported_list.append(openai_moderations_object)
                            elif (
                                isinstance(callback, str)
                                and callback == "lakera_prompt_injection"
                            ):
                                from enterprise.enterprise_hooks.lakera_ai import (
                                    _ENTERPRISE_lakeraAI_Moderation,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use LakeraAI Prompt Injection"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                lakera_moderations_object = (
                                    _ENTERPRISE_lakeraAI_Moderation()
                                )
                                imported_list.append(lakera_moderations_object)
                            elif (
                                isinstance(callback, str)
                                and callback == "google_text_moderation"
                            ):
                                from enterprise.enterprise_hooks.google_text_moderation import (
                                    _ENTERPRISE_GoogleTextModeration,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use Google Text Moderation"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                google_text_moderation_obj = (
                                    _ENTERPRISE_GoogleTextModeration()
                                )
                                imported_list.append(google_text_moderation_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "llmguard_moderations"
                            ):
                                from enterprise.enterprise_hooks.llm_guard import (
                                    _ENTERPRISE_LLMGuard,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use Llm Guard"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                llm_guard_moderation_obj = _ENTERPRISE_LLMGuard()
                                imported_list.append(llm_guard_moderation_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "blocked_user_check"
                            ):
                                from enterprise.enterprise_hooks.blocked_user_list import (
                                    _ENTERPRISE_BlockedUserList,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use ENTERPRISE BlockedUser"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                blocked_user_list = _ENTERPRISE_BlockedUserList(
                                    prisma_client=prisma_client
                                )
                                imported_list.append(blocked_user_list)
                            elif (
                                isinstance(callback, str)
                                and callback == "banned_keywords"
                            ):
                                from enterprise.enterprise_hooks.banned_keywords import (
                                    _ENTERPRISE_BannedKeywords,
                                )

                                if premium_user != True:
                                    raise Exception(
                                        "Trying to use ENTERPRISE BannedKeyword"
                                        + CommonProxyErrors.not_premium_user.value
                                    )

                                banned_keywords_obj = _ENTERPRISE_BannedKeywords()
                                imported_list.append(banned_keywords_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "detect_prompt_injection"
                            ):
                                from litellm.proxy.hooks.prompt_injection_detection import (
                                    _OPTIONAL_PromptInjectionDetection,
                                )

                                prompt_injection_params = None
                                if "prompt_injection_params" in litellm_settings:
                                    prompt_injection_params_in_config = (
                                        litellm_settings["prompt_injection_params"]
                                    )
                                    prompt_injection_params = (
                                        LiteLLMPromptInjectionParams(
                                            **prompt_injection_params_in_config
                                        )
                                    )

                                prompt_injection_detection_obj = (
                                    _OPTIONAL_PromptInjectionDetection(
                                        prompt_injection_params=prompt_injection_params,
                                    )
                                )
                                imported_list.append(prompt_injection_detection_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "batch_redis_requests"
                            ):
                                from litellm.proxy.hooks.batch_redis_get import (
                                    _PROXY_BatchRedisRequests,
                                )

                                batch_redis_obj = _PROXY_BatchRedisRequests()
                                imported_list.append(batch_redis_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "azure_content_safety"
                            ):
                                from litellm.proxy.hooks.azure_content_safety import (
                                    _PROXY_AzureContentSafety,
                                )

                                azure_content_safety_params = litellm_settings[
                                    "azure_content_safety_params"
                                ]
                                for k, v in azure_content_safety_params.items():
                                    if (
                                        v is not None
                                        and isinstance(v, str)
                                        and v.startswith("os.environ/")
                                    ):
                                        azure_content_safety_params[k] = (
                                            litellm.get_secret(v)
                                        )

                                azure_content_safety_obj = _PROXY_AzureContentSafety(
                                    **azure_content_safety_params,
                                )
                                imported_list.append(azure_content_safety_obj)
                            else:
                                imported_list.append(
                                    get_instance_fn(
                                        value=callback,
                                        config_file_path=config_file_path,
                                    )
                                )
                        litellm.callbacks = imported_list  # type: ignore
                    else:
                        litellm.callbacks = [
                            get_instance_fn(
                                value=value,
                                config_file_path=config_file_path,
                            )
                        ]
                    verbose_proxy_logger.debug(
                        f"{blue_color_code} Initialized Callbacks - {litellm.callbacks} {reset_color_code}"
                    )
                elif key == "post_call_rules":
                    litellm.post_call_rules = [
                        get_instance_fn(value=value, config_file_path=config_file_path)
                    ]
                    verbose_proxy_logger.debug(
                        f"litellm.post_call_rules: {litellm.post_call_rules}"
                    )
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
                    == KeyManagementSystem.AWS_SECRET_MANAGER.value
                ):
                    ### LOAD FROM AWS SECRET MANAGER ###
                    load_aws_secret_manager(use_aws_secret_manager=True)
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
        router = litellm.Router(**router_params)  # type:ignore
        return router, model_list, general_settings

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
                        # decode base64
                        try:
                            decoded_b64 = base64.b64decode(v)
                        except Exception as e:
                            verbose_proxy_logger.error(
                                "Error decoding value - {}".format(v)
                            )
                            continue
                        # decrypt value
                        _value = decrypt_value(value=decoded_b64, master_key=master_key)
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
                            if isinstance(v, str):
                                # decode base64
                                decoded_b64 = base64.b64decode(v)
                                # decrypt value
                                _litellm_params[k] = decrypt_value(
                                    value=decoded_b64, master_key=master_key  # type: ignore
                                )
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
                    llm_router = litellm.Router(model_list=_model_list)
                    verbose_proxy_logger.debug(f"updated llm_router: {llm_router}")
            else:
                verbose_proxy_logger.debug(f"len new_models: {len(new_models)}")
                ## DELETE MODEL LOGIC
                await self._delete_deployment(db_models=new_models)

                ## ADD MODEL LOGIC
                self._add_deployment(db_models=new_models)

        except Exception as e:
            verbose_proxy_logger.error(
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
                if v is not None:
                    decoded_b64 = base64.b64decode(v)
                    value = decrypt_value(value=decoded_b64, master_key=master_key)  # type: ignore
                    os.environ[k] = value
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
                and general_settings["alerting"] is not None
                and isinstance(general_settings["alerting"], list)
                and _general_settings["alerting"] is not None
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
            new_models = await prisma_client.db.litellm_proxymodeltable.find_many()
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
            verbose_proxy_logger.error(
                "{}\nTraceback:{}".format(str(e), traceback.format_exc())
            )


proxy_config = ProxyConfig()


def _duration_in_seconds(duration: str):
    match = re.match(r"(\d+)([smhd]?)", duration)
    if not match:
        raise ValueError("Invalid duration format")

    value, unit = match.groups()
    value = int(value)

    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    else:
        raise ValueError("Unsupported duration unit")


async def generate_key_helper_fn(
    request_type: Literal[
        "user", "key"
    ],  # identifies if this request is from /user/new or /key/generate
    duration: Optional[str],
    models: list,
    aliases: dict,
    config: dict,
    spend: float,
    key_max_budget: Optional[float] = None,  # key_max_budget is used to Budget Per key
    key_budget_duration: Optional[str] = None,
    budget_id: Optional[float] = None,  # budget id <-> LiteLLM_BudgetTable
    soft_budget: Optional[
        float
    ] = None,  # soft_budget is used to set soft Budgets Per user
    max_budget: Optional[float] = None,  # max_budget is used to Budget Per user
    budget_duration: Optional[str] = None,  # max_budget is used to Budget Per user
    token: Optional[str] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    user_email: Optional[str] = None,
    user_role: Optional[str] = None,
    max_parallel_requests: Optional[int] = None,
    metadata: Optional[dict] = {},
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    query_type: Literal["insert_data", "update_data"] = "insert_data",
    update_key_values: Optional[dict] = None,
    key_alias: Optional[str] = None,
    allowed_cache_controls: Optional[list] = [],
    permissions: Optional[dict] = {},
    model_max_budget: Optional[dict] = {},
    teams: Optional[list] = None,
    organization_id: Optional[str] = None,
    table_name: Optional[Literal["key", "user"]] = None,
    send_invite_email: Optional[bool] = None,
):
    global prisma_client, custom_db_client, user_api_key_cache, litellm_proxy_admin_name, premium_user

    if prisma_client is None and custom_db_client is None:
        raise Exception(
            f"Connect Proxy to database to generate keys - https://docs.litellm.ai/docs/proxy/virtual_keys "
        )

    if token is None:
        token = f"sk-{secrets.token_urlsafe(16)}"

    if duration is None:  # allow tokens that never expire
        expires = None
    else:
        duration_s = _duration_in_seconds(duration=duration)
        expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    if key_budget_duration is None:  # one-time budget
        key_reset_at = None
    else:
        duration_s = _duration_in_seconds(duration=key_budget_duration)
        key_reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    if budget_duration is None:  # one-time budget
        reset_at = None
    else:
        duration_s = _duration_in_seconds(duration=budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    permissions_json = json.dumps(permissions)
    metadata_json = json.dumps(metadata)
    model_max_budget_json = json.dumps(model_max_budget)
    user_role = user_role
    tpm_limit = tpm_limit
    rpm_limit = rpm_limit
    allowed_cache_controls = allowed_cache_controls

    try:
        # Create a new verification token (you may want to enhance this logic based on your needs)
        user_data = {
            "max_budget": max_budget,
            "user_email": user_email,
            "user_id": user_id,
            "team_id": team_id,
            "organization_id": organization_id,
            "user_role": user_role,
            "spend": spend,
            "models": models,
            "max_parallel_requests": max_parallel_requests,
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "budget_duration": budget_duration,
            "budget_reset_at": reset_at,
            "allowed_cache_controls": allowed_cache_controls,
        }
        if teams is not None:
            user_data["teams"] = teams
        key_data = {
            "token": token,
            "key_alias": key_alias,
            "expires": expires,
            "models": models,
            "aliases": aliases_json,
            "config": config_json,
            "spend": spend,
            "max_budget": key_max_budget,
            "user_id": user_id,
            "team_id": team_id,
            "max_parallel_requests": max_parallel_requests,
            "metadata": metadata_json,
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "budget_duration": key_budget_duration,
            "budget_reset_at": key_reset_at,
            "allowed_cache_controls": allowed_cache_controls,
            "permissions": permissions_json,
            "model_max_budget": model_max_budget_json,
            "budget_id": budget_id,
        }

        if (
            litellm.get_secret("DISABLE_KEY_NAME", False) == True
        ):  # allow user to disable storing abbreviated key name (shown in UI, to help figure out which key spent how much)
            pass
        else:
            key_data["key_name"] = f"sk-...{token[-4:]}"
        saved_token = copy.deepcopy(key_data)
        if isinstance(saved_token["aliases"], str):
            saved_token["aliases"] = json.loads(saved_token["aliases"])
        if isinstance(saved_token["config"], str):
            saved_token["config"] = json.loads(saved_token["config"])
        if isinstance(saved_token["metadata"], str):
            saved_token["metadata"] = json.loads(saved_token["metadata"])
        if isinstance(saved_token["permissions"], str):
            if (
                "get_spend_routes" in saved_token["permissions"]
                and premium_user != True
            ):
                raise ValueError(
                    "get_spend_routes permission is only available for LiteLLM Enterprise users"
                )

            saved_token["permissions"] = json.loads(saved_token["permissions"])
        if isinstance(saved_token["model_max_budget"], str):
            saved_token["model_max_budget"] = json.loads(
                saved_token["model_max_budget"]
            )

        if saved_token.get("expires", None) is not None and isinstance(
            saved_token["expires"], datetime
        ):
            saved_token["expires"] = saved_token["expires"].isoformat()
        if prisma_client is not None:
            if (
                table_name is None or table_name == "user"
            ):  # do not auto-create users for `/key/generate`
                ## CREATE USER (If necessary)
                if query_type == "insert_data":
                    user_row = await prisma_client.insert_data(
                        data=user_data, table_name="user"
                    )
                    ## use default user model list if no key-specific model list provided
                    if len(user_row.models) > 0 and len(key_data["models"]) == 0:  # type: ignore
                        key_data["models"] = user_row.models
                elif query_type == "update_data":
                    user_row = await prisma_client.update_data(
                        data=user_data,
                        table_name="user",
                        update_key_values=update_key_values,
                    )
            if user_id == litellm_proxy_budget_name or (
                table_name is not None and table_name == "user"
            ):
                # do not create a key for litellm_proxy_budget_name or if table name is set to just 'user'
                # we only need to ensure this exists in the user table
                # the LiteLLM_VerificationToken table will increase in size if we don't do this check
                return user_data

            ## CREATE KEY
            verbose_proxy_logger.debug("prisma_client: Creating Key= %s", key_data)
            create_key_response = await prisma_client.insert_data(
                data=key_data, table_name="key"
            )
            key_data["token_id"] = getattr(create_key_response, "token", None)
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal Server Error."},
        )

    # Add budget related info in key_data - this ensures it's returned
    key_data["budget_id"] = budget_id

    if request_type == "user":
        # if this is a /user/new request update the key_date with user_data fields
        key_data.update(user_data)
    return key_data


async def delete_verification_token(tokens: List, user_id: Optional[str] = None):
    global prisma_client
    try:
        if prisma_client:
            # Assuming 'db' is your Prisma Client instance
            # check if admin making request - don't filter by user-id
            if user_id == litellm_proxy_admin_name:
                deleted_tokens = await prisma_client.delete_data(tokens=tokens)
            # else
            else:
                deleted_tokens = await prisma_client.delete_data(
                    tokens=tokens, user_id=user_id
                )
                _num_deleted_tokens = deleted_tokens.get("deleted_keys", 0)
                if _num_deleted_tokens != len(tokens):
                    raise Exception(
                        "Failed to delete all tokens. Tried to delete tokens that don't belong to user: "
                        + str(user_id)
                    )
        else:
            raise Exception("DB not connected. prisma_client is None")
    except Exception as e:
        traceback.print_exc()
        raise e
    return deleted_tokens


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
    generate_feedback_box()
    user_model = model
    user_debug = debug
    if debug == True:  # this needs to be first, so users can see Router init debugg
        from litellm._logging import (
            verbose_router_logger,
            verbose_proxy_logger,
            verbose_logger,
        )
        import logging

        # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS
        verbose_logger.setLevel(level=logging.INFO)  # sets package logs to info
        verbose_router_logger.setLevel(level=logging.INFO)  # set router logs to info
        verbose_proxy_logger.setLevel(level=logging.INFO)  # set proxy logs to info
    if detailed_debug == True:
        from litellm._logging import (
            verbose_router_logger,
            verbose_proxy_logger,
            verbose_logger,
        )
        import logging

        verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
        verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
        verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug
    elif debug == False and detailed_debug == False:
        # users can control proxy debugging using env variable = 'LITELLM_LOG'
        litellm_log_setting = os.environ.get("LITELLM_LOG", "")
        if litellm_log_setting != None:
            if litellm_log_setting.upper() == "INFO":
                from litellm._logging import verbose_router_logger, verbose_proxy_logger
                import logging

                # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS

                verbose_router_logger.setLevel(
                    level=logging.INFO
                )  # set router logs to info
                verbose_proxy_logger.setLevel(
                    level=logging.INFO
                )  # set proxy logs to info
            elif litellm_log_setting.upper() == "DEBUG":
                from litellm._logging import verbose_router_logger, verbose_proxy_logger
                import logging

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


async def async_data_generator(
    response, user_api_key_dict: UserAPIKeyAuth, request_data: dict
):
    verbose_proxy_logger.debug("inside generator")
    try:
        start_time = time.time()
        async for chunk in response:

            ### CALL HOOKS ### - modify outgoing data
            chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                user_api_key_dict=user_api_key_dict, response=chunk
            )

            chunk = chunk.model_dump_json(exclude_none=True)
            try:
                yield f"data: {chunk}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        # Streaming is done, yield the [DONE] chunk
        done_message = "[DONE]"
        yield f"data: {done_message}\n\n"
    except Exception as e:
        traceback.print_exc()
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_data,
        )
        verbose_proxy_logger.debug(
            f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
        )
        router_model_names = llm_router.model_names if llm_router is not None else []
        if user_debug:
            traceback.print_exc()

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


def parse_cache_control(cache_control):
    cache_dict = {}
    directives = cache_control.split(", ")

    for directive in directives:
        if "=" in directive:
            key, value = directive.split("=")
            cache_dict[key] = value
        else:
            cache_dict[directive] = True

    return cache_dict


def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    verbose_proxy_logger.debug("Backing off... this was attempt # %s", details["tries"])


@router.on_event("startup")
async def startup_event():
    global prisma_client, master_key, use_background_health_checks, llm_router, llm_model_list, general_settings, proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time, litellm_proxy_admin_name, db_writer_client, store_model_in_db
    import json

    ### LOAD MASTER KEY ###
    # check if master key set in environment - load from there
    master_key = litellm.get_secret("LITELLM_MASTER_KEY", None)
    # check if DATABASE_URL in environment - load from there
    if prisma_client is None:
        prisma_setup(database_url=os.getenv("DATABASE_URL"))

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
    else:
        # if not, assume it's a json string
        worker_config = json.loads(os.getenv("WORKER_CONFIG"))
        await initialize(**worker_config)

    ## COST TRACKING ##
    cost_tracking()

    ## Error Tracking ##
    error_tracking()

    ## UPDATE SLACK ALERTING ##
    proxy_logging_obj.slack_alerting_instance.update_values(llm_router=llm_router)

    db_writer_client = HTTPHandler()

    proxy_logging_obj._init_litellm_callbacks()  # INITIALIZE LITELLM CALLBACKS ON SERVER STARTUP <- do this to catch any logging errors on startup, not when calls are being made

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
        # add master key to db
        if os.getenv("PROXY_ADMIN_ID", None) is not None:
            litellm_proxy_admin_name = os.getenv(
                "PROXY_ADMIN_ID", litellm_proxy_admin_name
            )
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

    verbose_proxy_logger.debug(
        f"custom_db_client client {custom_db_client}. Master_key: {master_key}"
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
    max_tries=litellm.num_retries or 3,  # maximum number of retries
    max_time=litellm.request_timeout or 60,  # maximum total time to retry for
    on_backoff=on_backoff,  # specifying the function to call on backoff
    giveup=lambda e: not (
        isinstance(e, ProxyException)
        and getattr(e, "message", None) is not None
        and isinstance(e.message, str)
        and "Max parallel request limit reached" in e.message
    ),  # the result of the logical expression is on the second position
)
async def chat_completion(
    request: Request,
    fastapi_response: Response,
    model: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global general_settings, user_debug, proxy_logging_obj, llm_model_list
    data = {}
    try:
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except:
            data = json.loads(body_str)

        # Azure OpenAI only: check if user passed api-version
        query_params = dict(request.query_params)
        if "api-version" in query_params:
            data["api_version"] = query_params["api-version"]

        # Include original request and headers in the data
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        ## Cache Controls
        headers = request.headers
        verbose_proxy_logger.debug("Request Headers: %s", headers)
        cache_control_header = headers.get("Cache-Control", None)
        if cache_control_header:
            cache_dict = parse_cache_control(cache_control_header)
            data["ttl"] = cache_dict.get("s-maxage")

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
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["user_api_end_user_max_budget"] = getattr(
            user_api_key_dict, "end_user_max_budget", None
        )
        data["metadata"]["litellm_api_version"] = version

        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_org_id"] = user_api_key_dict.org_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                _is_valid_team_configs(
                    team_id=team_id, team_config=team_config, request_data=data
                )
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        ### END-USER SPECIFIC PARAMS ###
        if user_api_key_dict.allowed_model_region is not None:
            data["allowed_model_region"] = user_api_key_dict.allowed_model_region

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

        ## LOGGING OBJECT ## - initialize logging object for logging success/failure events for call
        data["litellm_call_id"] = str(uuid.uuid4())
        logging_obj, data = litellm.utils.function_setup(
            original_function="acompletion",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.now(),
            **data,
        )

        data["litellm_logging_obj"] = logging_obj

        ### CALL HOOKS ### - modify/reject incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="completion"
        )

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
        router_model_names = llm_router.model_names if llm_router is not None else []
        # skip router if user passed their key
        if "api_key" in data:
            tasks.append(litellm.acompletion(**data))
        elif "," in data["model"] and llm_router is not None:
            if (
                data.get("fastest_response", None) is not None
                and data["fastest_response"] == True
            ):
                tasks.append(llm_router.abatch_completion_fastest_response(**data))
            else:
                _models_csv_string = data.pop("model")
                _models = [model.strip() for model in _models_csv_string.split(",")]
                tasks.append(llm_router.abatch_completion(models=_models, **data))
        elif "user_config" in data:
            # initialize a new router instance. make request using this Router
            router_config = data.pop("user_config")
            user_router = litellm.Router(**router_config)
            tasks.append(user_router.acompletion(**data))
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            tasks.append(llm_router.acompletion(**data))
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            tasks.append(llm_router.acompletion(**data))
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            tasks.append(llm_router.acompletion(**data, specific_deployment=True))
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            tasks.append(llm_router.acompletion(**data))
        elif user_model is not None:  # `litellm --model <your-model-name>`
            tasks.append(litellm.acompletion(**data))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "chat_completion: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

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
        fastest_response_batch_completion = hidden_params.get(
            "fastest_response_batch_completion", None
        )

        # Post Call Processing
        if llm_router is not None:
            data["deployment"] = llm_router.get_deployment(model_id=model_id)
        data["litellm_status"] = "success"  # used for alerting

        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
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

        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
            )
        )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            user_api_key_dict=user_api_key_dict, response=response
        )

        return response
    except RejectedRequestError as e:
        _data = e.request_data
        _data["litellm_status"] = "fail"  # used for alerting
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
        data["litellm_status"] = "fail"  # used for alerting
        traceback.print_exc()
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
        if user_debug:
            traceback.print_exc()

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
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
    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
    data = {}
    try:
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except:
            data = json.loads(body_str)

        data["user"] = data.get("user", user_api_key_dict.user_id)
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["litellm_api_version"] = version
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["endpoint"] = str(request.url)

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
            llm_response = asyncio.create_task(litellm.atext_completion(**data))
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.atext_completion(**data))
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            llm_response = asyncio.create_task(llm_router.atext_completion(**data))
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            llm_response = asyncio.create_task(
                llm_router.atext_completion(**data, specific_deployment=True)
            )
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            llm_response = asyncio.create_task(llm_router.atext_completion(**data))
        elif user_model is not None:  # `litellm --model <your-model-name>`
            llm_response = asyncio.create_task(litellm.atext_completion(**data))
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

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

        verbose_proxy_logger.debug("final response: %s", response)
        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
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
        fastapi_response.headers.update(
            get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
            )
        )

        return response
    except RejectedRequestError as e:
        _data = e.request_data
        _data["litellm_status"] = "fail"  # used for alerting
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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.debug("EXCEPTION RAISED IN PROXY MAIN.PY")
        litellm_debug_info = getattr(e, "litellm_debug_info", "")
        verbose_proxy_logger.debug(
            "\033[1;31mAn error occurred: %s %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
            litellm_debug_info,
        )
        traceback.print_exc()
        error_traceback = traceback.format_exc()
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
    global proxy_logging_obj
    data: Any = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        data["model"] = (
            general_settings.get("embedding_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["litellm_api_version"] = version
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        router_model_names = llm_router.model_names if llm_router is not None else []
        if (
            "input" in data
            and isinstance(data["input"], list)
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

        ## ROUTE TO CORRECT ENDPOINT ##
        # skip router if user passed their key
        if "api_key" in data:
            response = await litellm.aembedding(**data)
        elif "user_config" in data:
            # initialize a new router instance. make request using this Router
            router_config = data.pop("user_config")
            user_router = litellm.Router(**router_config)
            response = await user_router.aembedding(**data)
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            response = await llm_router.aembedding(**data)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            response = await llm_router.aembedding(
                **data
            )  # ensure this goes the llm_router, router will do the correct alias mapping
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aembedding(**data, specific_deployment=True)
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aembedding(**data)
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.aembedding(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "embeddings: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        litellm_debug_info = getattr(e, "litellm_debug_info", "")
        verbose_proxy_logger.debug(
            "\033[1;31mAn error occurred: %s %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
            litellm_debug_info,
        )
        traceback.print_exc()
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
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        data["model"] = (
            general_settings.get("image_generation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

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
        # skip router if user passed their key
        if "api_key" in data:
            response = await litellm.aimage_generation(**data)
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            response = await llm_router.aimage_generation(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aimage_generation(
                **data, specific_deployment=True
            )
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            response = await llm_router.aimage_generation(
                **data
            )  # ensure this goes the llm_router, router will do the correct alias mapping
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aimage_generation(**data)
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.aimage_generation(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "image_generation: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if user_model:
            data["model"] = user_model

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
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
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        router_model_names = llm_router.model_names if llm_router is not None else []

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )

        ## ROUTE TO CORRECT ENDPOINT ##
        # skip router if user passed their key
        if "api_key" in data:
            response = await litellm.aspeech(**data)
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            response = await llm_router.aspeech(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aspeech(**data, specific_deployment=True)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            response = await llm_router.aspeech(
                **data
            )  # ensure this goes the llm_router, router will do the correct alias mapping
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aspeech(**data)
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.aspeech(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "audio_speech: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

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
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            fastest_response_batch_completion=None,
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
        traceback.print_exc()
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
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        data["model"] = (
            general_settings.get("moderation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
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
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)
        data["metadata"]["file_name"] = file.filename

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        router_model_names = llm_router.model_names if llm_router is not None else []

        assert (
            file.filename is not None
        )  # make sure filename passed in (needed for type)

        _original_filename = file.filename
        file_extension = os.path.splitext(file.filename)[1]
        # rename the file to a random hash file name -> we eventuall remove the file and don't want to remove any local files
        file.filename = f"tmp-request" + str(uuid.uuid4()) + file_extension

        # IMP - Asserts that we've renamed the uploaded file, since we run os.remove(file.filename), we should rename the original file
        assert file.filename != _original_filename

        with open(file.filename, "wb+") as f:
            f.write(await file.read())
            try:
                data["file"] = open(file.filename, "rb")
                ### CALL HOOKS ### - modify incoming data / reject request before calling the model
                data = await proxy_logging_obj.pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    call_type="audio_transcription",
                )

                ## ROUTE TO CORRECT ENDPOINT ##
                # skip router if user passed their key
                if "api_key" in data:
                    response = await litellm.atranscription(**data)
                elif (
                    llm_router is not None and data["model"] in router_model_names
                ):  # model in router model list
                    response = await llm_router.atranscription(**data)

                elif (
                    llm_router is not None
                    and data["model"] in llm_router.deployment_names
                ):  # model in router deployments, calling a specific deployment on the router
                    response = await llm_router.atranscription(
                        **data, specific_deployment=True
                    )
                elif (
                    llm_router is not None
                    and llm_router.model_group_alias is not None
                    and data["model"] in llm_router.model_group_alias
                ):  # model set in model_group_alias
                    response = await llm_router.atranscription(
                        **data
                    )  # ensure this goes the llm_router, router will do the correct alias mapping
                elif (
                    llm_router is not None
                    and data["model"] not in router_model_names
                    and llm_router.default_deployment is not None
                ):  # model in router deployments, calling a specific deployment on the router
                    response = await llm_router.atranscription(**data)
                elif user_model is not None:  # `litellm --model <your-model-name>`
                    response = await litellm.atranscription(**data)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "audio_transcriptions: Invalid model name passed in model="
                            + data.get("model", "")
                        },
                    )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                os.remove(file.filename)  # Delete the saved file

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data = orjson.loads(body)

        # Include original request and headers in the data
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        data["metadata"]["litellm_api_version"] = version
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
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.aget_assistants(
            custom_llm_provider="openai", client=None, **data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data = orjson.loads(body)

        # Include original request and headers in the data
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "litellm_metadata" not in data:
            data["litellm_metadata"] = {}
        data["litellm_metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
        data["litellm_metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["litellm_metadata"]["headers"] = _headers
        data["litellm_metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["litellm_metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["litellm_metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["litellm_metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["litellm_metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["litellm_metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["litellm_metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.acreate_thread(
            custom_llm_provider="openai", client=None, **data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
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
        data["metadata"]["litellm_api_version"] = version
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.aget_thread(
            custom_llm_provider="openai", thread_id=thread_id, client=None, **data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "litellm_metadata" not in data:
            data["litellm_metadata"] = {}
        data["litellm_metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["litellm_metadata"]["litellm_api_version"] = version
        data["litellm_metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["litellm_metadata"]["headers"] = _headers
        data["litellm_metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["litellm_metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["litellm_metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["litellm_metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["litellm_metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["litellm_metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["litellm_metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.a_add_message(
            custom_llm_provider="openai", thread_id=thread_id, client=None, **data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
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
        data["metadata"]["litellm_api_version"] = version
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.aget_messages(
            custom_llm_provider="openai", thread_id=thread_id, client=None, **data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
    "/v1/threads/{thread_id}/runs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["assistants"],
)
@router.get(
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
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "litellm_metadata" not in data:
            data["litellm_metadata"] = {}
        data["litellm_metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["litellm_metadata"]["litellm_api_version"] = version
        data["litellm_metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["litellm_metadata"]["headers"] = _headers
        data["litellm_metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["litellm_metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["litellm_metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["litellm_metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["litellm_metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["litellm_metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["litellm_metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        if llm_router is None:
            raise HTTPException(
                status_code=500, detail={"error": CommonProxyErrors.no_llm_router.value}
            )
        response = await llm_router.arun_thread(
            custom_llm_provider="openai", thread_id=thread_id, client=None, **data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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

#                          /v1/batches Endpoints


######################################################################
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
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        form_data = await request.form()
        data = {key: value for key, value in form_data.items() if key != "file"}

        # Include original request and headers in the data
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
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
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        _create_batch_data = CreateBatchRequest(**data)

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        response = await litellm.acreate_batch(
            custom_llm_provider="openai", **_create_batch_data
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
    "/v1/batches{batch_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
@router.get(
    "/batches{batch_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["batch"],
)
async def retrieve_batch(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
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
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        form_data = await request.form()
        data = {key: value for key, value in form_data.items() if key != "file"}

        # Include original request and headers in the data
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
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
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        _retrieve_batch_request = RetrieveBatchRequest(
            batch_id=batch_id,
        )

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        response = await litellm.aretrieve_batch(
            custom_llm_provider="openai", **_retrieve_batch_request
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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


######################################################################

#                          /v1/files Endpoints


######################################################################
@router.post(
    "/v1/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.post(
    "/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
async def create_file(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Upload a file that can be used across - Assistants API, Batch API 
    This is the equivalent of POST https://api.openai.com/v1/files

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/files/create

    Example Curl
    ```
    curl https://api.openai.com/v1/files \
        -H "Authorization: Bearer sk-1234" \
        -F purpose="batch" \
        -F file="@mydata.jsonl"

    ```
    """
    global proxy_logging_obj
    data: Dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        form_data = await request.form()
        data = {key: value for key, value in form_data.items() if key != "file"}

        # Include original request and headers in the data
        data["proxy_server_request"] = {  # type: ignore
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
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
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        _create_file_request = CreateFileRequest()

        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        response = await litellm.acreate_file(
            custom_llm_provider="openai", **_create_file_request
        )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
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
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        if data.get("user", None) is None and user_api_key_dict.user_id is not None:
            data["user"] = user_api_key_dict.user_id

        data["model"] = (
            general_settings.get("moderation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        data["metadata"]["litellm_api_version"] = version
        data["metadata"]["user_api_key_metadata"] = user_api_key_dict.metadata
        _headers = dict(request.headers)
        _headers.pop(
            "authorization", None
        )  # do not store the original `sk-..` api key in the db
        data["metadata"]["headers"] = _headers
        data["metadata"]["global_max_parallel_requests"] = general_settings.get(
            "global_max_parallel_requests", None
        )
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
        )
        data["metadata"]["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        data["metadata"]["endpoint"] = str(request.url)

        ### TEAM-SPECIFIC PARAMS ###
        if user_api_key_dict.team_id is not None:
            team_config = await proxy_config.load_team_config(
                team_id=user_api_key_dict.team_id
            )
            if len(team_config) == 0:
                pass
            else:
                team_id = team_config.pop("team_id", None)
                data["metadata"]["team_id"] = team_id
                data = {
                    **team_config,
                    **data,
                }  # add the team-specific configs to the completion call

        router_model_names = llm_router.model_names if llm_router is not None else []

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="moderation"
        )

        start_time = time.time()

        ## ROUTE TO CORRECT ENDPOINT ##
        # skip router if user passed their key
        if "api_key" in data:
            response = await litellm.amoderation(**data)
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            response = await llm_router.amoderation(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.amoderation(**data, specific_deployment=True)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            response = await llm_router.amoderation(
                **data
            )  # ensure this goes the llm_router, router will do the correct alias mapping
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.amoderation(**data)
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.amoderation(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "moderations: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

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
        data["litellm_status"] = "fail"  # used for alerting
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
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


#### KEY MANAGEMENT ####


@router.post(
    "/key/generate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GenerateKeyResponse,
)
async def generate_key_fn(
    data: GenerateKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    Authorization: Optional[str] = Header(None),
):
    """
    Generate an API key based on the provided data.

    Docs: https://docs.litellm.ai/docs/proxy/virtual_keys

    Parameters:
    - duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
    - key_alias: Optional[str] - User defined key alias
    - team_id: Optional[str] - The team id of the key
    - user_id: Optional[str] - The user id of the key
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - aliases: Optional[dict] - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
    - config: Optional[dict] - any key-specific configs, overrides config in config.yaml
    - spend: Optional[int] - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend
    - send_invite_email: Optional[bool] - Whether to send an invite email to the user_id, with the generate key
    - max_budget: Optional[float] - Specify max budget for a given key.
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - metadata: Optional[dict] - Metadata for key, store information for key. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - permissions: Optional[dict] - key-specific permissions. Currently just used for turning off pii masking (if connected). Example - {"pii": false}
    - model_max_budget: Optional[dict] - key-specific model budget in USD. Example - {"text-davinci-002": 0.5, "gpt-3.5-turbo": 0.5}. IF null or {} then no model specific budget.

    Examples:

    1. Allow users to turn on/off pii masking

    ```bash
    curl --location 'http://0.0.0.0:8000/key/generate' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "permissions": {"allow_pii_controls": true}
    }'
    ```

    Returns:
    - key: (str) The generated api key
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    """
    try:
        global user_custom_key_generate
        verbose_proxy_logger.debug("entered /key/generate")

        if user_custom_key_generate is not None:
            result = await user_custom_key_generate(data)
            decision = result.get("decision", True)
            message = result.get("message", "Authentication Failed - Custom Auth Rule")
            if not decision:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=message
                )
        # check if user set default key/generate params on config.yaml
        if litellm.default_key_generate_params is not None:
            for elem in data:
                key, value = elem
                if value is None and key in [
                    "max_budget",
                    "user_id",
                    "team_id",
                    "max_parallel_requests",
                    "tpm_limit",
                    "rpm_limit",
                    "budget_duration",
                ]:
                    setattr(
                        data, key, litellm.default_key_generate_params.get(key, None)
                    )
                elif key == "models" and value == []:
                    setattr(data, key, litellm.default_key_generate_params.get(key, []))
                elif key == "metadata" and value == {}:
                    setattr(data, key, litellm.default_key_generate_params.get(key, {}))

        # check if user set default key/generate params on config.yaml
        if litellm.upperbound_key_generate_params is not None:
            for elem in data:
                # if key in litellm.upperbound_key_generate_params, use the min of value and litellm.upperbound_key_generate_params[key]
                key, value = elem
                if (
                    value is not None
                    and getattr(litellm.upperbound_key_generate_params, key, None)
                    is not None
                ):
                    # if value is float/int
                    if key in [
                        "max_budget",
                        "max_parallel_requests",
                        "tpm_limit",
                        "rpm_limit",
                    ]:
                        if value > getattr(litellm.upperbound_key_generate_params, key):
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": f"{key} is over max limit set in config - user_value={value}; max_value={getattr(litellm.upperbound_key_generate_params, key)}"
                                },
                            )
                    elif key == "budget_duration":
                        # budgets are in 1s, 1m, 1h, 1d, 1m (30s, 30m, 30h, 30d, 30m)
                        # compare the duration in seconds and max duration in seconds
                        upperbound_budget_duration = _duration_in_seconds(
                            duration=getattr(
                                litellm.upperbound_key_generate_params, key
                            )
                        )
                        user_set_budget_duration = _duration_in_seconds(duration=value)
                        if user_set_budget_duration > upperbound_budget_duration:
                            raise HTTPException(
                                status_code=400,
                                detail={
                                    "error": f"Budget duration is over max limit set in config - user_value={user_set_budget_duration}; max_value={upperbound_budget_duration}"
                                },
                            )

        # TODO: @ishaan-jaff: Migrate all budget tracking to use LiteLLM_BudgetTable
        _budget_id = None
        if prisma_client is not None and data.soft_budget is not None:
            # create the Budget Row for the LiteLLM Verification Token
            budget_row = LiteLLM_BudgetTable(
                soft_budget=data.soft_budget,
                model_max_budget=data.model_max_budget or {},
            )
            new_budget = prisma_client.jsonify_object(
                budget_row.json(exclude_none=True)
            )

            _budget = await prisma_client.db.litellm_budgettable.create(
                data={
                    **new_budget,  # type: ignore
                    "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                    "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                }
            )
            _budget_id = getattr(_budget, "budget_id", None)
        data_json = data.json()  # type: ignore
        # if we get max_budget passed to /key/generate, then use it as key_max_budget. Since generate_key_helper_fn is used to make new users
        if "max_budget" in data_json:
            data_json["key_max_budget"] = data_json.pop("max_budget", None)
        if _budget_id is not None:
            data_json["budget_id"] = _budget_id

        if "budget_duration" in data_json:
            data_json["key_budget_duration"] = data_json.pop("budget_duration", None)

        response = await generate_key_helper_fn(
            request_type="key", **data_json, table_name="key"
        )

        response["soft_budget"] = (
            data.soft_budget
        )  # include the user-input soft budget in the response

        if data.send_invite_email is True:
            if "email" not in general_settings.get("alerting", []):
                raise ValueError(
                    "Email alerting not setup on config.yaml. Please set `alerting=['email']. \nDocs: https://docs.litellm.ai/docs/proxy/email`"
                )
            event = WebhookEvent(
                event="key_created",
                event_group="key",
                event_message=f"API Key Created",
                token=response.get("token", ""),
                spend=response.get("spend", 0.0),
                max_budget=response.get("max_budget", 0.0),
                user_id=response.get("user_id", None),
                team_id=response.get("team_id", "Default Team"),
                key_alias=response.get("key_alias", None),
            )

            # If user configured email alerting - send an Email letting their end-user know the key was created
            asyncio.create_task(
                proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                    webhook_event=event,
                )
            )

        return GenerateKeyResponse(**response)
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/key/update", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def update_key_fn(request: Request, data: UpdateKeyRequest):
    """
    Update an existing key
    """
    global prisma_client
    try:
        data_json: dict = data.json()
        key = data_json.pop("key")
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

        if "duration" in non_default_values:
            duration = non_default_values.pop("duration")
            duration_s = _duration_in_seconds(duration=duration)
            expires = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
            non_default_values["expires"] = expires

        response = await prisma_client.update_data(
            token=key, data={**non_default_values, "token": key}
        )

        # Delete - key from cache, since it's been updated!
        # key updated - a new model could have been added to this key. it should not block requests after this is done
        user_api_key_cache.delete_cache(key)
        hashed_token = hash_token(key)
        user_api_key_cache.delete_cache(hashed_token)

        return {"key": key, **response["data"]}
        # update based on remaining passed in values
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/key/delete", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def delete_key_fn(
    data: KeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a key from the key management system.

    Parameters::
    - keys (List[str]): A list of keys or hashed keys to delete. Example {"keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e17519f44683334df5291321d97b8bf1098cd490e49e215f6fea935aa28be"]}

    Returns:
    - deleted_keys (List[str]): A list of deleted keys. Example {"deleted_keys": ["sk-QWrxEynunsNpV1zT48HIrw", "837e17519f44683334df5291321d97b8bf1098cd490e49e215f6fea935aa28be"]}


    Raises:
        HTTPException: If an error occurs during key deletion.
    """
    try:
        global user_api_key_cache
        keys = data.keys
        if len(keys) == 0:
            raise ProxyException(
                message=f"No keys provided, passed in: keys={keys}",
                type="auth_error",
                param="keys",
                code=status.HTTP_400_BAD_REQUEST,
            )

        ## only allow user to delete keys they own
        user_id = user_api_key_dict.user_id
        verbose_proxy_logger.debug(
            f"user_api_key_dict.user_role: {user_api_key_dict.user_role}"
        )
        if (
            user_api_key_dict.user_role is not None
            and user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            user_id = None  # unless they're admin

        number_deleted_keys = await delete_verification_token(
            tokens=keys, user_id=user_id
        )
        verbose_proxy_logger.debug(
            f"/key/delete - deleted_keys={number_deleted_keys['deleted_keys']}"
        )

        try:
            assert len(keys) == number_deleted_keys["deleted_keys"]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Not all keys passed in were deleted. This probably means you don't have access to delete all the keys passed in. Keys passed in={len(keys)}, Deleted keys ={number_deleted_keys['deleted_keys']}"
                },
            )

        for key in keys:
            user_api_key_cache.delete_cache(key)
            # remove hash token from cache
            hashed_token = hash_token(key)
            user_api_key_cache.delete_cache(hashed_token)

        verbose_proxy_logger.debug(
            f"/keys/delete - cache after delete: {user_api_key_cache.in_memory_cache.cache_dict}"
        )

        return {"deleted_keys": keys}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/v2/key/info", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def info_key_fn_v2(
    data: Optional[KeyRequest] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieve information about a list of keys.

    **New endpoint**. Currently admin only.
    Parameters:
        keys: Optional[list] = body parameter representing the key(s) in the request
        user_api_key_dict: UserAPIKeyAuth = Dependency representing the user's API key
    Returns:
        Dict containing the key and its associated information

    Example Curl:
    ```
    curl -X GET "http://0.0.0.0:8000/key/info" \
    -H "Authorization: Bearer sk-1234" \
    -d {"keys": ["sk-1", "sk-2", "sk-3"]}
    ```
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Malformed request. No keys passed in."},
            )

        key_info = await prisma_client.get_data(
            token=data.keys, table_name="key", query_type="find_all"
        )
        filtered_key_info = []
        for k in key_info:
            try:
                k = k.model_dump()  # noqa
            except:
                # if using pydantic v1
                k = k.dict()
            filtered_key_info.append(k)
        return {"key": data.keys, "info": filtered_key_info}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.get(
    "/key/info", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def info_key_fn(
    key: Optional[str] = fastapi.Query(
        default=None, description="Key in the request parameters"
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Retrieve information about a key.
    Parameters:
        key: Optional[str] = Query parameter representing the key in the request
        user_api_key_dict: UserAPIKeyAuth = Dependency representing the user's API key
    Returns:
        Dict containing the key and its associated information

    Example Curl:
    ```
    curl -X GET "http://0.0.0.0:8000/key/info?key=sk-02Wr4IAlN3NvPXvL5JVvDA" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Curl - if no key is passed, it will use the Key Passed in Authorization Header
    ```
    curl -X GET "http://0.0.0.0:8000/key/info" \
-H "Authorization: Bearer sk-02Wr4IAlN3NvPXvL5JVvDA"
    ```
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        if key == None:
            key = user_api_key_dict.api_key
        key_info = await prisma_client.get_data(token=key)
        ## REMOVE HASHED TOKEN INFO BEFORE RETURNING ##
        try:
            key_info = key_info.model_dump()  # noqa
        except:
            # if using pydantic v1
            key_info = key_info.dict()
        key_info.pop("token")
        return {"key": key, "info": key_info}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


#### SPEND MANAGEMENT #####


@router.get(
    "/spend/keys",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def spend_key_fn():
    """
    View all keys created, ordered by spend

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/keys" \
-H "Authorization: Bearer sk-1234"
    ```
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        key_info = await prisma_client.get_data(table_name="key", query_type="find_all")
        return key_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/spend/users",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def spend_user_fn(
    user_id: Optional[str] = fastapi.Query(
        default=None,
        description="Get User Table row for user_id",
    ),
):
    """
    View all users created, ordered by spend

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/users" \
-H "Authorization: Bearer sk-1234"
    ```

    View User Table row for user_id
    ```
    curl -X GET "http://0.0.0.0:8000/spend/users?user_id=1234" \
-H "Authorization: Bearer sk-1234"
    ```
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if user_id is not None:
            user_info = await prisma_client.get_data(
                table_name="user", query_type="find_unique", user_id=user_id
            )
            return [user_info]
        else:
            user_info = await prisma_client.get_data(
                table_name="user", query_type="find_all"
            )

        return user_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/spend/tags",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def view_spend_tags(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing key spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view key spend",
    ),
):
    """
    LiteLLM Enterprise - View Spend Per Request Tag

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/tags" \
-H "Authorization: Bearer sk-1234"
    ```

    Spend with Start Date and End Date
    ```
    curl -X GET "http://0.0.0.0:8000/spend/tags?start_date=2022-01-01&end_date=2022-02-01" \
-H "Authorization: Bearer sk-1234"
    ```
    """

    from enterprise.utils import get_spend_by_tags

    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        # run the following SQL query on prisma
        """
        SELECT
        jsonb_array_elements_text(request_tags) AS individual_request_tag,
        COUNT(*) AS log_count,
        SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        GROUP BY individual_request_tag;
        """
        response = await get_spend_by_tags(
            start_date=start_date, end_date=end_date, prisma_client=prisma_client
        )

        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/tags Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/tags Error" + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/global/activity",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_activity(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of API Requests, total tokens through proxy

    {
        "daily_data": [
                const chartdata = [
                {
                date: 'Jan 22',
                api_requests: 10,
                total_tokens: 2000
                },
                {
                date: 'Jan 23',
                api_requests: 10,
                total_tokens: 12
                },
        ],
        "sum_api_requests": 20,
        "sum_total_tokens": 2012
    }
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    global prisma_client, llm_router
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT
            date_trunc('day', "startTime") AS date,
            COUNT(*) AS api_requests,
            SUM(total_tokens) AS total_tokens
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime" BETWEEN $1::date AND $2::date + interval '1 day'
        GROUP BY date_trunc('day', "startTime")
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj
        )

        if db_response is None:
            return []

        sum_api_requests = 0
        sum_total_tokens = 0
        daily_data = []
        for row in db_response:
            # cast date to datetime
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            daily_data.append(row)
            sum_api_requests += row.get("api_requests", 0)
            sum_total_tokens += row.get("total_tokens", 0)

        # sort daily_data by date
        daily_data = sorted(daily_data, key=lambda x: x["date"])

        data_to_return = {
            "daily_data": daily_data,
            "sum_api_requests": sum_api_requests,
            "sum_total_tokens": sum_total_tokens,
        }

        return data_to_return

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/activity/model",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_activity_model(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of API Requests, total tokens through proxy - Grouped by MODEL

    [
        {
            "model": "gpt-4",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    api_requests: 10,
                    total_tokens: 2000
                    },
                    {
                    date: 'Jan 23',
                    api_requests: 10,
                    total_tokens: 12
                    },
            ],
            "sum_api_requests": 20,
            "sum_total_tokens": 2012

        },
        {
            "model": "azure/gpt-4-turbo",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    api_requests: 10,
                    total_tokens: 2000
                    },
                    {
                    date: 'Jan 23',
                    api_requests: 10,
                    total_tokens: 12
                    },
            ],
            "sum_api_requests": 20,
            "sum_total_tokens": 2012

        },
    ]
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    global prisma_client, llm_router, premium_user
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT
            model_group,
            date_trunc('day', "startTime") AS date,
            COUNT(*) AS api_requests,
            SUM(total_tokens) AS total_tokens
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime" BETWEEN $1::date AND $2::date + interval '1 day'
        GROUP BY model_group, date_trunc('day', "startTime")
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj
        )
        if db_response is None:
            return []

        model_ui_data: dict = (
            {}
        )  # {"gpt-4": {"daily_data": [], "sum_api_requests": 0, "sum_total_tokens": 0}}

        for row in db_response:
            _model = row["model_group"]
            if _model not in model_ui_data:
                model_ui_data[_model] = {
                    "daily_data": [],
                    "sum_api_requests": 0,
                    "sum_total_tokens": 0,
                }
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            model_ui_data[_model]["daily_data"].append(row)
            model_ui_data[_model]["sum_api_requests"] += row.get("api_requests", 0)
            model_ui_data[_model]["sum_total_tokens"] += row.get("total_tokens", 0)

        # sort mode ui data by sum_api_requests -> get top 10 models
        model_ui_data = dict(
            sorted(
                model_ui_data.items(),
                key=lambda x: x[1]["sum_api_requests"],
                reverse=True,
            )[:10]
        )

        response = []
        for model, data in model_ui_data.items():
            _sort_daily_data = sorted(data["daily_data"], key=lambda x: x["date"])

            response.append(
                {
                    "model": model,
                    "daily_data": _sort_daily_data,
                    "sum_api_requests": data["sum_api_requests"],
                    "sum_total_tokens": data["sum_total_tokens"],
                }
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


@router.get(
    "/global/activity/exceptions/deployment",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_activity_exceptions_per_deployment(
    model_group: str = fastapi.Query(
        description="Filter by model group",
    ),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of 429 errors - Grouped by deployment

    [
        {
            "deployment": "https://azure-us-east-1.openai.azure.com/",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    num_rate_limit_exceptions: 10
                    },
                    {
                    date: 'Jan 23',
                    num_rate_limit_exceptions: 12
                    },
            ],
            "sum_num_rate_limit_exceptions": 20,

        },
        {
            "deployment": "https://azure-us-east-1.openai.azure.com/",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    num_rate_limit_exceptions: 10,
                    },
                    {
                    date: 'Jan 23',
                    num_rate_limit_exceptions: 12
                    },
            ],
            "sum_num_rate_limit_exceptions": 20,

        },
    ]
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    global prisma_client, llm_router, premium_user
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT
            api_base,
            date_trunc('day', "startTime")::date AS date,
            COUNT(*) AS num_rate_limit_exceptions
        FROM
            "LiteLLM_ErrorLogs"
        WHERE
            "startTime" >= $1::date
            AND "startTime" < ($2::date + INTERVAL '1 day')
            AND model_group = $3
            AND status_code = '429'
        GROUP BY
            api_base,
            date_trunc('day', "startTime")
        ORDER BY
            date;
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj, model_group
        )
        if db_response is None:
            return []

        model_ui_data: dict = (
            {}
        )  # {"gpt-4": {"daily_data": [], "sum_api_requests": 0, "sum_total_tokens": 0}}

        for row in db_response:
            _model = row["api_base"]
            if _model not in model_ui_data:
                model_ui_data[_model] = {
                    "daily_data": [],
                    "sum_num_rate_limit_exceptions": 0,
                }
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            model_ui_data[_model]["daily_data"].append(row)
            model_ui_data[_model]["sum_num_rate_limit_exceptions"] += row.get(
                "num_rate_limit_exceptions", 0
            )

        # sort mode ui data by sum_api_requests -> get top 10 models
        model_ui_data = dict(
            sorted(
                model_ui_data.items(),
                key=lambda x: x[1]["sum_num_rate_limit_exceptions"],
                reverse=True,
            )[:10]
        )

        response = []
        for model, data in model_ui_data.items():
            _sort_daily_data = sorted(data["daily_data"], key=lambda x: x["date"])

            response.append(
                {
                    "api_base": model,
                    "daily_data": _sort_daily_data,
                    "sum_num_rate_limit_exceptions": data[
                        "sum_num_rate_limit_exceptions"
                    ],
                }
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


@router.get(
    "/global/activity/exceptions",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_activity_exceptions(
    model_group: str = fastapi.Query(
        description="Filter by model group",
    ),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of API Requests, total tokens through proxy

    {
        "daily_data": [
                const chartdata = [
                {
                date: 'Jan 22',
                num_rate_limit_exceptions: 10,
                },
                {
                date: 'Jan 23',
                num_rate_limit_exceptions: 10,
                },
        ],
        "sum_api_exceptions": 20,
    }
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    global prisma_client, llm_router
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT
            date_trunc('day', "startTime")::date AS date,
            COUNT(*) AS num_rate_limit_exceptions
        FROM
            "LiteLLM_ErrorLogs"
        WHERE
            "startTime" >= $1::date
            AND "startTime" < ($2::date + INTERVAL '1 day')
            AND model_group = $3
            AND status_code = '429'
        GROUP BY
            date_trunc('day', "startTime")
        ORDER BY
            date;
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj, model_group
        )

        if db_response is None:
            return []

        sum_num_rate_limit_exceptions = 0
        daily_data = []
        for row in db_response:
            # cast date to datetime
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            daily_data.append(row)
            sum_num_rate_limit_exceptions += row.get("num_rate_limit_exceptions", 0)

        # sort daily_data by date
        daily_data = sorted(daily_data, key=lambda x: x["date"])

        data_to_return = {
            "daily_data": daily_data,
            "sum_num_rate_limit_exceptions": sum_num_rate_limit_exceptions,
        }

        return data_to_return

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/spend/provider",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_spend_provider(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get breakdown of spend per provider
    [
        {
            "provider": "Azure OpenAI",
            "spend": 20
        },
        {
            "provider": "OpenAI",
            "spend": 10
        },
        {
            "provider": "VertexAI",
            "spend": 30
        }
    ]
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    global prisma_client, llm_router
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """

        SELECT
        model_id,
        SUM(spend) AS spend
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime" BETWEEN $1::date AND $2::date AND length(model_id) > 0
        GROUP BY model_id
        """

        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj
        )
        if db_response is None:
            return []

        ###################################
        # Convert model_id -> to Provider #
        ###################################

        # we use the in memory router for this
        ui_response = []
        provider_spend_mapping: defaultdict = defaultdict(int)
        for row in db_response:
            _model_id = row["model_id"]
            _provider = "Unknown"
            if llm_router is not None:
                _deployment = llm_router.get_deployment(model_id=_model_id)
                if _deployment is not None:
                    try:
                        _, _provider, _, _ = litellm.get_llm_provider(
                            model=_deployment.litellm_params.model,
                            custom_llm_provider=_deployment.litellm_params.custom_llm_provider,
                            api_base=_deployment.litellm_params.api_base,
                            litellm_params=_deployment.litellm_params,
                        )
                        provider_spend_mapping[_provider] += row["spend"]
                    except:
                        pass

        for provider, spend in provider_spend_mapping.items():
            ui_response.append({"provider": provider, "spend": spend})

        return ui_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/spend/report",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_spend_report(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get Daily Spend per Team, based on specific startTime and endTime. Per team, view usage by each key, model
    [
        {
            "group-by-day": "2024-05-10",
            "teams": [
                {
                    "team_name": "team-1"
                    "spend": 10,
                    "keys": [
                        "key": "1213",
                        "usage": {
                            "model-1": {
                                    "cost": 12.50,
                                    "input_tokens": 1000,
                                    "output_tokens": 5000,
                                    "requests": 100
                                },
                                "audio-modelname1": {
                                "cost": 25.50,
                                "seconds": 25,
                                "requests": 50
                        },
                        }
                    }
            ]
        ]
    }
    """
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        # first get data from spend logs -> SpendByModelApiKey
        # then read data from "SpendByModelApiKey" to format the response obj
        sql_query = """

        WITH SpendByModelApiKey AS (
            SELECT
                date_trunc('day', sl."startTime") AS group_by_day,
                COALESCE(tt.team_alias, 'Unassigned Team') AS team_name,
                sl.model,
                sl.api_key,
                SUM(sl.spend) AS model_api_spend,
                SUM(sl.total_tokens) AS model_api_tokens
            FROM 
                "LiteLLM_SpendLogs" sl
            LEFT JOIN 
                "LiteLLM_TeamTable" tt 
            ON 
                sl.team_id = tt.team_id
            WHERE
                sl."startTime" BETWEEN $1::date AND $2::date
            GROUP BY
                date_trunc('day', sl."startTime"),
                tt.team_alias,
                sl.model,
                sl.api_key
        )
            SELECT
                group_by_day,
                jsonb_agg(jsonb_build_object(
                    'team_name', team_name,
                    'total_spend', total_spend,
                    'metadata', metadata
                )) AS teams
            FROM (
                SELECT
                    group_by_day,
                    team_name,
                    SUM(model_api_spend) AS total_spend,
                    jsonb_agg(jsonb_build_object(
                        'model', model,
                        'api_key', api_key,
                        'spend', model_api_spend,
                        'total_tokens', model_api_tokens
                    )) AS metadata
                FROM 
                    SpendByModelApiKey
                GROUP BY
                    group_by_day,
                    team_name
            ) AS aggregated
            GROUP BY
                group_by_day
            ORDER BY
                group_by_day;
            """

        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj
        )
        if db_response is None:
            return []

        return db_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/spend/tags",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def global_view_spend_tags(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing key spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view key spend",
    ),
):
    """
    LiteLLM Enterprise - View Spend Per Request Tag. Used by LiteLLM UI

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/tags" \
-H "Authorization: Bearer sk-1234"
    ```

    Spend with Start Date and End Date
    ```
    curl -X GET "http://0.0.0.0:8000/spend/tags?start_date=2022-01-01&end_date=2022-02-01" \
-H "Authorization: Bearer sk-1234"
    ```
    """

    from enterprise.utils import ui_get_spend_by_tags

    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if end_date is None or start_date is None:
            raise ProxyException(
                message="Please provide start_date and end_date",
                type="bad_request",
                param=None,
                code=status.HTTP_400_BAD_REQUEST,
            )
        response = await ui_get_spend_by_tags(
            start_date=start_date, end_date=end_date, prisma_client=prisma_client
        )

        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/tags Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/tags Error" + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def _get_spend_report_for_time_range(
    start_date: str,
    end_date: str,
):
    global prisma_client
    if prisma_client is None:
        verbose_proxy_logger.error(
            f"Database not connected. Connect a database to your proxy for weekly, monthly spend reports"
        )
        return None

    try:
        sql_query = """
        SELECT
            t.team_alias,
            SUM(s.spend) AS total_spend
        FROM
            "LiteLLM_SpendLogs" s
        LEFT JOIN
            "LiteLLM_TeamTable" t ON s.team_id = t.team_id
        WHERE
            s."startTime"::DATE >= $1::date AND s."startTime"::DATE <= $2::date
        GROUP BY
            t.team_alias
        ORDER BY
            total_spend DESC;
        """
        response = await prisma_client.db.query_raw(sql_query, start_date, end_date)

        # get spend per tag for today
        sql_query = """
        SELECT 
        jsonb_array_elements_text(request_tags) AS individual_request_tag,
        SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime"::DATE >= $1::date AND "startTime"::DATE <= $2::date
        GROUP BY individual_request_tag
        ORDER BY total_spend DESC;
        """

        spend_per_tag = await prisma_client.db.query_raw(
            sql_query, start_date, end_date
        )

        return response, spend_per_tag
    except Exception as e:
        verbose_proxy_logger.error("Exception in _get_daily_spend_reports", e)  # noqa


@router.post(
    "/spend/calculate",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {
            "cost": {
                "description": "The calculated cost",
                "example": 0.0,
                "type": "float",
            }
        }
    },
)
async def calculate_spend(request: Request):
    """
    Accepts all the params of completion_cost.

    Calculate spend **before** making call:

    Note: If you see a spend of $0.0 you need to set custom_pricing for your model: https://docs.litellm.ai/docs/proxy/custom_pricing

    ```
    curl --location 'http://localhost:4000/spend/calculate'
    --header 'Authorization: Bearer sk-1234'
    --header 'Content-Type: application/json'
    --data '{
        "model": "anthropic.claude-v2",
        "messages": [{"role": "user", "content": "Hey, how'''s it going?"}]
    }'
    ```

    Calculate spend **after** making call:

    ```
    curl --location 'http://localhost:4000/spend/calculate'
    --header 'Authorization: Bearer sk-1234'
    --header 'Content-Type: application/json'
    --data '{
        "completion_response": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there, how may I assist you today?"
                },
                "logprobs": null,
                "finish_reason": "stop"
            }]
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
    }'
    ```
    """
    from litellm import completion_cost

    data = await request.json()
    if "completion_response" in data:
        data["completion_response"] = litellm.ModelResponse(
            **data["completion_response"]
        )
    return {"cost": completion_cost(**data)}


@router.get(
    "/spend/logs",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def view_spend_logs(
    api_key: Optional[str] = fastapi.Query(
        default=None,
        description="Get spend logs based on api key",
    ),
    user_id: Optional[str] = fastapi.Query(
        default=None,
        description="Get spend logs based on user_id",
    ),
    request_id: Optional[str] = fastapi.Query(
        default=None,
        description="request_id to get spend logs for specific request_id. If none passed then pass spend logs for all requests",
    ),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing key spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view key spend",
    ),
):
    """
    View all spend logs, if request_id is provided, only logs for that request_id will be returned

    Example Request for all logs
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Request for specific request_id
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs?request_id=chatcmpl-6dcb2540-d3d7-4e49-bb27-291f863f112e" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Request for specific api_key
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs?api_key=sk-Fn8Ej39NkBQmUagFEoUWPQ" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Request for specific user_id
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs?user_id=ishaan@berri.ai" \
-H "Authorization: Bearer sk-1234"
    ```
    """
    global prisma_client
    try:
        verbose_proxy_logger.debug("inside view_spend_logs")
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        spend_logs = []
        if (
            start_date is not None
            and isinstance(start_date, str)
            and end_date is not None
            and isinstance(end_date, str)
        ):
            # Convert the date strings to datetime objects
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            filter_query = {
                "startTime": {
                    "gte": start_date_obj,  # Greater than or equal to Start Date
                    "lte": end_date_obj,  # Less than or equal to End Date
                }
            }

            if api_key is not None and isinstance(api_key, str):
                filter_query["api_key"] = api_key  # type: ignore
            elif request_id is not None and isinstance(request_id, str):
                filter_query["request_id"] = request_id  # type: ignore
            elif user_id is not None and isinstance(user_id, str):
                filter_query["user"] = user_id  # type: ignore

            # SQL query
            response = await prisma_client.db.litellm_spendlogs.group_by(
                by=["api_key", "user", "model", "startTime"],
                where=filter_query,  # type: ignore
                sum={
                    "spend": True,
                },
            )

            if (
                isinstance(response, list)
                and len(response) > 0
                and isinstance(response[0], dict)
            ):
                result: dict = {}
                for record in response:
                    dt_object = datetime.strptime(
                        str(record["startTime"]), "%Y-%m-%dT%H:%M:%S.%fZ"
                    )  # type: ignore
                    date = dt_object.date()
                    if date not in result:
                        result[date] = {"users": {}, "models": {}}
                    api_key = record["api_key"]
                    user_id = record["user"]
                    model = record["model"]
                    result[date]["spend"] = (
                        result[date].get("spend", 0) + record["_sum"]["spend"]
                    )
                    result[date][api_key] = (
                        result[date].get(api_key, 0) + record["_sum"]["spend"]
                    )
                    result[date]["users"][user_id] = (
                        result[date]["users"].get(user_id, 0) + record["_sum"]["spend"]
                    )
                    result[date]["models"][model] = (
                        result[date]["models"].get(model, 0) + record["_sum"]["spend"]
                    )
                return_list = []
                final_date = None
                for k, v in sorted(result.items()):
                    return_list.append({**v, "startTime": k})
                    final_date = k

                end_date_date = end_date_obj.date()
                if final_date is not None and final_date < end_date_date:
                    current_date = final_date + timedelta(days=1)
                    while current_date <= end_date_date:
                        # Represent current_date as string because original response has it this way
                        return_list.append(
                            {
                                "startTime": current_date,
                                "spend": 0,
                                "users": {},
                                "models": {},
                            }
                        )  # If no data, will stay as zero
                        current_date += timedelta(days=1)  # Move on to the next day

                return return_list

            return response

        elif api_key is not None and isinstance(api_key, str):
            if api_key.startswith("sk-"):
                hashed_token = prisma_client.hash_token(token=api_key)
            else:
                hashed_token = api_key
            spend_log = await prisma_client.get_data(
                table_name="spend",
                query_type="find_all",
                key_val={"key": "api_key", "value": hashed_token},
            )
            if isinstance(spend_log, list):
                return spend_log
            else:
                return [spend_log]
        elif request_id is not None:
            spend_log = await prisma_client.get_data(
                table_name="spend",
                query_type="find_unique",
                key_val={"key": "request_id", "value": request_id},
            )
            return [spend_log]
        elif user_id is not None:
            spend_log = await prisma_client.get_data(
                table_name="spend",
                query_type="find_all",
                key_val={"key": "user", "value": user_id},
            )
            if isinstance(spend_log, list):
                return spend_log
            else:
                return [spend_log]
        else:
            spend_logs = await prisma_client.get_data(
                table_name="spend", query_type="find_all"
            )

            return spend_log

        return None

    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/logs Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/logs Error" + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/global/spend/reset",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_reset():
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Globally reset spend for All API Keys and Teams, maintain LiteLLM_SpendLogs

    1. LiteLLM_SpendLogs will maintain the logs on spend, no data gets deleted from there
    2. LiteLLM_VerificationTokens spend will be set = 0
    3. LiteLLM_TeamTable spend will be set = 0

    """
    global prisma_client
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_401_UNAUTHORIZED,
        )

    await prisma_client.db.litellm_verificationtoken.update_many(
        data={"spend": 0.0}, where={}
    )
    await prisma_client.db.litellm_teamtable.update_many(data={"spend": 0.0}, where={})

    return {
        "message": "Spend for all API Keys and Teams reset successfully",
        "status": "success",
    }


@router.get(
    "/global/spend/logs",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_logs(
    api_key: str = fastapi.Query(
        default=None,
        description="API Key to get global spend (spend per day for last 30d). Admin-only endpoint",
    )
):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get global spend (spend per day for last 30d). Admin-only endpoint

    More efficient implementation of /spend/logs, by creating a view over the spend logs table.
    """
    global prisma_client
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    if api_key is None:
        sql_query = """SELECT * FROM "MonthlyGlobalSpend" ORDER BY "date";"""

        response = await prisma_client.db.query_raw(query=sql_query)

        return response
    else:
        sql_query = """
            SELECT * FROM "MonthlyGlobalSpendPerKey"
            WHERE "api_key" = $1
            ORDER BY "date";
            """

        response = await prisma_client.db.query_raw(sql_query, api_key)

        return response
    return


@router.get(
    "/global/spend",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend():
    """
    [BETA] This is a beta endpoint. It will change.

    View total spend across all proxy keys
    """
    global prisma_client
    total_spend = 0.0
    total_proxy_budget = 0.0

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})
    sql_query = """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""
    response = await prisma_client.db.query_raw(query=sql_query)
    if response is not None:
        if isinstance(response, list) and len(response) > 0:
            total_spend = response[0].get("total_spend", 0.0)

    return {"spend": total_spend, "max_budget": litellm.max_budget}


@router.get(
    "/global/spend/keys",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_keys(
    limit: int = fastapi.Query(
        default=None,
        description="Number of keys to get. Will return Top 'n' keys.",
    )
):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get the top 'n' keys with the highest spend, ordered by spend.
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})
    sql_query = f"""SELECT * FROM "Last30dKeysBySpend" LIMIT {limit};"""

    response = await prisma_client.db.query_raw(query=sql_query)

    return response


@router.get(
    "/global/spend/teams",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_per_team():
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get daily spend, grouped by `team_id` and `date`
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})
    sql_query = """
        SELECT
            t.team_alias as team_alias,
            DATE(s."startTime") AS spend_date,
            SUM(s.spend) AS total_spend
        FROM
            "LiteLLM_SpendLogs" s
        LEFT JOIN
            "LiteLLM_TeamTable" t ON s.team_id = t.team_id
        WHERE
            s."startTime" >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY
            t.team_alias,
            DATE(s."startTime")
        ORDER BY
            spend_date;
        """
    response = await prisma_client.db.query_raw(query=sql_query)

    # transform the response for the Admin UI
    spend_by_date = {}
    team_aliases = set()
    total_spend_per_team = {}
    for row in response:
        row_date = row["spend_date"]
        if row_date is None:
            continue
        team_alias = row["team_alias"]
        if team_alias is None:
            team_alias = "Unassigned"
        team_aliases.add(team_alias)
        if row_date in spend_by_date:
            # get the team_id for this entry
            # get the spend for this entry
            spend = row["total_spend"]
            spend = round(spend, 2)
            current_date_entries = spend_by_date[row_date]
            current_date_entries[team_alias] = spend
        else:
            spend = row["total_spend"]
            spend = round(spend, 2)
            spend_by_date[row_date] = {team_alias: spend}

        if team_alias in total_spend_per_team:
            total_spend_per_team[team_alias] += spend
        else:
            total_spend_per_team[team_alias] = spend

    total_spend_per_team_ui = []
    # order the elements in total_spend_per_team by spend
    total_spend_per_team = dict(
        sorted(total_spend_per_team.items(), key=lambda item: item[1], reverse=True)
    )
    for team_id in total_spend_per_team:
        # only add first 10 elements to total_spend_per_team_ui
        if len(total_spend_per_team_ui) >= 10:
            break
        if team_id is None:
            team_id = "Unassigned"
        total_spend_per_team_ui.append(
            {"team_id": team_id, "total_spend": total_spend_per_team[team_id]}
        )

    # sort spend_by_date by it's key (which is a date)

    response_data = []
    for key in spend_by_date:
        value = spend_by_date[key]
        response_data.append({"date": key, **value})

    return {
        "daily_spend": response_data,
        "teams": list(team_aliases),
        "total_spend_per_team": total_spend_per_team_ui,
    }


@router.post(
    "/global/spend/end_users",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_end_users(data: Optional[GlobalEndUsersSpend] = None):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get the top 'n' keys with the highest spend, ordered by spend.
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    """
    Gets the top 100 end-users for a given api key
    """
    startTime = None
    endTime = None
    selected_api_key = None
    if data is not None:
        startTime = data.startTime
        endTime = data.endTime
        selected_api_key = data.api_key

    startTime = startTime or datetime.now() - timedelta(days=30)
    endTime = endTime or datetime.now()

    sql_query = """
SELECT end_user, COUNT(*) AS total_count, SUM(spend) AS total_spend
FROM "LiteLLM_SpendLogs"
WHERE "startTime" >= $1::timestamp
  AND "startTime" < $2::timestamp
  AND (
    CASE
      WHEN $3::TEXT IS NULL THEN TRUE
      ELSE api_key = $3
    END
  )
GROUP BY end_user
ORDER BY total_spend DESC
LIMIT 100
    """
    response = await prisma_client.db.query_raw(
        sql_query, startTime, endTime, selected_api_key
    )

    return response


@router.get(
    "/global/spend/models",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_models(
    limit: int = fastapi.Query(
        default=None,
        description="Number of models to get. Will return Top 'n' models.",
    )
):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get the top 'n' keys with the highest spend, ordered by spend.
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    sql_query = f"""SELECT * FROM "Last30dModelsBySpend" LIMIT {limit};"""

    response = await prisma_client.db.query_raw(query=sql_query)

    return response


@router.post(
    "/global/predict/spend/logs",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_predict_spend_logs(request: Request):
    from enterprise.utils import _forecast_daily_cost

    data = await request.json()
    data = data.get("data")
    return _forecast_daily_cost(data)


#### INTERNAL USER MANAGEMENT ####
@router.post(
    "/user/new",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewUserResponse,
)
async def new_user(data: NewUserRequest):
    """
    Use this to create a new INTERNAL user with a budget.
    Internal Users can access LiteLLM Admin UI to make keys, request access to models.
    This creates a new user and generates a new api key for the new user. The new api key is returned.

    Returns user id, budget + new key.

    Parameters:
    - user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
    - user_alias: Optional[str] - A descriptive name for you to know who this user id refers to.
    - teams: Optional[list] - specify a list of team id's a user belongs to.
    - organization_id: Optional[str] - specify the org a user belongs to.
    - user_email: Optional[str] - Specify a user email.
    - send_invite_email: Optional[bool] - Specify if an invite email should be sent.
    - user_role: Optional[str] - Specify a user role - "admin", "app_owner", "app_user"
    - max_budget: Optional[float] - Specify max budget for a given user.
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - tpm_limit: Optional[int] - Specify tpm limit for a given user (Tokens per minute)
    - rpm_limit: Optional[int] - Specify rpm limit for a given user (Requests per minute)
    - auto_create_key: bool - Default=True. Flag used for returning a key as part of the /user/new response

    Returns:
    - key: (str) The generated api key for the user
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    - max_budget: (float|None) Max budget for given user.
    """
    data_json = data.json()  # type: ignore
    if "user_id" in data_json and data_json["user_id"] is None:
        data_json["user_id"] = str(uuid.uuid4())
    auto_create_key = data_json.pop("auto_create_key", True)
    if auto_create_key == False:
        data_json["table_name"] = (
            "user"  # only create a user, don't create key if 'auto_create_key' set to False
        )
    response = await generate_key_helper_fn(request_type="user", **data_json)

    # Admin UI Logic
    # if team_id passed add this user to the team
    if data_json.get("team_id", None) is not None:
        await team_member_add(
            data=TeamMemberAddRequest(
                team_id=data_json.get("team_id", None),
                member=Member(
                    user_id=data_json.get("user_id", None),
                    role="user",
                    user_email=data_json.get("user_email", None),
                ),
            )
        )

    if data.send_invite_email is True:
        # check if user has setup email alerting
        if "email" not in general_settings.get("alerting", []):
            raise ValueError(
                "Email alerting not setup on config.yaml. Please set `alerting=['email']. \nDocs: https://docs.litellm.ai/docs/proxy/email`"
            )

        event = WebhookEvent(
            event="internal_user_created",
            event_group="internal_user",
            event_message=f"Welcome to LiteLLM Proxy",
            token=response.get("token", ""),
            spend=response.get("spend", 0.0),
            max_budget=response.get("max_budget", 0.0),
            user_id=response.get("user_id", None),
            user_email=response.get("user_email", None),
            team_id=response.get("team_id", "Default Team"),
            key_alias=response.get("key_alias", None),
        )

        # If user configured email alerting - send an Email letting their end-user know the key was created
        asyncio.create_task(
            proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                webhook_event=event,
            )
        )

    return NewUserResponse(
        key=response.get("token", ""),
        expires=response.get("expires", None),
        max_budget=response["max_budget"],
        user_id=response["user_id"],
        user_role=response.get("user_role", None),
        user_email=response.get("user_email", None),
        teams=response.get("teams", None),
        team_id=response.get("team_id", None),
        metadata=response.get("metadata", None),
        models=response.get("models", None),
        tpm_limit=response.get("tpm_limit", None),
        rpm_limit=response.get("rpm_limit", None),
        budget_duration=response.get("budget_duration", None),
    )


@router.post(
    "/user/auth",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_auth(request: Request):
    """
    Allows UI ("https://dashboard.litellm.ai/", or self-hosted - os.getenv("LITELLM_HOSTED_UI")) to request a magic link to be sent to user email, for auth to proxy.

    Only allows emails from accepted email subdomains.

    Rate limit: 1 request every 60s.

    Only works, if you enable 'allow_user_auth' in general settings:
    e.g.:
    ```yaml
    general_settings:
        allow_user_auth: true
    ```

    Requirements:
    SMTP server details saved in .env:
    - os.environ["SMTP_HOST"]
    - os.environ["SMTP_PORT"]
    - os.environ["SMTP_USERNAME"]
    - os.environ["SMTP_PASSWORD"]
    - os.environ["SMTP_SENDER_EMAIL"]
    """
    global prisma_client

    data = await request.json()  # type: ignore
    user_email = data["user_email"]
    page_params = data["page"]
    if user_email is None:
        raise HTTPException(status_code=400, detail="User email is none")

    if prisma_client is None:  # if no db connected, raise an error
        raise Exception("No connected db.")

    ### Check if user email in user table
    response = await prisma_client.get_generic_data(
        key="user_email", value=user_email, table_name="users"
    )
    ### if so - generate a 24 hr key with that user id
    if response is not None:
        user_id = response.user_id
        response = await generate_key_helper_fn(
            request_type="key",
            **{"duration": "24hr", "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": user_id},  # type: ignore
        )
    else:  ### else - create new user
        response = await generate_key_helper_fn(
            request_type="key",
            **{"duration": "24hr", "models": [], "aliases": {}, "config": {}, "spend": 0, "user_email": user_email},  # type: ignore
        )

    base_url = os.getenv("LITELLM_HOSTED_UI", "https://dashboard.litellm.ai/")

    params = {
        "sender_name": "LiteLLM Proxy",
        "receiver_email": user_email,
        "subject": "Your Magic Link",
        "html": f"<strong> Follow this  link, to login:\n\n{base_url}user/?token={response['token']}&user_id={response['user_id']}&page={page_params}</strong>",
    }

    await send_email(**params)
    return "Email sent!"


@router.get(
    "/user/available_roles",
    tags=["Internal User management"],
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth)],
)
async def ui_get_available_role(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Endpoint used by Admin UI to show all available roles to assign a user
    return {
        "proxy_admin": {
            "description": "Proxy Admin role",
            "ui_label": "Admin"
        }
    }
    """

    _data_to_return = {}
    for role in LitellmUserRoles:

        # We only show a subset of roles on UI
        if role in [
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        ]:
            _data_to_return[role.value] = {
                "description": role.description,
                "ui_label": role.ui_label,
            }
    return _data_to_return


@router.get(
    "/user/info",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_info(
    user_id: Optional[str] = fastapi.Query(
        default=None, description="User ID in the request parameters"
    ),
    view_all: bool = fastapi.Query(
        default=False,
        description="set to true to View all users. When using view_all, don't pass user_id",
    ),
    page: Optional[int] = fastapi.Query(
        default=0,
        description="Page number for pagination. Only use when view_all is true",
    ),
    page_size: Optional[int] = fastapi.Query(
        default=25,
        description="Number of items per page. Only use when view_all is true",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use this to get user information. (user row + all user key info)

    Example request
    ```
    curl -X GET 'http://localhost:8000/user/info?user_id=krrish7%40berri.ai' \
    --header 'Authorization: Bearer sk-1234'
    ```
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        ## GET USER ROW ##
        if user_id is not None:
            user_info = await prisma_client.get_data(user_id=user_id)
        elif view_all == True:
            if page is None:
                page = 0
            if page_size is None:
                page_size = 25
            offset = (page) * page_size  # default is 0
            limit = page_size  # default is 10
            user_info = await prisma_client.get_data(
                table_name="user", query_type="find_all", offset=offset, limit=limit
            )
            return user_info
        else:
            user_info = None
        ## GET ALL TEAMS ##
        team_list = []
        team_id_list = []
        # _DEPRECATED_ check if user in 'member' field
        teams_1 = await prisma_client.get_data(
            user_id=user_id, table_name="team", query_type="find_all"
        )

        if teams_1 is not None and isinstance(teams_1, list):
            team_list = teams_1
            for team in teams_1:
                team_id_list.append(team.team_id)

        if user_info is not None:
            # *NEW* get all teams in user 'teams' field
            teams_2 = await prisma_client.get_data(
                team_id_list=user_info.teams, table_name="team", query_type="find_all"
            )

            if teams_2 is not None and isinstance(teams_2, list):
                for team in teams_2:
                    if team.team_id not in team_id_list:
                        team_list.append(team)
                        team_id_list.append(team.team_id)
        elif (
            user_api_key_dict.user_id is not None and user_id is None
        ):  # the key querying the endpoint is the one asking for it's teams
            caller_user_info = await prisma_client.get_data(
                user_id=user_api_key_dict.user_id
            )
            # *NEW* get all teams in user 'teams' field
            if (
                getattr(caller_user_info, "user_role", None)
                == LitellmUserRoles.PROXY_ADMIN
            ):
                teams_2 = await prisma_client.get_data(
                    table_name="team",
                    query_type="find_all",
                    team_id_list=None,
                )
            else:
                teams_2 = await prisma_client.get_data(
                    team_id_list=caller_user_info.teams,
                    table_name="team",
                    query_type="find_all",
                )

            if teams_2 is not None and isinstance(teams_2, list):
                for team in teams_2:
                    if team.team_id not in team_id_list:
                        team_list.append(team)
                        team_id_list.append(team.team_id)

        ## GET ALL KEYS ##
        keys = await prisma_client.get_data(
            user_id=user_id,
            table_name="key",
            query_type="find_all",
            expires=datetime.now(),
        )

        if user_info is None:
            ## make sure we still return a total spend ##
            spend = 0
            for k in keys:
                spend += getattr(k, "spend", 0)
            user_info = {"spend": spend}

        ## REMOVE HASHED TOKEN INFO before returning ##
        returned_keys = []
        for key in keys:
            if (
                key.token == litellm_master_key_hash
                and general_settings.get("disable_master_key_return", False)
                == True  ## [IMPORTANT] used by hosted proxy-ui to prevent sharing master key on ui
            ):
                continue

            try:
                key = key.model_dump()  # noqa
            except:
                # if using pydantic v1
                key = key.dict()
            if (
                "team_id" in key
                and key["team_id"] is not None
                and key["team_id"] != "litellm-dashboard"
            ):
                team_info = await prisma_client.get_data(
                    team_id=key["team_id"], table_name="team"
                )
                team_alias = getattr(team_info, "team_alias", None)
                key["team_alias"] = team_alias
            else:
                key["team_alias"] = "None"
            returned_keys.append(key)

        response_data = {
            "user_id": user_id,
            "user_info": user_info,
            "keys": returned_keys,
            "teams": team_list,
        }
        return response_data
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/user/update",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_update(data: UpdateUserRequest):
    """
    Example curl 

    ```
    curl --location 'http://0.0.0.0:4000/user/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "test-litellm-user-4",
        "user_role": "proxy_admin_viewer"
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
        verbose_proxy_logger.debug("/user/update: Received data = %s", data)
        if data.user_id is not None and len(data.user_id) > 0:
            non_default_values["user_id"] = data.user_id  # type: ignore
            verbose_proxy_logger.debug("In update user, user_id condition block.")
            response = await prisma_client.update_data(
                user_id=data.user_id,
                data=non_default_values,
                table_name="user",
            )
            verbose_proxy_logger.debug(
                f"received response from updating prisma client. response={response}"
            )
        elif data.user_email is not None:
            non_default_values["user_id"] = str(uuid.uuid4())
            non_default_values["user_email"] = data.user_email
            ## user email is not unique acc. to prisma schema -> future improvement
            ### for now: check if it exists in db, if not - insert it
            existing_user_rows = await prisma_client.get_data(
                key_val={"user_email": data.user_email},
                table_name="user",
                query_type="find_all",
            )
            if existing_user_rows is None or (
                isinstance(existing_user_rows, list) and len(existing_user_rows) == 0
            ):
                response = await prisma_client.insert_data(
                    data=non_default_values, table_name="user"
                )
            elif isinstance(existing_user_rows, list) and len(existing_user_rows) > 0:
                for existing_user in existing_user_rows:
                    response = await prisma_client.update_data(
                        user_id=existing_user.user_id,
                        data=non_default_values,
                        table_name="user",
                    )
        return response
        # update based on remaining passed in values
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/user/request_model",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_request_model(request: Request):
    """
    Allow a user to create a request to access a model
    """
    global prisma_client
    try:
        data_json = await request.json()

        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        non_default_values = {k: v for k, v in data_json.items() if v is not None}
        new_models = non_default_values.get("models", None)
        user_id = non_default_values.get("user_id", None)
        justification = non_default_values.get("justification", None)

        response = await prisma_client.insert_data(
            data={
                "models": new_models,
                "justification": justification,
                "user_id": user_id,
                "status": "pending",
                "request_id": str(uuid.uuid4()),
            },
            table_name="user_notification",
        )
        return {"status": "success"}
        # update based on remaining passed in values
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.get(
    "/user/get_requests",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def user_get_requests():
    """
    Get all "Access" requests made by proxy users, access requests are requests for accessing models
    """
    global prisma_client
    try:

        # get the row from db
        if prisma_client is None:
            raise Exception("Not connected to DB!")

        # TODO: Optimize this so we don't read all the data here, eventually move to pagination
        response = await prisma_client.get_data(
            query_type="find_all",
            table_name="user_notification",
        )
        return {"requests": response}
        # update based on remaining passed in values
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.get(
    "/user/get_users",
    tags=["Internal User management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_users(
    role: str = fastapi.Query(
        default=None,
        description="Either 'proxy_admin', 'proxy_viewer', 'app_owner', 'app_user'",
    )
):
    """
    [BETA] This could change without notice. Give feedback - https://github.com/BerriAI/litellm/issues

    Get all users who are a specific `user_role`.

    Used by the UI to populate the user lists.

    Currently - admin-only endpoint.
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": f"No db connected. prisma client={prisma_client}"},
        )
    all_users = await prisma_client.get_data(
        table_name="user", query_type="find_all", key_val={"user_role": role}
    )

    return all_users


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
    NOTE: This used to be called `/end_user/new`, we will still be maintaining compatibility for /end_user/XXX for these endpoints

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
        traceback.print_exc()
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
        traceback.print_exc()
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


#### TEAM MANAGEMENT ####


@router.post(
    "/team/new",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_TeamTable,
)
async def new_team(
    data: NewTeamRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Allow users to create a new team. Apply user permissions to their team.

    [ASK FOR HELP](https://github.com/BerriAI/litellm/issues)

    Parameters:
    - team_alias: Optional[str] - User defined team alias
    - team_id: Optional[str] - The team id of the user. If none passed, we'll generate it.
    - members_with_roles: List[{"role": "admin" or "user", "user_id": "<user-id>"}] - A list of users and their roles in the team. Get user_id when making a new user via `/user/new`.
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.

    Returns:
    - team_id: (str) Unique team id - used for tracking spend across multiple keys for same team id.

    _deprecated_params:
    - admins: list - A list of user_id's for the admin role
    - users: list - A list of user_id's for the user role

    Example Request:
    ```
    curl --location 'http://0.0.0.0:4000/team/new' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data '{
      "team_alias": "my-new-team_2",
      "members_with_roles": [{"role": "admin", "user_id": "user-1234"},
        {"role": "user", "user_id": "user-2434"}]
    }'

    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        data.team_id = str(uuid.uuid4())
    else:
        # Check if team_id exists already
        _existing_team_id = await prisma_client.get_data(
            team_id=data.team_id, table_name="team", query_type="find_unique"
        )
        if _existing_team_id is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Team id = {data.team_id} already exists. Please use a different team id."
                },
            )

    if (
        user_api_key_dict.user_role is None
        or user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
    ):  # don't restrict proxy admin
        if (
            data.tpm_limit is not None
            and user_api_key_dict.tpm_limit is not None
            and data.tpm_limit > user_api_key_dict.tpm_limit
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"tpm limit higher than user max. User tpm limit={user_api_key_dict.tpm_limit}. User role={user_api_key_dict.user_role}"
                },
            )

        if (
            data.rpm_limit is not None
            and user_api_key_dict.rpm_limit is not None
            and data.rpm_limit > user_api_key_dict.rpm_limit
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"rpm limit higher than user max. User rpm limit={user_api_key_dict.rpm_limit}. User role={user_api_key_dict.user_role}"
                },
            )

        if (
            data.max_budget is not None
            and user_api_key_dict.max_budget is not None
            and data.max_budget > user_api_key_dict.max_budget
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"max budget higher than user max. User max budget={user_api_key_dict.max_budget}. User role={user_api_key_dict.user_role}"
                },
            )

        if data.models is not None and len(user_api_key_dict.models) > 0:
            for m in data.models:
                if m not in user_api_key_dict.models:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"Model not in allowed user models. User allowed models={user_api_key_dict.models}. User id={user_api_key_dict.user_id}"
                        },
                    )

    if user_api_key_dict.user_id is not None:
        creating_user_in_list = False
        for member in data.members_with_roles:
            if member.user_id == user_api_key_dict.user_id:
                creating_user_in_list = True

        if creating_user_in_list == False:
            data.members_with_roles.append(
                Member(role="admin", user_id=user_api_key_dict.user_id)
            )

    ## ADD TO MODEL TABLE
    _model_id = None
    if data.model_aliases is not None and isinstance(data.model_aliases, dict):
        litellm_modeltable = LiteLLM_ModelTable(
            model_aliases=json.dumps(data.model_aliases),
            created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
            updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
        )
        model_dict = await prisma_client.db.litellm_modeltable.create(
            {**litellm_modeltable.json(exclude_none=True)}  # type: ignore
        )  # type: ignore

        _model_id = model_dict.id

    ## ADD TO TEAM TABLE
    complete_team_data = LiteLLM_TeamTable(
        **data.json(),
        model_id=_model_id,
    )

    # If budget_duration is set, set `budget_reset_at`
    if complete_team_data.budget_duration is not None:
        duration_s = _duration_in_seconds(duration=complete_team_data.budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)
        complete_team_data.budget_reset_at = reset_at

    team_row = await prisma_client.insert_data(
        data=complete_team_data.json(exclude_none=True), table_name="team"
    )

    ## ADD TEAM ID TO USER TABLE ##
    for user in complete_team_data.members_with_roles:
        ## add team id to user row ##
        await prisma_client.update_data(
            user_id=user.user_id,
            data={"user_id": user.user_id, "teams": [team_row.team_id]},
            update_key_values_custom_query={
                "teams": {
                    "push ": [team_row.team_id],
                }
            },
        )
    try:
        return team_row.model_dump()
    except Exception as e:
        return team_row.dict()


@router.post(
    "/team/update", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def update_team(
    data: UpdateTeamRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use `/team/member_add` AND `/team/member/delete` to add/remove new team members  

    You can now update team budget / rate limits via /team/update

    Parameters:
    - team_id: str - The team id of the user. Required param.
    - team_alias: Optional[str] - User defined team alias
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }
    - tpm_limit: Optional[int] - The TPM (Tokens Per Minute) limit for this team - all keys with this team_id will have at max this TPM limit
    - rpm_limit: Optional[int] - The RPM (Requests Per Minute) limit for this team - all keys associated with this team_id will have at max this RPM limit
    - max_budget: Optional[float] - The maximum budget allocated to the team - all keys for this team_id will have at max this max_budget
    - models: Optional[list] - A list of models associated with the team - all keys for this team_id will have at most, these models. If empty, assumes all models are allowed.
    - blocked: bool - Flag indicating if the team is blocked or not - will stop all calls from keys with this team_id.

    Example - update team TPM Limit

    ```
    curl --location 'http://0.0.0.0:8000/team/update' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data-raw '{
        "team_id": "litellm-test-client-id-new",
        "tpm_limit": 100
    }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})
    verbose_proxy_logger.debug("/team/update - %s", data)

    existing_team_row = await prisma_client.get_data(
        team_id=data.team_id, table_name="team", query_type="find_unique"
    )
    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Team not found, passed team_id={data.team_id}"},
        )

    updated_kv = data.json(exclude_none=True)

    # Check budget_duration and budget_reset_at
    if data.budget_duration is not None:
        duration_s = _duration_in_seconds(duration=data.budget_duration)
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

        # set the budget_reset_at in DB
        updated_kv["budget_reset_at"] = reset_at

    team_row = await prisma_client.update_data(
        update_key_values=updated_kv,
        data=updated_kv,
        table_name="team",
        team_id=data.team_id,
    )

    return team_row


@router.post(
    "/team/member_add",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def team_member_add(
    data: TeamMemberAddRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]

    Add new members (either via user_email or user_id) to a team

    If user doesn't exist, new user row will also be added to User Table

    ```

    curl -X POST 'http://0.0.0.0:4000/team/member_add' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849", "member": {"role": "user", "user_id": "krrish247652@berri.ai"}}'

    ```
    """
    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    if data.member is None:
        raise HTTPException(status_code=400, detail={"error": "No member passed in"})

    existing_team_row = await prisma_client.get_data(  # type: ignore
        team_id=data.team_id, table_name="team", query_type="find_unique"
    )
    if existing_team_row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Team not found for team_id={getattr(data, 'team_id', None)}"
            },
        )

    new_member = data.member

    existing_team_row.members_with_roles.append(new_member)

    complete_team_data = LiteLLM_TeamTable(
        **_get_pydantic_json_dict(existing_team_row),
    )

    team_row = await prisma_client.update_data(
        update_key_values=complete_team_data.json(exclude_none=True),
        data=complete_team_data.json(exclude_none=True),
        table_name="team",
        team_id=data.team_id,
    )

    ## ADD USER, IF NEW ##
    user_data = {  # type: ignore
        "teams": [team_row["team_id"]],
        "models": team_row["data"].models,
    }
    if new_member.user_id is not None:
        user_data["user_id"] = new_member.user_id  # type: ignore
        await prisma_client.update_data(
            user_id=new_member.user_id,
            data=user_data,
            update_key_values_custom_query={
                "teams": {
                    "push": [team_row["team_id"]],
                }
            },
            table_name="user",
        )
    elif new_member.user_email is not None:
        user_data["user_id"] = str(uuid.uuid4())
        user_data["user_email"] = new_member.user_email
        ## user email is not unique acc. to prisma schema -> future improvement
        ### for now: check if it exists in db, if not - insert it
        existing_user_row = await prisma_client.get_data(
            key_val={"user_email": new_member.user_email},
            table_name="user",
            query_type="find_all",
        )
        if existing_user_row is None or (
            isinstance(existing_user_row, list) and len(existing_user_row) == 0
        ):

            await prisma_client.insert_data(data=user_data, table_name="user")

    # Check if trying to set a budget for team member
    if data.max_budget_in_team is not None and new_member.user_id is not None:
        # create a new budget item for this member
        response = await prisma_client.db.litellm_budgettable.create(
            data={
                "max_budget": data.max_budget_in_team,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )

        _budget_id = response.budget_id
        await prisma_client.db.litellm_teammembership.create(
            data={
                "team_id": data.team_id,
                "user_id": new_member.user_id,
                "budget_id": _budget_id,
            }
        )

    return team_row


@router.post(
    "/team/member_delete",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def team_member_delete(
    data: TeamMemberDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]

    delete members (either via user_email or user_id) from a team

    If user doesn't exist, an exception will be raised
    ```
    curl -X POST 'http://0.0.0.0:8000/team/update' \

    -H 'Authorization: Bearer sk-1234' \

    -H 'Content-Type: application/json' \

    -D '{
        "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "user_id": "krrish247652@berri.ai"
    }'
    ```
    """
    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    if data.user_id is None and data.user_email is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Either user_id or user_email needs to be passed in"},
        )

    existing_team_row = await prisma_client.get_data(  # type: ignore
        team_id=data.team_id, table_name="team", query_type="find_unique"
    )

    ## DELETE MEMBER FROM TEAM
    new_team_members = []
    for m in existing_team_row.members_with_roles:
        if (
            data.user_id is not None
            and m["user_id"] is not None
            and data.user_id == m["user_id"]
        ):
            continue
        elif (
            data.user_email is not None
            and m["user_email"] is not None
            and data.user_email == m["user_email"]
        ):
            continue
        new_team_members.append(m)
    existing_team_row.members_with_roles = new_team_members
    complete_team_data = LiteLLM_TeamTable(
        **existing_team_row.model_dump(),
    )

    team_row = await prisma_client.update_data(
        update_key_values=complete_team_data.json(exclude_none=True),
        data=complete_team_data.json(exclude_none=True),
        table_name="team",
        team_id=data.team_id,
    )

    ## DELETE TEAM ID from USER ROW, IF EXISTS ##
    # get user row
    key_val = {}
    if data.user_id is not None:
        key_val["user_id"] = data.user_id
    elif data.user_email is not None:
        key_val["user_email"] = data.user_email
    existing_user_rows = await prisma_client.get_data(
        key_val=key_val,
        table_name="user",
        query_type="find_all",
    )
    user_data = {  # type: ignore
        "teams": [],
        "models": team_row["data"].models,
    }
    if existing_user_rows is not None and (
        isinstance(existing_user_rows, list) and len(existing_user_rows) > 0
    ):
        for existing_user in existing_user_rows:
            team_list = []
            if hasattr(existing_user, "teams"):
                team_list = existing_user.teams
                team_list.remove(data.team_id)
                user_data["user_id"] = existing_user.user_id
                await prisma_client.update_data(
                    user_id=existing_user.user_id,
                    data=user_data,
                    update_key_values_custom_query={
                        "teams": {
                            "set": [team_row["team_id"]],
                        }
                    },
                    table_name="user",
                )

    return team_row["data"]


@router.post(
    "/team/delete", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def delete_team(
    data: DeleteTeamRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    delete team and associated team keys

    ```
    curl --location 'http://0.0.0.0:8000/team/delete' \

    --header 'Authorization: Bearer sk-1234' \

    --header 'Content-Type: application/json' \

    --data-raw '{
        "team_ids": ["45e3e396-ee08-4a61-a88e-16b3ce7e0849"]
    }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_ids is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    # check that all teams passed exist
    for team_id in data.team_ids:
        team_row = await prisma_client.get_data(  # type: ignore
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if team_row is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Team not found, passed team_id={team_id}"},
            )

    ## DELETE ASSOCIATED KEYS
    await prisma_client.delete_data(team_id_list=data.team_ids, table_name="key")
    ## DELETE TEAMS
    deleted_teams = await prisma_client.delete_data(
        team_id_list=data.team_ids, table_name="team"
    )
    return deleted_teams


@router.get(
    "/team/info", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def team_info(
    team_id: str = fastapi.Query(
        default=None, description="Team ID in the request parameters"
    )
):
    """
    get info on team + related keys

    ```
    curl --location 'http://localhost:4000/team/info' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "teams": ["<team-id>",..]
    }'
    ```
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
                },
            )
        if team_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Malformed request. No team id passed in."},
            )

        team_info = await prisma_client.get_data(
            team_id=team_id, table_name="team", query_type="find_unique"
        )
        if team_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": f"Team not found, passed team id: {team_id}."},
            )

        ## GET ALL KEYS ##
        keys = await prisma_client.get_data(
            team_id=team_id,
            table_name="key",
            query_type="find_all",
            expires=datetime.now(),
        )

        if team_info is None:
            ## make sure we still return a total spend ##
            spend = 0
            for k in keys:
                spend += getattr(k, "spend", 0)
            team_info = {"spend": spend}

        ## REMOVE HASHED TOKEN INFO before returning ##
        for key in keys:
            try:
                key = key.model_dump()  # noqa
            except:
                # if using pydantic v1
                key = key.dict()
            key.pop("token", None)
        return {"team_id": team_id, "team_info": team_info, "keys": keys}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.post(
    "/team/block", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def block_team(
    data: BlockTeamRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Blocks all calls from keys with this team id.
    """
    global prisma_client

    if prisma_client is None:
        raise Exception("No DB Connected.")

    record = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"blocked": True}  # type: ignore
    )

    return record


@router.post(
    "/team/unblock", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def unblock_team(
    data: BlockTeamRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Blocks all calls from keys with this team id.
    """
    global prisma_client

    if prisma_client is None:
        raise Exception("No DB Connected.")

    record = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": data.team_id}, data={"blocked": False}  # type: ignore
    )

    return record


@router.get(
    "/team/list", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def list_team(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Admin-only] List all available teams

    ```
    curl --location --request GET 'http://0.0.0.0:4000/team/list' \
        --header 'Authorization: Bearer sk-1234'
    ```
    """
    global prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Admin-only endpoint. Your user role={}".format(
                    user_api_key_dict.user_role
                )
            },
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    response = await prisma_client.db.litellm_teamtable.find_many()

    return response


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
                if isinstance(v, str):
                    encrypted_value = encrypt_value(value=v, master_key=master_key)  # type: ignore
                    model_params.litellm_params[k] = base64.b64encode(
                        encrypted_value
                    ).decode("utf-8")
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
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
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
                if isinstance(v, str):
                    encrypted_value = encrypt_value(value=v, master_key=master_key)  # type: ignore
                    model_params.litellm_params[k] = base64.b64encode(
                        encrypted_value
                    ).decode("utf-8")

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
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
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
                "error": f"Invalid llm model list. llm_model_list={llm_model_list}"
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
        if debug == True:
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
            except:
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
                litellm_model_info = litellm.get_model_info(model=litellm_model)
            except:
                litellm_model_info = {}
        for k, v in litellm_model_info.items():
            if k not in model_info:
                model_info[k] = v
        _model["model_info"] = model_info
        # don't return the api key / vertex credentials
        _model["litellm_params"].pop("api_key", None)
        _model["litellm_params"].pop("vertex_credentials", None)

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

    sql_query = """
        SELECT
            api_base,
            model_group,
            model,
            DATE_TRUNC('day', "startTime")::DATE AS day,
            AVG(EXTRACT(epoch FROM ("endTime" - "startTime")) / "completion_tokens") AS avg_latency_per_token
        FROM
            "LiteLLM_SpendLogs"
        WHERE
            "startTime" BETWEEN $2::timestamp AND $3::timestamp
            AND "model_group" = $1 AND "cache_hit" != 'True'
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
        sql_query, _selected_model_group, startTime, endTime
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
):
    global prisma_client, llm_router, proxy_logging_obj
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
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
GROUP BY
    api_base
ORDER BY
    slow_count DESC;
    """

    db_response = await prisma_client.db.query_raw(
        sql_query, alerting_threshold, _selected_model_group, startTime, endTime
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

    """
    """
    sql_query = """
        WITH cte AS (
            SELECT 
                CASE WHEN api_base = '' THEN litellm_model_name ELSE CONCAT(litellm_model_name, '-', api_base) END AS combined_model_api_base,
                exception_type,
                COUNT(*) AS num_rate_limit_exceptions
            FROM "LiteLLM_ErrorLogs"
            WHERE "startTime" >= $1::timestamp AND "endTime" <= $2::timestamp AND model_group = $3
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
        sql_query, startTime, endTime, _selected_model_group
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
    description="Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
@router.get(
    "/v1/model/info",
    description="Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def model_info_v1(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global llm_model_list, general_settings, user_config_file_path, proxy_config

    if llm_model_list is None:
        raise HTTPException(
            status_code=500, detail={"error": "LLM Model List not loaded in"}
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
                litellm_model_info = litellm.get_model_info(model=litellm_model)
            except:
                litellm_model_info = {}
        for k, v in litellm_model_info.items():
            if k not in model_info:
                model_info[k] = v
        model["model_info"] = model_info
        # don't return the api key
        model["litellm_params"].pop("api_key", None)

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
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
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
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.get(
    "/ollama_logs", dependencies=[Depends(user_api_key_auth)], tags=["experimental"]
)
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser("~/.ollama/logs/server.log")
    return FileResponse(filepath)


#### LOGIN ENDPOINTS ####


@app.get("/sso/key/generate", tags=["experimental"])
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
                type="auth_error",
                param="premium_user",
                code=status.HTTP_403_FORBIDDEN,
            )

    ####### Detect DB + MASTER KEY in .env #######
    if prisma_client is None or master_key is None:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=missing_keys_html_form, status_code=200)

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
                type="auth_error",
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
                type="auth_error",
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
        from fastapi_sso.sso.generic import create_provider, DiscoveryDocument

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
                type="auth_error",
                param="GENERIC_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_authorization_endpoint is None:
            raise ProxyException(
                message="GENERIC_AUTHORIZATION_ENDPOINT not set. Set it in .env file",
                type="auth_error",
                param="GENERIC_AUTHORIZATION_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_token_endpoint is None:
            raise ProxyException(
                message="GENERIC_TOKEN_ENDPOINT not set. Set it in .env file",
                type="auth_error",
                param="GENERIC_TOKEN_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_userinfo_endpoint is None:
            raise ProxyException(
                message="GENERIC_USERINFO_ENDPOINT not set. Set it in .env file",
                type="auth_error",
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
    global premium_user
    try:
        import multipart
    except ImportError:
        subprocess.run(["pip", "install", "python-multipart"])
    global master_key
    if master_key is None:
        raise ProxyException(
            message="Master Key not set for Proxy. Please set Master Key to use Admin UI. Set `LITELLM_MASTER_KEY` in .env or set general_settings:master_key in config.yaml.  https://docs.litellm.ai/docs/proxy/virtual_keys. If set, use `--detailed_debug` to debug issue.",
            type="auth_error",
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
            type="auth_error",
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
        user_role = "app_owner"
        user_id = username
        key_user_id = user_id
        if (
            os.getenv("PROXY_ADMIN_ID", None) is not None
            and os.environ["PROXY_ADMIN_ID"] == user_id
        ) or user_id == "admin":
            # checks if user is admin
            user_role = "app_admin"
            key_user_id = os.getenv("PROXY_ADMIN_ID", "default_user_id")

        # Admin is Authe'd in - generate key for the UI to access Proxy

        # ensure this user is set as the proxy admin, in this route there is no sso, we can assume this user is only the admin
        await user_update(
            data=UpdateUserRequest(
                user_id=key_user_id,
                user_role=LitellmUserRoles.PROXY_ADMIN,
            )
        )
        if os.getenv("DATABASE_URL") is not None:
            response = await generate_key_helper_fn(
                request_type="key",
                **{"user_role": LitellmUserRoles.PROXY_ADMIN, "duration": "2hr", "key_max_budget": 5, "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": key_user_id, "team_id": "litellm-dashboard"},  # type: ignore
            )
        else:
            raise ProxyException(
                message="No Database connected. Set DATABASE_URL in .env. If set, use `--detailed_debug` to debug issue.",
                type="auth_error",
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

        jwt_token = jwt.encode(
            {
                "user_id": user_id,
                "key": key,
                "user_email": user_id,
                "user_role": "app_admin",  # this is the path without sso - we can assume only admins will use this
                "login_method": "username_password",
                "premium_user": premium_user,
            },
            "secret",
            algorithm="HS256",
        )
        litellm_dashboard_ui += "?userID=" + user_id + "&token=" + jwt_token
        return RedirectResponse(url=litellm_dashboard_ui, status_code=303)
    elif _user_row is not None:
        user_id = getattr(_user_row, "user_id", "unknown")
        user_role = getattr(_user_row, "user_role", "unknown")
        user_email = getattr(_user_row, "user_email", "unknown")
        _password = getattr(_user_row, "password", "unknown")

        # check if password == _user_row.password
        if secrets.compare_digest(password, _password):
            if os.getenv("DATABASE_URL") is not None:
                response = await generate_key_helper_fn(
                    request_type="key",
                    **{  # type: ignore
                        "user_role": user_role,
                        "duration": "2hr",
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
                    type="auth_error",
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

            jwt_token = jwt.encode(
                {
                    "user_id": user_id,
                    "key": key,
                    "user_email": user_email,
                    "user_role": user_role,
                    "login_method": "username_password",
                    "premium_user": premium_user,
                },
                "secret",
                algorithm="HS256",
            )
            litellm_dashboard_ui += "?userID=" + user_id + "&token=" + jwt_token
            return RedirectResponse(url=litellm_dashboard_ui, status_code=303)
        else:
            raise ProxyException(
                message=f"Invalid credentials used to access UI. Passed in username: {username}, passed in password: {password}.\nNot valid credentials for {username}",
                type="auth_error",
                param="invalid_credentials",
                code=status.HTTP_401_UNAUTHORIZED,
            )
    else:
        raise ProxyException(
            message=f"Invalid credentials used to access UI. Passed in username: {username}, passed in password: {password}.\nCheck 'UI_USERNAME', 'UI_PASSWORD' in .env file",
            type="auth_error",
            param="invalid_credentials",
            code=status.HTTP_401_UNAUTHORIZED,
        )


@app.get("/onboarding/get_token", include_in_schema=False)
async def onboarding(invite_link: str):
    """
    - Get the invite link
    - Validate it's still 'valid'
    - Invalidate the link (prevents abuse)
    - Get user from db
    - Pass in user_email if set
    """
    global prisma_client
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
            "duration": "2hr",
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

    jwt_token = jwt.encode(
        {
            "user_id": user_obj.user_id,
            "key": key,
            "user_email": user_obj.user_email,
            "user_role": user_obj.user_role,
            "login_method": "username_password",
            "premium_user": premium_user,
        },
        "secret",
        algorithm="HS256",
    )

    litellm_dashboard_ui += "?token={}&user_email={}".format(jwt_token, user_email)
    return {
        "login_url": litellm_dashboard_ui,
        "token": jwt_token,
        "user_email": user_email,
    }


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


@app.get("/sso/callback", tags=["experimental"])
async def auth_callback(request: Request):
    """Verify login"""
    global general_settings, ui_access_mode, premium_user
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)
    # get url from request
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
                type="auth_error",
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
                type="auth_error",
                param="MICROSOFT_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if microsoft_tenant is None:
            raise ProxyException(
                message="MICROSOFT_TENANT not set. Set it in .env file",
                type="auth_error",
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
        from fastapi_sso.sso.generic import create_provider, DiscoveryDocument, OpenID

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
                type="auth_error",
                param="GENERIC_CLIENT_SECRET",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_authorization_endpoint is None:
            raise ProxyException(
                message="GENERIC_AUTHORIZATION_ENDPOINT not set. Set it in .env file",
                type="auth_error",
                param="GENERIC_AUTHORIZATION_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_token_endpoint is None:
            raise ProxyException(
                message="GENERIC_TOKEN_ENDPOINT not set. Set it in .env file",
                type="auth_error",
                param="GENERIC_TOKEN_ENDPOINT",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if generic_userinfo_endpoint is None:
            raise ProxyException(
                message="GENERIC_USERINFO_ENDPOINT not set. Set it in .env file",
                type="auth_error",
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

    # generic client id
    if generic_client_id is not None:
        user_id = getattr(result, "id", None)
        user_email = getattr(result, "email", None)
        user_role = getattr(result, generic_user_role_attribute_name, None)

    if user_id is None:
        user_id = getattr(result, "first_name", "") + getattr(result, "last_name", "")

    user_info = None
    user_id_models: List = []

    # User might not be already created on first generation of key
    # But if it is, we want their models preferences
    default_ui_key_values = {
        "duration": "2hr",
        "key_max_budget": 0.01,
        "aliases": {},
        "config": {},
        "spend": 0,
        "team_id": "litellm-dashboard",
    }
    user_defined_values = {
        "models": user_id_models,
        "user_id": user_id,
        "user_email": user_email,
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
                    "models": getattr(user_info, "models", []),
                    "user_id": getattr(user_info, "user_id", user_id),
                    "user_email": getattr(user_info, "user_id", user_email),
                    "user_role": getattr(user_info, "user_role", None),
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
                }
    except Exception as e:
        pass

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

    jwt_token = jwt.encode(
        {
            "user_id": user_id,
            "key": key,
            "user_email": user_email,
            "user_role": user_role,
            "login_method": "sso",
            "premium_user": premium_user,
        },
        "secret",
        algorithm="HS256",
    )
    litellm_dashboard_ui += "?userID=" + user_id + "&token=" + jwt_token
    return RedirectResponse(url=litellm_dashboard_ui)


#### INVITATION MANAGEMENT ####


@router.post(
    "/invitation/new",
    tags=["Invite Links"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InvitationModel,
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
                if isinstance(v, str):
                    encrypted_value = encrypt_value(value=v, master_key=master_key)  # type: ignore
                    _updated_environment_variables[k] = base64.b64encode(
                        encrypted_value
                    ).decode("utf-8")

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
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
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
    }

    return_val = []

    for field_name, field_info in ConfigGeneralSettings.model_fields.items():
        if field_name in allowed_args:

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
            )
            return_val.append(_response_obj)

    return return_val


@router.post(
    "/config/field/delete",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
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
            if _callback == "openmeter":
                env_vars = [
                    "OPENMETER_API_KEY",
                ]
                env_vars_dict = {}
                for _var in env_vars:
                    env_variable = environment_variables.get(_var, None)
                    if env_variable is None:
                        env_vars_dict[_var] = None
                    else:
                        # decode + decrypt the value
                        decoded_b64 = base64.b64decode(env_variable)
                        _decrypted_value = decrypt_value(
                            value=decoded_b64, master_key=master_key
                        )
                        env_vars_dict[_var] = _decrypted_value

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
                        decoded_b64 = base64.b64decode(env_variable)
                        _decrypted_value = decrypt_value(
                            value=decoded_b64, master_key=master_key
                        )
                        _langfuse_env_vars[_var] = _decrypted_value

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
                    decoded_b64 = base64.b64decode(env_variable)
                    _decrypted_value = decrypt_value(
                        value=decoded_b64, master_key=master_key
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
                decoded_b64 = base64.b64decode(env_variable)
                _decrypted_value = decrypt_value(
                    value=decoded_b64, master_key=master_key
                )
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
        }
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_400_BAD_REQUEST,
        )


@router.get(
    "/config/yaml",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
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


#### BASIC ENDPOINTS ####
@router.get(
    "/test",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def test_endpoint(request: Request):
    """
    [DEPRECATED] use `/health/liveliness` instead.

    A test endpoint that pings the proxy server to check if it's healthy.

    Parameters:
        request (Request): The incoming request.

    Returns:
        dict: A dictionary containing the route of the request URL.
    """
    # ping the proxy server to check if its healthy
    return {"route": request.url.path}


@router.get(
    "/health/services",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def health_services_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    service: Literal[
        "slack_budget_alerts", "langfuse", "slack", "openmeter", "webhook", "email"
    ] = fastapi.Query(description="Specify the service being hit."),
):
    """
    Hidden endpoint.

    Used by the UI to let user check if slack alerting is working as expected.
    """
    try:
        global general_settings, proxy_logging_obj

        if service is None:
            raise HTTPException(
                status_code=400, detail={"error": "Service must be specified."}
            )
        if service not in [
            "slack_budget_alerts",
            "email",
            "langfuse",
            "slack",
            "openmeter",
            "webhook",
        ]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Service must be in list. Service={service}. List={['slack_budget_alerts']}"
                },
            )

        if service == "openmeter":
            _ = await litellm.acompletion(
                model="openai/litellm-mock-response-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                user="litellm:/health/services",
                mock_response="This is a mock response",
            )
            return {
                "status": "success",
                "message": "Mock LLM request made - check openmeter.",
            }

        if service == "langfuse":
            from litellm.integrations.langfuse import LangFuseLogger

            langfuse_logger = LangFuseLogger()
            langfuse_logger.Langfuse.auth_check()
            _ = litellm.completion(
                model="openai/litellm-mock-response-model",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                user="litellm:/health/services",
                mock_response="This is a mock response",
            )
            return {
                "status": "success",
                "message": "Mock LLM request made - check langfuse.",
            }

        if service == "webhook":
            user_info = CallInfo(
                token=user_api_key_dict.token or "",
                spend=1,
                max_budget=0,
                user_id=user_api_key_dict.user_id,
                key_alias=user_api_key_dict.key_alias,
                team_id=user_api_key_dict.team_id,
            )
            await proxy_logging_obj.budget_alerts(
                type="user_budget",
                user_info=user_info,
            )

        if service == "slack" or service == "slack_budget_alerts":
            if "slack" in general_settings.get("alerting", []):
                # test_message = f"""\n🚨 `ProjectedLimitExceededError` 💸\n\n`Key Alias:` litellm-ui-test-alert \n`Expected Day of Error`: 28th March \n`Current Spend`: $100.00 \n`Projected Spend at end of month`: $1000.00 \n`Soft Limit`: $700"""
                # check if user has opted into unique_alert_webhooks
                if (
                    proxy_logging_obj.slack_alerting_instance.alert_to_webhook_url
                    is not None
                ):
                    for (
                        alert_type
                    ) in proxy_logging_obj.slack_alerting_instance.alert_to_webhook_url:
                        """
                        "llm_exceptions",
                        "llm_too_slow",
                        "llm_requests_hanging",
                        "budget_alerts",
                        "db_exceptions",
                        """
                        # only test alert if it's in active alert types
                        if (
                            proxy_logging_obj.slack_alerting_instance.alert_types
                            is not None
                            and alert_type
                            not in proxy_logging_obj.slack_alerting_instance.alert_types
                        ):
                            continue
                        test_message = "default test message"
                        if alert_type == "llm_exceptions":
                            test_message = f"LLM Exception test alert"
                        elif alert_type == "llm_too_slow":
                            test_message = f"LLM Too Slow test alert"
                        elif alert_type == "llm_requests_hanging":
                            test_message = f"LLM Requests Hanging test alert"
                        elif alert_type == "budget_alerts":
                            test_message = f"Budget Alert test alert"
                        elif alert_type == "db_exceptions":
                            test_message = f"DB Exception test alert"
                        elif alert_type == "outage_alerts":
                            test_message = f"Outage Alert Exception test alert"
                        elif alert_type == "daily_reports":
                            test_message = f"Daily Reports test alert"

                        await proxy_logging_obj.alerting_handler(
                            message=test_message, level="Low", alert_type=alert_type
                        )
                else:
                    await proxy_logging_obj.alerting_handler(
                        message="This is a test slack alert message",
                        level="Low",
                        alert_type="budget_alerts",
                    )

                if prisma_client is not None:
                    asyncio.create_task(
                        proxy_logging_obj.slack_alerting_instance.send_monthly_spend_report()
                    )
                    asyncio.create_task(
                        proxy_logging_obj.slack_alerting_instance.send_weekly_spend_report()
                    )

                alert_types = (
                    proxy_logging_obj.slack_alerting_instance.alert_types or []
                )
                alert_types = list(alert_types)
                return {
                    "status": "success",
                    "alert_types": alert_types,
                    "message": "Mock Slack Alert sent, verify Slack Alert Received on your channel",
                }
            else:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": '"{}" not in proxy config: general_settings. Unable to test this.'.format(
                            service
                        )
                    },
                )
        if service == "email":
            webhook_event = WebhookEvent(
                event="key_created",
                event_group="key",
                event_message="Test Email Alert",
                token=user_api_key_dict.token or "",
                key_alias="Email Test key (This is only a test alert key. DO NOT USE THIS IN PRODUCTION.)",
                spend=0,
                max_budget=0,
                user_id=user_api_key_dict.user_id,
                user_email=os.getenv("TEST_EMAIL_ADDRESS"),
                team_id=user_api_key_dict.team_id,
            )

            # use create task - this can take 10 seconds. don't keep ui users waiting for notification to check their email
            asyncio.create_task(
                proxy_logging_obj.slack_alerting_instance.send_key_created_or_user_invited_email(
                    webhook_event=webhook_event
                )
            )

            return {
                "status": "success",
                "message": "Mock Email Alert sent, verify Email Alert Received",
            }

    except Exception as e:
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type="auth_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type="auth_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/health", tags=["health"], dependencies=[Depends(user_api_key_auth)])
async def health_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Specify the model name (optional)"
    ),
):
    """
    🚨 USE `/health/liveliness` to health check the proxy 🚨

    See more 👉 https://docs.litellm.ai/docs/proxy/health


    Check the health of all the endpoints in config.yaml

    To run health checks in the background, add this to config.yaml:
    ```
    general_settings:
        # ... other settings
        background_health_checks: True
    ```
    else, the health checks will be run on models when /health is called.
    """
    global health_check_results, use_background_health_checks, user_model, llm_model_list
    try:
        if llm_model_list is None:
            # if no router set, check if user set a model using litellm --model ollama/llama2
            if user_model is not None:
                healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                    model_list=[], cli_model=user_model
                )
                return {
                    "healthy_endpoints": healthy_endpoints,
                    "unhealthy_endpoints": unhealthy_endpoints,
                    "healthy_count": len(healthy_endpoints),
                    "unhealthy_count": len(unhealthy_endpoints),
                }
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Model list not initialized"},
            )
        _llm_model_list = copy.deepcopy(llm_model_list)
        ### FILTER MODELS FOR ONLY THOSE USER HAS ACCESS TO ###
        if len(user_api_key_dict.models) > 0:
            allowed_model_names = user_api_key_dict.models
        else:
            allowed_model_names = []  #
        if use_background_health_checks:
            return health_check_results
        else:
            healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                _llm_model_list, model
            )

            return {
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": unhealthy_endpoints,
                "healthy_count": len(healthy_endpoints),
                "unhealthy_count": len(unhealthy_endpoints),
            }
    except Exception as e:
        traceback.print_exc()
        raise e


db_health_cache = {"status": "unknown", "last_updated": datetime.now()}


def _db_health_readiness_check():
    global db_health_cache, prisma_client

    # Note - Intentionally don't try/except this so it raises an exception when it fails

    # if timedelta is less than 2 minutes return DB Status
    time_diff = datetime.now() - db_health_cache["last_updated"]
    if db_health_cache["status"] != "unknown" and time_diff < timedelta(minutes=2):
        return db_health_cache
    prisma_client.health_check()
    db_health_cache = {"status": "connected", "last_updated": datetime.now()}
    return db_health_cache


@router.get(
    "/active/callbacks",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def active_callbacks():
    """
    Returns a list of active callbacks on litellm.callbacks, litellm.input_callback, litellm.failure_callback, litellm.success_callback
    """
    global proxy_logging_obj
    _alerting = str(general_settings.get("alerting"))
    # get success callbacks

    litellm_callbacks = [str(x) for x in litellm.callbacks]
    litellm_input_callbacks = [str(x) for x in litellm.input_callback]
    litellm_failure_callbacks = [str(x) for x in litellm.failure_callback]
    litellm_success_callbacks = [str(x) for x in litellm.success_callback]
    litellm_async_success_callbacks = [str(x) for x in litellm._async_success_callback]
    litellm_async_failure_callbacks = [str(x) for x in litellm._async_failure_callback]
    litellm_async_input_callbacks = [str(x) for x in litellm._async_input_callback]

    all_litellm_callbacks = (
        litellm_callbacks
        + litellm_input_callbacks
        + litellm_failure_callbacks
        + litellm_success_callbacks
        + litellm_async_success_callbacks
        + litellm_async_failure_callbacks
        + litellm_async_input_callbacks
    )

    alerting = proxy_logging_obj.alerting
    _num_alerting = 0
    if alerting and isinstance(alerting, list):
        _num_alerting = len(alerting)

    return {
        "alerting": _alerting,
        "litellm.callbacks": litellm_callbacks,
        "litellm.input_callback": litellm_input_callbacks,
        "litellm.failure_callback": litellm_failure_callbacks,
        "litellm.success_callback": litellm_success_callbacks,
        "litellm._async_success_callback": litellm_async_success_callbacks,
        "litellm._async_failure_callback": litellm_async_failure_callbacks,
        "litellm._async_input_callback": litellm_async_input_callbacks,
        "all_litellm_callbacks": all_litellm_callbacks,
        "num_callbacks": len(all_litellm_callbacks),
        "num_alerting": _num_alerting,
    }


@router.get(
    "/health/readiness",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_readiness():
    """
    Unprotected endpoint for checking if worker can receive requests
    """
    global general_settings
    try:
        # get success callback
        success_callback_names = []

        try:
            # this was returning a JSON of the values in some of the callbacks
            # all we need is the callback name, hence we do str(callback)
            success_callback_names = [str(x) for x in litellm.success_callback]
        except:
            # don't let this block the /health/readiness response, if we can't convert to str -> return litellm.success_callback
            success_callback_names = litellm.success_callback

        # check Cache
        cache_type = None
        if litellm.cache is not None:
            from litellm.caching import RedisSemanticCache

            cache_type = litellm.cache.type

            if isinstance(litellm.cache.cache, RedisSemanticCache):
                # ping the cache
                # TODO: @ishaan-jaff - we should probably not ping the cache on every /health/readiness check
                try:
                    index_info = await litellm.cache.cache._index_info()
                except Exception as e:
                    index_info = "index does not exist - error: " + str(e)
                cache_type = {"type": cache_type, "index_info": index_info}

        # check DB
        if prisma_client is not None:  # if db passed in, check if it's connected
            db_health_status = _db_health_readiness_check()
            return {
                "status": "healthy",
                "db": "connected",
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
                **db_health_status,
            }
        else:
            return {
                "status": "healthy",
                "db": "Not connected",
                "cache": cache_type,
                "litellm_version": version,
                "success_callbacks": success_callback_names,
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service Unhealthy ({str(e)})")


@router.get(
    "/health/liveliness",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_liveliness():
    """
    Unprotected endpoint for checking if worker is alive
    """
    return "I'm alive!"


@router.get(
    "/cache/ping",
    tags=["caching"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_ping():
    """
    Endpoint for checking if cache can be pinged
    """
    try:
        litellm_cache_params = {}
        specific_cache_params = {}

        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )

        for k, v in vars(litellm.cache).items():
            try:
                if k == "cache":
                    continue
                litellm_cache_params[k] = str(copy.deepcopy(v))
            except Exception:
                litellm_cache_params[k] = "<unable to copy or convert>"
        for k, v in vars(litellm.cache.cache).items():
            try:
                specific_cache_params[k] = str(v)
            except Exception:
                specific_cache_params[k] = "<unable to copy or convert>"
        if litellm.cache.type == "redis":
            # ping the redis cache
            ping_response = await litellm.cache.ping()
            verbose_proxy_logger.debug(
                "/cache/ping: ping_response: " + str(ping_response)
            )
            # making a set cache call
            # add cache does not return anything
            await litellm.cache.async_add_cache(
                result="test_key",
                model="test-model",
                messages=[{"role": "user", "content": "test from litellm"}],
            )
            verbose_proxy_logger.debug("/cache/ping: done with set_cache()")
            return {
                "status": "healthy",
                "cache_type": litellm.cache.type,
                "ping_response": True,
                "set_cache_response": "success",
                "litellm_cache_params": litellm_cache_params,
                "redis_cache_params": specific_cache_params,
            }
        else:
            return {
                "status": "healthy",
                "cache_type": litellm.cache.type,
                "litellm_cache_params": litellm_cache_params,
            }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unhealthy ({str(e)}).Cache parameters: {litellm_cache_params}.specific_cache_params: {specific_cache_params}",
        )


@router.post(
    "/cache/delete",
    tags=["caching"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_delete(request: Request):
    """
    Endpoint for deleting a key from the cache. All responses from litellm proxy have `x-litellm-cache-key` in the headers

    Parameters:
    - **keys**: *Optional[List[str]]* - A list of keys to delete from the cache. Example {"keys": ["key1", "key2"]}

    ```shell
    curl -X POST "http://0.0.0.0:4000/cache/delete" \
    -H "Authorization: Bearer sk-1234" \
    -d '{"keys": ["key1", "key2"]}'
    ```

    """
    try:
        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )

        request_data = await request.json()
        keys = request_data.get("keys", None)

        if litellm.cache.type == "redis":
            await litellm.cache.delete_cache_keys(keys=keys)
            return {
                "status": "success",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Cache type {litellm.cache.type} does not support deleting a key. only `redis` is supported",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache Delete Failed({str(e)})",
        )


@router.get(
    "/cache/redis/info",
    tags=["caching"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_redis_info():
    """
    Endpoint for getting /redis/info
    """
    try:
        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )
        if litellm.cache.type == "redis":
            client_list = litellm.cache.cache.client_list()
            redis_info = litellm.cache.cache.info()
            num_clients = len(client_list)
            return {
                "num_clients": num_clients,
                "clients": client_list,
                "info": redis_info,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Cache type {litellm.cache.type} does not support flushing",
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unhealthy ({str(e)})",
        )


@router.post(
    "/cache/flushall",
    tags=["caching"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_flushall():
    """
    A function to flush all items from the cache. (All items will be deleted from the cache with this)
    Raises HTTPException if the cache is not initialized or if the cache type does not support flushing.
    Returns a dictionary with the status of the operation.

    Usage:
    ```
    curl -X POST http://0.0.0.0:4000/cache/flushall -H "Authorization: Bearer sk-1234"
    ```
    """
    try:
        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )
        if litellm.cache.type == "redis":
            litellm.cache.cache.flushall()
            return {
                "status": "success",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Cache type {litellm.cache.type} does not support flushing",
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unhealthy ({str(e)})",
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
@router.get("/token/generate", dependencies=[Depends(user_api_key_auth)])
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


def _has_user_setup_sso():
    """
    Check if the user has set up single sign-on (SSO) by verifying the presence of Microsoft client ID, Google client ID, and UI username environment variables.
    Returns a boolean indicating whether SSO has been set up.
    """
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    ui_username = os.getenv("UI_USERNAME", None)

    sso_setup = (
        (microsoft_client_id is not None)
        or (google_client_id is not None)
        or (ui_username is not None)
    )

    return sso_setup


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
    prisma_client = None
    custom_db_client = None


app.include_router(router)
