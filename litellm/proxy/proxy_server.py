import sys, os, platform, time, copy, re, asyncio, inspect
import threading, ast
import shutil, random, traceback, requests
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Callable
import secrets, subprocess
import hashlib, uuid
import warnings
import importlib
import warnings
import backoff


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
    import yaml
    import orjson
    import logging
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from argon2 import PasswordHasher
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
    _read_request_body,
    _is_valid_team_configs,
    _is_user_proxy_admin,
    _is_projected_spend_over_limit,
    _get_projected_spend_over_limit,
)
from litellm.proxy.secret_managers.google_kms import load_google_kms
import pydantic
from litellm.proxy._types import *
from litellm.caching import DualCache
from litellm.proxy.health_check import perform_health_check
from litellm._logging import verbose_router_logger, verbose_proxy_logger

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
    Depends,
    BackgroundTasks,
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
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
import json
import logging
from typing import Union

ui_link = f"/ui/"
ui_message = (
    f"👉 [```LiteLLM Admin Panel on /ui```]({ui_link}). Create, Edit Keys with SSO"
)
app = FastAPI(
    docs_url="/",
    title="LiteLLM API",
    description=f"Proxy Server to call 100+ LLMs in the OpenAI format\n\n{ui_message}",
    version=version,
    root_path=os.environ.get(
        "SERVER_ROOT_PATH", ""
    ),  # check if user passed root path, FastAPI defaults this value to ""
)


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

    def to_dict(self) -> dict:
        """Converts the ProxyException instance to a dictionary."""
        return {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }


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
ph = PasswordHasher()
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
user_api_key_cache = DualCache()
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
litellm_master_key_hash = None
### INITIALIZE GLOBAL LOGGING OBJECT ###
proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
### REDIS QUEUE ###
async_result = None
celery_app_conn = None
celery_fn = None  # Redis Queue for handling requests
### logger ###


def usage_telemetry(
    feature: str,
):  # helps us know if people are using this feature. Set `litellm --telemetry False` to your cli call to turn this off
    if user_telemetry:
        data = {"feature": feature}  # "local_proxy_server"
        threading.Thread(
            target=litellm.utils.litellm_telemetry, args=(data,), daemon=True
        ).start()


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


async def user_api_key_auth(
    request: Request, api_key: str = fastapi.Security(api_key_header)
) -> UserAPIKeyAuth:
    global master_key, prisma_client, llm_model_list, user_custom_auth, custom_db_client
    try:
        if isinstance(api_key, str):
            passed_in_key = api_key
            api_key = _get_bearer_token(api_key=api_key)

        ### USER-DEFINED AUTH FUNCTION ###
        if user_custom_auth is not None:
            response = await user_custom_auth(request=request, api_key=api_key)
            return UserAPIKeyAuth.model_validate(response)

        ### LITELLM-DEFINED AUTH FUNCTION ###
        if master_key is None:
            if isinstance(api_key, str):
                return UserAPIKeyAuth(api_key=api_key)
            else:
                return UserAPIKeyAuth()

        route: str = request.url.path
        if route == "/user/auth":
            if general_settings.get("allow_user_auth", False) == True:
                return UserAPIKeyAuth()
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="'allow_user_auth' not set or set to False",
                )
        elif (
            route == "/routes"
            or route == "/"
            or route == "/health/liveliness"
            or route == "/health/readiness"
            or route == "/test"
            or route == "/config/yaml"
        ):
            """
            Unprotected endpoints
            """
            return UserAPIKeyAuth()
        elif route.startswith("/config/"):
            raise Exception(f"Only admin can modify config")

        if api_key is None:  # only require api key if master key is set
            raise Exception(f"No api key passed in.")

        if api_key == "":
            # missing 'Bearer ' prefix
            raise Exception(
                f"Malformed API Key passed in. Ensure Key has `Bearer ` prefix. Passed in: {passed_in_key}"
            )

        ### CHECK IF ADMIN ###
        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        try:
            is_master_key_valid = ph.verify(litellm_master_key_hash, api_key)
        except Exception as e:
            is_master_key_valid = False

        if is_master_key_valid:
            return UserAPIKeyAuth(
                api_key=master_key,
                user_role="proxy_admin",
                user_id=litellm_proxy_admin_name,
            )

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
        valid_token = user_api_key_cache.get_cache(key=api_key)
        if valid_token is None:
            ## check db
            verbose_proxy_logger.debug(f"api key: {api_key}")
            if prisma_client is not None:
                valid_token = await prisma_client.get_data(
                    token=api_key, table_name="combined_view"
                )
            elif custom_db_client is not None:
                try:
                    valid_token = await custom_db_client.get_data(
                        key=api_key, table_name="key"
                    )
                except:
                    # (Patch: For DynamoDB Backwards Compatibility)
                    valid_token = await custom_db_client.get_data(
                        key=original_api_key, table_name="key"
                    )
            verbose_proxy_logger.debug(f"Token from db: {valid_token}")
        elif valid_token is not None:
            verbose_proxy_logger.debug(f"API Key Cache Hit!")
        if valid_token:
            # Got Valid Token from Cache, DB
            # Run checks for
            # 1. If token can call model
            # 2. If user_id for this token is in budget
            # 3. If 'user' passed to /chat/completions, /embeddings endpoint is in budget
            # 4. If token is expired
            # 5. If token spend is under Budget for the token
            # 6. If token spend per model is under budget per model
            # 7. If token spend is under team budget
            # 8. If team spend is under team budget
            request_data = await _read_request_body(
                request=request
            )  # request data, used across all checks. Making this easily available

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
                if model is not None and model not in filtered_models:
                    raise ValueError(
                        f"API Key not allowed to access model. This token can only access models={valid_token.models}. Tried to access {model}"
                    )
                valid_token.models = filtered_models
                verbose_proxy_logger.debug(
                    f"filtered allowed_models: {filtered_models}; valid_token.models: {valid_token.models}"
                )

            # Check 2. If user_id for this token is in budget
            ## Check 2.1 If global proxy is in budget
            ## Check 2.2 [OPTIONAL - checked only if litellm.max_user_budget is not None] If 'user' passed in /chat/completions is in budget
            if valid_token.user_id is not None:
                user_id_list = [valid_token.user_id, litellm_proxy_budget_name]
                if (
                    litellm.max_user_budget is not None
                ):  # Check if 'user' passed in /chat/completions is in budget, only checked if litellm.max_user_budget is set
                    user_passed_to_chat_completions = request_data.get("user", None)
                    if user_passed_to_chat_completions is not None:
                        user_id_list.append(user_passed_to_chat_completions)

                user_id_information = None
                for id in user_id_list:
                    value = user_api_key_cache.get_cache(key=id)
                    if value is not None:
                        if user_id_information is None:
                            user_id_information = []
                        user_id_information.append(value)
                if user_id_information is None or (
                    isinstance(user_id_information, list)
                    and len(user_id_information) < 2
                ):
                    if prisma_client is not None:
                        user_id_information = await prisma_client.get_data(
                            user_id_list=[
                                valid_token.user_id,
                                litellm_proxy_budget_name,
                            ],
                            table_name="user",
                            query_type="find_all",
                        )
                        for _id in user_id_information:
                            user_api_key_cache.set_cache(
                                key=_id["user_id"], value=_id, ttl=600
                            )
                    if custom_db_client is not None:
                        user_id_information = await custom_db_client.get_data(
                            key=valid_token.user_id, table_name="user"
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
                                asyncio.create_task(
                                    proxy_logging_obj.budget_alerts(
                                        user_max_budget=user_max_budget,
                                        user_current_spend=user_current_spend,
                                        type="user_and_proxy_budget",
                                        user_info=_user,
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
                            asyncio.create_task(
                                proxy_logging_obj.budget_alerts(
                                    user_max_budget=user_max_budget,
                                    user_current_spend=user_current_spend,
                                    type="user_budget",
                                    user_info=user_id_information,
                                )
                            )

                            if user_current_spend > user_max_budget:
                                raise Exception(
                                    f"ExceededBudget: User {valid_token.user_id} has exceeded their budget. Current spend: {user_current_spend}; Max Budget: {user_max_budget}"
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
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        user_max_budget=valid_token.max_budget,
                        user_current_spend=valid_token.spend,
                        type="token_budget",
                        user_info=valid_token,
                    )
                )

                if valid_token.spend > valid_token.max_budget:
                    raise Exception(
                        f"ExceededTokenBudget: Current spend for token: {valid_token.spend}; Max Budget for Token: {valid_token.max_budget}"
                    )

            # Check 5. Token Model Spend is under Model budget
            max_budget_per_model = valid_token.model_max_budget
            spend_per_model = valid_token.model_spend

            if max_budget_per_model is not None and spend_per_model is not None:
                current_model = request_data.get("model")
                if current_model is not None:
                    current_model_spend = spend_per_model.get(current_model, None)
                    current_model_budget = max_budget_per_model.get(current_model, None)

                    if (
                        current_model_spend is not None
                        and current_model_budget is not None
                    ):
                        if current_model_spend > current_model_budget:
                            raise Exception(
                                f"ExceededModelBudget: Current spend for model: {current_model_spend}; Max Budget for Model: {current_model_budget}"
                            )

            # Check 6. Token spend is under Team budget
            if (
                valid_token.spend is not None
                and hasattr(valid_token, "team_max_budget")
                and valid_token.team_max_budget is not None
            ):
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        user_max_budget=valid_token.team_max_budget,
                        user_current_spend=valid_token.spend,
                        type="token_budget",
                        user_info=valid_token,
                    )
                )

                if valid_token.spend >= valid_token.team_max_budget:
                    raise Exception(
                        f"ExceededTokenBudget: Current spend for token: {valid_token.spend}; Max Budget for Team: {valid_token.team_max_budget}"
                    )

            # Check 7. Team spend is under Team budget
            if (
                hasattr(valid_token, "team_spend")
                and valid_token.team_spend is not None
                and hasattr(valid_token, "team_max_budget")
                and valid_token.team_max_budget is not None
            ):
                asyncio.create_task(
                    proxy_logging_obj.budget_alerts(
                        user_max_budget=valid_token.team_max_budget,
                        user_current_spend=valid_token.team_spend,
                        type="token_budget",
                        user_info=valid_token,
                    )
                )

                if valid_token.team_spend >= valid_token.team_max_budget:
                    raise Exception(
                        f"ExceededTokenBudget: Current Team Spend: {valid_token.team_spend}; Max Budget for Team: {valid_token.team_max_budget}"
                    )

            # Token passed all checks
            api_key = valid_token.token

            # Add hashed token to cache
            user_api_key_cache.set_cache(key=api_key, value=valid_token, ttl=600)
            valid_token_dict = _get_pydantic_json_dict(valid_token)
            valid_token_dict.pop("token", None)
            """
            asyncio create task to update the user api key cache with the user db table as well

            This makes the user row data accessible to pre-api call hooks.
            """
            if prisma_client is not None:
                asyncio.create_task(
                    _cache_user_row(
                        user_id=valid_token.user_id,
                        cache=user_api_key_cache,
                        db=prisma_client,
                    )
                )
            elif custom_db_client is not None:
                asyncio.create_task(
                    _cache_user_row(
                        user_id=valid_token.user_id,
                        cache=user_api_key_cache,
                        db=custom_db_client,
                    )
                )
            if (
                (
                    route.startswith("/key/")
                    or route.startswith("/user/")
                    or route.startswith("/model/")
                    or route.startswith("/spend/")
                )
                and (not is_master_key_valid)
                and (not _is_user_proxy_admin(user_id_information))
            ):
                allow_user_auth = False
                if (
                    general_settings.get("allow_user_auth", False) == True
                    or _has_user_setup_sso() == True
                ):
                    allow_user_auth = True  # user can create and delete their own keys
                # enters this block when allow_user_auth is set to False
                if route == "/key/info":
                    # check if user can access this route
                    query_params = request.query_params
                    key = query_params.get("key")
                    if (
                        key is not None
                        and prisma_client.hash_token(token=key) != api_key
                    ):
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
                elif route == "/user/update":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="only proxy admin can update user settings. Tried calling `/user/update`",
                    )
                elif route == "/model/info":
                    # /model/info just shows models user has access to
                    pass
                elif route == "/user/request_model":
                    pass  # this allows any user to request a model through the UI
                elif allow_user_auth == True and route == "/key/generate":
                    pass
                elif allow_user_auth == True and route == "/key/delete":
                    pass
                elif route == "/spend/logs":
                    # check if user can access this route
                    # user can only access this route if
                    # - api_key they need logs for has the same user_id as the one used for auth
                    query_params = request.query_params
                    if query_params.get("api_key") is not None:
                        api_key = query_params.get("api_key")
                        token_info = await prisma_client.get_data(
                            token=api_key, table_name="key", query_type="find_unique"
                        )
                        if secrets.compare_digest(
                            token_info.user_id, valid_token.user_id
                        ):
                            pass
                    elif query_params.get("user_id") is not None:
                        user_id = query_params.get("user_id")
                        # check if user id == token.user_id
                        if secrets.compare_digest(user_id, valid_token.user_id):
                            pass
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="user not allowed to access this key's info",
                        )
                else:
                    raise Exception(
                        f"Only master key can be used to generate, delete, update or get info for new keys/users. Value of allow_user_auth={allow_user_auth}"
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
                "/key",
                "/spend",
                "/user",
                "/model/info",
                "/v2/model/info",
                "/v2/key/info",
                "/models",
                "/v1/models",
                "/global/spend/logs",
                "/global/spend/keys",
                "/global/spend/models",
                "/global/predict/spend/logs",
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
                if _is_user_proxy_admin(user_id_information):
                    return UserAPIKeyAuth(
                        api_key=api_key, user_role="proxy_admin", **valid_token_dict
                    )
                else:
                    raise Exception(
                        f"This key is made for LiteLLM UI, Tried to access route: {route}. Not allowed"
                    )
        if valid_token_dict is not None:
            return UserAPIKeyAuth(api_key=api_key, **valid_token_dict)
        else:
            raise Exception()
    except Exception as e:
        # verbose_proxy_logger.debug(f"An exception occurred - {traceback.format_exc()}")
        traceback.print_exc()
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


async def _PROXY_track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    verbose_proxy_logger.debug(f"INSIDE _PROXY_track_cost_callback")
    global prisma_client, custom_db_client
    try:
        # check if it has collected an entire stream response
        verbose_proxy_logger.debug(f"Proxy: In track_cost_callback for {kwargs}")
        verbose_proxy_logger.debug(
            f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}"
        )
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
        user_id = proxy_server_request.get("body", {}).get("user", None)
        user_id = user_id or kwargs["litellm_params"]["metadata"].get(
            "user_api_key_user_id", None
        )
        team_id = kwargs["litellm_params"]["metadata"].get("user_api_key_team_id", None)
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

            verbose_proxy_logger.info(
                f"response_cost {response_cost}, for user_id {user_id}"
            )
            verbose_proxy_logger.debug(
                f"user_api_key {user_api_key}, prisma_client: {prisma_client}, custom_db_client: {custom_db_client}"
            )
            if user_api_key is not None:
                ## UPDATE DATABASE
                await update_database(
                    token=user_api_key,
                    response_cost=response_cost,
                    user_id=user_id,
                    team_id=team_id,
                    kwargs=kwargs,
                    completion_response=completion_response,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                raise Exception("User API key missing from custom callback.")
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
        user_id = user_id or "not-found"
        asyncio.create_task(
            proxy_logging_obj.budget_alerts(
                user_max_budget=0,
                user_current_spend=0,
                type="failed_tracking",
                user_info=user_id,
                error_message=error_msg,
            )
        )
        verbose_proxy_logger.debug(f"error in tracking cost callback - {error_msg}")


async def update_database(
    token,
    response_cost,
    user_id=None,
    team_id=None,
    kwargs=None,
    completion_response=None,
    start_time=None,
    end_time=None,
):
    try:
        verbose_proxy_logger.info(
            f"Enters prisma db call, response_cost: {response_cost}, token: {token}; user_id: {user_id}; team_id: {team_id}"
        )

        ### [TODO] STEP 1: GET KEY + USER SPEND ### (key, user)

        ### [TODO] STEP 2: UPDATE SPEND ### (key, user, spend logs)

        ### UPDATE USER SPEND ###
        async def _update_user_db():
            """
            - Update that user's row
            - Update litellm-proxy-budget row (global proxy spend)
            """
            user_ids = [user_id, litellm_proxy_budget_name]
            data_list = []
            try:
                for id in user_ids:
                    if id is None:
                        continue
                    if prisma_client is not None:
                        existing_spend_obj = await prisma_client.get_data(user_id=id)
                    elif (
                        custom_db_client is not None and id != litellm_proxy_budget_name
                    ):
                        existing_spend_obj = await custom_db_client.get_data(
                            key=id, table_name="user"
                        )
                    verbose_proxy_logger.debug(
                        f"Updating existing_spend_obj: {existing_spend_obj}"
                    )
                    if existing_spend_obj is None:
                        # if user does not exist in LiteLLM_UserTable, create a new user
                        existing_spend = 0
                        max_user_budget = None
                        if litellm.max_user_budget is not None:
                            max_user_budget = litellm.max_user_budget
                        existing_spend_obj = LiteLLM_UserTable(
                            user_id=id,
                            spend=0,
                            max_budget=max_user_budget,
                            user_email=None,
                        )
                    else:
                        existing_spend = existing_spend_obj.spend

                    # Calculate the new cost by adding the existing cost and response_cost
                    existing_spend_obj.spend = existing_spend + response_cost

                    # track cost per model, for the given user
                    spend_per_model = existing_spend_obj.model_spend or {}
                    current_model = kwargs.get("model")

                    if current_model is not None and spend_per_model is not None:
                        if spend_per_model.get(current_model) is None:
                            spend_per_model[current_model] = response_cost
                        else:
                            spend_per_model[current_model] += response_cost
                    existing_spend_obj.model_spend = spend_per_model

                    valid_token = user_api_key_cache.get_cache(key=id)
                    if valid_token is not None and isinstance(valid_token, dict):
                        user_api_key_cache.set_cache(
                            key=id, value=existing_spend_obj.json()
                        )

                    verbose_proxy_logger.debug(
                        f"user - new cost: {existing_spend_obj.spend}, user_id: {id}"
                    )
                    data_list.append(existing_spend_obj)

                    if custom_db_client is not None and user_id is not None:
                        new_spend = data_list[0].spend
                        await custom_db_client.update_data(
                            key=user_id, value={"spend": new_spend}, table_name="user"
                        )
                # Update the cost column for the given user id
                if prisma_client is not None:
                    await prisma_client.update_data(
                        data_list=data_list,
                        query_type="update_many",
                        table_name="user",
                    )
            except Exception as e:
                verbose_proxy_logger.info(
                    f"Update User DB call failed to execute {str(e)}"
                )

        ### UPDATE KEY SPEND ###
        async def _update_key_db():
            try:
                verbose_proxy_logger.debug(
                    f"adding spend to key db. Response cost: {response_cost}. Token: {token}."
                )
                if prisma_client is not None:
                    # Fetch the existing cost for the given token
                    existing_spend_obj = await prisma_client.get_data(token=token)
                    verbose_proxy_logger.debug(
                        f"_update_key_db: existing spend: {existing_spend_obj}"
                    )
                    if existing_spend_obj is None:
                        existing_spend = 0
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
                                soft_budget_limit=existing_spend_obj.litellm_budget_table.soft_budget,
                            )
                            == True
                        )
                    ):
                        key_alias = existing_spend_obj.key_alias
                        projected_spend, projected_exceeded_date = (
                            _get_projected_spend_over_limit(
                                current_spend=new_spend,
                                soft_budget_limit=existing_spend_obj.litellm_budget_table.soft_budget,
                            )
                        )
                        soft_limit = existing_spend_obj.litellm_budget_table.soft_budget
                        user_info = {
                            "key_alias": key_alias,
                            "projected_spend": projected_spend,
                            "projected_exceeded_date": projected_exceeded_date,
                        }
                        # alert user
                        asyncio.create_task(
                            proxy_logging_obj.budget_alerts(
                                type="projected_limit_exceeded",
                                user_info=user_info,
                                user_max_budget=soft_limit,
                                user_current_spend=new_spend,
                            )
                        )
                        # set cooldown on alert
                        soft_budget_cooldown = True
                    # track cost per model, for the given key
                    spend_per_model = existing_spend_obj.model_spend or {}
                    current_model = kwargs.get("model")
                    if current_model is not None and spend_per_model is not None:
                        if spend_per_model.get(current_model) is None:
                            spend_per_model[current_model] = response_cost
                        else:
                            spend_per_model[current_model] += response_cost

                    verbose_proxy_logger.debug(
                        f"new cost: {new_spend}, new spend per model: {spend_per_model}"
                    )
                    # Update the cost column for the given token
                    await prisma_client.update_data(
                        token=token,
                        data={
                            "spend": new_spend,
                            "model_spend": spend_per_model,
                            "soft_budget_cooldown": soft_budget_cooldown,
                        },
                    )

                    valid_token = user_api_key_cache.get_cache(key=token)
                    if valid_token is not None:
                        valid_token.spend = new_spend
                        valid_token.model_spend = spend_per_model
                        user_api_key_cache.set_cache(key=token, value=valid_token)
                elif custom_db_client is not None:
                    # Fetch the existing cost for the given token
                    existing_spend_obj = await custom_db_client.get_data(
                        key=token, table_name="key"
                    )
                    verbose_proxy_logger.debug(
                        f"_update_key_db existing spend: {existing_spend_obj}"
                    )
                    if existing_spend_obj is None:
                        existing_spend = 0
                    else:
                        existing_spend = existing_spend_obj.spend
                    # Calculate the new cost by adding the existing cost and response_cost
                    new_spend = existing_spend + response_cost

                    verbose_proxy_logger.debug(f"new cost: {new_spend}")
                    # Update the cost column for the given token
                    await custom_db_client.update_data(
                        key=token, value={"spend": new_spend}, table_name="key"
                    )

                    valid_token = user_api_key_cache.get_cache(key=token)
                    if valid_token is not None:
                        valid_token.spend = new_spend
                        user_api_key_cache.set_cache(key=token, value=valid_token)
            except Exception as e:
                traceback.print_exc()
                verbose_proxy_logger.info(
                    f"Update Key DB Call failed to execute - {str(e)}"
                )

        ### UPDATE SPEND LOGS ###
        async def _insert_spend_log_to_db():
            try:
                # Helper to generate payload to log
                verbose_proxy_logger.debug("inserting spend log to db")
                payload = get_logging_payload(
                    kwargs=kwargs,
                    response_obj=completion_response,
                    start_time=start_time,
                    end_time=end_time,
                )

                payload["spend"] = response_cost
                if prisma_client is not None:
                    await prisma_client.insert_data(data=payload, table_name="spend")
                elif custom_db_client is not None:
                    await custom_db_client.insert_data(payload, table_name="spend")

            except Exception as e:
                verbose_proxy_logger.info(
                    f"Update Spend Logs DB failed to execute - {str(e)}"
                )

        ### UPDATE KEY SPEND ###
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
                    # Fetch the existing cost for the given token
                    existing_spend_obj = await prisma_client.get_data(
                        team_id=team_id, table_name="team"
                    )
                    verbose_proxy_logger.debug(
                        f"_update_team_db: existing spend: {existing_spend_obj}"
                    )
                    if existing_spend_obj is None:
                        # the team does not exist in the db - return
                        verbose_proxy_logger.debug(
                            "team_id does not exist in db, not tracking spend for team"
                        )
                        return
                    else:
                        existing_spend = existing_spend_obj.spend
                    # Calculate the new cost by adding the existing cost and response_cost
                    new_spend = existing_spend + response_cost
                    spend_per_model = getattr(existing_spend_obj, "model_spend", {})
                    # track cost per model, for the given team
                    spend_per_model = existing_spend_obj.model_spend or {}
                    current_model = kwargs.get("model")
                    if current_model is not None and spend_per_model is not None:
                        if spend_per_model.get(current_model) is None:
                            spend_per_model[current_model] = response_cost
                        else:
                            spend_per_model[current_model] += response_cost

                    verbose_proxy_logger.debug(f"new cost: {new_spend}")
                    # Update the cost column for the given token
                    await prisma_client.update_data(
                        team_id=team_id,
                        data={"spend": new_spend, "model_spend": spend_per_model},
                        table_name="team",
                    )

                elif custom_db_client is not None:
                    # Fetch the existing cost for the given token
                    existing_spend_obj = await custom_db_client.get_data(
                        key=token, table_name="key"
                    )
                    verbose_proxy_logger.debug(
                        f"_update_key_db existing spend: {existing_spend_obj}"
                    )
                    if existing_spend_obj is None:
                        existing_spend = 0
                    else:
                        existing_spend = existing_spend_obj.spend
                    # Calculate the new cost by adding the existing cost and response_cost
                    new_spend = existing_spend + response_cost

                    verbose_proxy_logger.debug(f"new cost: {new_spend}")
                    # Update the cost column for the given token
                    await custom_db_client.update_data(
                        key=token, value={"spend": new_spend}, table_name="key"
                    )

                    valid_token = user_api_key_cache.get_cache(key=token)
                    if valid_token is not None:
                        valid_token.spend = new_spend
                        user_api_key_cache.set_cache(key=token, value=valid_token)
            except Exception as e:
                verbose_proxy_logger.info(
                    f"Update Team DB failed to execute - {str(e)}"
                )

        asyncio.create_task(_update_user_db())
        asyncio.create_task(_update_key_db())
        asyncio.create_task(_update_team_db())
        asyncio.create_task(_insert_spend_log_to_db())
        verbose_proxy_logger.info("Successfully updated spend in all 3 tables")
    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error updating Prisma database: {traceback.format_exc()}"
        )
        pass


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
        if (
            prisma_client is not None
            and litellm.get_secret("SAVE_CONFIG_TO_DB", False) == True
        ):
            prisma_setup(database_url=None)  # in case it's not been connected yet
            _tasks = []
            keys = [
                "model_list",
                "general_settings",
                "router_settings",
                "litellm_settings",
            ]
            for k in keys:
                response = prisma_client.get_generic_data(
                    key="param_name", value=k, table_name="config"
                )
                _tasks.append(response)

            responses = await asyncio.gather(*_tasks)

        return config

    async def save_config(self, new_config: dict):
        global prisma_client, llm_router, user_config_file_path, llm_model_list, general_settings
        # Load existing config
        backup_config = await self.get_config()

        # Save the updated config
        ## YAML
        with open(f"{user_config_file_path}", "w") as config_file:
            yaml.dump(new_config, config_file, default_flow_style=False)

        # update Router - verifies if this is a valid config
        try:
            (
                llm_router,
                llm_model_list,
                general_settings,
            ) = await proxy_config.load_config(
                router=llm_router, config_file_path=user_config_file_path
            )
        except Exception as e:
            traceback.print_exc()
            # Revert to old config instead
            with open(f"{user_config_file_path}", "w") as config_file:
                yaml.dump(backup_config, config_file, default_flow_style=False)
            raise HTTPException(status_code=400, detail="Invalid config passed in")

        ## DB - writes valid config to db
        """
        - Do not write restricted params like 'api_key' to the database
        - if api_key is passed, save that to the local environment or connected secret manage (maybe expose `litellm.save_secret()`)
        """
        if (
            prisma_client is not None
            and litellm.get_secret("SAVE_CONFIG_TO_DB", default_value=False) == True
        ):
            ### KEY REMOVAL ###
            models = new_config.get("model_list", [])
            for m in models:
                if m.get("litellm_params", {}).get("api_key", None) is not None:
                    # pop the key
                    api_key = m["litellm_params"].pop("api_key")
                    # store in local env
                    key_name = f"LITELLM_MODEL_KEY_{uuid.uuid4()}"
                    os.environ[key_name] = api_key
                    # save the key name (not the value)
                    m["litellm_params"]["api_key"] = f"os.environ/{key_name}"
            await prisma_client.insert_data(data=new_config, table_name="config")

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

    async def load_config(
        self, router: Optional[litellm.Router], config_file_path: str
    ):
        """
        Load config values into proxy global state
        """
        global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, user_custom_key_generate, use_background_health_checks, health_check_interval, use_queue, custom_db_client, proxy_budget_rescheduler_max_time, proxy_budget_rescheduler_min_time, ui_access_mode, litellm_master_key_hash

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

                    verbose_proxy_logger.debug(f"passed cache type={cache_type}")

                    if cache_type == "redis" or cache_type == "redis-semantic":
                        cache_host = litellm.get_secret("REDIS_HOST", None)
                        cache_port = litellm.get_secret("REDIS_PORT", None)
                        cache_password = litellm.get_secret("REDIS_PASSWORD", None)

                        cache_params.update(
                            {
                                "type": cache_type,
                                "host": cache_host,
                                "port": cache_port,
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
                    litellm.cache = Cache(**cache_params)
                    print(  # noqa
                        f"{blue_color_code}Set Cache on LiteLLM Proxy: {vars(litellm.cache.cache)}{reset_color_code}"
                    )
                elif key == "callbacks":
                    if isinstance(value, list):
                        imported_list: List[Any] = []
                        for callback in value:  # ["presidio", <my-custom-callback>]
                            if isinstance(callback, str) and callback == "presidio":
                                from litellm.proxy.hooks.presidio_pii_masking import (
                                    _OPTIONAL_PresidioPIIMasking,
                                )

                                pii_masking_object = _OPTIONAL_PresidioPIIMasking()
                                imported_list.append(pii_masking_object)
                            elif (
                                isinstance(callback, str)
                                and callback == "llamaguard_moderations"
                            ):
                                from litellm.proxy.enterprise.enterprise_hooks.llama_guard import (
                                    _ENTERPRISE_LlamaGuard,
                                )

                                llama_guard_object = _ENTERPRISE_LlamaGuard()
                                imported_list.append(llama_guard_object)
                            elif (
                                isinstance(callback, str)
                                and callback == "google_text_moderation"
                            ):
                                from litellm.proxy.enterprise.enterprise_hooks.google_text_moderation import (
                                    _ENTERPRISE_GoogleTextModeration,
                                )

                                google_text_moderation_obj = (
                                    _ENTERPRISE_GoogleTextModeration()
                                )
                                imported_list.append(google_text_moderation_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "llmguard_moderations"
                            ):
                                from litellm.proxy.enterprise.enterprise_hooks.llm_guard import (
                                    _ENTERPRISE_LLMGuard,
                                )

                                llm_guard_moderation_obj = _ENTERPRISE_LLMGuard()
                                imported_list.append(llm_guard_moderation_obj)
                            elif (
                                isinstance(callback, str)
                                and callback == "blocked_user_check"
                            ):
                                from litellm.proxy.enterprise.enterprise_hooks.blocked_user_list import (
                                    _ENTERPRISE_BlockedUserList,
                                )

                                blocked_user_list = _ENTERPRISE_BlockedUserList()
                                imported_list.append(blocked_user_list)
                            elif (
                                isinstance(callback, str)
                                and callback == "banned_keywords"
                            ):
                                from litellm.proxy.enterprise.enterprise_hooks.banned_keywords import (
                                    _ENTERPRISE_BannedKeywords,
                                )

                                banned_keywords_obj = _ENTERPRISE_BannedKeywords()
                                imported_list.append(banned_keywords_obj)
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

                    # intialize success callbacks
                    for callback in value:
                        # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                        if "." in callback:
                            litellm.success_callback.append(
                                get_instance_fn(value=callback)
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.success_callback.append(callback)
                    print(  # noqa
                        f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}"
                    )  # noqa
                elif key == "failure_callback":
                    litellm.failure_callback = []

                    # intialize success callbacks
                    for callback in value:
                        # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                        if "." in callback:
                            litellm.failure_callback.append(
                                get_instance_fn(value=callback)
                            )
                        # these are litellm callbacks - "langfuse", "sentry", "wandb"
                        else:
                            litellm.failure_callback.append(callback)
                    verbose_proxy_logger.debug(
                        f"{blue_color_code} Initialized Success Callbacks - {litellm.failure_callback} {reset_color_code}"
                    )
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
                else:
                    raise ValueError("Invalid Key Management System selected")
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
            )
            ### CONNECT TO DATABASE ###
            database_url = general_settings.get("database_url", None)
            if database_url and database_url.startswith("os.environ/"):
                verbose_proxy_logger.debug(f"GOING INTO LITELLM.GET_SECRET!")
                database_url = litellm.get_secret(database_url)
                verbose_proxy_logger.debug(f"RETRIEVED DB URL: {database_url}")
            ### MASTER KEY ###
            master_key = general_settings.get(
                "master_key", litellm.get_secret("LITELLM_MASTER_KEY", None)
            )
            if master_key and master_key.startswith("os.environ/"):
                master_key = litellm.get_secret(master_key)

            if master_key is not None and isinstance(master_key, str):
                litellm_master_key_hash = ph.hash(master_key)
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
                verbose_proxy_logger.debug(f"database_args: {database_args}")
                custom_db_client = DBClient(
                    custom_db_args=database_args, custom_db_type=database_type
                )
            ## COST TRACKING ##
            cost_tracking()
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
    duration: Optional[str],
    models: list,
    aliases: dict,
    config: dict,
    spend: float,
    key_max_budget: Optional[float] = None,  # key_max_budget is used to Budget Per key
    key_budget_duration: Optional[str] = None,
    key_soft_budget: Optional[
        float
    ] = None,  # key_soft_budget is used to Budget Per key
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
    table_name: Optional[Literal["key", "user"]] = None,
):
    global prisma_client, custom_db_client, user_api_key_cache

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
        expires = datetime.utcnow() + timedelta(seconds=duration_s)

    if key_budget_duration is None:  # one-time budget
        key_reset_at = None
    else:
        duration_s = _duration_in_seconds(duration=key_budget_duration)
        key_reset_at = datetime.utcnow() + timedelta(seconds=duration_s)

    if budget_duration is None:  # one-time budget
        reset_at = None
    else:
        duration_s = _duration_in_seconds(duration=budget_duration)
        reset_at = datetime.utcnow() + timedelta(seconds=duration_s)

    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    permissions_json = json.dumps(permissions)
    metadata_json = json.dumps(metadata)
    model_max_budget_json = json.dumps(model_max_budget)

    user_id = user_id or str(uuid.uuid4())
    user_role = user_role or "app_user"
    tpm_limit = tpm_limit
    rpm_limit = rpm_limit
    allowed_cache_controls = allowed_cache_controls

    # TODO: @ishaan-jaff: Migrate all budget tracking to use LiteLLM_BudgetTable
    _budget_id = None
    if prisma_client is not None and key_soft_budget is not None:
        # create the Budget Row for the LiteLLM Verification Token
        budget_row = LiteLLM_BudgetTable(
            soft_budget=key_soft_budget,
            model_max_budget=model_max_budget or {},
        )
        new_budget = prisma_client.jsonify_object(budget_row.json(exclude_none=True))

        _budget = await prisma_client.db.litellm_budgettable.create(
            data={
                **new_budget,  # type: ignore
                "created_by": user_id,
                "updated_by": user_id,
            }
        )
        _budget_id = getattr(_budget, "budget_id", None)

    try:
        # Create a new verification token (you may want to enhance this logic based on your needs)
        user_data = {
            "max_budget": max_budget,
            "user_email": user_email,
            "user_id": user_id,
            "team_id": team_id,
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
            "budget_id": _budget_id,
        }
        if (
            general_settings.get("allow_user_auth", False) == True
            or _has_user_setup_sso() == True
        ):
            key_data["key_name"] = f"sk-...{token[-4:]}"
        saved_token = copy.deepcopy(key_data)
        if isinstance(saved_token["aliases"], str):
            saved_token["aliases"] = json.loads(saved_token["aliases"])
        if isinstance(saved_token["config"], str):
            saved_token["config"] = json.loads(saved_token["config"])
        if isinstance(saved_token["metadata"], str):
            saved_token["metadata"] = json.loads(saved_token["metadata"])
        if isinstance(saved_token["permissions"], str):
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
            ## CREATE USER (If necessary)
            verbose_proxy_logger.debug(f"prisma_client: Creating User={user_data}")
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
                return key_data

            ## CREATE KEY
            verbose_proxy_logger.debug(f"prisma_client: Creating Key={key_data}")
            await prisma_client.insert_data(data=key_data, table_name="key")
        elif custom_db_client is not None:
            ## CREATE USER (If necessary)
            verbose_proxy_logger.debug(f"CustomDBClient: Creating User={user_data}")
            user_row = await custom_db_client.insert_data(
                value=user_data, table_name="user"
            )
            if user_row is None:
                # GET USER ROW
                user_row = await custom_db_client.get_data(
                    key=user_id, table_name="user"
                )

            ## use default user model list if no key-specific model list provided
            if len(user_row.models) > 0 and len(key_data["models"]) == 0:  # type: ignore
                key_data["models"] = user_row.models
            ## CREATE KEY
            verbose_proxy_logger.debug(f"CustomDBClient: Creating Key={key_data}")
            await custom_db_client.insert_data(value=key_data, table_name="key")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Add budget related info in key_data - this ensures it's returned
    key_data["soft_budget"] = key_soft_budget
    return key_data


async def delete_verification_token(tokens: List, user_id: Optional[str] = None):
    global prisma_client
    try:
        if prisma_client:
            # Assuming 'db' is your Prisma Client instance
            deleted_tokens = await prisma_client.delete_data(
                tokens=tokens, user_id=user_id
            )
        else:
            raise Exception
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
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
    usage_telemetry(feature="local_proxy_server")


# for streaming
def data_generator(response):
    verbose_proxy_logger.debug("inside generator")
    for chunk in response:
        verbose_proxy_logger.debug(f"returned chunk: {chunk}")
        try:
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"


async def async_data_generator(response, user_api_key_dict):
    verbose_proxy_logger.debug("inside generator")
    try:
        start_time = time.time()
        async for chunk in response:
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
            user_api_key_dict=user_api_key_dict, original_exception=e
        )
        verbose_proxy_logger.debug(
            f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
        )
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
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


def select_data_generator(response, user_api_key_dict):
    return async_data_generator(response=response, user_api_key_dict=user_api_key_dict)


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
    verbose_proxy_logger.debug(f"Backing off... this was attempt #{details['tries']}")


@router.on_event("startup")
async def startup_event():
    global prisma_client, master_key, use_background_health_checks, llm_router, llm_model_list, general_settings, proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time, litellm_proxy_admin_name
    import json

    ### LOAD MASTER KEY ###
    # check if master key set in environment - load from there
    master_key = litellm.get_secret("LITELLM_MASTER_KEY", None)
    # check if DATABASE_URL in environment - load from there
    if prisma_client is None:
        prisma_setup(database_url=os.getenv("DATABASE_URL"))

    ### LOAD CONFIG ###
    worker_config = litellm.get_secret("WORKER_CONFIG")
    verbose_proxy_logger.debug(f"worker_config: {worker_config}")
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
    proxy_logging_obj._init_litellm_callbacks()  # INITIALIZE LITELLM CALLBACKS ON SERVER STARTUP <- do this to catch any logging errors on startup, not when calls are being made

    if use_background_health_checks:
        asyncio.create_task(
            _run_background_health_check()
        )  # start the background health check coroutine.

    verbose_proxy_logger.debug(f"prisma client - {prisma_client}")
    if prisma_client is not None:
        await prisma_client.connect()

    verbose_proxy_logger.debug(f"custom_db_client client - {custom_db_client}")
    if custom_db_client is not None:
        verbose_proxy_logger.debug(f"custom_db_client connecting - {custom_db_client}")
        await custom_db_client.connect()

    if prisma_client is not None and master_key is not None:
        # add master key to db
        if os.getenv("PROXY_ADMIN_ID", None) is not None:
            litellm_proxy_admin_name = os.getenv("PROXY_ADMIN_ID")

        asyncio.create_task(
            generate_key_helper_fn(
                duration=None,
                models=[],
                aliases={},
                config={},
                spend=0,
                token=master_key,
                user_id=litellm_proxy_admin_name,
                user_role="proxy_admin",
                query_type="update_data",
                update_key_values={
                    "user_role": "proxy_admin",
                },
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
            duration=None, models=[], aliases={}, config={}, spend=0, token=master_key
        )

    ### CHECK IF VIEW EXISTS ###
    if prisma_client is not None:
        create_view_response = await prisma_client.check_view_exists()

    ### START BUDGET SCHEDULER ###
    if prisma_client is not None:
        scheduler = AsyncIOScheduler()
        interval = random.randint(
            proxy_budget_rescheduler_min_time, proxy_budget_rescheduler_max_time
        )  # random interval, so multiple workers avoid resetting budget at the same time
        scheduler.add_job(
            reset_budget, "interval", seconds=interval, args=[prisma_client]
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
    if len(user_api_key_dict.models) > 0:
        all_models = user_api_key_dict.models
    else:
        ## if no specific model access
        if general_settings.get("infer_model_from_keys", False):
            all_models = litellm.utils.get_valid_models()
        if llm_model_list:
            all_models = list(
                set(all_models + [m["model_name"] for m in llm_model_list])
            )
        if user_model is not None:
            all_models += [user_model]
    verbose_proxy_logger.debug(f"all_models: {all_models}")
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
async def completion(
    request: Request,
    fastapi_response: Response,
    model: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
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
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
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
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="completion"
        )

        start_time = time.time()

        ### ROUTE THE REQUESTs ###
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        # skip router if user passed their key
        if "api_key" in data:
            response = await litellm.atext_completion(**data)
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            response = await llm_router.atext_completion(**data)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            response = await llm_router.atext_completion(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.atext_completion(
                **data, specific_deployment=True
            )
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.atext_completion(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid model name passed in"},
            )

        if hasattr(response, "_hidden_params"):
            model_id = response._hidden_params.get("model_id", None) or ""
        else:
            model_id = ""

        verbose_proxy_logger.debug(f"final response: {response}")
        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = {"x-litellm-model-id": model_id}
            selected_data_generator = select_data_generator(
                response=response, user_api_key_dict=user_api_key_dict
            )

            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                headers=custom_headers,
            )

        fastapi_response.headers["x-litellm-model-id"] = model_id
        return response
    except Exception as e:
        verbose_proxy_logger.debug(f"EXCEPTION RAISED IN PROXY MAIN.PY")
        verbose_proxy_logger.debug(
            f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
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
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global general_settings, user_debug, proxy_logging_obj, llm_model_list
    try:
        data = {}
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
        verbose_proxy_logger.debug(f"Request Headers: {headers}")
        cache_control_header = headers.get("Cache-Control", None)
        if cache_control_header:
            cache_dict = parse_cache_control(cache_control_header)
            data["ttl"] = cache_dict.get("s-maxage")

        verbose_proxy_logger.debug(f"receiving data: {data}")
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
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
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
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="completion"
        )

        tasks = []
        tasks.append(proxy_logging_obj.during_call_hook(data=data))

        start_time = time.time()

        ### ROUTE THE REQUEST ###
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        # skip router if user passed their key
        if "api_key" in data:
            tasks.append(litellm.acompletion(**data))
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
        elif user_model is not None:  # `litellm --model <your-model-name>`
            tasks.append(litellm.acompletion(**data))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid model name passed in"},
            )

        # wait for call to end
        responses = await asyncio.gather(
            *tasks
        )  # run the moderation check in parallel to the actual llm api call
        response = responses[1]

        # Post Call Processing
        data["litellm_status"] = "success"  # used for alerting
        if hasattr(response, "_hidden_params"):
            model_id = response._hidden_params.get("model_id", None) or ""
        else:
            model_id = ""

        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = {"x-litellm-model-id": model_id}
            selected_data_generator = select_data_generator(
                response=response, user_api_key_dict=user_api_key_dict
            )
            return StreamingResponse(
                selected_data_generator,
                media_type="text/event-stream",
                headers=custom_headers,
            )

        fastapi_response.headers["x-litellm-model-id"] = model_id

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            user_api_key_dict=user_api_key_dict, response=response
        )

        return response
    except Exception as e:
        traceback.print_exc()
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
        )
        verbose_proxy_logger.debug(
            f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
        )
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and data.get("model", "") in router_model_names:
            verbose_proxy_logger.debug("Results from router")
            verbose_proxy_logger.debug("\nRouter stats")
            verbose_proxy_logger.debug("\nTotal Calls made")
            for key, value in llm_router.total_calls.items():
                verbose_proxy_logger.debug(f"{key}: {value}")
            verbose_proxy_logger.debug("\nSuccess Calls made")
            for key, value in llm_router.success_calls.items():
                verbose_proxy_logger.debug(f"{key}: {value}")
            verbose_proxy_logger.debug("\nFail Calls made")
            for key, value in llm_router.fail_calls.items():
                verbose_proxy_logger.debug(f"{key}: {value}")
        if user_debug:
            traceback.print_exc()

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"

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
    model: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global proxy_logging_obj
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
        data["metadata"]["user_api_key_alias"] = getattr(
            user_api_key_dict, "key_alias", None
        )
        data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        data["metadata"]["user_api_key_team_id"] = getattr(
            user_api_key_dict, "team_id", None
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

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
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

        start_time = time.time()

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
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.aembedding(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid model name passed in"},
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
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
            error_msg = f"{str(e)}\n\n{error_traceback}"
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
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global proxy_logging_obj
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

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="embeddings"
        )

        start_time = time.time()

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
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.aimage_generation(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid model name passed in"},
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
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
            error_msg = f"{str(e)}\n\n{error_traceback}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


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
    file: UploadFile = File(...),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Same params as:

    https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=curl
    """
    global proxy_logging_obj
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        form_data = await request.form()
        data: Dict = {key: value for key, value in form_data.items() if key != "file"}

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

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )

        assert (
            file.filename is not None
        )  # make sure filename passed in (needed for type)

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
                elif user_model is not None:  # `litellm --model <your-model-name>`
                    response = await litellm.atranscription(**data)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"error": "Invalid model name passed in"},
                    )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                os.remove(file.filename)  # Delete the saved file

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
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
            error_msg = f"{str(e)}\n\n{error_traceback}"
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

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )

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
        elif user_model is not None:  # `litellm --model <your-model-name>`
            response = await litellm.amoderation(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid model name passed in"},
            )

        ### ALERTING ###
        data["litellm_status"] = "success"  # used for alerting

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
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
            error_msg = f"{str(e)}\n\n{error_traceback}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
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
                if value is not None and key in litellm.upperbound_key_generate_params:
                    # if value is float/int
                    if key in [
                        "max_budget",
                        "max_parallel_requests",
                        "tpm_limit",
                        "rpm_limit",
                    ]:
                        if value > litellm.upperbound_key_generate_params[key]:
                            # directly compare floats/ints
                            setattr(
                                data, key, litellm.upperbound_key_generate_params[key]
                            )
                    elif key == "budget_duration":
                        # budgets are in 1s, 1m, 1h, 1d, 1m (30s, 30m, 30h, 30d, 30m)
                        # compare the duration in seconds and max duration in seconds
                        upperbound_budget_duration = _duration_in_seconds(
                            duration=litellm.upperbound_key_generate_params[key]
                        )
                        user_set_budget_duration = _duration_in_seconds(duration=value)
                        if user_set_budget_duration > upperbound_budget_duration:
                            setattr(
                                data, key, litellm.upperbound_key_generate_params[key]
                            )

        data_json = data.json()  # type: ignore

        # if we get max_budget passed to /key/generate, then use it as key_max_budget. Since generate_key_helper_fn is used to make new users
        if "max_budget" in data_json:
            data_json["key_max_budget"] = data_json.pop("max_budget", None)
        if "soft_budget" in data_json:
            data_json["key_soft_budget"] = data_json.pop("soft_budget", None)

        if "budget_duration" in data_json:
            data_json["key_budget_duration"] = data_json.pop("budget_duration", None)

        response = await generate_key_helper_fn(**data_json)
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
        response = await prisma_client.update_data(
            token=key, data={**non_default_values, "token": key}
        )
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
            and user_api_key_dict.user_role == "proxy_admin"
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
                    "error": "Not all keys passed in were deleted. This probably means you don't have access to delete all the keys passed in."
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
    tags=["budget & spend Tracking"],
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
    tags=["budget & spend Tracking"],
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
    tags=["budget & spend Tracking"],
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

    from litellm.proxy.enterprise.utils import get_spend_by_tags

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
        sql_query = """SELECT * FROM "MonthlyGlobalSpend";"""

        response = await prisma_client.db.query_raw(query=sql_query)

        return response
    else:
        sql_query = (
            """
            SELECT * FROM "MonthlyGlobalSpendPerKey"
            WHERE "api_key" = '"""
            + api_key
            + """'
            ORDER BY "date";
        """
        )

        response = await prisma_client.db.query_raw(query=sql_query)

        return response
    return


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

    if data is None:
        sql_query = f"""SELECT * FROM "Last30dTopEndUsersSpend";"""

        response = await prisma_client.db.query_raw(query=sql_query)
    else:
        """
        Gets the top 100 end-users for a given api key
        """
        current_date = datetime.now()
        past_date = current_date - timedelta(days=30)
        response = await prisma_client.db.litellm_spendlogs.group_by(  # type: ignore
            by=["end_user"],
            where={
                "AND": [{"startTime": {"gte": past_date}}, {"api_key": data.api_key}]  # type: ignore
            },
            sum={"spend": True},
            order={"_sum": {"spend": "desc"}},  # type: ignore
            take=100,
            count=True,
        )
        if response is not None and isinstance(response, list):
            new_response = []
            for r in response:
                new_r = r
                new_r["total_spend"] = r["_sum"]["spend"]
                new_r["total_count"] = r["_count"]["_all"]
                new_r.pop("_sum")
                new_r.pop("_count")
                new_response.append(new_r)

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
    from litellm.proxy.enterprise.utils import _forecast_daily_cost

    data = await request.json()
    data = data.get("data")
    return _forecast_daily_cost(data)


#### USER MANAGEMENT ####
@router.post(
    "/user/new",
    tags=["user management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewUserResponse,
)
async def new_user(data: NewUserRequest):
    """
    Use this to create a new user with a budget. This creates a new user and generates a new api key for the new user. The new api key is returned.

    Returns user id, budget + new key.

    Parameters:
    - user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
    - user_email: Optional[str] - Specify a user email.
    - user_role: Optional[str] - Specify a user role - "admin", "app_owner", "app_user"
    - max_budget: Optional[float] - Specify max budget for a given user.
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - tpm_limit: Optional[int] - Specify tpm limit for a given user (Tokens per minute)
    - rpm_limit: Optional[int] - Specify rpm limit for a given user (Requests per minute)

    Returns:
    - key: (str) The generated api key for the user
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    - max_budget: (float|None) Max budget for given user.
    """
    data_json = data.json()  # type: ignore
    if "user_role" in data_json:
        user_role = data_json["user_role"]
        if user_role is not None:
            if user_role not in ["admin", "app_owner", "app_user"]:
                raise ProxyException(
                    message=f"Invalid user role, passed in {user_role}. Must be one of 'admin', 'app_owner', 'app_user'",
                    type="invalid_user_role",
                    param="user_role",
                    code=status.HTTP_400_BAD_REQUEST,
                )
    response = await generate_key_helper_fn(**data_json)
    return NewUserResponse(
        key=response["token"],
        expires=response["expires"],
        user_id=response["user_id"],
        max_budget=response["max_budget"],
    )


@router.post(
    "/user/auth", tags=["user management"], dependencies=[Depends(user_api_key_auth)]
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
            **{"duration": "24hr", "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": user_id}  # type: ignore
        )
    else:  ### else - create new user
        response = await generate_key_helper_fn(
            **{"duration": "24hr", "models": [], "aliases": {}, "config": {}, "spend": 0, "user_email": user_email}  # type: ignore
        )

    base_url = os.getenv("LITELLM_HOSTED_UI", "https://dashboard.litellm.ai/")

    params = {
        "sender_name": "LiteLLM Proxy",
        "sender_email": os.getenv("SMTP_SENDER_EMAIL"),
        "receiver_email": user_email,
        "subject": "Your Magic Link",
        "html": f"<strong> Follow this  link, to login:\n\n{base_url}user/?token={response['token']}&user_id={response['user_id']}&page={page_params}</strong>",
    }

    await send_email(**params)
    return "Email sent!"


@router.get(
    "/user/info", tags=["user management"], dependencies=[Depends(user_api_key_auth)]
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
        for key in keys:
            try:
                key = key.model_dump()  # noqa
            except:
                # if using pydantic v1
                key = key.dict()
            key.pop("token", None)

        response_data = {
            "user_id": user_id,
            "user_info": user_info,
            "keys": keys,
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
    "/user/update", tags=["user management"], dependencies=[Depends(user_api_key_auth)]
)
async def user_update(data: UpdateUserRequest):
    """
    [TODO]: Use this to update user budget
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
        verbose_proxy_logger.debug(f"/user/update: Received data = {data}")
        if data.user_id is not None and len(data.user_id) > 0:
            non_default_values["user_id"] = data.user_id  # type: ignore
            verbose_proxy_logger.debug(f"In update user, user_id condition block.")
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
    tags=["user management"],
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
    tags=["user management"],
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


@router.post(
    "/user/block",
    tags=["user management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def block_user(data: BlockUsers):
    """
    [BETA] Reject calls with this user id

    ```
    curl -X POST "http://0.0.0.0:8000/user/block"
    -H "Authorization: Bearer sk-1234"
    -D '{
    "user_ids": [<user_id>, ...]
    }'
    ```
    """
    from litellm.proxy.enterprise.enterprise_hooks.blocked_user_list import (
        _ENTERPRISE_BlockedUserList,
    )

    if not any(isinstance(x, _ENTERPRISE_BlockedUserList) for x in litellm.callbacks):
        blocked_user_list = _ENTERPRISE_BlockedUserList()
        litellm.callbacks.append(blocked_user_list)  # type: ignore

    if litellm.blocked_user_list is None:
        litellm.blocked_user_list = data.user_ids
    elif isinstance(litellm.blocked_user_list, list):
        litellm.blocked_user_list = litellm.blocked_user_list + data.user_ids
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "`blocked_user_list` must be a list or not set. Filepaths can't be updated."
            },
        )

    return {"blocked_users": litellm.blocked_user_list}


@router.post(
    "/user/unblock",
    tags=["user management"],
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
    from litellm.proxy.enterprise.enterprise_hooks.blocked_user_list import (
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


@router.get(
    "/user/get_users",
    tags=["user management"],
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

    Parameters:
    - team_alias: Optional[str] - User defined team alias
    - team_id: Optional[str] - The team id of the user. If none passed, we'll generate it.
    - members_with_roles: list - A list of dictionaries, mapping user_id to role in team (either 'admin' or 'user')
    - metadata: Optional[dict] - Metadata for team, store information for team. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }

    Returns:
    - team_id: (str) Unique team id - used for tracking spend across multiple keys for same team id.

    _deprecated_params: 
    - admins: list - A list of user_id's for the admin role 
    - users: list - A list of user_id's for the user role 

    Example Request:
    ```
    curl --location 'http://0.0.0.0:8000/team/new' \
    
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

    if (
        user_api_key_dict.user_role is None
        or user_api_key_dict.user_role != "proxy_admin"
    ):  # don't restrict proxy admin
        if (
            data.tpm_limit is not None
            and user_api_key_dict.tpm_limit is not None
            and data.tpm_limit > user_api_key_dict.tpm_limit
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"tpm limit higher than user max. User tpm limit={user_api_key_dict.tpm_limit}"
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
                    "error": f"rpm limit higher than user max. User rpm limit={user_api_key_dict.rpm_limit}"
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
                    "error": f"max budget higher than user max. User max budget={user_api_key_dict.max_budget}"
                },
            )

        if data.models is not None:
            for m in data.models:
                if m not in user_api_key_dict.models:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"Model not in allowed user models. User allowed models={user_api_key_dict.models}"
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
        max_parallel_requests=user_api_key_dict.max_parallel_requests,
        budget_duration=user_api_key_dict.budget_duration,
        budget_reset_at=user_api_key_dict.budget_reset_at,
        model_id=_model_id,
    )

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
    return team_row.model_dump()


@router.post(
    "/team/update", tags=["team management"], dependencies=[Depends(user_api_key_auth)]
)
async def update_team(
    data: UpdateTeamRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA]
    [RECOMMENDED] - use `/team/member_add` to add new team members instead 

    You can now update team budget / rate limits via /team/update

    ```
    curl --location 'http://0.0.0.0:8000/team/update' \
    
    --header 'Authorization: Bearer sk-1234' \
        
    --header 'Content-Type: application/json' \
    
    --data-raw '{
        "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "members_with_roles": [{"role": "admin", "user_id": "5c4a0aa3-a1e1-43dc-bd87-3c2da8382a3a"}, {"role": "user", "user_id": "krrish247652@berri.ai"}]
    }'
    ```
    """
    global prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if data.team_id is None:
        raise HTTPException(status_code=400, detail={"error": "No team id passed in"})

    existing_team_row = await prisma_client.get_data(
        team_id=data.team_id, table_name="team", query_type="find_unique"
    )

    updated_kv = data.json(exclude_none=True)
    team_row = await prisma_client.update_data(
        update_key_values=updated_kv,
        data=updated_kv,
        table_name="team",
        team_id=data.team_id,
    )

    ## ADD NEW USERS ##
    existing_user_id_list = []
    ## Get new users
    for user in existing_team_row.members_with_roles:
        if user["user_id"] is not None:
            existing_user_id_list.append(user["user_id"])

    ## Update new user rows with team id (info used by /user/info to show all teams, user is a part of)
    if data.members_with_roles is not None:
        for user in data.members_with_roles:
            if user.user_id not in existing_user_id_list:
                await prisma_client.update_data(
                    user_id=user.user_id,
                    data={
                        "user_id": user.user_id,
                        "teams": [team_row["team_id"]],
                        "models": team_row["data"].models,
                    },
                    update_key_values_custom_query={
                        "teams": {
                            "push": [team_row["team_id"]],
                        }
                    },
                    table_name="user",
                )

    ## REMOVE DELETED USERS ##
    ### Get list of deleted users (old list - new list)
    deleted_user_id_list = []
    new_user_id_list = []
    ## Get old user list
    if data.members_with_roles is not None:
        for user in data.members_with_roles:
            new_user_id_list.append(user.user_id)
    ## Get diff
    if existing_team_row.members_with_roles is not None:
        for user in existing_team_row.members_with_roles:
            if user["user_id"] not in new_user_id_list:
                deleted_user_id_list.append(user["user_id"])

    ## SET UPDATED LIST
    if len(deleted_user_id_list) > 0:
        # get the deleted users
        existing_user_rows = await prisma_client.get_data(
            user_id_list=deleted_user_id_list, table_name="user", query_type="find_all"
        )
        for user in existing_user_rows:
            if data.team_id in user["teams"]:
                user["teams"].remove(data.team_id)
            await prisma_client.update_data(
                user_id=user["user_id"],
                data=user,
                update_key_values={"user_id": user["user_id"], "teams": user["teams"]},
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
    curl -X POST 'http://0.0.0.0:8000/team/update' \
    
    -H 'Authorization: Bearer sk-1234' \
        
    -H 'Content-Type: application/json' \
    
    -D '{
        "team_id": "45e3e396-ee08-4a61-a88e-16b3ce7e0849",
        "member": {"role": "user", "user_id": "krrish247652@berri.ai"}
    }'
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

    new_member = data.member

    existing_team_row.members_with_roles.append(new_member)

    complete_team_data = LiteLLM_TeamTable(
        **existing_team_row.model_dump(),
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
        "member": {"role": "user", "user_id": "krrish247652@berri.ai"}
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

    ## DELETE ASSOCIATED KEYS
    await prisma_client.delete_data(team_id_list=data.team_ids, table_name="key")
    ## DELETE TEAMS
    await prisma_client.delete_data(team_id_list=data.team_ids, table_name="team")


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
        or user_api_key_dict.user_role != "proxy_admin"
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
        budget_row = LiteLLM_BudgetTable(**data.json(exclude_none=True))

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
    "/budget/info",
    tags=["organization management"],
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


#### MODEL MANAGEMENT ####


#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post(
    "/model/new",
    description="Allows adding new models to the model list in the config.yaml",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def add_new_model(model_params: ModelParams):
    global llm_router, llm_model_list, general_settings, user_config_file_path, proxy_config
    try:
        # Load existing config
        config = await proxy_config.get_config()

        verbose_proxy_logger.debug(f"User config path: {user_config_file_path}")

        verbose_proxy_logger.debug(f"Loaded config: {config}")
        # Add the new model to the config
        model_info = model_params.model_info.json()
        model_info = {k: v for k, v in model_info.items() if v is not None}
        config["model_list"].append(
            {
                "model_name": model_params.model_name,
                "litellm_params": model_params.litellm_params,
                "model_info": model_info,
            }
        )

        verbose_proxy_logger.debug(f"updated model list: {config['model_list']}")

        # Save new config
        await proxy_config.save_config(new_config=config)
        return {"message": "Model added successfully"}

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
)
async def model_info_v2(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global llm_model_list, general_settings, user_config_file_path, proxy_config

    # Load existing config
    config = await proxy_config.get_config()

    all_models = config.get("model_list", [])
    if user_model is not None:
        # if user does not use a config.yaml, https://github.com/BerriAI/litellm/issues/2061
        all_models += [user_model]

    # check all models user has access to in user_api_key_dict
    user_models = []
    if len(user_api_key_dict.models) > 0:
        user_models = user_api_key_dict.models

    # for all models check if the user has access, and mark it as "user_access": `True` or `False`
    for model in all_models:
        model_name = model.get("model_name", None)
        if model_name is not None:
            user_has_access = model_name in user_models
            if (
                user_models == []
            ):  # if user_api_key_dict.models == [], user has access to all models
                user_has_access = True
            model["user_access"] = user_has_access

    # fill in model info based on config.yaml and litellm model_prices_and_context_window.json
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

    verbose_proxy_logger.debug(f"all_models: {all_models}")
    return {"data": all_models}


@router.get(
    "/model/metrics",
    description="View number of requests & avg latency per model on config.yaml",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def model_metrics(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    global prisma_client
    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    sql_query = """
        SELECT
            CASE WHEN api_base = '' THEN model ELSE CONCAT(model, '-', api_base) END AS combined_model_api_base,
            COUNT(*) AS num_requests,
            AVG(EXTRACT(epoch FROM ("endTime" - "startTime"))) AS avg_latency_seconds
        FROM
            "LiteLLM_SpendLogs"
        WHERE
            "startTime" >= NOW() - INTERVAL '10000 hours'
        GROUP BY
            CASE WHEN api_base = '' THEN model ELSE CONCAT(model, '-', api_base) END
        ORDER BY
            num_requests DESC
        LIMIT 50;
    """

    db_response = await prisma_client.db.query_raw(query=sql_query)
    response: List[dict] = []
    if response is not None:
        # loop through all models
        for model_data in db_response:
            model = model_data.get("combined_model_api_base", "")
            num_requests = model_data.get("num_requests", 0)
            avg_latency_seconds = model_data.get("avg_latency_seconds", 0)
            response.append(
                {
                    "model": model,
                    "num_requests": num_requests,
                    "avg_latency_seconds": avg_latency_seconds,
                }
            )
    return response


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

    # Load existing config
    config = await proxy_config.get_config()

    if len(user_api_key_dict.models) > 0:
        model_names = user_api_key_dict.models
        all_models = [m for m in config["model_list"] if m in model_names]
    else:
        all_models = config["model_list"]
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

    verbose_proxy_logger.debug(f"all_models: {all_models}")
    return {"data": all_models}


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
        if not os.path.exists(user_config_file_path):
            raise HTTPException(status_code=404, detail="Config file does not exist.")

        # Load existing config
        config = await proxy_config.get_config()

        # If model_list is not in the config, nothing can be deleted
        if len(config.get("model_list", [])) == 0:
            raise HTTPException(
                status_code=400, detail="No model list available in the config."
            )

        # Check if the model with the specified model_id exists
        model_to_delete = None
        for model in config["model_list"]:
            if model.get("model_info", {}).get("id", None) == model_info.id:
                model_to_delete = model
                break

        # If the model was not found, return an error
        if model_to_delete is None:
            raise HTTPException(
                status_code=400, detail="Model with given model_id not found."
            )

        # Remove model from the list and save the updated config
        config["model_list"].remove(model_to_delete)

        # Save updated config
        config = await proxy_config.save_config(new_config=config)
        return {"message": "Model deleted successfully"}

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


#### EXPERIMENTAL QUEUING ####
async def _litellm_chat_completions_worker(data, user_api_key_dict):
    """
    worker to make litellm completions calls
    """
    while True:
        try:
            ### CALL HOOKS ### - modify incoming data before calling the model
            data = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict, data=data, call_type="completion"
            )

            verbose_proxy_logger.debug(f"_litellm_chat_completions_worker started")
            ### ROUTE THE REQUEST ###
            router_model_names = (
                [m["model_name"] for m in llm_model_list]
                if llm_model_list is not None
                else []
            )
            if (
                llm_router is not None and data["model"] in router_model_names
            ):  # model in router model list
                response = await llm_router.acompletion(**data)
            elif (
                llm_router is not None and data["model"] in llm_router.deployment_names
            ):  # model in router deployments, calling a specific deployment on the router
                response = await llm_router.acompletion(
                    **data, specific_deployment=True
                )
            elif (
                llm_router is not None
                and llm_router.model_group_alias is not None
                and data["model"] in llm_router.model_group_alias
            ):  # model set in model_group_alias
                response = await llm_router.acompletion(**data)
            else:  # router is not set
                response = await litellm.acompletion(**data)

            verbose_proxy_logger.debug(f"final response: {response}")
            return response
        except HTTPException as e:
            verbose_proxy_logger.debug(
                f"EXCEPTION RAISED IN _litellm_chat_completions_worker - {e.status_code}; {e.detail}"
            )
            if (
                e.status_code == 429
                and "Max parallel request limit reached" in e.detail
            ):
                verbose_proxy_logger.debug(f"Max parallel request limit reached!")
                timeout = litellm._calculate_retry_after(
                    remaining_retries=3, max_retries=3, min_timeout=1
                )
                await asyncio.sleep(timeout)
            else:
                raise e


@router.post(
    "/queue/chat/completions",
    tags=["experimental"],
    dependencies=[Depends(user_api_key_auth)],
)
async def async_queue_request(
    request: Request,
    model: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    global general_settings, user_debug, proxy_logging_obj
    """
    v2 attempt at a background worker to handle queuing. 

    Just supports /chat/completion calls currently.

    Now using a FastAPI background task + /chat/completions compatible endpoint
    """
    try:
        data = {}
        data = await request.json()  # type: ignore

        # Include original request and headers in the data
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data),  # use copy instead of deepcopy
        }

        verbose_proxy_logger.debug(f"receiving data: {data}")
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

        response = await asyncio.wait_for(
            _litellm_chat_completions_worker(
                data=data, user_api_key_dict=user_api_key_dict
            ),
            timeout=litellm.request_timeout,
        )

        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            return StreamingResponse(
                async_data_generator(
                    user_api_key_dict=user_api_key_dict, response=response
                ),
                media_type="text/event-stream",
            )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
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
    microsoft_client_id = os.getenv("MICROSOFT_CLIENT_ID", None)
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", None)
    generic_client_id = os.getenv("GENERIC_CLIENT_ID", None)
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


@router.post(
    "/login", include_in_schema=False
)  # hidden since this is a helper for UI sso login
async def login(request: Request):
    try:
        import multipart
    except ImportError:
        subprocess.run(["pip", "install", "python-multipart"])
    global master_key
    form = await request.form()
    username = str(form.get("username"))
    password = str(form.get("password"))
    ui_username = os.getenv("UI_USERNAME", "admin")
    ui_password = os.getenv("UI_PASSWORD", None)
    if ui_password is None:
        ui_password = str(master_key) if master_key is not None else None
    if ui_password is None:
        raise ProxyException(
            message="set Proxy master key to use UI. https://docs.litellm.ai/docs/proxy/virtual_keys",
            type="auth_error",
            param="UI_PASSWORD",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
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
                user_role="proxy_admin",
            )
        )
        if os.getenv("DATABASE_URL") is not None:
            response = await generate_key_helper_fn(
                **{"user_role": "proxy_admin", "duration": "1hr", "key_max_budget": 5, "models": [], "aliases": {}, "config": {}, "spend": 0, "user_id": key_user_id, "team_id": "litellm-dashboard"}  # type: ignore
            )
        else:
            response = {
                "token": "sk-gm",
                "user_id": "litellm-dashboard",
            }
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
            },
            "secret",
            algorithm="HS256",
        )
        litellm_dashboard_ui += "?userID=" + user_id + "&token=" + jwt_token
        return RedirectResponse(url=litellm_dashboard_ui, status_code=303)
    else:
        raise ProxyException(
            message=f"Invalid credentials used to access UI. Passed in username: {username}, passed in password: {password}.\nCheck 'UI_USERNAME', 'UI_PASSWORD' in .env file",
            type="auth_error",
            param="invalid_credentials",
            code=status.HTTP_401_UNAUTHORIZED,
        )


@app.get("/get_image", include_in_schema=False)
def get_image():
    """Get logo to show on admin UI"""
    from fastapi.responses import FileResponse

    # get current_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_logo = os.path.join(current_dir, "logo.jpg")

    logo_path = os.getenv("UI_LOGO_PATH", default_logo)
    verbose_proxy_logger.debug(f"Reading logo from {logo_path}")

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
    global general_settings, ui_access_mode
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
        verbose_proxy_logger.debug(f"calling generic_sso.verify_and_process")
        result = await generic_sso.verify_and_process(
            request, params={"include_client_id": generic_include_client_id}
        )
        verbose_proxy_logger.debug(f"generic result: {result}")

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
        "duration": "1hr",
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
                }
                user_role = getattr(user_info, "user_role", None)
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
    response = await generate_key_helper_fn(
        **default_ui_key_values, **user_defined_values  # type: ignore
    )
    key = response["token"]  # type: ignore
    user_id = response["user_id"]  # type: ignore
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
        },
        "secret",
        algorithm="HS256",
    )
    litellm_dashboard_ui += "?userID=" + user_id + "&token=" + jwt_token
    # if a user has logged in they should be allowed to create keys - this ensures that it's set to True
    general_settings["allow_user_auth"] = True
    return RedirectResponse(url=litellm_dashboard_ui)


#### BASIC ENDPOINTS ####
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
    global llm_router, llm_model_list, general_settings, proxy_config, proxy_logging_obj
    try:
        # Load existing config
        config = await proxy_config.get_config()

        backup_config = copy.deepcopy(config)
        verbose_proxy_logger.debug(f"Loaded config: {config}")

        # update the general settings
        if config_info.general_settings is not None:
            config.setdefault("general_settings", {})
            updated_general_settings = config_info.general_settings.dict(
                exclude_none=True
            )
            config["general_settings"] = {
                **updated_general_settings,
                **config["general_settings"],
            }

        if config_info.environment_variables is not None:
            config.setdefault("environment_variables", {})
            updated_environment_variables = config_info.environment_variables
            config["environment_variables"] = {
                **updated_environment_variables,
                **config["environment_variables"],
            }

        # update the litellm settings
        if config_info.litellm_settings is not None:
            config.setdefault("litellm_settings", {})
            updated_litellm_settings = config_info.litellm_settings
            config["litellm_settings"] = {
                **updated_litellm_settings,
                **config["litellm_settings"],
            }

        # Save the updated config
        await proxy_config.save_config(new_config=config)

        # Test new connections
        ## Slack
        if "slack" in config.get("general_settings", {}).get("alerting", []):
            await proxy_logging_obj.alerting_handler(
                message="This is a test", level="Low"
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
    service: Literal["slack_budget_alerts"] = fastapi.Query(
        description="Specify the service being hit."
    ),
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

        if service not in ["slack_budget_alerts"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Service must be in list. Service={service}. List={['slack_budget_alerts']}"
                },
            )

        if "slack" in general_settings.get("alerting", []):
            test_message = f"""\n🚨 `ProjectedLimitExceededError` 💸\n\n`Key Alias:` litellm-ui-test-alert \n`Expected Day of Error`: 28th March \n`Current Spend`: $100.00 \n`Projected Spend at end of month`: $1000.00 \n`Soft Limit`: $700"""
            await proxy_logging_obj.alerting_handler(message=test_message, level="Low")
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": '"slack" not in proxy config: general_settings. Unable to test this.'
                },
            )
    except Exception as e:
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


@router.get("/health", tags=["health"], dependencies=[Depends(user_api_key_auth)])
async def health_endpoint(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = fastapi.Query(
        None, description="Specify the model name (optional)"
    ),
):
    """
    Check the health of all the endpoints in config.yaml

    To run health checks in the background, add this to config.yaml:
    ```
    general_settings:
        # ... other settings
        background_health_checks: True
    ```
    else, the health checks will be run on models when /health is called.
    """
    global health_check_results, use_background_health_checks, user_model
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

        ### FILTER MODELS FOR ONLY THOSE USER HAS ACCESS TO ###
        if len(user_api_key_dict.models) > 0:
            allowed_model_names = user_api_key_dict.models
        else:
            allowed_model_names = []  #
        if use_background_health_checks:
            return health_check_results
        else:
            healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                llm_model_list, model
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


@router.get(
    "/health/readiness",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def health_readiness():
    """
    Unprotected endpoint for checking if worker can receive requests
    """
    global prisma_client

    cache_type = None
    if litellm.cache is not None:
        from litellm.caching import RedisSemanticCache

        cache_type = litellm.cache.type

        if isinstance(litellm.cache.cache, RedisSemanticCache):
            # ping the cache
            try:
                index_info = await litellm.cache.cache._index_info()
            except Exception as e:
                index_info = "index does not exist - error: " + str(e)
            cache_type = {"type": cache_type, "index_info": index_info}
    if prisma_client is not None:  # if db passed in, check if it's connected
        await prisma_client.health_check()  # test the db connection
        response_object = {"db": "connected"}

        return {
            "status": "healthy",
            "db": "connected",
            "cache": cache_type,
            "litellm_version": version,
            "success_callbacks": litellm.success_callback,
        }
    else:
        return {
            "status": "healthy",
            "db": "Not connected",
            "cache": cache_type,
            "litellm_version": version,
            "success_callbacks": litellm.success_callback,
        }
    raise HTTPException(status_code=503, detail="Service Unhealthy")


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
    if prisma_client:
        verbose_proxy_logger.debug("Disconnecting from Prisma")
        await prisma_client.disconnect()

    if litellm.cache is not None:
        await litellm.cache.disconnect()
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
