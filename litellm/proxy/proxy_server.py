import sys, os, platform, time, copy, re, asyncio, inspect
import threading, ast
import shutil, random, traceback, requests
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import secrets, subprocess
import hashlib, uuid
import warnings
import importlib

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
)
from litellm.proxy.secret_managers.google_kms import load_google_kms
import pydantic
from litellm.proxy._types import *
from litellm.caching import DualCache
from litellm.proxy.health_check import perform_health_check
from litellm._logging import verbose_router_logger, verbose_proxy_logger

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
)
from fastapi.routing import APIRouter
from fastapi.security import OAuth2PasswordBearer
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, FileResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import json
import logging
from typing import Union

app = FastAPI(
    docs_url="/",
    title="LiteLLM API",
    description="Proxy Server to call 100+ LLMs in the OpenAI format\n\nAdmin Panel on [https://dashboard.litellm.ai/admin](https://dashboard.litellm.ai/admin)",
)
router = APIRouter()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from typing import Dict

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
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
user_api_key_cache = DualCache()
user_custom_auth = None
use_background_health_checks = None
use_queue = False
health_check_interval = None
health_check_results = {}
queue: List = []
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


def _get_bearer_token(api_key: str):
    assert api_key.startswith("Bearer ")  # ensure Bearer token passed in
    api_key = api_key.replace("Bearer ", "")  # extract the token
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

        if api_key is None:  # only require api key if master key is set
            raise Exception(f"No api key passed in.")

        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        is_master_key_valid = secrets.compare_digest(api_key, master_key)
        if is_master_key_valid:
            return UserAPIKeyAuth(api_key=master_key)

        if route.startswith("/config/") and not is_master_key_valid:
            raise Exception(f"Only admin can modify config")

        if (
            (route.startswith("/key/") or route.startswith("/user/"))
            or route.startswith("/model/")
            and not is_master_key_valid
            and general_settings.get("allow_user_auth", False) != True
        ):
            raise Exception(
                f"If master key is set, only master key can be used to generate, delete, update or get info for new keys/users"
            )

        if (
            prisma_client is None and custom_db_client is None
        ):  # if both master key + user key submitted, and user key != master key, and no db connected, raise an error
            raise Exception("No connected db.")

        ## check for cache hit (In-Memory Cache)
        valid_token = user_api_key_cache.get_cache(key=api_key)
        verbose_proxy_logger.debug(f"valid_token from cache: {valid_token}")
        if valid_token is None:
            ## check db
            verbose_proxy_logger.debug(f"api key: {api_key}")
            if prisma_client is not None:
                valid_token = await prisma_client.get_data(
                    token=api_key,
                )

                expires = datetime.utcnow().replace(tzinfo=timezone.utc)
            elif custom_db_client is not None:
                valid_token = await custom_db_client.get_data(
                    key=api_key, table_name="key"
                )
            # Token exists, now check expiration.
            if valid_token.expires is not None:
                expiry_time = datetime.fromisoformat(valid_token.expires)
                if expiry_time >= datetime.utcnow():
                    # Token exists and is not expired.
                    return response
                else:
                    # Token exists but is expired.
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="expired user key",
                    )
            verbose_proxy_logger.debug(f"valid token from prisma: {valid_token}")
            user_api_key_cache.set_cache(key=api_key, value=valid_token, ttl=60)
        elif valid_token is not None:
            verbose_proxy_logger.debug(f"API Key Cache Hit!")
        if valid_token:
            litellm.model_alias_map = valid_token.aliases
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
                if model and model not in valid_token.models:
                    raise Exception(f"Token not allowed to access model")
            api_key = valid_token.token
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
            return UserAPIKeyAuth(api_key=api_key, **valid_token_dict)
        else:
            raise Exception(f"Invalid token")
    except Exception as e:
        # verbose_proxy_logger.debug(f"An exception occurred - {traceback.format_exc()}")
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid user key",
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
            verbose_proxy_logger.debug(
                f"Error when initializing prisma, Ensure you run pip install prisma {str(e)}"
            )


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
            if (track_cost_callback) not in litellm.success_callback:  # type: ignore
                litellm.success_callback.append(track_cost_callback)  # type: ignore


async def track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    global prisma_client, custom_db_client
    try:
        # check if it has collected an entire stream response
        verbose_proxy_logger.debug(
            f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}"
        )
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost
            completion_response = kwargs["complete_streaming_response"]
            response_cost = litellm.completion_cost(
                completion_response=completion_response
            )
            verbose_proxy_logger.debug(f"streaming response_cost {response_cost}")
            user_api_key = kwargs["litellm_params"]["metadata"].get(
                "user_api_key", None
            )
            user_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_user_id", None
            )
            if user_api_key and (
                prisma_client is not None or custom_db_client is not None
            ):
                await update_database(token=user_api_key, response_cost=response_cost)
        elif kwargs["stream"] == False:  # for non streaming responses
            response_cost = litellm.completion_cost(
                completion_response=completion_response
            )
            user_api_key = kwargs["litellm_params"]["metadata"].get(
                "user_api_key", None
            )
            user_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_user_id", None
            )
            if user_api_key and (
                prisma_client is not None or custom_db_client is not None
            ):
                await update_database(
                    token=user_api_key, response_cost=response_cost, user_id=user_id
                )
    except Exception as e:
        verbose_proxy_logger.debug(f"error in tracking cost callback - {str(e)}")


async def update_database(token, response_cost, user_id=None):
    try:
        verbose_proxy_logger.debug(
            f"Enters prisma db call, token: {token}; user_id: {user_id}"
        )

        ### UPDATE USER SPEND ###
        async def _update_user_db():
            if user_id is None:
                return
            if prisma_client is not None:
                existing_spend_obj = await prisma_client.get_data(user_id=user_id)
            elif custom_db_client is not None:
                existing_spend_obj = await custom_db_client.get_data(
                    key=user_id, table_name="user"
                )
            if existing_spend_obj is None:
                existing_spend = 0
            else:
                existing_spend = existing_spend_obj.spend

            # Calculate the new cost by adding the existing cost and response_cost
            new_spend = existing_spend + response_cost

            verbose_proxy_logger.debug(f"new cost: {new_spend}")
            # Update the cost column for the given user id
            if prisma_client is not None:
                await prisma_client.update_data(
                    user_id=user_id, data={"spend": new_spend}
                )
            elif custom_db_client is not None:
                await custom_db_client.update_data(
                    key=user_id, value={"spend": new_spend}, table_name="user"
                )

        ### UPDATE KEY SPEND ###
        async def _update_key_db():
            if prisma_client is not None:
                # Fetch the existing cost for the given token
                existing_spend_obj = await prisma_client.get_data(token=token)
                verbose_proxy_logger.debug(f"existing spend: {existing_spend_obj}")
                if existing_spend_obj is None:
                    existing_spend = 0
                else:
                    existing_spend = existing_spend_obj.spend
                # Calculate the new cost by adding the existing cost and response_cost
                new_spend = existing_spend + response_cost

                verbose_proxy_logger.debug(f"new cost: {new_spend}")
                # Update the cost column for the given token
                await prisma_client.update_data(token=token, data={"spend": new_spend})
            elif custom_db_client is not None:
                # Fetch the existing cost for the given token
                existing_spend_obj = await custom_db_client.get_data(
                    key=token, table_name="key"
                )
                verbose_proxy_logger.debug(f"existing spend: {existing_spend_obj}")
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

        tasks = []
        tasks.append(_update_user_db())
        tasks.append(_update_key_db())
        await asyncio.gather(*tasks)
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

    async def load_config(
        self, router: Optional[litellm.Router], config_file_path: str
    ):
        """
        Load config values into proxy global state
        """
        global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, use_background_health_checks, health_check_interval, use_queue, custom_db_client

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
                if key == "cache":
                    print(f"{blue_color_code}\nSetting Cache on Proxy")  # noqa
                    from litellm.caching import Cache

                    cache_params = {}
                    if "cache_params" in litellm_settings:
                        cache_params_in_config = litellm_settings["cache_params"]
                        # overwrie cache_params with cache_params_in_config
                        cache_params.update(cache_params_in_config)

                    cache_type = cache_params.get("type", "redis")

                    verbose_proxy_logger.debug(f"passed cache type={cache_type}")

                    if cache_type == "redis":
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
                    litellm.callbacks = [
                        get_instance_fn(value=value, config_file_path=config_file_path)
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
                    verbose_proxy_logger.debug(
                        f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}"
                    )
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
                else:
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
            ### CUSTOM API KEY AUTH ###
            ## pass filepath
            custom_auth = general_settings.get("custom_auth", None)
            if custom_auth is not None:
                user_custom_auth = get_instance_fn(
                    value=custom_auth, config_file_path=config_file_path
                )
            ## dynamodb
            database_type = general_settings.get("database_type", None)
            if database_type is not None and (
                database_type == "dynamo_db" or database_type == "dynamodb"
            ):
                database_args = general_settings.get("database_args", None)
                custom_db_client = DBClient(
                    custom_db_args=database_args, custom_db_type=database_type
                )
            ## COST TRACKING ##
            cost_tracking()
            ### BACKGROUND HEALTH CHECKS ###
            # Enable background health checks
            use_background_health_checks = general_settings.get(
                "background_health_checks", False
            )
            health_check_interval = general_settings.get("health_check_interval", 300)

        router_params: dict = {
            "num_retries": 3,
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


async def generate_key_helper_fn(
    duration: Optional[str],
    models: list,
    aliases: dict,
    config: dict,
    spend: float,
    max_budget: Optional[float] = None,
    token: Optional[str] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    max_parallel_requests: Optional[int] = None,
    metadata: Optional[dict] = {},
):
    global prisma_client, custom_db_client

    if prisma_client is None and custom_db_client is None:
        raise Exception(
            f"Connect Proxy to database to generate keys - https://docs.litellm.ai/docs/proxy/virtual_keys "
        )

    if token is None:
        token = f"sk-{secrets.token_urlsafe(16)}"

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

    if duration is None:  # allow tokens that never expire
        expires = None
    else:
        duration_s = _duration_in_seconds(duration=duration)
        expires = datetime.utcnow() + timedelta(seconds=duration_s)

    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    metadata_json = json.dumps(metadata)
    user_id = user_id or str(uuid.uuid4())
    try:
        # Create a new verification token (you may want to enhance this logic based on your needs)
        user_data = {
            "max_budget": max_budget,
            "user_email": user_email,
            "user_id": user_id,
            "spend": spend,
        }
        key_data = {
            "token": token,
            "expires": expires,
            "models": models,
            "aliases": aliases_json,
            "config": config_json,
            "spend": spend,
            "user_id": user_id,
            "max_parallel_requests": max_parallel_requests,
            "metadata": metadata_json,
        }
        if prisma_client is not None:
            verification_token_data = dict(key_data)
            verification_token_data.update(user_data)
            verbose_proxy_logger.debug("PrismaClient: Before Insert Data")
            await prisma_client.insert_data(data=verification_token_data)
        elif custom_db_client is not None:
            ## CREATE USER (If necessary)
            await custom_db_client.insert_data(value=user_data, table_name="user")
            ## CREATE KEY
            await custom_db_client.insert_data(value=key_data, table_name="key")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return {
        "token": token,
        "expires": expires,
        "user_id": user_id,
        "max_budget": max_budget,
    }


async def delete_verification_token(tokens: List):
    global prisma_client
    try:
        if prisma_client:
            # Assuming 'db' is your Prisma Client instance
            deleted_tokens = await prisma_client.delete_data(tokens=tokens)
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
        from litellm._logging import verbose_router_logger, verbose_proxy_logger
        import logging

        # this must ALWAYS remain logging.INFO, DO NOT MODIFY THIS

        verbose_router_logger.setLevel(level=logging.INFO)  # set router logs to info
        verbose_proxy_logger.setLevel(level=logging.INFO)  # set proxy logs to info
    if detailed_debug == True:
        from litellm._logging import verbose_router_logger, verbose_proxy_logger
        import logging

        verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to info
        verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug
        litellm.set_verbose = True
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
                litellm.set_verbose = True

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
        os.environ[
            "AZURE_API_VERSION"
        ] = api_version  # set this for azure - litellm can read this from the env
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
            verbose_proxy_logger.debug(f"returned chunk: {chunk}")
            try:
                yield f"data: {json.dumps(chunk.dict())}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        ### ALERTING ###
        end_time = time.time()
        asyncio.create_task(
            proxy_logging_obj.response_taking_too_long(
                start_time=start_time, end_time=end_time, type="slow_response"
            )
        )

        # Streaming is done, yield the [DONE] chunk
        done_message = "[DONE]"
        yield f"data: {done_message}\n\n"
    except Exception as e:
        yield f"data: {str(e)}\n\n"


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


@router.on_event("startup")
async def startup_event():
    global prisma_client, master_key, use_background_health_checks, llm_router, llm_model_list, general_settings
    import json

    ### LOAD MASTER KEY ###
    # check if master key set in environment - load from there
    master_key = litellm.get_secret("LITELLM_MASTER_KEY", None)

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

    if custom_db_client is not None:
        await custom_db_client.connect()

    if prisma_client is not None and master_key is not None:
        # add master key to db
        await generate_key_helper_fn(
            duration=None, models=[], aliases={}, config={}, spend=0, token=master_key
        )

    if custom_db_client is not None and master_key is not None:
        # add master key to db
        await generate_key_helper_fn(
            duration=None, models=[], aliases={}, config={}, spend=0, token=master_key
        )


#### API ENDPOINTS ####
@router.get(
    "/v1/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)
@router.get(
    "/models", dependencies=[Depends(user_api_key_auth)], tags=["model management"]
)  # if project requires model list
def model_list():
    global llm_model_list, general_settings
    all_models = []
    if general_settings.get("infer_model_from_keys", False):
        all_models = litellm.utils.get_valid_models()
    if llm_model_list:
        all_models = list(set(all_models + [m["model_name"] for m in llm_model_list]))
    if user_model is not None:
        all_models += [user_model]
    verbose_proxy_logger.debug(f"all_models: {all_models}")
    ### CHECK OLLAMA MODELS ###
    try:
        response = requests.get("http://0.0.0.0:11434/api/tags")
        models = response.json()["models"]
        ollama_models = ["ollama/" + m["name"].replace(":latest", "") for m in models]
        all_models.extend(ollama_models)
    except Exception as e:
        pass
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
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
            data["metadata"]["headers"] = dict(request.headers)
        else:
            data["metadata"] = {
                "user_api_key": user_api_key_dict.api_key,
                "user_api_key_user_id": user_api_key_dict.user_id,
            }
            data["metadata"]["headers"] = dict(request.headers)
        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

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
        else:  # router is not set
            response = await litellm.atext_completion(**data)

        if hasattr(response, "_hidden_params"):
            model_id = response._hidden_params.get("model_id", None) or ""
        else:
            model_id = ""

        verbose_proxy_logger.debug(f"final response: {response}")
        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = {"x-litellm-model-id": model_id}
            return StreamingResponse(
                async_data_generator(
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                ),
                media_type="text/event-stream",
                headers=custom_headers,
            )

        ### ALERTING ###
        end_time = time.time()
        asyncio.create_task(
            proxy_logging_obj.response_taking_too_long(
                start_time=start_time, end_time=end_time, type="slow_response"
            )
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
        error_msg = f"{str(e)}\n\n{error_traceback}"
        try:
            status = e.status_code  # type: ignore
        except:
            status = 500
        raise HTTPException(status_code=status, detail=error_msg)


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

        if "metadata" in data:
            verbose_proxy_logger.debug(f'received metadata: {data["metadata"]}')
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
            data["metadata"]["headers"] = dict(request.headers)
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id

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

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="completion"
        )

        start_time = time.time()

        ### ROUTE THE REQUEST ###
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        # skip router if user passed their key
        if "api_key" in data:
            response = await litellm.acompletion(**data)
        elif "user_config" in data:
            # initialize a new router instance. make request using this Router
            router_config = data.pop("user_config")
            user_router = litellm.Router(**router_config)
            response = await user_router.acompletion(**data)
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            response = await llm_router.acompletion(**data)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            response = await llm_router.acompletion(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            response = await llm_router.acompletion(**data, specific_deployment=True)
        else:  # router is not set
            response = await litellm.acompletion(**data)

        if hasattr(response, "_hidden_params"):
            model_id = response._hidden_params.get("model_id", None) or ""
        else:
            model_id = ""

        if (
            "stream" in data and data["stream"] == True
        ):  # use generate_responses to stream responses
            custom_headers = {"x-litellm-model-id": model_id}
            return StreamingResponse(
                async_data_generator(
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                ),
                media_type="text/event-stream",
                headers=custom_headers,
            )

        ### ALERTING ###
        end_time = time.time()
        asyncio.create_task(
            proxy_logging_obj.response_taking_too_long(
                start_time=start_time, end_time=end_time, type="slow_response"
            )
        )

        fastapi_response.headers["x-litellm-model-id"] = model_id
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
            raise e
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"
            try:
                status = e.status_code  # type: ignore
            except:
                status = 500
            raise HTTPException(status_code=status, detail=error_msg)


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
async def embeddings(
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
            general_settings.get("embedding_model", None)  # server default
            or user_model  # model name passed via cli args
            or data["model"]  # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id

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
        else:
            response = await litellm.aembedding(**data)

        ### ALERTING ###
        end_time = time.time()
        asyncio.create_task(
            proxy_logging_obj.response_taking_too_long(
                start_time=start_time, end_time=end_time, type="slow_response"
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
        )
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"
            try:
                status = e.status_code  # type: ignore
            except:
                status = 500
            raise HTTPException(status_code=status, detail=error_msg)


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["image generation"],
)
@router.post(
    "/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["image generation"],
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
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id

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
        else:
            response = await litellm.aimage_generation(**data)

        ### ALERTING ###
        end_time = time.time()
        asyncio.create_task(
            proxy_logging_obj.response_taking_too_long(
                start_time=start_time, end_time=end_time, type="slow_response"
            )
        )

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e
        )
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"
            try:
                status = e.status_code  # type: ignore
            except:
                status = 500
            raise HTTPException(status_code=status, detail=error_msg)


#### KEY MANAGEMENT ####


@router.post(
    "/key/generate",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GenerateKeyResponse,
)
async def generate_key_fn(
    request: Request,
    data: GenerateKeyRequest,
    Authorization: Optional[str] = Header(None),
):
    """
    Generate an API key based on the provided data.

    Docs: https://docs.litellm.ai/docs/proxy/virtual_keys

    Parameters:
    - duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"). **(Default is set to 1 hour.)**
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - aliases: Optional[dict] - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
    - config: Optional[dict] - any key-specific configs, overrides config in config.yaml
    - spend: Optional[int] - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - metadata: Optional[dict] - Metadata for key, store information for key. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }

    Returns:
    - key: (str) The generated api key
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    """
    verbose_proxy_logger.debug("entered /key/generate")
    data_json = data.json()  # type: ignore
    response = await generate_key_helper_fn(**data_json)
    return GenerateKeyResponse(
        key=response["token"], expires=response["expires"], user_id=response["user_id"]
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

        non_default_values = {k: v for k, v in data_json.items() if v is not None}
        response = await prisma_client.update_data(
            token=key, data={**non_default_values, "token": key}
        )
        return {"key": key, **non_default_values}
        # update based on remaining passed in values
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.post(
    "/key/delete", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def delete_key_fn(request: Request, data: DeleteKeyRequest):
    try:
        keys = data.keys

        deleted_keys = await delete_verification_token(tokens=keys)
        assert len(keys) == deleted_keys
        return {"deleted_keys": keys}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/key/info", tags=["key management"], dependencies=[Depends(user_api_key_auth)]
)
async def info_key_fn(
    key: str = fastapi.Query(..., description="Key in the request parameters")
):
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        key_info = await prisma_client.get_data(token=key)
        return {"key": key, "info": key_info}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


#### USER MANAGEMENT ####
@router.post(
    "/user/new",
    tags=["user management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewUserResponse,
)
async def new_user(data: NewUserRequest):
    """
    Use this to create a new user with a budget.

    Returns user id, budget + new key.

    Parameters:
    - user_id: Optional[str] - Specify a user id. If not set, a unique id will be generated.
    - max_budget: Optional[float] - Specify max budget for a given user.
    - duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"). **(Default is set to 1 hour.)**
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - aliases: Optional[dict] - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
    - config: Optional[dict] - any key-specific configs, overrides config in config.yaml
    - spend: Optional[int] - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend
    - max_parallel_requests: Optional[int] - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
    - metadata: Optional[dict] - Metadata for key, store information for key. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }

    Returns:
    - key: (str) The generated api key
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    - max_budget: (float|None) Max budget for given user.
    """
    data_json = data.json()  # type: ignore
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
    user_id: str = fastapi.Query(..., description="User ID in the request parameters")
):
    """
    Use this to get user information. (user row + all user key info)
    """
    global prisma_client
    try:
        if prisma_client is None:
            raise Exception(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        ## GET USER ROW ##
        user_info = await prisma_client.get_data(user_id=user_id)
        ## GET ALL KEYS ##
        keys = await prisma_client.get_data(
            user_id=user_id, table_name="key", query_type="find_all"
        )
        return {"user_id": user_id, "user_info": user_info, "keys": keys}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.post(
    "/user/update", tags=["user management"], dependencies=[Depends(user_api_key_auth)]
)
async def user_update(request: Request):
    """
    [TODO]: Use this to update user budget
    """
    pass


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
            raise e
        else:
            raise HTTPException(
                status_code=500, detail=f"Internal Server Error: {str(e)}"
            )


#### [BETA] - This is a beta endpoint, format might change based on user feedback https://github.com/BerriAI/litellm/issues/933. If you need a stable endpoint use /model/info
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
async def model_info_v1(request: Request):
    global llm_model_list, general_settings, user_config_file_path, proxy_config

    # Load existing config
    config = await proxy_config.get_config()

    all_models = config["model_list"]
    for model in all_models:
        # provided model_info in config.yaml
        model_info = model.get("model_info", {})

        # read litellm model_prices_and_context_window.json to get the following:
        # input_cost_per_token, output_cost_per_token, max_tokens
        litellm_model_info = get_litellm_model_info(model=model)
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

    except HTTPException as e:
        # Re-raise the HTTP exceptions to be handled by FastAPI
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


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

        if "metadata" in data:
            verbose_proxy_logger.debug(f'received metadata: {data["metadata"]}')
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}
            data["metadata"]["headers"] = dict(request.headers)
            data["metadata"]["user_api_key_user_id"] = user_api_key_dict.user_id

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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/ollama_logs", dependencies=[Depends(user_api_key_auth)], tags=["experimental"]
)
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser("~/.ollama/logs/server.log")
    return FileResponse(filepath)


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
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred - {str(e)}")


@router.get("/config/yaml", tags=["config.yaml"])
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


@router.get("/test", tags=["health"])
async def test_endpoint(request: Request):
    """
    A test endpoint that pings the proxy server to check if it's healthy.

    Parameters:
        request (Request): The incoming request.

    Returns:
        dict: A dictionary containing the route of the request URL.
    """
    # ping the proxy server to check if its healthy
    return {"route": request.url.path}


@router.get("/health", tags=["health"], dependencies=[Depends(user_api_key_auth)])
async def health_endpoint(
    request: Request,
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


@router.get("/health/readiness", tags=["health"])
async def health_readiness():
    """
    Unprotected endpoint for checking if worker can receive requests
    """
    global prisma_client
    if prisma_client is not None:  # if db passed in, check if it's connected
        if prisma_client.db.is_connected() == True:
            return {"status": "healthy", "db": "connected"}
    else:
        return {"status": "healthy", "db": "Not connected"}
    raise HTTPException(status_code=503, detail="Service Unhealthy")


@router.get("/health/liveliness", tags=["health"])
async def health_liveliness():
    """
    Unprotected endpoint for checking if worker is alive
    """
    return "I'm alive!"


@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"


@router.get("/routes")
async def get_routes():
    """
    Get a list of available routes in the FastAPI application.
    """
    routes = []
    for route in app.routes:
        route_info = {
            "path": route.path,
            "methods": route.methods,
            "name": route.name,
            "endpoint": route.endpoint.__name__ if route.endpoint else None,
        }
        routes.append(route_info)

    return {"routes": routes}


@router.on_event("shutdown")
async def shutdown_event():
    global prisma_client, master_key, user_custom_auth
    if prisma_client:
        verbose_proxy_logger.debug("Disconnecting from Prisma")
        await prisma_client.disconnect()

    ## RESET CUSTOM VARIABLES ##
    cleanup_router_config_variables()


def cleanup_router_config_variables():
    global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, use_background_health_checks, health_check_interval

    # Set all variables to None
    master_key = None
    user_config_file_path = None
    otel_logging = None
    user_custom_auth = None
    user_custom_auth_path = None
    use_background_health_checks = None
    health_check_interval = None


app.include_router(router)
