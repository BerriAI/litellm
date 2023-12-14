import sys, os, platform, time, copy, re, asyncio, inspect
import threading, ast
import shutil, random, traceback, requests
from datetime import datetime, timedelta
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
    import uvicorn
    import fastapi
    import appdirs
    import backoff
    import yaml
    import rq
    import orjson
except ImportError:
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "uvicorn",
            "fastapi",
            "appdirs",
            "backoff",
            "pyyaml", 
            "rq",
            "orjson"
        ]
    )
    import uvicorn
    import fastapi
    import appdirs
    import backoff
    import yaml
    import orjson

    warnings.warn(
        "Installed runtime dependencies for proxy server. Specify these dependencies explicitly with `pip install litellm[proxy]`"
    )

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

    print()
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")
    print("\033[1;37m" + "# {:^59} #\033[0m".format(message))
    print(
        "\033[1;37m"
        + "# {:^59} #\033[0m".format("https://github.com/BerriAI/litellm/issues/new")
    )
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")
    print()
    print(" Thank you for using LiteLLM! - Krrish & Ishaan")
    print()
    print()
    print()
    print(
        "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"
    )
    print()
    print()

import litellm
from litellm.proxy.utils import (
    PrismaClient, 
    get_instance_fn, 
    ProxyLogging
)
import pydantic
from litellm.proxy._types import *
from litellm.caching import DualCache
from litellm.proxy.health_check import perform_health_check
litellm.suppress_debug_info = True
from fastapi import FastAPI, Request, HTTPException, status, Depends, BackgroundTasks, Header
from fastapi.routing import APIRouter
from fastapi.security import OAuth2PasswordBearer
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, FileResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import json
import logging
from typing import Union
# from litellm.proxy.queue import start_rq_worker_in_background

app = FastAPI(docs_url="/", title="LiteLLM API")
router = APIRouter()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def log_input_output(request, response, custom_logger=None):  
    if custom_logger is not None:
        custom_logger(request, response)
    global otel_logging
    if otel_logging != True:
        return
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    # Initialize OpenTelemetry components
    otlp_host = os.environ.get("OTEL_ENDPOINT", "localhost:4317")
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_host, insecure=True)
    resource = Resource.create({
        "service.name": "LiteLLM Proxy",
    })
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    span_processor = SimpleSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    with tracer.start_as_current_span("litellm-completion") as current_span:
        input_event_attributes = {f"{key}": str(value) for key, value in dict(request).items() if value is not None}
        # Log the input event with attributes
        current_span.add_event(
            name="LiteLLM: Request Input",
            attributes=input_event_attributes
        )
        event_headers = {f"{key}": str(value) for key, value in dict(request.headers).items() if value is not None}
        current_span.add_event(
            name="LiteLLM: Request Headers",
            attributes=event_headers
        )

        input_event_attributes.update(event_headers)

        input_event_attributes.update({f"{key}": str(value) for key, value in dict(response).items()})
        current_span.add_event(
            name="LiteLLM: Request Outpu",
            attributes=input_event_attributes
        )
        return True

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
user_config_file_path = f"config_{time.time()}.yaml"
local_logging = True # writes logs to a local api_log.json file for debugging
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
user_api_key_cache = DualCache()
user_custom_auth = None
use_background_health_checks = None
health_check_interval = None
health_check_results = {}
### INITIALIZE GLOBAL LOGGING OBJECT ###
proxy_logging_obj = ProxyLogging(user_api_key_cache=user_api_key_cache)
### REDIS QUEUE ### 
async_result = None
celery_app_conn = None 
celery_fn = None # Redis Queue for handling requests
#### HELPER FUNCTIONS ####
def print_verbose(print_statement):
    try:
        global user_debug
        if user_debug:
            print(print_statement)
    except:
        pass

def usage_telemetry(
    feature: str,
):  # helps us know if people are using this feature. Set `litellm --telemetry False` to your cli call to turn this off
    if user_telemetry:
        data = {"feature": feature}  # "local_proxy_server"
        threading.Thread(
            target=litellm.utils.litellm_telemetry, args=(data,), daemon=True
        ).start()

def _get_bearer_token(api_key: str): 
    assert api_key.startswith("Bearer ") # ensure Bearer token passed in
    api_key = api_key.replace("Bearer ", "") # extract the token
    return api_key

def _get_pydantic_json_dict(pydantic_obj: BaseModel) -> dict: 
    try:
        return pydantic_obj.model_dump() # type: ignore
    except:
        # if using pydantic v1
        return pydantic_obj.dict()

async def user_api_key_auth(request: Request, api_key: str = fastapi.Security(api_key_header)) -> UserAPIKeyAuth:
    global master_key, prisma_client, llm_model_list, user_custom_auth
    try: 
        if isinstance(api_key, str): 
            api_key = _get_bearer_token(api_key=api_key)
        ### USER-DEFINED AUTH FUNCTION ###
        if user_custom_auth:
            response = await user_custom_auth(request=request, api_key=api_key)
            return UserAPIKeyAuth.model_validate(response)
        
        if master_key is None:
            if isinstance(api_key, str):
                return UserAPIKeyAuth(api_key=api_key)
            else:
                return UserAPIKeyAuth()
            
        if api_key is None: # only require api key if master key is set
            raise Exception(f"No api key passed in.")

        route: str = request.url.path

        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        is_master_key_valid = secrets.compare_digest(api_key, master_key)
        if is_master_key_valid:
            return UserAPIKeyAuth(api_key=master_key)
        
        if route.startswith("/key/") and not is_master_key_valid:
            raise Exception(f"If master key is set, only master key can be used to generate, delete, update or get info for new keys")

        if prisma_client is None: # if both master key + user key submitted, and user key != master key, and no db connected, raise an error
            raise Exception("No connected db.")
        
        ## check for cache hit (In-Memory Cache)
        valid_token = user_api_key_cache.get_cache(key=api_key)
        print(f"valid_token from cache: {valid_token}")
        if valid_token is None: 
            ## check db 
            print(f"api key: {api_key}")
            valid_token = await prisma_client.get_data(token=api_key, expires=datetime.utcnow())
            print(f"valid token from prisma: {valid_token}")
            user_api_key_cache.set_cache(key=api_key, value=valid_token, ttl=60)
        elif valid_token is not None: 
            print(f"API Key Cache Hit!")
        if valid_token:
            litellm.model_alias_map = valid_token.aliases
            config = valid_token.config
            if config != {}:
                model_list = config.get("model_list", [])
                llm_model_list =  model_list
                print("\n new llm router model list", llm_model_list)
            if len(valid_token.models) == 0: # assume an empty model list means all models are allowed to be called
                api_key = valid_token.token
                valid_token_dict = _get_pydantic_json_dict(valid_token)
                valid_token_dict.pop("token", None)
                return UserAPIKeyAuth(api_key=api_key, **valid_token_dict)
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
            return UserAPIKeyAuth(api_key=api_key, **valid_token_dict)
        else: 
            raise Exception(f"Invalid token")
    except Exception as e: 
        print(f"An exception occurred - {traceback.format_exc()}")
        if isinstance(e, HTTPException): 
            raise e
        else: 
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid user key",
            )

def prisma_setup(database_url: Optional[str]): 
    global prisma_client, proxy_logging_obj, user_api_key_cache

    proxy_logging_obj._init_litellm_callbacks()
    if database_url is not None:
        try: 
            prisma_client = PrismaClient(database_url=database_url, proxy_logging_obj=proxy_logging_obj)
        except Exception as e:
            print("Error when initializing prisma, Ensure you run pip install prisma", e)

def celery_setup(use_queue: bool): 
    global celery_fn, celery_app_conn, async_result
    if use_queue:
        from litellm.proxy.queue.celery_worker import start_worker
        from litellm.proxy.queue.celery_app import celery_app, process_job
        from celery.result import AsyncResult
        start_worker(os.getcwd())
        celery_fn = process_job
        async_result = AsyncResult
        celery_app_conn = celery_app

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

        if KVUri is not None and client_id is not None and client_secret is not None and tenant_id is not None: 
            # Initialize the ClientSecretCredential
            credential = ClientSecretCredential(client_id=client_id, client_secret=client_secret, tenant_id=tenant_id)

            # Create the SecretClient using the credential
            client = SecretClient(vault_url=KVUri, credential=credential)
        
            litellm.secret_manager_client = client
        else: 
            raise Exception(f"Missing KVUri or client_id or client_secret or tenant_id from environment")
    except Exception as e: 
        print("Error when loading keys from Azure Key Vault. Ensure you run `pip install azure-identity azure-keyvault-secrets`")

def cost_tracking(): 
    global prisma_client
    if prisma_client is not None:
        if isinstance(litellm.success_callback, list):
            print("setting litellm success callback to track cost")
            if (track_cost_callback) not in litellm.success_callback: # type: ignore
                litellm.success_callback.append(track_cost_callback) # type: ignore
            else:
                litellm.success_callback = track_cost_callback # type: ignore

async def track_cost_callback(
    kwargs,                                       # kwargs to completion
    completion_response: litellm.ModelResponse,           # response from completion
    start_time = None,
    end_time = None,                              # start/end time for completion
):
    global prisma_client
    try:
        # check if it has collected an entire stream response
        print(f"kwargs stream: {kwargs.get('stream', None)} + complete streaming response: {kwargs.get('complete_streaming_response', None)}")
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost 
            completion_response=kwargs["complete_streaming_response"]
            response_cost = litellm.completion_cost(completion_response=completion_response)
            print("streaming response_cost", response_cost)
            user_api_key = kwargs["litellm_params"]["metadata"].get("user_api_key", None)
            if user_api_key and prisma_client: 
                await update_prisma_database(token=user_api_key, response_cost=response_cost)
        elif kwargs["stream"] == False: # for non streaming responses
            response_cost = litellm.completion_cost(completion_response=completion_response)
            user_api_key = kwargs["litellm_params"]["metadata"].get("user_api_key", None)
            if user_api_key and prisma_client: 
                await update_prisma_database(token=user_api_key, response_cost=response_cost)
    except Exception as e:
        print(f"error in tracking cost callback - {str(e)}")

async def update_prisma_database(token, response_cost):
    try:
        print(f"Enters prisma db call, token: {token}")
        # Fetch the existing cost for the given token
        existing_spend_obj = await prisma_client.get_data(token=token)
        print(f"existing spend: {existing_spend_obj}")
        if existing_spend_obj is None: 
            existing_spend = 0
        else:
            existing_spend = existing_spend_obj.spend
        # Calculate the new cost by adding the existing cost and response_cost
        new_spend = existing_spend + response_cost

        print(f"new cost: {new_spend}")
        # Update the cost column for the given token
        await prisma_client.update_data(token=token, data={"spend": new_spend})
    except Exception as e:
        print(f"Error updating Prisma database: {traceback.format_exc()}")
        pass

def run_ollama_serve():
    try:
        command = ['ollama', 'serve']

        with open(os.devnull, 'w') as devnull:
            process = subprocess.Popen(command, stdout=devnull, stderr=devnull)
    except Exception as e:
        print(f"""
            LiteLLM Warning: proxy started with `ollama` model\n`ollama serve` failed with Exception{e}. \nEnsure you run `ollama serve`
        """)

async def _run_background_health_check():
    """
    Periodically run health checks in the background on the endpoints. 

    Update health_check_results, based on this.
    """
    global health_check_results, llm_model_list, health_check_interval
    while True:
        healthy_endpoints, unhealthy_endpoints = await perform_health_check(model_list=llm_model_list)

        # Update the global variable with the health check results
        health_check_results["healthy_endpoints"] = healthy_endpoints
        health_check_results["unhealthy_endpoints"] = unhealthy_endpoints
        health_check_results["healthy_count"] = len(healthy_endpoints)
        health_check_results["unhealthy_count"] = len(unhealthy_endpoints)

        await asyncio.sleep(health_check_interval)

def load_router_config(router: Optional[litellm.Router], config_file_path: str):
    global master_key, user_config_file_path, otel_logging, user_custom_auth, user_custom_auth_path, use_background_health_checks, health_check_interval
    config = {}
    try: 
        if os.path.exists(config_file_path):
            user_config_file_path = config_file_path
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            raise Exception(f"Path to config does not exist, Current working directory: {os.getcwd()}, 'os.path.exists({config_file_path})' returned False")
    except Exception as e:
        raise Exception(f"Exception while reading Config: {e}")
    
    ## PRINT YAML FOR CONFIRMING IT WORKS 
    printed_yaml = copy.deepcopy(config)
    printed_yaml.pop("environment_variables", None)

    print_verbose(f"Loaded config YAML (api_key and environment_variables are not shown):\n{json.dumps(printed_yaml, indent=2)}")

    ## ENVIRONMENT VARIABLES
    environment_variables = config.get('environment_variables', None)
    if environment_variables: 
        for key, value in environment_variables.items(): 
            os.environ[key] = value

    ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
    litellm_settings = config.get('litellm_settings', None)
    if litellm_settings is None: 
        litellm_settings = {}
    if litellm_settings: 
        # ANSI escape code for blue text
        blue_color_code = "\033[94m"
        reset_color_code = "\033[0m"
        for key, value in litellm_settings.items(): 
            if key == "cache":
                print(f"{blue_color_code}\nSetting Cache on Proxy")
                from litellm.caching import Cache
                if isinstance(value, dict):
                    cache_type = value.get("type", "redis")
                else:
                    cache_type = "redis" # default to using redis on cache
                cache_responses = True
                cache_host = litellm.get_secret("REDIS_HOST", None)
                cache_port = litellm.get_secret("REDIS_PORT", None)
                cache_password = litellm.get_secret("REDIS_PASSWORD", None)

                # Assuming cache_type, cache_host, cache_port, and cache_password are strings
                print(f"{blue_color_code}Cache Type:{reset_color_code} {cache_type}")
                print(f"{blue_color_code}Cache Host:{reset_color_code} {cache_host}")
                print(f"{blue_color_code}Cache Port:{reset_color_code} {cache_port}")
                print(f"{blue_color_code}Cache Password:{reset_color_code} {cache_password}")
                print()

                ## to pass a complete url, or set ssl=True, etc. just set it as `os.environ[REDIS_URL] = <your-redis-url>`, _redis.py checks for REDIS specific environment variables
                litellm.cache = Cache(
                    type=cache_type,
                    host=cache_host,
                    port=cache_port,
                    password=cache_password
                )
                print(f"{blue_color_code}Set Cache on LiteLLM Proxy: {litellm.cache.cache}{reset_color_code} {cache_password}")
            elif key == "callbacks":
                litellm.callbacks = [get_instance_fn(value=value, config_file_path=config_file_path)]
                print_verbose(f"{blue_color_code} Initialized Callbacks - {litellm.callbacks} {reset_color_code}")
            elif key == "success_callback":
                litellm.success_callback = []
                
                # intialize success callbacks
                for callback in value:
                    # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                    if "." in callback: 
                        litellm.success_callback.append(get_instance_fn(value=callback))
                    # these are litellm callbacks - "langfuse", "sentry", "wandb"
                    else:
                        litellm.success_callback.append(callback)
                        if callback == "traceloop":
                            from traceloop.sdk import Traceloop
                            print_verbose(f"{blue_color_code} Initializing Traceloop SDK - \nRunning:`Traceloop.init(app_name='Litellm-Server', disable_batch=True)`")
                            Traceloop.init(app_name="Litellm-Server", disable_batch=True)
                print_verbose(f"{blue_color_code} Initialized Success Callbacks - {litellm.success_callback} {reset_color_code}")
            elif key == "failure_callback":
                litellm.failure_callback = []
                
                # intialize success callbacks
                for callback in value:
                    # user passed custom_callbacks.async_on_succes_logger. They need us to import a function
                    if "." in callback: 
                        litellm.failure_callback.append(get_instance_fn(value=callback))
                    # these are litellm callbacks - "langfuse", "sentry", "wandb"
                    else:
                        litellm.failure_callback.append(callback)
                print_verbose(f"{blue_color_code} Initialized Success Callbacks - {litellm.failure_callback} {reset_color_code}")
            else:
                setattr(litellm, key, value)



    ## GENERAL SERVER SETTINGS (e.g. master key,..) # do this after initializing litellm, to ensure sentry logging works for proxylogging
    general_settings = config.get("general_settings", {})
    if general_settings is None: 
        general_settings = {}
    if general_settings: 
        ### LOAD FROM AZURE KEY VAULT ###
        use_azure_key_vault = general_settings.get("use_azure_key_vault", False)
        load_from_azure_key_vault(use_azure_key_vault=use_azure_key_vault)
        ### CONNECT TO DATABASE ###
        database_url = general_settings.get("database_url", None)
        if database_url and database_url.startswith("os.environ/"): 
            database_url = litellm.get_secret(database_url)
        prisma_setup(database_url=database_url)
        ## COST TRACKING ## 
        cost_tracking()
        ### START REDIS QUEUE ###
        use_queue = general_settings.get("use_queue", False)
        celery_setup(use_queue=use_queue)
        ### MASTER KEY ###
        master_key = general_settings.get("master_key", None)
        if master_key and master_key.startswith("os.environ/"): 
            master_key = litellm.get_secret(master_key)
        #### OpenTelemetry Logging (OTEL) ########
        otel_logging =  general_settings.get("otel", False)
        if otel_logging == True:
            print("\nOpenTelemetry Logging Activated")
        ### CUSTOM API KEY AUTH ###
        custom_auth = general_settings.get("custom_auth", None)
        if custom_auth:
            user_custom_auth = get_instance_fn(value=custom_auth, config_file_path=config_file_path)
        ### BACKGROUND HEALTH CHECKS ###
        # Enable background health checks
        use_background_health_checks = general_settings.get("background_health_checks", False)
        health_check_interval = general_settings.get("health_check_interval", 300)

    router_params: dict = {
        "num_retries": 3, 
        "cache_responses": litellm.cache != None # cache if user passed in cache values
    }
    ## MODEL LIST
    model_list = config.get('model_list', None)
    if model_list:
        router_params["model_list"] = model_list
        print(f"\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m")
        for model in model_list:
            print(f"\033[32m    {model.get('model_name', '')}\033[0m")
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

        available_args = [
            x for x in arg_spec.args if x not in exclude_args
        ]

        for k, v in router_settings.items(): 
            if k in available_args: 
                router_params[k] = v
    
    router = litellm.Router(**router_params) # type:ignore
    return router, model_list, general_settings

async def generate_key_helper_fn(duration: Optional[str], models: list, aliases: dict, config: dict, spend: float, token: Optional[str]=None, user_id: Optional[str]=None, max_parallel_requests: Optional[int]=None):
    global prisma_client

    if prisma_client is None: 
        raise Exception(f"Connect Proxy to database to generate keys - https://docs.litellm.ai/docs/proxy/virtual_keys ")
    
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
        
    if duration is None: # allow tokens that never expire 
        expires = None
    else: 
        duration_s = _duration_in_seconds(duration=duration)
        expires = datetime.utcnow() + timedelta(seconds=duration_s)
    
    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    user_id = user_id or str(uuid.uuid4())
    try:
        # Create a new verification token (you may want to enhance this logic based on your needs)
        verification_token_data = {
            "token": token, 
            "expires": expires,
            "models": models,
            "aliases": aliases_json,
            "config": config_json,
            "spend": spend, 
            "user_id": user_id, 
            "max_parallel_requests": max_parallel_requests
        }
        new_verification_token = await prisma_client.insert_data(data=verification_token_data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return {"token": token, "expires": new_verification_token.expires, "user_id": user_id}



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

def initialize(
    model=None,
    alias=None,
    api_base=None,
    api_version=None,
    debug=False,
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
    global user_model, user_api_base, user_debug, user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers, experimental, llm_model_list, llm_router, general_settings, master_key, user_custom_auth
    generate_feedback_box()
    user_model = model
    user_debug = debug
    if debug==True:  # this needs to be first, so users can see Router init debugg
        litellm.set_verbose = True
    dynamic_config = {"general": {}, user_model: {}}
    if config:
        llm_router, llm_model_list, general_settings = load_router_config(router=llm_router, config_file_path=config)
    else: 
        # reset auth if config not passed, needed for consecutive tests on proxy
        master_key = None 
        user_custom_auth = None
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
    if use_queue: 
        celery_setup(use_queue=use_queue)
    if experimental: 
        pass
    user_telemetry = telemetry
    usage_telemetry(feature="local_proxy_server")
    curl_command = """
    curl --location 'http://0.0.0.0:8000/chat/completions' \\
    --header 'Content-Type: application/json' \\
    --data ' {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
    }'
    \n
    """
    print()
    print(f"\033[1;34mLiteLLM: Test your local proxy with: \"litellm --test\" This runs an openai.ChatCompletion request to your proxy [In a new terminal tab]\033[0m\n")
    print(f"\033[1;34mLiteLLM: Curl Command Test for your local proxy\n {curl_command} \033[0m\n")
    print("\033[1;34mDocs: https://docs.litellm.ai/docs/simple_proxy\033[0m\n")
# for streaming
def data_generator(response):
    print_verbose("inside generator")
    for chunk in response:
        print_verbose(f"returned chunk: {chunk}")
        try:
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"

async def async_data_generator(response, user_api_key_dict):
    print_verbose("inside generator")
    async for chunk in response:
        print_verbose(f"returned chunk: {chunk}")
        try:
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"

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
    
@router.on_event("startup")
async def startup_event():
    global prisma_client, master_key, use_background_health_checks
    import json

    ### LOAD CONFIG ### 
    worker_config = litellm.get_secret("WORKER_CONFIG")
    print_verbose(f"worker_config: {worker_config}")
    # check if it's a valid file path
    if os.path.isfile(worker_config):
        initialize(config=worker_config)
    else:
        # if not, assume it's a json string
        worker_config = json.loads(os.getenv("WORKER_CONFIG"))
        initialize(**worker_config)
    
    
    if use_background_health_checks:
        asyncio.create_task(_run_background_health_check()) # start the background health check coroutine. 

    print_verbose(f"prisma client - {prisma_client}")
    if prisma_client: 
        await prisma_client.connect()
    
    if prisma_client is not None and master_key is not None: 
        # add master key to db
        await generate_key_helper_fn(duration=None, models=[], aliases={}, config={}, spend=0, token=master_key)


#### API ENDPOINTS ####
@router.get("/v1/models", dependencies=[Depends(user_api_key_auth)])
@router.get("/models", dependencies=[Depends(user_api_key_auth)])  # if project requires model list
def model_list():
    global llm_model_list, general_settings    
    all_models = []
    if general_settings.get("infer_model_from_keys", False):
        all_models = litellm.utils.get_valid_models()
    if llm_model_list: 
        all_models = list(set(all_models + [m["model_name"] for m in llm_model_list]))
    if user_model is not None:
        all_models += [user_model]
    print_verbose(f"all_models: {all_models}")
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

@router.post("/v1/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/engines/{model:path}/completions", dependencies=[Depends(user_api_key_auth)])
async def completion(request: Request, model: Optional[str] = None, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth), background_tasks: BackgroundTasks = BackgroundTasks()):
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
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}

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
        data = await proxy_logging_obj.pre_call_hook(user_api_key_dict=user_api_key_dict, data=data, call_type="completion")

        ### ROUTE THE REQUEST ###
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data["model"] in router_model_names: # model in router model list 
                response = await llm_router.atext_completion(**data)
        elif llm_router is not None and data["model"] in llm_router.deployment_names: # model in router deployments, calling a specific deployment on the router
            response = await llm_router.atext_completion(**data, specific_deployment = True)
        elif llm_router is not None and llm_router.model_group_alias is not None and data["model"] in llm_router.model_group_alias: # model set in model_group_alias
            response = await llm_router.atext_completion(**data)
        else: # router is not set
            response = await litellm.atext_completion(**data)
        
        print(f"final response: {response}")
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(async_data_generator(user_api_key_dict=user_api_key_dict, response=response), media_type='text/event-stream')
        
        background_tasks.add_task(log_input_output, request, response) # background task for logging to OTEL 
        return response
    except Exception as e: 
        print(f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`")
        traceback.print_exc()
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        try:
            status = e.status_code  # type: ignore
        except:
            status = 500
        raise HTTPException(
            status_code=status,
            detail=error_msg
        )

@router.post("/v1/chat/completions", dependencies=[Depends(user_api_key_auth)], tags=["chat/completions"])
@router.post("/chat/completions", dependencies=[Depends(user_api_key_auth)], tags=["chat/completions"])
@router.post("/openai/deployments/{model:path}/chat/completions", dependencies=[Depends(user_api_key_auth)], tags=["chat/completions"]) # azure compatible endpoint
async def chat_completion(request: Request, model: Optional[str] = None, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth), background_tasks: BackgroundTasks = BackgroundTasks()):
    global general_settings, user_debug, proxy_logging_obj
    try: 
        data = {}
        data = await request.json() # type: ignore 

        # Include original request and headers in the data
        data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": copy.copy(data)  # use copy instead of deepcopy
        }

        print_verbose(f"receiving data: {data}")
        data["model"] = (
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )

        # users can pass in 'user' param to /chat/completions. Don't override it
        if data.get("user", None) is None:
            # if users are using user_api_key_auth, set `user` in `data`
            data["user"] = user_api_key_dict.user_id

        if "metadata" in data:
            print(f'received metadata: {data["metadata"]}')
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["headers"] = dict(request.headers)
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}
            data["metadata"]["headers"] = dict(request.headers)
        
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
        data = await proxy_logging_obj.pre_call_hook(user_api_key_dict=user_api_key_dict, data=data, call_type="completion")

        ### ROUTE THE REQUEST ###
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data["model"] in router_model_names: # model in router model list 
                response = await llm_router.acompletion(**data)
        elif llm_router is not None and data["model"] in llm_router.deployment_names: # model in router deployments, calling a specific deployment on the router
            response = await llm_router.acompletion(**data, specific_deployment = True)
        elif llm_router is not None and llm_router.model_group_alias is not None and data["model"] in llm_router.model_group_alias: # model set in model_group_alias
            response = await llm_router.acompletion(**data)
        else: # router is not set
            response = await litellm.acompletion(**data)
        
        print(f"final response: {response}")
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(async_data_generator(user_api_key_dict=user_api_key_dict, response=response), media_type='text/event-stream')
        
        background_tasks.add_task(log_input_output, request, response) # background task for logging to OTEL 
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(user_api_key_dict=user_api_key_dict, original_exception=e) 
        print(f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`")
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data.get("model", "") in router_model_names: 
            print("Results from router")
            print("\nRouter stats")
            print("\nTotal Calls made")
            for key, value in llm_router.total_calls.items():
                print(f"{key}: {value}")
            print("\nSuccess Calls made")
            for key, value in llm_router.success_calls.items():
                print(f"{key}: {value}")
            print("\nFail Calls made")
            for key, value in llm_router.fail_calls.items():
                print(f"{key}: {value}")
        if user_debug: 
            traceback.print_exc()
        
        if isinstance(e, HTTPException):
            raise e
        else:
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"
            try:
                status = e.status_code # type: ignore
            except:
                status = 500
            raise HTTPException(
                status_code=status,
                detail=error_msg
            )

@router.post("/v1/embeddings", dependencies=[Depends(user_api_key_auth)], response_class=ORJSONResponse)
@router.post("/embeddings", dependencies=[Depends(user_api_key_auth)], response_class=ORJSONResponse)
async def embeddings(request: Request, user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth), background_tasks: BackgroundTasks = BackgroundTasks()): 
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
            "body": copy.copy(data)  # use copy instead of deepcopy
        }

        data["user"] = data.get("user", user_api_key_dict.user_id)
        data["model"] = (
            general_settings.get("embedding_model", None) # server default
            or user_model # model name passed via cli args
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict.api_key
            data["metadata"]["headers"] = dict(request.headers)
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict.api_key}
            data["metadata"]["headers"] = dict(request.headers)
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if "input" in data and isinstance(data['input'], list) and isinstance(data['input'][0], list) and isinstance(data['input'][0][0], int): # check if array of tokens passed in
            # check if non-openai/azure model called - e.g. for langchain integration
            if llm_model_list is not None and data["model"] in router_model_names: 
                for m in llm_model_list: 
                    if m["model_name"] == data["model"] and (m["litellm_params"]["model"] in litellm.open_ai_embedding_models 
                                                             or m["litellm_params"]["model"].startswith("azure/")):
                        pass
                    else: 
                        # non-openai/azure embedding model called with token input
                        input_list = []
                        for i in data["input"]: 
                            input_list.append(litellm.decode(model="gpt-3.5-turbo", tokens=i))
                        data["input"] = input_list
                        break
        
        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        data = await proxy_logging_obj.pre_call_hook(user_api_key_dict=user_api_key_dict, data=data, call_type="embeddings")
        ## ROUTE TO CORRECT ENDPOINT ##
        if llm_router is not None and data["model"] in router_model_names: # model in router model list 
            response = await llm_router.aembedding(**data)
        elif llm_router is not None and data["model"] in llm_router.deployment_names: # model in router deployments, calling a specific deployment on the router
            response = await llm_router.aembedding(**data, specific_deployment = True)
        elif llm_router is not None and llm_router.model_group_alias is not None and data["model"] in llm_router.model_group_alias: # model set in model_group_alias
            response = await llm_router.aembedding(**data) # ensure this goes the llm_router, router will do the correct alias mapping
        else:
            response = await litellm.aembedding(**data)
        background_tasks.add_task(log_input_output, request, response) # background task for logging to OTEL 

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(user_api_key_dict=user_api_key_dict, original_exception=e) 
        traceback.print_exc()
        raise e

#### KEY MANAGEMENT #### 

@router.post("/key/generate", tags=["key management"], dependencies=[Depends(user_api_key_auth)], response_model=GenerateKeyResponse)
async def generate_key_fn(request: Request, data: GenerateKeyRequest, Authorization: Optional[str] = Header(None)): 
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

    Returns:
    - key: (str) The generated api key 
    - expires: (datetime) Datetime object for when key expires.
    - user_id: (str) Unique user id - used for tracking spend across multiple keys for same user id.
    """
    # data = await request.json()
    data_json = data.json()   # type: ignore
    response = await generate_key_helper_fn(**data_json)
    return GenerateKeyResponse(key=response["token"], expires=response["expires"], user_id=response["user_id"])

@router.post("/key/update", tags=["key management"], dependencies=[Depends(user_api_key_auth)])
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
        print(f"non_default_values: {non_default_values}")
        response = await prisma_client.update_data(token=key, data={**non_default_values, "token": key})
        return {"key": key, **non_default_values}
        # update based on remaining passed in values 
    except Exception as e: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )

@router.post("/key/delete", tags=["key management"], dependencies=[Depends(user_api_key_auth)])
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

@router.get("/key/info", tags=["key management"], dependencies=[Depends(user_api_key_auth)])
async def info_key_fn(key: str = fastapi.Query(..., description="Key in the request parameters")): 
    global prisma_client
    try: 
        if prisma_client is None: 
            raise Exception(f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys")
        key_info = await prisma_client.get_data(token=key)
        return {"key": key, "info": key_info}
    except Exception as e: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )

#### MODEL MANAGEMENT #### 
        
#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post("/model/new", description="Allows adding new models to the model list in the config.yaml", tags=["model management"], dependencies=[Depends(user_api_key_auth)])
async def add_new_model(model_params: ModelParams):
    global llm_router, llm_model_list, general_settings, user_config_file_path
    try:
        print_verbose(f"User config path: {user_config_file_path}")
        # Load existing config
        if os.path.exists(f"{user_config_file_path}"):
            with open(f"{user_config_file_path}", "r") as config_file:
                config = yaml.safe_load(config_file)
        else: 
            config = {"model_list": []} 
        
        print_verbose(f"Loaded config: {config}")
        # Add the new model to the config
        model_info = model_params.model_info.json()
        model_info = {k: v for k, v in model_info.items() if v is not None}
        config['model_list'].append({
            'model_name': model_params.model_name,
            'litellm_params': model_params.litellm_params,
            'model_info': model_info
        })

        # Save the updated config
        with open(f"{user_config_file_path}", "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

        # update Router 
        llm_router, llm_model_list, general_settings = load_router_config(router=llm_router, config_file_path=user_config_file_path)


        return {"message": "Model added successfully"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

#### [BETA] - This is a beta endpoint, format might change based on user feedback https://github.com/BerriAI/litellm/issues/933. If you need a stable endpoint use /model/info
@router.get("/model/info", description="Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)", tags=["model management"], dependencies=[Depends(user_api_key_auth)])
async def model_info_v1(request: Request):
    global llm_model_list, general_settings, user_config_file_path
    # Load existing config
    with open(f"{user_config_file_path}", "r") as config_file:
        config = yaml.safe_load(config_file)
    all_models = config['model_list']
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

    print_verbose(f"all_models: {all_models}")
    return {
        "data": all_models
    }


#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/933
@router.get("/v1/model/info", description="Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)", tags=["model management"], dependencies=[Depends(user_api_key_auth)])
async def model_info(request: Request):
    global llm_model_list, general_settings, user_config_file_path
    # Load existing config
    with open(f"{user_config_file_path}", "r") as config_file:
        config = yaml.safe_load(config_file)
    all_models = config['model_list']

    for model in all_models:
        # get the model cost map info 
        ## make an api call
        data = copy.deepcopy(model["litellm_params"])
        data["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
        data["max_tokens"] = 10
        print(f"data going to litellm acompletion: {data}")
        response = await litellm.acompletion(**data)
        response_model = response["model"]
        print(f"response model: {response_model}; response - {response}")
        litellm_model_info = litellm.get_model_info(response_model)
        model_info = model.get("model_info", {})
        for k, v in litellm_model_info.items(): 
            if k not in model_info: 
                model_info[k] = v
        model["model_info"] = model_info
        # don't return the api key
        model["litellm_params"].pop("api_key", None)
        
    # all_models = list(set([m["model_name"] for m in llm_model_list]))
    print_verbose(f"all_models: {all_models}")
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

#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/964
@router.post("/model/delete", description="Allows deleting models in the model list in the config.yaml", tags=["model management"], dependencies=[Depends(user_api_key_auth)])
async def delete_model(model_info: ModelInfoDelete):
    global llm_router, llm_model_list, general_settings, user_config_file_path
    try:
        if not os.path.exists(user_config_file_path):
            raise HTTPException(status_code=404, detail="Config file does not exist.")

        with open(user_config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        # If model_list is not in the config, nothing can be deleted
        if 'model_list' not in config:
            raise HTTPException(status_code=404, detail="No model list available in the config.")

        # Check if the model with the specified model_id exists
        model_to_delete = None
        for model in config['model_list']:
            if model.get('model_info', {}).get('id', None) == model_info.id:
                model_to_delete = model
                break

        # If the model was not found, return an error
        if model_to_delete is None:
            raise HTTPException(status_code=404, detail="Model with given model_id not found.")

        # Remove model from the list and save the updated config
        config['model_list'].remove(model_to_delete)
        with open(user_config_file_path, "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

        # Update Router
        llm_router, llm_model_list, general_settings = load_router_config(router=llm_router, config_file_path=user_config_file_path)

        return {"message": "Model deleted successfully"}

    except HTTPException as e:
        # Re-raise the HTTP exceptions to be handled by FastAPI
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

#### EXPERIMENTAL QUEUING #### 
@router.post("/queue/request", dependencies=[Depends(user_api_key_auth)])
async def async_queue_request(request: Request): 
    global celery_fn, llm_model_list
    if celery_fn is not None: 
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except: 
            data = json.loads(body_str)
        data["model"] = (
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or data["model"] # default passed in http request
        )
        data["llm_model_list"] = llm_model_list
        print(f"data: {data}")
        job = celery_fn.apply_async(kwargs=data)
        return {"id": job.id, "url": f"/queue/response/{job.id}", "eta": 5, "status": "queued"}
    else: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Queue not initialized"},
        )

@router.get("/queue/response/{task_id}", dependencies=[Depends(user_api_key_auth)])
async def async_queue_response(request: Request, task_id: str): 
    global celery_app_conn, async_result
    try: 
        if celery_app_conn is not None and async_result is not None: 
            job = async_result(task_id, app=celery_app_conn) 
            if job.ready():
                return {"status": "finished", "result": job.result}
            else:
                return {'status': 'queued'}
        else:
            raise Exception()
    except Exception as e: 
        return {"status": "finished", "result": str(e)}


@router.get("/ollama_logs", dependencies=[Depends(user_api_key_auth)])
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser("~/.ollama/logs/server.log")
    return FileResponse(filepath)


#### BASIC ENDPOINTS #### 

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
async def health_endpoint(request: Request, model: Optional[str] = fastapi.Query(None, description="Specify the model name (optional)")):
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
    global health_check_results, use_background_health_checks

    if llm_model_list is None: 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Model list not initialized"},
        )
    
    if use_background_health_checks:
        return health_check_results
    else:
        healthy_endpoints, unhealthy_endpoints = await perform_health_check(llm_model_list, model)

        return {
            "healthy_endpoints": healthy_endpoints,
            "unhealthy_endpoints": unhealthy_endpoints,
            "healthy_count": len(healthy_endpoints),
            "unhealthy_count": len(unhealthy_endpoints),
        }

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
        print("Disconnecting from Prisma")
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
