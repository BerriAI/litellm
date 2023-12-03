import sys, os, platform, time, copy, re, asyncio
import threading, ast
import shutil, random, traceback, requests
from datetime import datetime, timedelta
from typing import Optional, List
import secrets, subprocess
import hashlib, uuid
import warnings
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
    PrismaClient
)
from litellm.caching import DualCache
litellm.suppress_debug_info = True
from fastapi import FastAPI, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.routing import APIRouter
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
def log_input_output(request, response):  
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
from pydantic import BaseModel         
######### Request Class Definition ######
class ProxyChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    tools: Optional[List[str]] = None
    tool_choice: Optional[str] = None
    functions: Optional[List[str]] = None  # soon to be deprecated
    function_call: Optional[str] = None # soon to be deprecated

    # Optional LiteLLM params
    caching: Optional[bool] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    num_retries: Optional[int] = None
    context_window_fallback_dict: Optional[Dict[str, str]] = None
    fallbacks: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = {}
    deployment_id: Optional[str] = None
    request_timeout: Optional[int] = None

    class Config:
        extra='allow' # allow params not defined here, these fall in litellm.completion(**kwargs)

class ModelParams(BaseModel):
    model_name: str
    litellm_params: dict
    model_info: Optional[dict]
    class Config:
        protected_namespaces = ()

class GenerateKeyRequest(BaseModel):
    duration: str = "1h"
    models: list = []
    aliases: dict = {}
    config: dict = {}
    spend: int = 0
    user_id: Optional[str]

class GenerateKeyResponse(BaseModel):
    key: str
    expires: str
    user_id: str

class _DeleteKeyObject(BaseModel):
    key: str

class DeleteKeyRequest(BaseModel):
    keys: List[_DeleteKeyObject]


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
### REDIS QUEUE ### 
async_result = None
celery_app_conn = None 
celery_fn = None # Redis Queue for handling requests
#### HELPER FUNCTIONS ####
def print_verbose(print_statement):
    global user_debug
    if user_debug:
        print(print_statement)

def usage_telemetry(
    feature: str,
):  # helps us know if people are using this feature. Set `litellm --telemetry False` to your cli call to turn this off
    if user_telemetry:
        data = {"feature": feature}  # "local_proxy_server"
        threading.Thread(
            target=litellm.utils.litellm_telemetry, args=(data,), daemon=True
        ).start()

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def user_api_key_auth(request: Request, api_key: str = fastapi.Security(api_key_header)):
    global master_key, prisma_client, llm_model_list
    if master_key is None:
        return {
            "api_key": None
        } 
    try: 
        route = request.url.path

        # note: never string compare api keys, this is vulenerable to a time attack. Use secrets.compare_digest instead
        is_master_key_valid = secrets.compare_digest(api_key, master_key) or secrets.compare_digest(api_key, "Bearer " + master_key)
        if is_master_key_valid:
            return {
                "api_key": master_key
            }
        
        if (route == "/key/generate" or route == "/key/delete" or route == "/key/info") and not is_master_key_valid:
            raise Exception(f"If master key is set, only master key can be used to generate, delete or get info for new keys")

        if prisma_client: 
            ## check for cache hit (In-Memory Cache)
            valid_token = user_api_key_cache.get_cache(key=api_key)
            if valid_token is None and "Bearer " in api_key: 
                ## check db 
                cleaned_api_key = api_key[len("Bearer "):]
                valid_token = await prisma_client.get_data(token=cleaned_api_key, expires=datetime.utcnow())
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
                    return_dict = {"api_key": valid_token.token}
                    if valid_token.user_id:
                        return_dict["user_id"] = valid_token.user_id
                    return return_dict
                else: 
                    data = await request.json()
                    model = data.get("model", None)
                    if model in litellm.model_alias_map:
                        model = litellm.model_alias_map[model]
                    if model and model not in valid_token.models:
                        raise Exception(f"Token not allowed to access model")
                return_dict = {"api_key": valid_token.token}
                if valid_token.user_id:
                    return_dict["user_id"] = valid_token.user_id
                return return_dict 
            else: 
                raise Exception(f"Invalid token")
    except Exception as e: 
        print(f"An exception occurred - {traceback.format_exc()}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "invalid user key"},
    )

def prisma_setup(database_url: Optional[str]): 
    global prisma_client
    if database_url:
        try: 
            prisma_client = PrismaClient(database_url=database_url)
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
    global prisma_client, master_key
    if prisma_client is not None and master_key is not None:
        if isinstance(litellm.success_callback, list):
            print("setting litellm success callback to track cost")
            if (track_cost_callback) not in litellm.success_callback: # type: ignore
                litellm.success_callback.append(track_cost_callback) # type: ignore
            else:
                litellm.success_callback = track_cost_callback # type: ignore

def track_cost_callback(
    kwargs,                                       # kwargs to completion
    completion_response: litellm.ModelResponse,           # response from completion
    start_time = None,
    end_time = None,                              # start/end time for completion
):
    global prisma_client
    try:
        # check if it has collected an entire stream response
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost 
            completion_response=kwargs["complete_streaming_response"]
            input_text = kwargs["messages"]
            output_text = completion_response["choices"][0]["message"]["content"]
            response_cost = litellm.completion_cost(
                model = kwargs["model"],
                messages = input_text,
                completion=output_text
            )
            print("streaming response_cost", response_cost)
        # for non streaming responses
        elif kwargs["stream"] is False: # regular response
            input_text = kwargs.get("messages", "")
            if isinstance(input_text, list): 
                input_text = "".join(m["content"] for m in input_text)
            print(f"received completion response: {completion_response}")
            response_cost = litellm.completion_cost(completion_response=completion_response, completion=input_text)
            print("regular response_cost", response_cost)
        user_api_key = kwargs["litellm_params"]["metadata"].get("user_api_key", None)
        if user_api_key and prisma_client: 
            # asyncio.run(update_prisma_database(user_api_key, response_cost))
            # Create new event loop for async function execution in the new thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                # Run the async function using the newly created event loop
                existing_spend_obj = new_loop.run_until_complete(prisma_client.get_data(token=user_api_key))
                if existing_spend_obj is None: 
                    existing_spend = 0
                else:
                    existing_spend = existing_spend_obj.spend
                # Calculate the new cost by adding the existing cost and response_cost
                new_spend = existing_spend + response_cost
                print(f"new cost: {new_spend}")
                # Update the cost column for the given token
                new_loop.run_until_complete(prisma_client.update_data(token=user_api_key, data={"spend": new_spend}))
                print(f"Prisma database updated for token {user_api_key}. New cost: {new_spend}")
            except Exception as e:
                print(f"error in creating async loop - {str(e)}")
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
        print(f"Prisma database updated for token {token}. New cost: {new_spend}")

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

def load_router_config(router: Optional[litellm.Router], config_file_path: str):
    global master_key, user_config_file_path, otel_logging
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
    for model in printed_yaml["model_list"]:
        model["litellm_params"].pop("api_key", None)

    print(f"Loaded config YAML (api_key and environment_variables are not shown):\n{json.dumps(printed_yaml, indent=2)}")

    ## ENVIRONMENT VARIABLES
    environment_variables = config.get('environment_variables', None)
    if environment_variables: 
        for key, value in environment_variables.items(): 
            os.environ[key] = value

    ## GENERAL SERVER SETTINGS (e.g. master key,..)
    general_settings = config.get("general_settings", {})
    if general_settings is None: 
        general_settings = {}
    if general_settings: 
        ### MASTER KEY ###
        master_key = general_settings.get("master_key", None)
        if master_key and master_key.startswith("os.environ/"): 
            master_key_env_name = master_key.replace("os.environ/", "")
            master_key = os.getenv(master_key_env_name)
        ### CONNECT TO DATABASE ###
        database_url = general_settings.get("database_url", None)
        prisma_setup(database_url=database_url)
        ## COST TRACKING ## 
        cost_tracking()
        ### START REDIS QUEUE ###
        use_queue = general_settings.get("use_queue", False)
        celery_setup(use_queue=use_queue)
        ### LOAD FROM AZURE KEY VAULT ###
        use_azure_key_vault = general_settings.get("use_azure_key_vault", False)
        load_from_azure_key_vault(use_azure_key_vault=use_azure_key_vault)

        #### OpenTelemetry Logging (OTEL) ########
        otel_logging =  general_settings.get("otel", False)
        if otel_logging == True:
            print("\nOpenTelemetry Logging Activated")

    ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
    litellm_settings = config.get('litellm_settings', None)
    if litellm_settings: 
        # ANSI escape code for blue text
        blue_color_code = "\033[94m"
        reset_color_code = "\033[0m"
        for key, value in litellm_settings.items(): 
            if key == "cache":
                print(f"{blue_color_code}\nSetting Cache on Proxy")
                from litellm.caching import Cache
                cache_type = value["type"]
                cache_host = os.environ.get("REDIS_HOST")
                cache_port = os.environ.get("REDIS_PORT")
                cache_password = os.environ.get("REDIS_PASSWORD")

                # Assuming cache_type, cache_host, cache_port, and cache_password are strings
                print(f"{blue_color_code}Cache Type:{reset_color_code} {cache_type}")
                print(f"{blue_color_code}Cache Host:{reset_color_code} {cache_host}")
                print(f"{blue_color_code}Cache Port:{reset_color_code} {cache_port}")
                print(f"{blue_color_code}Cache Password:{reset_color_code} {cache_password}")
                print()

                litellm.cache = Cache(
                    type=cache_type,
                    host=cache_host,
                    port=cache_port,
                    password=cache_password
                )
            else:
                setattr(litellm, key, value)
                
    ## MODEL LIST
    model_list = config.get('model_list', None)
    if model_list:
        router = litellm.Router(model_list=model_list, num_retries=3)
        print(f"\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m")
        for model in model_list:
            print(f"\033[32m    {model.get('model_name', '')}\033[0m")
            litellm_model_name = model["litellm_params"]["model"]
            if "ollama" in litellm_model_name: 
                run_ollama_serve()
    return router, model_list, general_settings

async def generate_key_helper_fn(duration_str: Optional[str], models: list, aliases: dict, config: dict, spend: float, token: Optional[str]=None, user_id: Optional[str]=None):
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
        
    if duration_str is None: # allow tokens that never expire 
        expires = None
    else: 
        duration = _duration_in_seconds(duration=duration_str)
        expires = datetime.utcnow() + timedelta(seconds=duration)
    
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
            "user_id": user_id
        }
        new_verification_token = await prisma_client.insert_data(data=verification_token_data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return {"token": new_verification_token.token, "expires": new_verification_token.expires, "user_id": user_id}

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
    model,
    alias,
    api_base,
    api_version,
    debug,
    temperature,
    max_tokens,
    request_timeout,
    max_budget,
    telemetry,
    drop_params,
    add_function_to_prompt,
    headers,
    save,
    config, 
    use_queue
):
    global user_model, user_api_base, user_debug, user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers, experimental, llm_model_list, llm_router, general_settings
    generate_feedback_box()
    user_model = model
    user_debug = debug
    dynamic_config = {"general": {}, user_model: {}}
    if config:
        llm_router, llm_model_list, general_settings = load_router_config(router=llm_router, config_file_path=config)
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
    if debug==True:  # litellm-specific param
        litellm.set_verbose = True
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


async def async_data_generator(response):
    print_verbose("inside generator")
    async for chunk in response:
        print_verbose(f"returned chunk: {chunk}")
        try:
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"

def litellm_completion(*args, **kwargs):
    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
    call_type = kwargs.pop("call_type")
    # override with user settings, these are params passed via cli
    if user_temperature: 
        kwargs["temperature"] = user_temperature
    if user_request_timeout:
        kwargs["request_timeout"] = user_request_timeout
    if user_max_tokens: 
        kwargs["max_tokens"] = user_max_tokens
    if user_api_base: 
        kwargs["api_base"] = user_api_base
    ## ROUTE TO CORRECT ENDPOINT ## 
    router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
    try:
        if llm_router is not None and kwargs["model"] in router_model_names: # model in router model list 
            if call_type == "chat_completion":
                response = llm_router.completion(*args, **kwargs)
            elif call_type == "text_completion":
                response = llm_router.text_completion(*args, **kwargs)
        else: 
            if call_type == "chat_completion":
                response = litellm.completion(*args, **kwargs)
            elif call_type == "text_completion":
                response = litellm.text_completion(*args, **kwargs)
    except Exception as e:
        raise e
    if 'stream' in kwargs and kwargs['stream'] == True: # use generate_responses to stream responses
        return StreamingResponse(data_generator(response), media_type='text/event-stream')
    return response

@app.on_event("startup")
async def startup_event():
    global prisma_client, master_key
    import json
    worker_config = json.loads(os.getenv("WORKER_CONFIG"))
    initialize(**worker_config)
    if prisma_client: 
        await prisma_client.connect()
    
    if prisma_client is not None and master_key is not None: 
        # add master key to db
        await generate_key_helper_fn(duration_str=None, models=[], aliases={}, config={}, spend=0, token=master_key)

@app.on_event("shutdown")
async def shutdown_event():
    global prisma_client
    if prisma_client:
        print("Disconnecting from Prisma")
        await prisma_client.disconnect()

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
async def completion(request: Request, model: Optional[str] = None, user_api_key_dict: dict = Depends(user_api_key_auth)):
    try: 
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except: 
            data = json.loads(body_str)
        
        data["user"] = user_api_key_dict.get("user_id", None)
        data["model"] = (
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        data["call_type"] = "text_completion"
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict["api_key"]
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict["api_key"]}

        return litellm_completion(
            **data
        )
    except Exception as e: 
        print(f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`")
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
async def chat_completion(request: Request, model: Optional[str] = None, user_api_key_dict: dict = Depends(user_api_key_auth), background_tasks: BackgroundTasks = BackgroundTasks()):
    global general_settings, user_debug
    try: 
        data = {}
        data = await request.json() # type: ignore 

        print_verbose(f"receiving data: {data}")
        data["model"] = (
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )

        data["user"] = user_api_key_dict.get("user_id", None)

        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict["api_key"]
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict["api_key"]}

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
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data["model"] in router_model_names: # model in router model list 
                response = await llm_router.acompletion(**data)
        else: 
            response = await litellm.acompletion(**data)
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(async_data_generator(response), media_type='text/event-stream')
        background_tasks.add_task(log_input_output, request, response) # background task for logging to OTEL 
        return response
    except Exception as e: 
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
async def embeddings(request: Request, user_api_key_dict: dict = Depends(user_api_key_auth), background_tasks: BackgroundTasks = BackgroundTasks()): 
    try: 

        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
        data = orjson.loads(body)

        data["user"] = user_api_key_dict.get("user_id", None)
        data["model"] = (
            general_settings.get("embedding_model", None) # server default
            or user_model # model name passed via cli args
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict["api_key"]
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict["api_key"]}

        ## ROUTE TO CORRECT ENDPOINT ##
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data["model"] in router_model_names: # model in router model list 
            response = await llm_router.aembedding(**data)
        else:
            response = await litellm.aembedding(**data)
        background_tasks.add_task(log_input_output, request, response) # background task for logging to OTEL 
        return response
    except Exception as e:
        traceback.print_exc()
        raise e
    except Exception as e: 
        pass

#### KEY MANAGEMENT #### 

@router.post("/key/generate", tags=["key management"], dependencies=[Depends(user_api_key_auth)], response_model=GenerateKeyResponse)
async def generate_key_fn(request: Request, data: GenerateKeyRequest): 
    """
    Generate an API key based on the provided data. 

    Docs: https://docs.litellm.ai/docs/proxy/virtual_keys

    Parameters:
    - duration: Optional[str] - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d"). **(Default is set to 1 hour.)**
    - models: Optional[list] - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
    - aliases: Optional[dict] - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
    - config: Optional[dict] - any key-specific configs, overrides config in config.yaml 
    - spend: Optional[int] - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend

    Returns:
    - key: The generated api key 
    - expires: Datetime object for when key expires. 
    """
    data = await request.json()

    duration_str = data.duration  # Default to 1 hour if duration is not provided
    models = data.models # Default to an empty list (meaning allow token to call all models)
    aliases = data.aliases # Default to an empty dict (no alias mappings, on top of anything in the config.yaml model_list)
    config = data.config
    spend = data.spend
    user_id = data.user_id
    if isinstance(models, list):
        response = await generate_key_helper_fn(duration_str=duration_str, models=models, aliases=aliases, config=config, spend=spend, user_id=user_id)
        return GenerateKeyResponse(key=response["token"], expires=response["expires"], user_id=response["user_id"])
    else: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "models param must be a list"},
        )

@router.post("/key/delete", tags=["key management"], dependencies=[Depends(user_api_key_auth)])
async def delete_key_fn(request: Request, data: DeleteKeyRequest): 
    try: 
        data = await request.json()

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
        # Load existing config
        if os.path.exists(f"{user_config_file_path}"):
            with open(f"{user_config_file_path}", "r") as config_file:
                config = yaml.safe_load(config_file)
        else: 
            config = {"model_list": []} 
        # Add the new model to the config
        config['model_list'].append({
            'model_name': model_params.model_name,
            'litellm_params': model_params.litellm_params,
            'model_info': model_params.model_info
        })

        # Save the updated config
        with open(f"{user_config_file_path}", "w") as config_file:
            yaml.dump(config, config_file, default_flow_style=False)

        # update Router 
        llm_router, llm_model_list, general_settings = load_router_config(router=llm_router, config_file_path=config)


        return {"message": "Model added successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

#### [BETA] - This is a beta endpoint, format might change based on user feedback. - https://github.com/BerriAI/litellm/issues/933
@router.get("/model/info", description="Provides more info about each model in /models, including config.yaml descriptions (except api key and api base)", tags=["model management"], dependencies=[Depends(user_api_key_auth)])
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

@router.get("/test")
async def test_endpoint(request: Request): 
    return {"route": request.url.path}

@router.get("/health", description="Check the health of all the endpoints in config.yaml", tags=["health"])
async def health_endpoint(request: Request, model: Optional[str] = fastapi.Query(None, description="Specify the model name (optional)")):
    global llm_model_list
    healthy_endpoints = []
    unhealthy_endpoints = []
    if llm_model_list: 
        for model_name in llm_model_list: 
            try: 
                if model is None or model == model_name["litellm_params"]["model"]: # if model specified, just call that one. 
                    litellm_params = model_name["litellm_params"]
                    model_name = litellm.utils.remove_model_id(litellm_params["model"]) # removes, ids set by litellm.router
                    if model_name not in litellm.all_embedding_models: # filter out embedding models
                        litellm_params["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
                        litellm_params["model"] = model_name
                        litellm.completion(**litellm_params)
                        cleaned_params = {}
                        for key in litellm_params:
                            if key != "api_key" and key != "messages":
                                cleaned_params[key] = litellm_params[key]
                        healthy_endpoints.append(cleaned_params)
            except Exception as e:
                print("Got Exception", e) 
                cleaned_params = {}
                for key in litellm_params:
                    if key != "api_key" and key != "messages":
                        cleaned_params[key] = litellm_params[key]
                unhealthy_endpoints.append(cleaned_params)
                pass
    return {
        "healthy_endpoints": healthy_endpoints,
        "unhealthy_endpoints": unhealthy_endpoints
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


app.include_router(router)
