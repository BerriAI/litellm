import sys, os, platform, time, copy, re, asyncio
import threading, ast
import shutil, random, traceback, requests
from datetime import datetime, timedelta
from typing import Optional
import secrets, subprocess
messages: list = []
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path - for litellm local dev

try:
    import uvicorn
    import fastapi
    import tomli as tomllib
    import appdirs
    import tomli_w
    import backoff
    import yaml
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
            "tomli",
            "appdirs",
            "tomli-w",
            "backoff",
            "pyyaml"
        ]
    )
    import uvicorn
    import fastapi
    import tomli as tomllib
    import appdirs
    import tomli_w
    import backoff
    import yaml

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
    print("\033[1;34mDocs: https://docs.litellm.ai/docs/simple_proxy\033[0m\n")
    print(f"\033[32mLiteLLM: Test your local endpoint with: \"litellm --test\" [In a new terminal tab]\033[0m\n")
    print()

import litellm
litellm.suppress_debug_info = True
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.routing import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import json
import logging

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

user_api_base = None
user_model = None
user_debug = False
user_max_tokens = None
user_request_timeout = None
user_temperature = None
user_telemetry = True
user_config = None
user_headers = None
local_logging = True # writes logs to a local api_log.json file for debugging
config_filename = "litellm.secrets.toml"
config_dir = os.getcwd()
config_dir = appdirs.user_config_dir("litellm")
user_config_path = os.getenv(
    "LITELLM_CONFIG_PATH", os.path.join(config_dir, config_filename)
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
experimental = False
#### GLOBAL VARIABLES ####
llm_router: Optional[litellm.Router] = None
llm_model_list: Optional[list] = None
server_settings: dict = {}
log_file = "api_log.json"
worker_config = None
master_key = None
prisma_client = None
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

async def user_api_key_auth(request: Request): 
    global master_key, prisma_client
    if master_key is None:
        return 
    try: 
        api_key = await oauth2_scheme(request=request)
        if api_key == master_key: 
            return
        if prisma_client: 
            valid_token = await prisma_client.litellm_verificationtoken.find_first(
                where={
                    "token": api_key,
                    "expires": {"gte": datetime.utcnow()}  # Check if the token is not expired
                }
            )
            if valid_token:
                litellm.model_alias_map = valid_token.aliases
                if len(valid_token.models) == 0: # assume an empty model list means all models are allowed to be called
                    return
                else: 
                    data = await request.json()
                    model = data.get("model", None)
                    if model in litellm.model_alias_map:
                        model = litellm.model_alias_map[model]
                    if model and model not in valid_token.models:
                        raise Exception(f"Token not allowed to access model")
                return 
            else: 
                raise Exception(f"Invalid token")
    except Exception as e: 
        print(f"An exception occurred - {e}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "invalid user key"},
    )

def prisma_setup(database_url: Optional[str]): 
    global prisma_client
    if database_url: 
        import os 
        os.environ["DATABASE_URL"] = database_url
        subprocess.run(['pip', 'install', 'prisma'])
        subprocess.run(['python3', '-m', 'pip', 'install', 'prisma'])
        subprocess.run(['prisma', 'db', 'push'])
        # Now you can import the Prisma Client
        from prisma import Client
        prisma_client = Client()


def load_router_config(router: Optional[litellm.Router], config_file_path: str):
    global master_key
    config = {}
    server_settings: dict = {} 
    try: 
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            raise Exception(f"Path to config does not exist, 'os.path.exists({config_file_path})' returned False")
    except Exception as e:
        raise Exception(f"Exception while reading Config: {e}")
    
    print(f"Loaded config YAML:\n{json.dumps(config, indent=2)}")

    ## ENVIRONMENT VARIABLES
    environment_variables = config.get('environment_variables', None)
    if environment_variables: 
        for key, value in environment_variables.items(): 
            os.environ[key] = value

    ## GENERAL SERVER SETTINGS (e.g. master key,..)
    general_settings = config.get("general_settings", None)
    if general_settings: 
        ### MASTER KEY ###
        master_key = general_settings.get("master_key", None)
        ### CONNECT TO DATABASE ###
        database_url = general_settings.get("database_url", None)
        prisma_setup(database_url=database_url)
        

    ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
    litellm_settings = config.get('litellm_settings', None)
    if litellm_settings: 
        for key, value in litellm_settings.items(): 
            setattr(litellm, key, value)
    ## MODEL LIST
    model_list = config.get('model_list', None)
    if model_list:
        router = litellm.Router(model_list=model_list)
        print(f"\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m")
        for model in model_list:
            print(f"\033[32m    {model.get('model_name', '')}\033[0m")

    return router, model_list, server_settings

async def generate_key_helper_fn(duration_str: str, models: list, aliases: dict):
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

    duration = _duration_in_seconds(duration=duration_str)
    expires = datetime.utcnow() + timedelta(seconds=duration)
    aliases_json = json.dumps(aliases)
    try:
        db = prisma_client
        # Create a new verification token (you may want to enhance this logic based on your needs)
        verification_token_data = {
            "token": token, 
            "expires": expires,
            "models": models,
            "aliases": aliases_json
        }
        print(f"verification_token_data: {verification_token_data}")
        new_verification_token = await db.litellm_verificationtoken.create( # type: ignore
           {**verification_token_data} # type: ignore
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return {"token": new_verification_token.token, "expires": new_verification_token.expires}

async def generate_key_cli_task(duration_str):
    task = asyncio.create_task(generate_key_helper_fn(duration_str=duration_str))
    await task


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
):
    global user_model, user_api_base, user_debug, user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers, experimental, llm_model_list, llm_router, server_settings
    generate_feedback_box()
    user_model = model
    user_debug = debug
    dynamic_config = {"general": {}, user_model: {}}
    if config:
        llm_router, llm_model_list, server_settings = load_router_config(router=llm_router, config_file_path=config)
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
    if experimental: 
        pass
    if save:
        save_params_to_config(dynamic_config)
        with open(user_config_path) as f:
            print(f.read())
        print("\033[1;32mDone successfully\033[0m")
    user_telemetry = telemetry
    usage_telemetry(feature="local_proxy_server")

# for streaming
def data_generator(response):
    print_verbose("inside generator")
    for chunk in response:
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
    global prisma_client
    import json
    worker_config = json.loads(os.getenv("WORKER_CONFIG"))
    initialize(**worker_config)
    if prisma_client: 
        await prisma_client.connect()

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
    global llm_model_list, server_settings    
    all_models = []
    if server_settings.get("infer_model_from_keys", False):
        all_models = litellm.utils.get_valid_models()
    if llm_model_list: 
        all_models += llm_model_list
    if user_model is not None:
        all_models += user_model
    ### CHECK OLLAMA MODELS ### 
    try:
        response = requests.get("http://0.0.0.0:11434/api/tags")
        models = response.json()["models"]
        ollama_models = [m["name"].replace(":latest", "") for m in models]
        all_models.extend(ollama_models)
    except Exception as e: 
        traceback.print_exc()
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
async def completion(request: Request, model: Optional[str] = None):
    try: 
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except: 
            data = json.loads(body_str)
        data["model"] = (
            server_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        data["call_type"] = "text_completion"
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
            status = status.HTTP_500_INTERNAL_SERVER_ERROR,
        raise HTTPException(
            status_code=status,
            detail=error_msg
        )
                              

@router.post("/v1/chat/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/chat/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/openai/deployments/{model:path}/chat/completions", dependencies=[Depends(user_api_key_auth)]) # azure compatible endpoint
async def chat_completion(request: Request, model: Optional[str] = None):
    global server_settings
    try: 
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except: 
            data = json.loads(body_str)
        data["model"] = (
            server_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )
        data["call_type"] = "chat_completion"
        return litellm_completion(
            **data
        )
    except Exception as e: 
        print(f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`")
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        try:
            status = e.status_code # type: ignore
        except:
            status = status.HTTP_500_INTERNAL_SERVER_ERROR,
        raise HTTPException(
            status_code=status,
            detail=error_msg
        )

@router.post("/key/generate", dependencies=[Depends(user_api_key_auth)])
async def generate_key_fn(request: Request): 
    data = await request.json()

    duration_str = data.get("duration", "1h")  # Default to 1 hour if duration is not provided
    models = data.get("models", []) # Default to an empty list (meaning allow token to call all models)
    aliases = data.get("aliases", {}) # Default to an empty dict (no alias mappings, on top of anything in the config.yaml model_list)
    if isinstance(models, list):
        response = await generate_key_helper_fn(duration_str=duration_str, models=models, aliases=aliases)
        return {"key": response["token"], "expires": response["expires"]}
    else: 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "models param must be a list"},
        )


@router.get("/ollama_logs", dependencies=[Depends(user_api_key_auth)])
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser("~/.ollama/logs/server.log")
    return FileResponse(filepath)


@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"


app.include_router(router)
