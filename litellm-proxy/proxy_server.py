import sys, os, platform, time, copy
import threading
import shutil, random, traceback

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
except ImportError:
    import subprocess
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
        ]
    )
    import uvicorn
    import fastapi
    import tomli as tomllib
    import appdirs
    import tomli_w

try:
    from .llm import litellm_completion
except ImportError as e:
    from llm import litellm_completion  # type: ignore

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


generate_feedback_box()

print()
print(
    "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"
)
print()
print("\033[1;34mDocs: https://docs.litellm.ai/docs/proxy_server\033[0m")
print()

import litellm
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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
model_router = litellm.Router()
config_filename = "litellm.secrets.toml"
config_dir = os.getcwd()
config_dir = appdirs.user_config_dir("litellm")
user_config_path = os.getenv(
    "LITELLM_CONFIG_PATH", os.path.join(config_dir, config_filename)
)
log_file = "api_log.json"


#### HELPER FUNCTIONS ####
def print_verbose(print_statement):
    global user_debug
    if user_debug:
        print(print_statement)


def find_avatar_url(role):
    role = role.replace(" ", "%20")
    avatar_filename = f"avatars/{role}.png"
    avatar_url = f"/static/{avatar_filename}"
    return avatar_url


def usage_telemetry(
    feature: str,
):  # helps us know if people are using this feature. Set `litellm --telemetry False` to your cli call to turn this off
    if user_telemetry:
        data = {"feature": feature}  # "local_proxy_server"
        threading.Thread(
            target=litellm.utils.litellm_telemetry, args=(data,), daemon=True
        ).start()


def add_keys_to_config(key, value):
    # Check if file exists
    if os.path.exists(user_config_path):
        # Load existing file
        with open(user_config_path, "rb") as f:
            config = tomllib.load(f)
    else:
        # File doesn't exist, create empty config
        config = {}

    # Add new key
    config.setdefault("keys", {})[key] = value

    # Write config to file
    with open(user_config_path, "wb") as f:
        tomli_w.dump(config, f)


def save_params_to_config(data: dict):
    # Check if file exists
    if os.path.exists(user_config_path):
        # Load existing file
        with open(user_config_path, "rb") as f:
            config = tomllib.load(f)
    else:
        # File doesn't exist, create empty config
        config = {}

    config.setdefault("general", {})

    ## general config
    general_settings = data["general"]

    for key, value in general_settings.items():
        config["general"][key] = value

    ## model-specific config
    config.setdefault("model", {})
    config["model"].setdefault(user_model, {})

    user_model_config = data[user_model]
    model_key = model_key = user_model_config.pop("alias", user_model)
    config["model"].setdefault(model_key, {})
    for key, value in user_model_config.items():
        config["model"][model_key][key] = value

    # Write config to file
    with open(user_config_path, "wb") as f:
        tomli_w.dump(config, f)


def load_config():
    try:
        global user_config, user_api_base, user_max_tokens, user_temperature, user_model, local_logging
        # As the .env file is typically much simpler in structure, we use load_dotenv here directly
        with open(user_config_path, "rb") as f:
            user_config = tomllib.load(f)

        ## load keys
        if "keys" in user_config:
            for key in user_config["keys"]:
                os.environ[key] = user_config["keys"][
                    key
                ]  # litellm can read keys from the environment
        ## settings
        if "general" in user_config:
            litellm.add_function_to_prompt = user_config["general"].get(
                "add_function_to_prompt", True
            )  # by default add function to prompt if unsupported by provider
            litellm.drop_params = user_config["general"].get(
                "drop_params", True
            )  # by default drop params if unsupported by provider
            litellm.model_fallbacks = user_config["general"].get(
                "fallbacks", None
            )  # fallback models in case initial completion call fails
            default_model = user_config["general"].get(
                "default_model", None
            )  # route all requests to this model.

            local_logging = user_config["general"].get("local_logging", True)

            if user_model is None:  # `litellm --model <model-name>`` > default_model.
                user_model = default_model

        ## load model config - to set this run `litellm --config`
        model_config = None
        if "model" in user_config:
            if user_model in user_config["model"]:
                model_config = user_config["model"][user_model]
            model_list = []
            for model in user_config["model"]:
                if "model_list" in user_config["model"][model]:
                    model_list.extend(user_config["model"][model]["model_list"])
            if len(model_list) > 0:
                model_router.set_model_list(model_list=model_list)

        print_verbose(f"user_config: {user_config}")
        print_verbose(f"model_config: {model_config}")
        print_verbose(f"user_model: {user_model}")
        if model_config is None:
            return

        user_max_tokens = model_config.get("max_tokens", None)
        user_temperature = model_config.get("temperature", None)
        user_api_base = model_config.get("api_base", None)

        ## custom prompt template
        if "prompt_template" in model_config:
            model_prompt_template = model_config["prompt_template"]
            if (
                len(model_prompt_template.keys()) > 0
            ):  # if user has initialized this at all
                litellm.register_prompt_template(
                    model=user_model,
                    initial_prompt_value=model_prompt_template.get(
                        "MODEL_PRE_PROMPT", ""
                    ),
                    roles={
                        "system": {
                            "pre_message": model_prompt_template.get(
                                "MODEL_SYSTEM_MESSAGE_START_TOKEN", ""
                            ),
                            "post_message": model_prompt_template.get(
                                "MODEL_SYSTEM_MESSAGE_END_TOKEN", ""
                            ),
                        },
                        "user": {
                            "pre_message": model_prompt_template.get(
                                "MODEL_USER_MESSAGE_START_TOKEN", ""
                            ),
                            "post_message": model_prompt_template.get(
                                "MODEL_USER_MESSAGE_END_TOKEN", ""
                            ),
                        },
                        "assistant": {
                            "pre_message": model_prompt_template.get(
                                "MODEL_ASSISTANT_MESSAGE_START_TOKEN", ""
                            ),
                            "post_message": model_prompt_template.get(
                                "MODEL_ASSISTANT_MESSAGE_END_TOKEN", ""
                            ),
                        },
                    },
                    final_prompt_value=model_prompt_template.get(
                        "MODEL_POST_PROMPT", ""
                    ),
                )
    except:
        pass


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
):
    global user_model, user_api_base, user_debug, user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers
    user_model = model
    user_debug = debug
    load_config()
    dynamic_config = {"general": {}, user_model: {}}
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
    if debug:  # litellm-specific param
        litellm.set_verbose = True
    if save:
        save_params_to_config(dynamic_config)
        with open(user_config_path) as f:
            print(f.read())
        print("\033[1;32mDone successfully\033[0m")
    user_telemetry = telemetry
    usage_telemetry(feature="local_proxy_server")


def track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    # track cost like this
    # {
    #     "Oct12": {
    #         "gpt-4": 10,
    #         "claude-2": 12.01,
    #     },
    #     "Oct 15": {
    #         "ollama/llama2": 0.0,
    #         "gpt2": 1.2
    #     }
    # }
    try:
        # for streaming responses
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost
            completion_response = kwargs["complete_streaming_response"]
            input_text = kwargs["messages"]
            output_text = completion_response["choices"][0]["message"]["content"]
            response_cost = litellm.completion_cost(
                model=kwargs["model"], messages=input_text, completion=output_text
            )
            model = kwargs["model"]

        # for non streaming responses
        else:
            # we pass the completion_response obj
            if kwargs["stream"] != True:
                response_cost = litellm.completion_cost(
                    completion_response=completion_response
                )
                model = completion_response["model"]

        # read/write from json for storing daily model costs
        cost_data = {}
        try:
            with open("costs.json") as f:
                cost_data = json.load(f)
        except FileNotFoundError:
            cost_data = {}
        import datetime

        date = datetime.datetime.now().strftime("%b-%d-%Y")
        if date not in cost_data:
            cost_data[date] = {}

        if kwargs["model"] in cost_data[date]:
            cost_data[date][kwargs["model"]]["cost"] += response_cost
            cost_data[date][kwargs["model"]]["num_requests"] += 1
        else:
            cost_data[date][kwargs["model"]] = {
                "cost": response_cost,
                "num_requests": 1,
            }

        with open("costs.json", "w") as f:
            json.dump(cost_data, f, indent=2)

    except:
        pass


def logger(
    kwargs,  # kwargs to completion
    completion_response=None,  # response from completion
    start_time=None,
    end_time=None,  # start/end time
):
    log_event_type = kwargs["log_event_type"]
    try:
        if log_event_type == "pre_api_call":
            inference_params = copy.deepcopy(kwargs)
            timestamp = inference_params.pop("start_time")
            dt_key = timestamp.strftime("%Y%m%d%H%M%S%f")[:23]
            log_data = {dt_key: {"pre_api_call": inference_params}}

            try:
                with open(log_file, "r") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {}

            existing_data.update(log_data)

            def write_to_log():
                with open(log_file, "w") as f:
                    json.dump(existing_data, f, indent=2)

            thread = threading.Thread(target=write_to_log, daemon=True)
            thread.start()
    except:
        pass


litellm.input_callback = [logger]
litellm.success_callback = [logger]
litellm.failure_callback = [logger]


#### API ENDPOINTS ####
@router.post("/v1/models")
@router.get("/models")  # if project requires model list
def model_list():
    if user_model != None:
        return dict(
            data=[
                {
                    "id": user_model,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "openai",
                }
            ],
            object="list",
        )
    else:
        all_models = litellm.utils.get_valid_models()
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


@router.post("/v1/completions")
@router.post("/completions")
async def completion(request: Request):
    data = await request.json()
    return litellm_completion(data=data, type="completion", user_model=user_model, user_temperature=user_temperature,
                              user_max_tokens=user_max_tokens, user_api_base=user_api_base, user_headers=user_headers,
                              user_debug=user_debug, model_router=model_router, user_request_timeout=user_request_timeout)


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completion(request: Request):
    data = await request.json()
    print_verbose(f"data passed in: {data}")
    return litellm_completion(data, type="chat_completion", user_model=user_model,
                              user_temperature=user_temperature, user_max_tokens=user_max_tokens,
                              user_api_base=user_api_base, user_headers=user_headers, user_debug=user_debug, model_router=model_router, user_request_timeout=user_request_timeout)


def print_cost_logs():
    with open("costs.json", "r") as f:
        # print this in green
        print("\033[1;32m")
        print(f.read())
        print("\033[0m")
    return


@router.get("/ollama_logs")
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser("~/.ollama/logs/server.log")
    return FileResponse(filepath)


@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"


app.include_router(router)
