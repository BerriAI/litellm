import threading
from typing import Callable, List, Optional

success_callback: List[str] = []
failure_callback: List[str] = []
set_verbose = False
telemetry = True
max_tokens = 256  # OpenAI Defaults
retry = True
api_key: Optional[str] = None
openai_key: Optional[str] = None
azure_key: Optional[str] = None
anthropic_key: Optional[str] = None
replicate_key: Optional[str] = None
cohere_key: Optional[str] = None
openrouter_key: Optional[str] = None
huggingface_key: Optional[str] = None
vertex_project: Optional[str] = None
vertex_location: Optional[str] = None
hugging_api_token: Optional[str] = None
togetherai_api_key: Optional[str] = None
caching = False
caching_with_models = False # if you want the caching key to be model + prompt
model_cost = {
    "gpt-3.5-turbo": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-35-turbo": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },  # azure model name
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k": {
        "max_tokens": 16000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
    },
    "gpt-35-turbo-16k": {
        "max_tokens": 16000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
    },  # azure model name
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
    },
    "gpt-4": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-0613": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-32k": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
    },
    "claude-instant-1": {
        "max_tokens": 100000,
        "input_cost_per_token": 0.00000163,
        "output_cost_per_token": 0.00000551,
    },
    "claude-2": {
        "max_tokens": 100000,
        "input_cost_per_token": 0.00001102,
        "output_cost_per_token": 0.00003268,
    },
    "text-bison-001": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.000004,
    },
    "chat-bison-001": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000002,
    },
    "command-nightly": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
    },
}


####### THREAD-SPECIFIC DATA ###################
class MyLocal(threading.local):
    def __init__(self):
        self.user = "Hello World"


_thread_context = MyLocal()


def identify(event_details):
    # Store user in thread local data
    if "user" in event_details:
        _thread_context.user = event_details["user"]


####### ADDITIONAL PARAMS ################### configurable params if you use proxy models like Helicone, map spend to org id, etc.
api_base = None
headers = None
api_version = None
organization = None
config_path = None
####### Secret Manager #####################
secret_manager_client = None
####### COMPLETION MODELS ###################
open_ai_chat_completion_models = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    #################
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]
open_ai_text_completion_models = ["text-davinci-003"]

cohere_models = [
    "command-nightly",
    "command",
    "command-light",
    "command-medium-beta",
    "command-xlarge-beta",
]

anthropic_models = ["claude-2", "claude-instant-1", "claude-instant-1.2"]

replicate_models = [
    "replicate/"
]  # placeholder, to make sure we accept any replicate model in our model_list

openrouter_models = [
    "google/palm-2-codechat-bison",
    "google/palm-2-chat-bison",
    "openai/gpt-3.5-turbo",
    "openai/gpt-3.5-turbo-16k",
    "openai/gpt-4-32k",
    "anthropic/claude-2",
    "anthropic/claude-instant-v1",
    "meta-llama/llama-2-13b-chat",
    "meta-llama/llama-2-70b-chat",
]

vertex_chat_models = ["chat-bison", "chat-bison@001"]


vertex_text_models = ["text-bison", "text-bison@001"]

huggingface_models = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-2-7b-chat",
    "meta-llama/Llama-2-13b",
    "meta-llama/Llama-2-13b-chat",
    "meta-llama/Llama-2-70b",
    "meta-llama/Llama-2-70b-chat",
]  # these have been tested on extensively. But by default all text2text-generation and text-generation models are supported by liteLLM. - https://docs.litellm.ai/docs/completion/supported

ai21_models = ["j2-ultra", "j2-mid", "j2-light"]

model_list = (
    open_ai_chat_completion_models
    + open_ai_text_completion_models
    + cohere_models
    + anthropic_models
    + replicate_models
    + openrouter_models
    + huggingface_models
    + vertex_chat_models
    + vertex_text_models
    + ai21_models
)

provider_list = [
    "openai",
    "cohere",
    "anthropic",
    "replicate",
    "huggingface",
    "together_ai",
    "openrouter",
    "vertex_ai",
    "ai21",
]
####### EMBEDDING MODELS ###################
open_ai_embedding_models = ["text-embedding-ada-002"]

from .timeout import timeout
from .testing import *
from .utils import (
    client,
    logging,
    exception_type,
    get_optional_params,
    modify_integration,
    token_counter,
    cost_per_token,
    completion_cost,
    get_litellm_params,
)
from .main import *  # type: ignore
from .integrations import *
from openai.error import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
)
