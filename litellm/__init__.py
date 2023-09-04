import threading, requests
from typing import Callable, List, Optional, Dict
from litellm.caching import Cache

input_callback: List[str] = []
success_callback: List[str] = []
failure_callback: List[str] = []
set_verbose = False
email: Optional[
    str
] = None  # for hosted dashboard. Learn more - https://docs.litellm.ai/docs/debugging/hosted_debugging
token: Optional[
    str
] = None  # for hosted dashboard. Learn more - https://docs.litellm.ai/docs/debugging/hosted_debugging
telemetry = True
max_tokens = 256  # OpenAI Defaults
retry = True
api_key: Optional[str] = None
openai_key: Optional[str] = None
azure_key: Optional[str] = None
anthropic_key: Optional[str] = None
replicate_key: Optional[str] = None
cohere_key: Optional[str] = None
ai21_key: Optional[str] = None
openrouter_key: Optional[str] = None
huggingface_key: Optional[str] = None
vertex_project: Optional[str] = None
vertex_location: Optional[str] = None
togetherai_api_key: Optional[str] = None
baseten_key: Optional[str] = None
aleph_alpha_key: Optional[str] = None
use_client = False
logging = True
caching = False # deprecated son
caching_with_models = False  # if you want the caching key to be model + prompt # deprecated soon
cache: Optional[Cache] = None # cache object
model_alias_map: Dict[str, str] = {}
def get_model_cost_map():
    url = "https://raw.githubusercontent.com/BerriAI/litellm/main/cookbook/community-resources/max_tokens.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if request is unsuccessful
        content = response.json()
        return content
    except requests.exceptions.RequestException as e:
        print("Error occurred:", e)
        return None
model_cost = get_model_cost_map()

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
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    #################
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
]
open_ai_text_completion_models = ["text-davinci-003", "curie-001", "babbage-001", "ada-001", "babbage-002", "davinci-002"]

cohere_models = [
    "command-nightly",
    "command",
    "command-light",
    "command-medium-beta",
    "command-xlarge-beta",
]

anthropic_models = ["claude-2", "claude-instant-1", "claude-instant-1.2"]

replicate_models = [
    "replicate/",
    "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    "a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52",
    "joehoover/instructblip-vicuna13b:c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe",
    "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
    "a16z-infra/llama-2-7b-chat:7b0bfc9aff140d5b75bacbed23e91fd3c34b01a1e958d32132de6e0a19796e2c",
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    "daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f",
    "replit/replit-code-v1-3b:b84f4c074b807211cd75e3e8b1589b6399052125b4c27106e43d47189e8415ad",
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

together_ai_models = [
    "togethercomputer/llama-2-70b-chat",
    "togethercomputer/Llama-2-7B-32K-Instruct",
    "togethercomputer/llama-2-7b",
]

aleph_alpha_models = [
    "luminous-base",
    "luminous-base-control",
    "luminous-extended",
    "luminous-extended-control",
    "luminous-supreme",
    "luminous-supreme-control"
]

baseten_models = ["qvv0xeq", "q841o8w", "31dxrj3"]  # FALCON 7B  # WizardLM  # Mosaic ML

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
    + together_ai_models
    + baseten_models
    + aleph_alpha_models
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
    "baseten",
    "azure",
    "sagemaker",
    "bedrock",
]

models_by_provider = {
    "openai": open_ai_chat_completion_models + open_ai_text_completion_models,
    "cohere": cohere_models,
    "anthropic": anthropic_models,
    "replicate": replicate_models,
    "huggingface": huggingface_models,
    "together_ai": together_ai_models,
    "baseten": baseten_models,
    "openrouter": openrouter_models,
    "vertex_ai": vertex_chat_models + vertex_text_models,
    "ai21": ai21_models,
}

####### EMBEDDING MODELS ###################
open_ai_embedding_models = ["text-embedding-ada-002"]

from .timeout import timeout
from .testing import *
from .utils import (
    client,
    exception_type,
    get_optional_params,
    modify_integration,
    token_counter,
    cost_per_token,
    completion_cost,
    get_litellm_params,
    Logging,
    acreate,
    get_model_list,
    completion_with_split_tests,
    get_max_tokens
)
from .main import *  # type: ignore
from .integrations import *
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
    ContextWindowExceededError
)
