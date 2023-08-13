import threading
success_callback = []
failure_callback = []
set_verbose=False
telemetry=True
max_tokens = 256 # OpenAI Defaults
retry = True
openai_key = None 
azure_key = None 
anthropic_key = None 
replicate_key = None 
cohere_key = None 
openrouter_key = None
vertex_project = None
vertex_location = None

hugging_api_token = None
model_cost = {
    "gpt-3.5-turbo": {"max_tokens": 4000, "input_cost_per_token": 0.0000015, "output_cost_per_token": 0.000002},
    "gpt-35-turbo": {"max_tokens": 4000, "input_cost_per_token": 0.0000015, "output_cost_per_token": 0.000002}, # azure model name
    "gpt-3.5-turbo-0613": {"max_tokens": 4000, "input_cost_per_token": 0.0000015, "output_cost_per_token": 0.000002},
    "gpt-3.5-turbo-0301": {"max_tokens": 4000, "input_cost_per_token": 0.0000015, "output_cost_per_token": 0.000002},
    "gpt-3.5-turbo-16k": {"max_tokens": 16000, "input_cost_per_token": 0.000003, "output_cost_per_token": 0.000004},
    "gpt-35-turbo-16k": {"max_tokens": 16000, "input_cost_per_token": 0.000003, "output_cost_per_token": 0.000004}, # azure model name
    "gpt-3.5-turbo-16k-0613": {"max_tokens": 16000, "input_cost_per_token": 0.000003, "output_cost_per_token": 0.000004},
    "gpt-4": {"max_tokens": 8000, "input_cost_per_token": 0.000003, "output_cost_per_token": 0.00006},
    "gpt-4-0613": {"max_tokens": 8000, "input_cost_per_token": 0.000003, "output_cost_per_token": 0.00006},
    "gpt-4-32k": {"max_tokens": 8000, "input_cost_per_token": 0.00006, "output_cost_per_token": 0.00012},
    "claude-instant-1": {"max_tokens": 100000, "input_cost_per_token": 0.00000163, "output_cost_per_token": 0.00000551},
    "claude-2": {"max_tokens": 100000, "input_cost_per_token": 0.00001102, "output_cost_per_token": 0.00003268},
    "text-bison-001": {"max_tokens": 8192, "input_cost_per_token": 0.000004, "output_cost_per_token": 0.000004},
    "chat-bison-001": {"max_tokens": 4096, "input_cost_per_token": 0.000002, "output_cost_per_token": 0.000002},
    "command-nightly": {"max_tokens": 4096, "input_cost_per_token": 0.000015, "output_cost_per_token": 0.000015},
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
  'gpt-3.5-turbo', 
  'gpt-3.5-turbo-16k-0613',
  'gpt-3.5-turbo-16k'
]
open_ai_text_completion_models = [
    'text-davinci-003'
]

cohere_models = [
    'command-nightly',
    "command", 
    "command-light", 
    "command-medium-beta", 
    "command-xlarge-beta"
]

anthropic_models = [
  "claude-2", 
  "claude-instant-1",
  "claude-instant-1.2"
]

replicate_models = [
    "replicate/"
] # placeholder, to make sure we accept any replicate model in our model_list 

openrouter_models = [
    'google/palm-2-codechat-bison',
    'google/palm-2-chat-bison',
    'openai/gpt-3.5-turbo',
    'openai/gpt-3.5-turbo-16k',
    'openai/gpt-4-32k',
    'anthropic/claude-2',
    'anthropic/claude-instant-v1',
    'meta-llama/llama-2-13b-chat',
    'meta-llama/llama-2-70b-chat'
]

vertex_models = [
    "chat-bison",
    "chat-bison@001"
]

model_list = open_ai_chat_completion_models + open_ai_text_completion_models + cohere_models + anthropic_models + replicate_models + openrouter_models + vertex_models

####### EMBEDDING MODELS ###################
open_ai_embedding_models = [
    'text-embedding-ada-002'
]

from .timeout import timeout
from .utils import client, logging, exception_type, get_optional_params, modify_integration, token_counter, cost_per_token, completion_cost, load_test_model, get_litellm_params
from .main import *  # Import all the symbols from main.py
from .integrations import *
from openai.error import AuthenticationError, InvalidRequestError, RateLimitError, ServiceUnavailableError, OpenAIError