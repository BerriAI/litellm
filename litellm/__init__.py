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
  "claude-instant-1"
]

replicate_models = [
    "replicate/"
] # placeholder, to make sure we accept any replicate model in our model_list 

model_list = open_ai_chat_completion_models + open_ai_text_completion_models + cohere_models + anthropic_models + replicate_models

####### EMBEDDING MODELS ###################
open_ai_embedding_models = [
    'text-embedding-ada-002'
]
from .timeout import timeout
from .utils import client, logging, exception_type, get_optional_params, modify_integration
from .main import *  # Import all the symbols from main.py
from .integrations import *
from openai.error import AuthenticationError, InvalidRequestError, RateLimitError, ServiceUnavailableError, OpenAIError