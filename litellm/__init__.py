success_callback = []
failure_callback = []
set_verbose=False
telemetry=True
max_tokens = 256 # OpenAI Defaults
retry = True # control tenacity retries. 
openai_key = None 
azure_key = None 
anthropic_key = None 
replicate_key = None 
cohere_key = None 
MAX_TOKENS = {
    'gpt-3.5-turbo': 4000,
    'gpt-3.5-turbo-0613': 4000,
    'gpt-3.5-turbo-0301': 4000,
    'gpt-3.5-turbo-16k': 16000,
    'gpt-3.5-turbo-16k-0613': 16000,
    'gpt-4': 8000,
    'gpt-4-0613': 8000,
    'gpt-4-32k': 32000,
    'claude-instant-1': 100000,
    'claude-2': 100000,
    'command-nightly': 4096,
    'replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1': 4096,
}
####### PROXY PARAMS ################### configurable params if you use proxy models like Helicone
api_base = None
headers = None
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
from .utils import client, logging, exception_type  # Import all the symbols from main.py
from .main import *  # Import all the symbols from main.py
from .integrations import *