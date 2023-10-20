#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	},
	"tpm": 1000000,
	"rpm": 9000
}]

router = Router(model_list=model_list, redis_host=os.getenv("REDIS_HOST"), redis_password=os.getenv("REDIS_PASSWORD"), redis_port=int(os.getenv("REDIS_PORT"))) # type: ignore

completions = [] 
with ThreadPoolExecutor(max_workers=100) as executor:
	kwargs = {
		"model": "gpt-3.5-turbo",
		"messages": [{"role": "user", "content": "Hey, how's it going?"}]
	}
	for _ in range(20):
		future = executor.submit(router.completion, **kwargs) # type: ignore
		completions.append(future)

# Retrieve the results from the futures
results = [future.result() for future in completions]

print(results)