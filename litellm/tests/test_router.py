#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback, asyncio
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

def test_multiple_deployments(): 
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

### FUNCTION CALLING 

def test_function_calling(): 
	model_list = [
		{
			"model_name": "gpt-3.5-turbo-0613",
			"litellm_params": {
				"model": "gpt-3.5-turbo-0613",
				"api_key": os.getenv("OPENAI_API_KEY"),
			},
			"tpm": 100000,
			"rpm": 10000,
		},
	]

	messages = [
		{"role": "user", "content": "What is the weather like in Boston?"}
	]
	functions = [
		{
		"name": "get_current_weather",
		"description": "Get the current weather in a given location",
		"parameters": {
			"type": "object",
			"properties": {
			"location": {
				"type": "string",
				"description": "The city and state, e.g. San Francisco, CA"
			},
			"unit": {
				"type": "string",
				"enum": ["celsius", "fahrenheit"]
			}
			},
			"required": ["location"]
		}
		}
	]

	router = Router(model_list=model_list)
	response = router.completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
	print(response)

### FUNCTION CALLING -> NORMAL COMPLETION
def test_litellm_params_not_overwritten_by_function_calling():
	try:
		model_list = [
			{
				"model_name": "gpt-3.5-turbo-0613",
				"litellm_params": {
					"model": "gpt-3.5-turbo-0613",
					"api_key": os.getenv("OPENAI_API_KEY"),
				},
				"tpm": 100000,
				"rpm": 10000,
			},
		]

		messages = [
			{"role": "user", "content": "What is the weather like in Boston?"}
		]
		functions = [
			{
			"name": "get_current_weather",
			"description": "Get the current weather in a given location",
			"parameters": {
				"type": "object",
				"properties": {
				"location": {
					"type": "string",
					"description": "The city and state, e.g. San Francisco, CA"
				},
				"unit": {
					"type": "string",
					"enum": ["celsius", "fahrenheit"]
				}
				},
				"required": ["location"]
			}
			}
		]

		router = Router(model_list=model_list)
		_ = router.completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
		response = router.completion(model="gpt-3.5-turbo-0613", messages=messages)
		assert response.choices[0].finish_reason != "function_call"
	except Exception as e:
		pytest.fail(f"Error occurred: {e}")

# test_litellm_params_not_overwritten_by_function_calling()

def test_acompletion_on_router(): 
	try:
		model_list = [
			{
				"model_name": "gpt-3.5-turbo",
				"litellm_params": {
					"model": "gpt-3.5-turbo-0613",
					"api_key": os.getenv("OPENAI_API_KEY"),
				},
				"tpm": 100000,
				"rpm": 10000,
			},
		]

		messages = [
			{"role": "user", "content": "What is the weather like in Boston?"}
		]

		async def get_response(): 
			router = Router(model_list=model_list)
			response = await router.acompletion(model="gpt-3.5-turbo", messages=messages)
			return response
		response = asyncio.run(get_response())

		assert isinstance(response['choices'][0]['message']['content'], str)
	except Exception as e:
		traceback.print_exc()
		pytest.fail(f"Error occurred: {e}")


def test_aembedding_on_router():
	try:
		model_list = [
			{
				"model_name": "text-embedding-ada-002",
				"litellm_params": {
					"model": "text-embedding-ada-002",
				},
				"tpm": 100000,
				"rpm": 10000,
			},
		]

		async def embedding_call():
			router = Router(model_list=model_list)
			response = await router.aembedding(
				model="text-embedding-ada-002",
				input=["good morning from litellm", "this is another item"],
			)
			print(response)
		asyncio.run(embedding_call())
	except Exception as e:
		traceback.print_exc()
		pytest.fail(f"Error occurred: {e}")
