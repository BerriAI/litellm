#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os, time
import traceback, asyncio
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
# import logging
# logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# def test_openai_only(): 
# 	from litellm import completion
# 	import time
# 	completions = []
# 	max_workers = 1000 # Adjust as needed
# 	start_time = time.time()
# 	print(f"Started test: {start_time}")
# 	with ThreadPoolExecutor(max_workers=max_workers) as executor:
# 		kwargs = {
# 			"model": "gpt-3.5-turbo",
# 			"messages": [{"role": "user", "content": """Context:

# In the historical era of Ancient Greece, a multitude of significant individuals lived, contributing immensely to various disciplines like science, politics, philosophy, and literature. For instance, Socrates, a renowned philosopher, primarily focused on ethics. His notable method, the Socratic Method, involved acknowledging one's own ignorance to stimulate critical thinking and illuminate ideas. His student, Plato, another prominent figure, founded the Academy in Athens. He proposed theories on justice, beauty, and equality, and also introduced the theory of forms, which is pivotal to understanding his philosophical insights. Another student of Socrates, Xenophon, distinguished himself more in the domain of history and military affairs.

# Aristotle, who studied under Plato, led an equally remarkable life. His extensive works have been influential across various domains, including science, logic, metaphysics, ethics, and politics. Perhaps most notably, a substantial portion of the Western intellectual tradition traces back to his writings. He later tutored Alexander the Great who went on to create one of the most vast empires in the world.

# In the domain of mathematics, Pythagoras and Euclid made significant contributions. Pythagoras is best known for the Pythagorean theorem, a fundamental principle in geometry, while Euclid, often regarded as the father of geometry, wrote "The Elements", a collection of definitions, axioms, theorems, and proofs. 

# Apart from these luminaries, the period also saw a number of influential political figures. Pericles, a prominent and influential Greek statesman, orator, and general of Athens during the Golden Age, specifically between the Persian and Peloponnesian wars, played a significant role in developing the Athenian democracy.

# The Ancient Greek era also witnessed extraordinary advancements in arts and literature. Homer, credited with the creation of the epic poems 'The Iliad' and 'The Odyssey,' is considered one of the greatest poets in history. The tragedies of Sophocles, Aeschylus, and Euripides left an indelible mark on the field of drama, and the comedies of Aristophanes remain influential even today.

# ---
# Question: 

# Who among the mentioned figures from Ancient Greece contributed to the domain of mathematics and what are their significant contributions?"""}],
# 		}
# 		for _ in range(10000):
# 			future = executor.submit(completion, **kwargs)
# 			completions.append(future)

# 	# Retrieve the results from the futures
# 	results = [future.result() for future in completions]
# 	end_time = time.time()

# 	print(f"Total Duration: {end_time-start_time}")

# test_openai_only()


# def test_multiple_deployments(): 
# 	import concurrent
# 	# litellm.set_verbose=True
# 	futures = {}
# 	model_list = [{ # list of model deployments 
# 		"model_name": "gpt-3.5-turbo", # openai model name 
# 		"litellm_params": { # params for litellm completion/embedding call 
# 			"model": "azure/chatgpt-v-2", 
# 			"api_key": os.getenv("AZURE_API_KEY"),
# 			"api_version": os.getenv("AZURE_API_VERSION"),
# 			"api_base": os.getenv("AZURE_API_BASE")
# 		},
# 		"tpm": 240000,
# 		"rpm": 1800
# 	}, {
# 		"model_name": "gpt-3.5-turbo", # openai model name 
# 		"litellm_params": { # params for litellm completion/embedding call 
# 			"model": "azure/chatgpt-functioncalling", 
# 			"api_key": os.getenv("AZURE_API_KEY"),
# 			"api_version": os.getenv("AZURE_API_VERSION"),
# 			"api_base": os.getenv("AZURE_API_BASE")
# 		},
# 		"tpm": 240000,
# 		"rpm": 1800
# 	}, {
# 		"model_name": "gpt-3.5-turbo", # openai model name 
# 		"litellm_params": { # params for litellm completion/embedding call 
# 			"model": "gpt-3.5-turbo", 
# 			"api_key": os.getenv("OPENAI_API_KEY"),
# 		},
# 		"tpm": 1000000,
# 		"rpm": 9000
# 	}]

# 	router = Router(model_list=model_list, redis_host=os.getenv("REDIS_HOST"), redis_password=os.getenv("REDIS_PASSWORD"), redis_port=int(os.getenv("REDIS_PORT"))) # type: ignore

# 	results = [] 
# 	with ThreadPoolExecutor(max_workers=10) as executor:
# 		kwargs = {
# 			"model": "gpt-3.5-turbo",
# 			"messages": [{"role": "user", "content": """Context:

# In the historical era of Ancient Greece, a multitude of significant individuals lived, contributing immensely to various disciplines like science, politics, philosophy, and literature. For instance, Socrates, a renowned philosopher, primarily focused on ethics. His notable method, the Socratic Method, involved acknowledging one's own ignorance to stimulate critical thinking and illuminate ideas. His student, Plato, another prominent figure, founded the Academy in Athens. He proposed theories on justice, beauty, and equality, and also introduced the theory of forms, which is pivotal to understanding his philosophical insights. Another student of Socrates, Xenophon, distinguished himself more in the domain of history and military affairs.

# Aristotle, who studied under Plato, led an equally remarkable life. His extensive works have been influential across various domains, including science, logic, metaphysics, ethics, and politics. Perhaps most notably, a substantial portion of the Western intellectual tradition traces back to his writings. He later tutored Alexander the Great who went on to create one of the most vast empires in the world.

# In the domain of mathematics, Pythagoras and Euclid made significant contributions. Pythagoras is best known for the Pythagorean theorem, a fundamental principle in geometry, while Euclid, often regarded as the father of geometry, wrote "The Elements", a collection of definitions, axioms, theorems, and proofs. 

# Apart from these luminaries, the period also saw a number of influential political figures. Pericles, a prominent and influential Greek statesman, orator, and general of Athens during the Golden Age, specifically between the Persian and Peloponnesian wars, played a significant role in developing the Athenian democracy.

# The Ancient Greek era also witnessed extraordinary advancements in arts and literature. Homer, credited with the creation of the epic poems 'The Iliad' and 'The Odyssey,' is considered one of the greatest poets in history. The tragedies of Sophocles, Aeschylus, and Euripides left an indelible mark on the field of drama, and the comedies of Aristophanes remain influential even today.

# ---
# Question: 

# Who among the mentioned figures from Ancient Greece contributed to the domain of mathematics and what are their significant contributions?"""}],
# 		}
# 		for _ in range(10):
# 			future = executor.submit(router.completion, **kwargs)
# 			futures[future] = future

# 		# Retrieve the results from the futures
# 		while futures:
# 			done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
# 			for future in done:
# 				try:
# 					result = future.result()
# 					print(f"result: {result}")
# 					results.append(result)
# 					del futures[future]
# 				except Exception as e:
# 					print(f"Exception: {e}; traceback: {traceback.format_exc()}")
# 					del futures[future]  # remove the done future

# 		# Check results
# 		print(results)


# test_multiple_deployments()
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

# ### FUNCTION CALLING -> NORMAL COMPLETION
# def test_litellm_params_not_overwritten_by_function_calling():
# 	try:
# 		model_list = [
# 			{
# 				"model_name": "gpt-3.5-turbo-0613",
# 				"litellm_params": {
# 					"model": "gpt-3.5-turbo-0613",
# 					"api_key": os.getenv("OPENAI_API_KEY"),
# 				},
# 				"tpm": 100000,
# 				"rpm": 10000,
# 			},
# 		]

# 		messages = [
# 			{"role": "user", "content": "What is the weather like in Boston?"}
# 		]
# 		functions = [
# 			{
# 			"name": "get_current_weather",
# 			"description": "Get the current weather in a given location",
# 			"parameters": {
# 				"type": "object",
# 				"properties": {
# 				"location": {
# 					"type": "string",
# 					"description": "The city and state, e.g. San Francisco, CA"
# 				},
# 				"unit": {
# 					"type": "string",
# 					"enum": ["celsius", "fahrenheit"]
# 				}
# 				},
# 				"required": ["location"]
# 			}
# 			}
# 		]

# 		router = Router(model_list=model_list)
# 		_ = router.completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
# 		response = router.completion(model="gpt-3.5-turbo-0613", messages=messages)
# 		assert response.choices[0].finish_reason != "function_call"
# 	except Exception as e:
# 		pytest.fail(f"Error occurred: {e}")

# test_litellm_params_not_overwritten_by_function_calling()

def test_acompletion_on_router(): 
	try:
		litellm.set_verbose = True
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
			{
				"model_name": "gpt-3.5-turbo",
				"litellm_params": {
					"model": "chatgpt-v-2",
					"api_key": os.getenv("AZURE_API_KEY"),
					"api_base": os.getenv("AZURE_API_BASE"),
					"api_version": os.getenv("AZURE_API_VERSION")
				},
				"tpm": 100000,
				"rpm": 10000,
			}
		]

		messages = [
			{"role": "user", "content": "What is the weather like in Boston?"}
		]
		start_time = time.time()
		async def get_response(): 
			router = Router(model_list=model_list, 
				   redis_host=os.environ["REDIS_HOST"], 
				   redis_password=os.environ["REDIS_PASSWORD"], 
				   redis_port=os.environ["REDIS_PORT"], 
				   cache_responses=True, 
				   timeout=30,
				   routing_strategy="usage-based-routing")
			response1 = await router.acompletion(model="gpt-3.5-turbo", messages=messages)
			print(f"response1: {response1}")
			response2 = await router.acompletion(model="gpt-3.5-turbo", messages=messages)
			print(f"response2: {response2}")
			assert response1["choices"][0]["message"]["content"] == response2["choices"][0]["message"]["content"]
		asyncio.run(get_response())
	except litellm.Timeout as e: 
		end_time = time.time()
		print(f"timeout error occurred: {end_time - start_time}")
		pass
	except Exception as e:
		traceback.print_exc()
		pytest.fail(f"Error occurred: {e}")

test_acompletion_on_router() 

def test_function_calling_on_router(): 
	try: 
		litellm.set_verbose = True
		model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]
		function1 = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ]
		router = Router(
			model_list=model_list,
			redis_host=os.getenv("REDIS_HOST"),
			redis_password=os.getenv("REDIS_PASSWORD"),
			redis_port=os.getenv("REDIS_PORT")
		)
		messages=[
                {
                    "role": "user",
                    "content": "what's the weather in boston"
                }
            ]
		response = router.completion(model="gpt-3.5-turbo", messages=messages, functions=function1)
		print(f"final returned response: {response}")
		assert isinstance(response["choices"][0]["message"]["function_call"], dict)
	except Exception as e: 
		print(f"An exception occurred: {e}")

# test_function_calling_on_router()

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
