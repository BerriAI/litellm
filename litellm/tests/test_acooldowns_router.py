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
import concurrent
from dotenv import load_dotenv
load_dotenv()

model_list = [{ # list of model deployments 
		"model_name": "gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-v-2", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800,  
	}, 
	{
		"model_name": "gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "gpt-3.5-turbo", 
			"api_key": os.getenv("OPENAI_API_KEY"),
		},
		"tpm": 1000000,
		"rpm": 9000
	}
]

kwargs = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hey, how's it going?"}],}


def test_multiple_deployments_sync(): 
	import concurrent, time
	litellm.set_verbose=False
	results = [] 
	router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=int(os.getenv("REDIS_PORT")),  # type: ignore
                routing_strategy="simple-shuffle",
                set_verbose=True,
                num_retries=1) # type: ignore
	try:
		for _ in range(3): 
			response = router.completion(**kwargs)
			results.append(response)
		print(results)
		router.reset()
	except Exception as e:
		print(f"FAILED TEST!")
		pytest.fail(f"An error occurred - {traceback.format_exc()}")

test_multiple_deployments_sync()


def test_multiple_deployments_parallel():
    litellm.set_verbose = False  # Corrected the syntax for setting verbose to False
    results = []
    futures = {}
    start_time = time.time()
    router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=int(os.getenv("REDIS_PORT")),  # type: ignore
                routing_strategy="simple-shuffle",
                set_verbose=True,
                num_retries=1) # type: ignore
    # Assuming you have an executor instance defined somewhere in your code
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(5):
            future = executor.submit(router.completion, **kwargs)
            futures[future] = future

        # Retrieve the results from the futures
        while futures:
            done, not_done = concurrent.futures.wait(futures.values(), timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                    results.append(result)
                    del futures[future]  # Remove the done future
                except Exception as e:
                    print(f"Exception: {e}; traceback: {traceback.format_exc()}")
                    del futures[future]  # Remove the done future with exception

            print(f"Remaining futures: {len(futures)}")
    router.reset()
    end_time = time.time()
    print(results)
    print(f"ELAPSED TIME: {end_time - start_time}")

# Assuming litellm, router, and executor are defined somewhere in your code

# test_multiple_deployments_parallel()