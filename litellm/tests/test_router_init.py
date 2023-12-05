# this tests if the router is intiaized correctly
import sys, os, time
import traceback, asyncio
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()


# everytime we load the router we should have 4 clients:
# Async
# Sync
# Async + Stream
# Sync + Stream


def test_init_clients():
	litellm.set_verbose = True
	try:
		model_list = [
			{ 
				"model_name": "gpt-3.5-turbo", 
				"litellm_params": { 
					"model": "azure/chatgpt-v-2", 
					"api_key": os.getenv("AZURE_API_KEY"),
					"api_version": os.getenv("AZURE_API_VERSION"),
					"api_base": os.getenv("AZURE_API_BASE")
				},
			},
		]


		router = Router(model_list=model_list)
		print(router.model_list)
		for elem in router.model_list:
			print(elem)
			assert elem["client"] is not None
			assert elem["async_client"] is not None
			assert elem["stream_client"] is not None
			assert elem["stream_async_client"] is not None

	except Exception as e:
		traceback.print_exc()
		pytest.fail(f"Error occurred: {e}")
# test_init_clients()
