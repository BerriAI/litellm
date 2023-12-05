# this tests if the router is initialized correctly
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

# every time we load the router we should have 4 clients:
# Async
# Sync
# Async + Stream
# Sync + Stream

def test_init_clients():
    litellm.set_verbose = True
    try:
        print("testing init 4 clients with diff timeouts")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "timeout": 0.01,
                    "stream_timeout": 0.000_001,
                    "max_retries": 7
                },
            },
        ]
        router = Router(model_list=model_list)
        for elem in router.model_list:
            assert elem["client"] is not None
            assert elem["async_client"] is not None
            assert elem["stream_client"] is not None
            assert elem["stream_async_client"] is not None
            
            # check if timeout for stream/non stream clients is set correctly
            async_client = elem["async_client"]
            stream_async_client = elem["stream_async_client"]

            assert async_client.timeout == 0.01
            assert stream_async_client.timeout == 0.000_001
        print("PASSED !")

    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")

test_init_clients()


def test_init_clients_basic():
    litellm.set_verbose = True
    try:
        print("Test basic client init")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
        ]
        router = Router(model_list=model_list)
        for elem in router.model_list:
            assert elem["client"] is not None
            assert elem["async_client"] is not None
            assert elem["stream_client"] is not None
            assert elem["stream_async_client"] is not None
        print("PASSED !")
            
        # see if we can init clients without timeout or max retries set
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")

test_init_clients_basic()

