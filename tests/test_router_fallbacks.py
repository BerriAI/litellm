#### What this tests ####
#    This tests calling router with fallback models 

import sys, os, time
import traceback, asyncio
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router

model_list = [
    { # list of model deployments 
		"model_name": "azure/gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-v-2", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800
	}, 
    { # list of model deployments 
		"model_name": "azure/gpt-3.5-turbo-context-fallback", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-v-2", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800
	}, 
	{
		"model_name": "azure/gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "azure/chatgpt-functioncalling", 
			"api_key": "bad-key",
			"api_version": os.getenv("AZURE_API_VERSION"),
			"api_base": os.getenv("AZURE_API_BASE")
		},
		"tpm": 240000,
		"rpm": 1800
	}, 
	{
		"model_name": "gpt-3.5-turbo", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "gpt-3.5-turbo", 
			"api_key": os.getenv("OPENAI_API_KEY"),
		},
		"tpm": 1000000,
		"rpm": 9000
	},
    {
		"model_name": "gpt-3.5-turbo-16k", # openai model name 
		"litellm_params": { # params for litellm completion/embedding call 
			"model": "gpt-3.5-turbo-16k", 
			"api_key": os.getenv("OPENAI_API_KEY"),
		},
		"tpm": 1000000,
		"rpm": 9000
	}
]



router = Router(model_list=model_list, 
                fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}], 
                context_window_fallbacks=[{"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}],
                set_verbose=True)

kwargs = {"model": "azure/gpt-3.5-turbo", "messages": [{"role": "user", "content":"Hey, how's it going?"}]}

def test_sync_fallbacks():        
    try:
        litellm.set_verbose = True
        response = router.completion(**kwargs)
        print(f"response: {response}")
        router.flush_cache()
    except Exception as e:
        print(e)
# test_sync_fallbacks() 

def test_async_fallbacks(): 
    litellm.set_verbose = False
    async def test_get_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await router.acompletion(**kwargs)
            # response = await response
            print(f"response: {response}")
            router.flush_cache()
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")

    asyncio.run(test_get_response())

# test_async_fallbacks()

def test_sync_context_window_fallbacks(): 
    try:
        sample_text = "Say error 50 times" * 10000
        kwargs["model"] = "azure/gpt-3.5-turbo-context-fallback"
        kwargs["messages"] = [{"role": "user", "content": sample_text}]
        response = router.completion(**kwargs)
        print(f"response: {response}")
        router.reset()
    except Exception as e:
        print(e)

# test_sync_context_window_fallbacks()

def test_dynamic_fallbacks_sync(): 
    """
    Allow setting the fallback in the router.completion() call. 
    """
    try:
          router = Router(model_list=model_list, set_verbose=True)
          kwargs = {}
          kwargs["model"] = "azure/gpt-3.5-turbo"
          kwargs["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
          kwargs["fallbacks"] = [{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}]
          response = router.completion(**kwargs)
          print(f"response: {response}")
          router.reset()
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")

# test_dynamic_fallbacks_sync()

def test_dynamic_fallbacks_async(): 
    """
    Allow setting the fallback in the router.completion() call. 
    """
    async def test_get_response():
         try: 
            router = Router(model_list=model_list, set_verbose=True)
            kwargs = {}
            kwargs["model"] = "azure/gpt-3.5-turbo"
            kwargs["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
            kwargs["fallbacks"] = [{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}]
            response = await router.acompletion(**kwargs)
            print(f"response: {response}")
            router.reset()
         except Exception as e:
              pytest.fail(f"An exception occurred - {e}")
    asyncio.run(test_get_response())

# test_dynamic_fallbacks_async()