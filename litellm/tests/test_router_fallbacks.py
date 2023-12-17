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
from litellm.integrations.custom_logger import CustomLogger

class MyCustomHandler(CustomLogger):
    success: bool = False
    failure: bool = False
    previous_models: int = 0

    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"previous_models: {kwargs['litellm_params']['metadata']['previous_models']}")
        self.previous_models += len(kwargs["litellm_params"]["metadata"]["previous_models"]) # {"previous_models": [{"model": litellm_model_name, "exception_type": AuthenticationError, "exception_string": <complete_traceback>}]}
        print(f"self.previous_models: {self.previous_models}")
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

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
			"api_key": os.getenv("AZURE_API_KEY"),
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



kwargs = {"model": "azure/gpt-3.5-turbo", "messages": [{"role": "user", "content":"Hey, how's it going?"}]}

def test_sync_fallbacks():        
    try:
        litellm.set_verbose = True
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        router = Router(model_list=model_list, 
                fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}], 
                context_window_fallbacks=[{"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}],
                set_verbose=False)
        response = router.completion(**kwargs)
        print(f"response: {response}")
        time.sleep(0.05) # allow a delay as success_callbacks are on a separate thread
        assert customHandler.previous_models == 1 # 0 retries, 1 fallback
        router.reset()
    except Exception as e:
        print(e)
# test_sync_fallbacks() 

def test_async_fallbacks(): 
    litellm.set_verbose = False
    router = Router(model_list=model_list, 
                fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}], 
                context_window_fallbacks=[{"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}],
                set_verbose=False)
    async def test_get_response():
        customHandler = MyCustomHandler()
        litellm.callbacks = [customHandler]
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await router.acompletion(**kwargs)
            print(f"customHandler.previous_models: {customHandler.previous_models}")
            time.sleep(0.05) # allow a delay as success_callbacks are on a separate thread
            assert customHandler.previous_models == 1 # 0 retries, 1 fallback
            router.reset()
        except litellm.Timeout as e: 
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")
        finally:
            router.reset()
    asyncio.run(test_get_response())

# test_async_fallbacks()

## COMMENTING OUT as the context size exceeds both gpt-3.5-turbo and gpt-3.5-turbo-16k, need a better message here
# def test_sync_context_window_fallbacks(): 
#     try:
#         customHandler = MyCustomHandler()
#         litellm.callbacks = [customHandler]
#         sample_text = "Say error 50 times" * 10000
#         kwargs["model"] = "azure/gpt-3.5-turbo-context-fallback"
#         kwargs["messages"] = [{"role": "user", "content": sample_text}]
#         router = Router(model_list=model_list, 
#                 fallbacks=[{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}], 
#                 context_window_fallbacks=[{"azure/gpt-3.5-turbo-context-fallback": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}],
#                 set_verbose=False)
#         response = router.completion(**kwargs)
#         print(f"response: {response}")
#         time.sleep(0.05) # allow a delay as success_callbacks are on a separate thread
#         assert customHandler.previous_models == 1 # 0 retries, 1 fallback
#         router.reset()
#     except Exception as e:
#         print(f"An exception occurred - {e}")
#     finally: 
#         router.reset()

# test_sync_context_window_fallbacks()

def test_dynamic_fallbacks_sync(): 
    """
    Allow setting the fallback in the router.completion() call. 
    """
    try:
          customHandler = MyCustomHandler()
          litellm.callbacks = [customHandler]
          router = Router(model_list=model_list, set_verbose=True)
          kwargs = {}
          kwargs["model"] = "azure/gpt-3.5-turbo"
          kwargs["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
          kwargs["fallbacks"] = [{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}]
          response = router.completion(**kwargs)
          print(f"response: {response}")
          time.sleep(0.05) # allow a delay as success_callbacks are on a separate thread
          assert customHandler.previous_models == 1 # 0 retries, 1 fallback
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
            customHandler = MyCustomHandler()
            litellm.callbacks = [customHandler]
            router = Router(model_list=model_list, set_verbose=True)
            kwargs = {}
            kwargs["model"] = "azure/gpt-3.5-turbo"
            kwargs["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
            kwargs["fallbacks"] = [{"azure/gpt-3.5-turbo": ["gpt-3.5-turbo"]}]
            response = await router.acompletion(**kwargs)
            print(f"response: {response}")
            time.sleep(0.05) # allow a delay as success_callbacks are on a separate thread
            assert customHandler.previous_models == 1 # 0 retries, 1 fallback
            router.reset()
         except Exception as e:
              pytest.fail(f"An exception occurred - {e}")
    asyncio.run(test_get_response())

# test_dynamic_fallbacks_async()