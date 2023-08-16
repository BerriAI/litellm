import litellm
import time 
from concurrent.futures import ThreadPoolExecutor
import traceback

def testing_batch_completion(*args, **kwargs):
  try:
    batch_models = args[0] if len(args) > 0 else kwargs.pop("models") ## expected input format- ["gpt-3.5-turbo", {"model": "qvv0xeq", "custom_llm_provider"="baseten"}...]
    batch_messages = args[1] if len(args) > 1 else kwargs.pop("messages")
    results = []
    completions = []
    exceptions = []
    times = []
    with ThreadPoolExecutor() as executor:
        for model in batch_models:
            kwargs_modified = dict(kwargs)
            args_modified = list(args)
            if len(args) > 0:
                args_modified[0] = model["model"]
            else: 
                kwargs_modified["model"] = model["model"] if isinstance(model, dict) and "model" in model else model # if model is a dictionary get it's value else assume it's a string
                kwargs_modified["custom_llm_provider"] = model["custom_llm_provider"] if isinstance(model, dict) and "model" in model else None
            for message_list in batch_messages:
                if len(args) > 1:
                    args_modified[1] = message_list
                    future = executor.submit(litellm.completion, *args_modified, **kwargs_modified)
                else:
                    kwargs_modified["messages"] = message_list
                    future = executor.submit(litellm.completion, *args_modified, **kwargs_modified)
            completions.append(future)
    
    # Retrieve the results and calculate elapsed time for each completion call
    for future in completions:
        start_time = time.time()
        try:
            result = future.result()
            end_time = time.time()
            elapsed_time = end_time - start_time
            result_dict = {"status": "succeeded", "response": future.result(), "response_time": elapsed_time}
            results.append(result_dict)
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            result_dict = {"status": "succeeded", "response": e, "response_time": elapsed_time}
            results.append(result_dict)
    return results
  except:
     traceback.print_exc()

def duration_test_model(original_function):
    def wrapper_function(*args, **kwargs):
        # Code to be executed before the original function
        duration = kwargs.pop("duration", None)
        interval = kwargs.pop("interval", None)
        results = []
        if duration and interval:
            start_time = time.time()
            end_time = start_time + duration  # default to 1hr duration
            while time.time() < end_time:
                result = original_function(*args, **kwargs)
                results.append(result)
                time.sleep(interval)
        else:
            result = original_function(*args, **kwargs)
            results = result
        return results

    # Return the wrapper function
    return wrapper_function

@duration_test_model
def load_test_model(model: str, custom_llm_provider: str = None, custom_api_base: str = None, prompt: str = None, num_calls: int = None, request_timeout: int = None):
  test_prompt = "Hey, how's it going"
  test_calls = 100
  if prompt:
     test_prompt = prompt
  if num_calls:
     test_calls = num_calls
  messages = [[{"role": "user", "content": test_prompt}] for _ in range(test_calls)]
  start_time = time.time()
  try:
    results = testing_batch_completion(models=[model], messages=messages, custom_llm_provider=custom_llm_provider, custom_api_base = custom_api_base, force_timeout=request_timeout)
    end_time = time.time() 
    response_time = end_time - start_time
    return {"total_response_time": response_time, "calls_made": test_calls, "prompt": test_prompt, "results": results}
  except Exception as e:
    traceback.print_exc()
    end_time = time.time() 
    response_time = end_time - start_time
    return {"total_response_time": response_time, "calls_made": test_calls, "prompt": test_prompt, "exception": e}