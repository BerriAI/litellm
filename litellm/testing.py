import litellm
import time 
from concurrent.futures import ThreadPoolExecutor

def batch_completion(*args, **kwargs):
  batch_messages = args[1] if len(args) > 1 else kwargs.get("messages")
  results = []
  completions = []
  exceptions = []
  times = []
  with ThreadPoolExecutor() as executor:
    for message_list in batch_messages:
      if len(args) > 1:
        args_modified = list(args)
        args_modified[1] = message_list
        future = executor.submit(litellm.completion, *args_modified)
      else:
        kwargs_modified = dict(kwargs)
        kwargs_modified["messages"] = message_list
        future = executor.submit(litellm.completion, *args, **kwargs_modified)
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

def load_test_model(model: str, custom_llm_provider: str = None, custom_api_base: str = None, prompt: str = None, num_calls: int = None, force_timeout: int = None):
  test_prompt = "Hey, how's it going"
  test_calls = 100
  if prompt:
     test_prompt = prompt
  if num_calls:
     test_calls = num_calls
  messages = [[{"role": "user", "content": test_prompt}] for _ in range(test_calls)]
  start_time = time.time()
  try:
    results = batch_completion(model=model, messages=messages, custom_llm_provider=custom_llm_provider, custom_api_base = custom_api_base, force_timeout=force_timeout)
    end_time = time.time() 
    response_time = end_time - start_time
    return {"total_response_time": response_time, "calls_made": test_calls, "prompt": test_prompt, "results": results}
  except Exception as e:
    end_time = time.time() 
    response_time = end_time - start_time
    return {"total_response_time": response_time, "calls_made": test_calls, "prompt": test_prompt, "exception": e}