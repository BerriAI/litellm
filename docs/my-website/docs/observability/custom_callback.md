# Custom Callbacks

:::info
**For PROXY** [Go Here](../proxy/logging.md#custom-callback-class-async)
::: 

## Callback Class
You can create a custom callback class to precisely log events as they occur in litellm. 

```python
import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm import completion, acompletion

class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")
    

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")
    
    #### ASYNC #### - for acompletion/aembeddings

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Failure")

customHandler = MyCustomHandler()

litellm.callbacks = [customHandler]

## sync 
response = completion(model="gpt-3.5-turbo", messages=[{ "role": "user", "content": "Hi 👋 - i'm openai"}],
                              stream=True)
for chunk in response: 
    continue


## async
import asyncio 

def async completion():
    response = await acompletion(model="gpt-3.5-turbo", messages=[{ "role": "user", "content": "Hi 👋 - i'm openai"}],
                              stream=True)
    async for chunk in response: 
        continue
asyncio.run(completion())
```

## Common Hooks

- `async_log_success_event` - Log successful API calls
- `async_log_failure_event` - Log failed API calls  
- `log_pre_api_call` - Log before API call
- `log_post_api_call` - Log after API call

**Proxy-only hooks** (only work with LiteLLM Proxy):
- `async_post_call_success_hook` - Access user data + modify responses
- `async_pre_call_hook` - Modify requests before sending

### Example: Modifying the Response in async_post_call_success_hook

You can use `async_post_call_success_hook` to add custom headers or metadata to the response before it is returned to the client. For example:

```python
async def async_post_call_success_hook(data, user_api_key_dict, response):
    # Add a custom header to the response
    additional_headers = getattr(response, "_hidden_params", {}).get("additional_headers", {}) or {}
    additional_headers["x-litellm-custom-header"] = "my-value"
    if not hasattr(response, "_hidden_params"):
        response._hidden_params = {}
    response._hidden_params["additional_headers"] = additional_headers
    return response
```

This allows you to inject custom metadata or headers into the response for downstream consumers. You can use this pattern to pass information to clients, proxies, or observability tools.

## Callback Functions
If you just want to log on a specific event (e.g. on input) - you can use callback functions. 

You can set custom callbacks to trigger for:
- `litellm.input_callback`   - Track inputs/transformed inputs before making the LLM API call
- `litellm.success_callback` - Track inputs/outputs after making LLM API call
- `litellm.failure_callback` - Track inputs/outputs + exceptions for litellm calls

## Defining a Custom Callback Function
Create a custom callback function that takes specific arguments:

```python
def custom_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    # Your custom code here
    print("LITELLM: in custom callback function")
    print("kwargs", kwargs)
    print("completion_response", completion_response)
    print("start_time", start_time)
    print("end_time", end_time)
```

### Setting the custom callback function
```python
import litellm
litellm.success_callback = [custom_callback]
```

## Using Your Custom Callback Function

```python
import litellm
from litellm import completion

# Assign the custom callback function
litellm.success_callback = [custom_callback]

response = completion(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Hi 👋 - i'm openai"
        }
    ]
)

print(response)

```

## Async Callback Functions 

We recommend using the Custom Logger class for async.

```python
from litellm.integrations.custom_logger import CustomLogger
from litellm import acompletion 

class MyCustomHandler(CustomLogger):
    #### ASYNC #### 
    


    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Failure")

import asyncio 
customHandler = MyCustomHandler()

litellm.callbacks = [customHandler]

def async completion():
    response = await acompletion(model="gpt-3.5-turbo", messages=[{ "role": "user", "content": "Hi 👋 - i'm openai"}],
                              stream=True)
    async for chunk in response: 
        continue
asyncio.run(completion())
```

**Functions**

If you just want to pass in an async function for logging. 

LiteLLM currently supports just async success callback functions for async completion/embedding calls. 

```python
import asyncio, litellm 

async def async_test_logging_fn(kwargs, completion_obj, start_time, end_time):
    print(f"On Async Success!")

async def test_chat_openai():
    try:
        # litellm.set_verbose = True
        litellm.success_callback = [async_test_logging_fn]
        response = await litellm.acompletion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm openai"
                              }],
                              stream=True)
        async for chunk in response: 
            continue
    except Exception as e:
        print(e)
        pytest.fail(f"An error occurred - {str(e)}")

asyncio.run(test_chat_openai())
```

## What's Available in kwargs?

The kwargs dictionary contains all the details about your API call:

```python
def custom_callback(kwargs, completion_response, start_time, end_time):
    # Access common data
    model = kwargs.get("model")
    messages = kwargs.get("messages", [])
    cost = kwargs.get("response_cost", 0)
    cache_hit = kwargs.get("cache_hit", False)
    
    # Access metadata you passed in
    metadata = kwargs.get("litellm_params", {}).get("metadata", {})
```

**Key fields in kwargs:**
- `model` - The model name
- `messages` - Input messages  
- `response_cost` - Calculated cost
- `cache_hit` - Whether response was cached
- `litellm_params.metadata` - Your custom metadata

## Practical Examples

### Track API Costs
```python
def track_cost_callback(kwargs, completion_response, start_time, end_time):
    cost = kwargs["response_cost"] # litellm calculates this for you
    print(f"Request cost: ${cost}")

litellm.success_callback = [track_cost_callback]

response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}])
```

### Log Inputs to LLMs
```python
def get_transformed_inputs(kwargs):
    params_to_model = kwargs["additional_args"]["complete_input_dict"]
    print("params to model", params_to_model)

litellm.input_callback = [get_transformed_inputs]

response = completion(model="claude-2", messages=[{"role": "user", "content": "Hello"}])
```

### Send to External Service
```python
import requests

def send_to_analytics(kwargs, completion_response, start_time, end_time):
    data = {
        "model": kwargs.get("model"),
        "cost": kwargs.get("response_cost", 0),
        "duration": (end_time - start_time).total_seconds()
    }
    requests.post("https://your-analytics.com/api", json=data)

litellm.success_callback = [send_to_analytics]
```

## Common Issues

### Callback Not Called
Make sure you:
1. Register callbacks correctly: `litellm.callbacks = [MyHandler()]`
2. Use the right hook names (check spelling)
3. Don't use proxy-only hooks in library mode

### Performance Issues  
- Use async hooks for I/O operations
- Don't block in callback functions
- Handle exceptions properly:

```python
class SafeHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            await external_service(response_obj)
        except Exception as e:
            print(f"Callback error: {e}")  # Log but don't break the flow
```

