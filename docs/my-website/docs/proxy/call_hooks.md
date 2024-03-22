# Modify / Reject Incoming Requests

- Modify data before making llm api calls on proxy
- Reject data before making llm api calls / before returning the response 

See a complete example with our [parallel request rate limiter](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/hooks/parallel_request_limiter.py)

## Quick Start

1. In your Custom Handler add a new `async_pre_call_hook` function

This function is called just before a litellm completion call is made, and allows you to modify the data going into the litellm call [**See Code**](https://github.com/BerriAI/litellm/blob/589a6ca863000ba8e92c897ba0f776796e7a5904/litellm/proxy/proxy_server.py#L1000)

```python
from litellm.integrations.custom_logger import CustomLogger
import litellm

# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger): # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        pass

    #### ASYNC #### 
    
    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_pre_api_call(self, model, messages, kwargs):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### CALL HOOKS - proxy only #### 

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal["completion", "embeddings"]):
        data["model"] = "my-new-model"
        return data 

proxy_handler_instance = MyCustomHandler()
```

2. Add this file to your proxy config

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo

litellm_settings:
  callbacks: custom_callbacks.proxy_handler_instance # sets litellm.callbacks = [proxy_handler_instance]
```

3. Start the server + test the request

```shell
$ litellm /path/to/config.yaml
```
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --data ' {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "good morning good sir"
        }
    ],
    "user": "ishaan-app",
    "temperature": 0.2
    }'
```


## [BETA] *NEW* async_moderation_hook 

Run a moderation check in parallel to the actual LLM API call. 

In your Custom Handler add a new `async_moderation_hook` function

- This is currently only supported for `/chat/completion` calls. 
- This function runs in parallel to the actual LLM API call. 
- If your `async_moderation_hook` raises an Exception, we will return that to the user. 


:::info

We might need to update the function schema in the future, to support multiple endpoints (e.g. accept a call_type). Please keep that in mind, while trying this feature

:::

See a complete example with our [Llama Guard content moderation hook](https://github.com/BerriAI/litellm/blob/main/enterprise/enterprise_hooks/llm_guard.py)

```python
from litellm.integrations.custom_logger import CustomLogger
import litellm
from fastapi import HTTPException

# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger): # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        pass

    #### ASYNC #### 
    
    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_pre_api_call(self, model, messages, kwargs):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### CALL HOOKS - proxy only #### 

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal["completion", "embeddings"]):
        data["model"] = "my-new-model"
        return data 
    
    async def async_moderation_hook( ### 👈 KEY CHANGE ###
        self,
        data: dict,
    ):
        messages = data["messages"]
        print(messages)
        if messages[0]["content"] == "hello world": 
            raise HTTPException(
                    status_code=400, detail={"error": "Violated content safety policy"}
                )

proxy_handler_instance = MyCustomHandler()
```


2. Add this file to your proxy config

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo

litellm_settings:
  callbacks: custom_callbacks.proxy_handler_instance # sets litellm.callbacks = [proxy_handler_instance]
```

3. Start the server + test the request

```shell
$ litellm /path/to/config.yaml
```
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --data ' {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "Hello world"
        }
    ],
    }'
```