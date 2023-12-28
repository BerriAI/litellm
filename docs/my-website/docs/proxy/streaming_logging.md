# [Tutorial] Streaming token usage Logging

### Step 1 - Create your custom `litellm` callback class
We use `litellm.integrations.custom_logger` for this, **more details about litellm custom callbacks [here](https://docs.litellm.ai/docs/observability/custom_callback)**

Define your custom callback class in a python file.

```python
from litellm.integrations.custom_logger import CustomLogger
import litellm

# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            # init logging config
            logging.basicConfig(
                    filename='cost.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
            )

            # check if it has collected an entire stream response
            if "complete_streaming_response" in kwargs:
                # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost 
                completion_response=kwargs["complete_streaming_response"]
                input_text = kwargs["messages"]
                output_text = completion_response["choices"][0]["message"]["content"]
                response_cost = litellm.completion_cost(
                    model = kwargs["model"],
                    messages = input_text,
                    completion=output_text
                )
                print("streaming response_cost", response_cost)
                logging.info(f"Model {kwargs['model']} Cost: ${response_cost:.8f}")

            # for non streaming responses
            else:
                # we pass the completion_response obj
                if kwargs["stream"] != True:
                    response_cost = litellm.completion_cost(completion_response=completion_response)
                    print("regular response_cost", response_cost)
                    logging.info(f"Model {completion_response.model} Cost: ${response_cost:.8f}")
        except:
            pass


    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Async Failure")

proxy_handler_instance = MyCustomHandler()

# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy
```

### Step 2 - Pass your custom callback class in `config.yaml`
We pass the custom callback class defined in **Step1** to the config.yaml. 
Set `callbacks` to `python_filename.logger_instance_name`

In the config below, we pass
- python_filename: `custom_callbacks.py`
- logger_instance_name: `proxy_handler_instance`. This is defined in Step 1

`callbacks: custom_callbacks.proxy_handler_instance`


```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo

litellm_settings:
  callbacks: custom_callbacks.proxy_handler_instance # sets litellm.callbacks = [proxy_handler_instance]

```

### Step 3 - Start proxy + test request
```shell
litellm --config proxy_config.yaml
```

```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
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