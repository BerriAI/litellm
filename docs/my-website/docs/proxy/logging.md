# Logging - Custom Callbacks, OpenTelemetry, Langfuse
Log Proxy Input, Output, Exceptions using Custom Callbacks, Langfuse, OpenTelemetry

## Custom Callbacks
Use this when you want to run custom callbacks in `python`

### Step 1 - Create your custom `litellm` callback class
We use `litellm.integrations.custom_logger` for this, **more details about litellm custom callbacks [here](https://docs.litellm.ai/docs/observability/custom_callback)**

Define your custom callback class in a python file.

Here's an example custom logger for tracking `key, user, model, prompt, response, tokens, cost`. We create a file called `custom_callbacks.py` and initialize `proxy_handler_instance` 

```python
from litellm.integrations.custom_logger import CustomLogger
import litellm

# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        # Logging key details: key, user, model, prompt, response, tokens, cost
        print("\nOn Success")
        # Access kwargs passed to litellm.completion()
        model = kwargs.get("model", None)
        messages = kwargs.get("messages", None)
        user = kwargs.get("user", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {}) # Headers passed to LiteLLM proxy

        # Calculate cost using litellm.completion_cost()
        cost = litellm.completion_cost(completion_response=response_obj)
        usage = response_obj["usage"] # Tokens used in response

        print(
            f"""
                Model: {model},
                Messages: {messages},
                User: {user},
                Usage: {usage},
                Cost: {cost},
                Response: {response}
                Proxy Metadata: {metadata}
            """
        )
        return

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

proxy_handler_instance = MyCustomHandler()

# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy
```

### Step 2 - Pass your custom callback class in `config.yaml`
We pass the custom callback class defined in **Step1** to the config.yaml. 

Set `callbacks` to `python_filename.logger_instance_name`

In the config below, the custom callback is defined in a file`custom_callbacks.py` and has an instance of `proxy_handler_instance = MyCustomHandler()`. 

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

#### Resulting Log on Proxy
```shell
On Success
    Model: gpt-3.5-turbo,
    Messages: [{'role': 'user', 'content': 'good morning good sir'}],
    User: ishaan-app,
    Usage: {'completion_tokens': 10, 'prompt_tokens': 11, 'total_tokens': 21},
    Cost: 3.65e-05,
    Response: {'id': 'chatcmpl-8S8avKJ1aVBg941y5xzGMSKrYCMvN', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': 'Good morning! How can I assist you today?', 'role': 'assistant'}}], 'created': 1701716913, 'model': 'gpt-3.5-turbo-0613', 'object': 'chat.completion', 'system_fingerprint': None, 'usage': {'completion_tokens': 10, 'prompt_tokens': 11, 'total_tokens': 21}}
    Proxy Metadata: {'user_api_key': None, 'headers': Headers({'host': '0.0.0.0:8000', 'user-agent': 'curl/7.88.1', 'accept': '*/*', 'authorization': 'Bearer sk-1234', 'content-length': '199', 'content-type': 'application/x-www-form-urlencoded'}), 'model_group': 'gpt-3.5-turbo', 'deployment': 'gpt-3.5-turbo-ModelID-gpt-3.5-turbo'}
```

## OpenTelemetry, ElasticSearch

### Step 1 Start OpenTelemetry Collecter Docker Container
This container sends logs to your selected destination 

#### Install OpenTelemetry Collecter Docker Image
```shell
docker pull otel/opentelemetry-collector:0.90.0
docker run -p 127.0.0.1:4317:4317 -p 127.0.0.1:55679:55679 otel/opentelemetry-collector:0.90.0
```

#### Set Destination paths on OpenTelemetry Collecter

Here's the OpenTelemetry yaml config to use with Elastic Search
```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
  
processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

exporters:
  logging:
    loglevel: debug
  otlphttp/elastic:
    endpoint: "<your elastic endpoint>"
    headers: 
      Authorization: "Bearer <elastic api key>"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [logging, otlphttp/elastic]
    traces:
      receivers: [otlp]
      exporters: [logging, otlphttp/elastic]
    logs: 
      receivers: [otlp]
      exporters: [logging,otlphttp/elastic]
```

#### Start the OpenTelemetry container with config
Run the following command to start your docker container. We pass `otel_config.yaml` from the previous step

```shell
docker run -p 4317:4317 \
    -v $(pwd)/otel_config.yaml:/etc/otel-collector-config.yaml \
    otel/opentelemetry-collector:latest \
    --config=/etc/otel-collector-config.yaml
```

### Step 2 Configure LiteLLM proxy to log on OpenTelemetry

#### Pip install opentelemetry
```shell
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp -U
```

#### Set (OpenTelemetry) `otel=True` on the proxy `config.yaml`
**Example config.yaml**

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-eu
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: 
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)

general_settings: 
  otel: True      # set OpenTelemetry=True, on litellm Proxy

```

#### Set OTEL collector endpoint
LiteLLM will read the `OTEL_ENDPOINT` environment variable to send data to your OTEL collector 

```python
os.environ['OTEL_ENDPOINT'] # defauls to 127.0.0.1:4317 if not provided
```

#### Start LiteLLM Proxy
```shell
litellm -config config.yaml
```

#### Run a test request to Proxy
```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Authorization: Bearer sk-1244' \
    --data ' {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "request from LiteLLM testing"
        }
    ]
    }'
```


#### Test & View Logs on OpenTelemetry Collecter
On successfull logging you should be able to see this log on your `OpenTelemetry Collecter` Docker Container
```shell
Events:
SpanEvent #0
     -> Name: LiteLLM: Request Input
     -> Timestamp: 2023-12-02 05:05:53.71063 +0000 UTC
     -> DroppedAttributesCount: 0
     -> Attributes::
          -> type: Str(http)
          -> asgi: Str({'version': '3.0', 'spec_version': '2.3'})
          -> http_version: Str(1.1)
          -> server: Str(('127.0.0.1', 8000))
          -> client: Str(('127.0.0.1', 62796))
          -> scheme: Str(http)
          -> method: Str(POST)
          -> root_path: Str()
          -> path: Str(/chat/completions)
          -> raw_path: Str(b'/chat/completions')
          -> query_string: Str(b'')
          -> headers: Str([(b'host', b'0.0.0.0:8000'), (b'user-agent', b'curl/7.88.1'), (b'accept', b'*/*'), (b'authorization', b'Bearer sk-1244'), (b'content-length', b'147'), (b'content-type', b'application/x-www-form-urlencoded')])
          -> state: Str({})
          -> app: Str(<fastapi.applications.FastAPI object at 0x1253dd960>)
          -> fastapi_astack: Str(<contextlib.AsyncExitStack object at 0x127c8b7c0>)
          -> router: Str(<fastapi.routing.APIRouter object at 0x1253dda50>)
          -> endpoint: Str(<function chat_completion at 0x1254383a0>)
          -> path_params: Str({})
          -> route: Str(APIRoute(path='/chat/completions', name='chat_completion', methods=['POST']))
SpanEvent #1
     -> Name: LiteLLM: Request Headers
     -> Timestamp: 2023-12-02 05:05:53.710652 +0000 UTC
     -> DroppedAttributesCount: 0
     -> Attributes::
          -> host: Str(0.0.0.0:8000)
          -> user-agent: Str(curl/7.88.1)
          -> accept: Str(*/*)
          -> authorization: Str(Bearer sk-1244)
          -> content-length: Str(147)
          -> content-type: Str(application/x-www-form-urlencoded)
SpanEvent #2
```

### View Log on Elastic Search
Here's the log view on Elastic Search. You can see the request `input`, `output` and `headers`

<Image img={require('../../img/elastic_otel.png')} />

## Logging Proxy Input/Output - Langfuse
We will use the `--config` to set `litellm.success_callback = ["langfuse"]` this will log all successfull LLM calls to langfuse

**Step 1** Install langfuse

```shell
pip install langfuse
```

**Step 2**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`
```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["langfuse"]
```

**Step 3**: Start the proxy, make a test request

Start proxy
```shell
litellm --config config.yaml --debug
```

Test Request
```
litellm --test
```

Expected output on Langfuse

<Image img={require('../../img/langfuse_small.png')} />
