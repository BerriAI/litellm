# Logging - OpenTelemetry, Langfuse, ElasticSearch 
Log Proxy Input, Output, Exceptions to Langfuse, OpenTelemetry
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
