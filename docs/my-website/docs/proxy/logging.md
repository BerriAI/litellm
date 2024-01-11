import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# 🔎 Logging - Custom Callbacks, Langfuse, s3 Bucket, Sentry, OpenTelemetry

Log Proxy Input, Output, Exceptions using Custom Callbacks, Langfuse, OpenTelemetry, LangFuse, DynamoDB, s3 Bucket

- [Async Custom Callbacks](#custom-callback-class-async)
- [Logging to Langfuse](#logging-proxy-inputoutput---langfuse)
- [Logging to s3 Buckets](#logging-proxy-inputoutput---s3-buckets)
- [Logging to DynamoDB](#logging-proxy-inputoutput---dynamodb)
- [Logging to Sentry](#logging-proxy-inputoutput---sentry)
- [Logging to Traceloop (OpenTelemetry)](#opentelemetry---traceloop)

## Custom Callback Class [Async]
Use this when you want to run custom callbacks in `python`

#### Step 1 - Create your custom `litellm` callback class
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
        print("On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success!")
        # log: key, user, model, prompt, response, tokens, cost
        # Access kwargs passed to litellm.completion()
        model = kwargs.get("model", None)
        messages = kwargs.get("messages", None)
        user = kwargs.get("user", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})   # headers passed to LiteLLM proxy, can be found here

        # Calculate cost using  litellm.completion_cost()
        cost = litellm.completion_cost(completion_response=response_obj)
        response = response_obj
        # tokens used in response 
        usage = response_obj["usage"]

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

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        try:
            print(f"On Async Failure !")
            print("\nkwargs", kwargs)
            # Access kwargs passed to litellm.completion()
            model = kwargs.get("model", None)
            messages = kwargs.get("messages", None)
            user = kwargs.get("user", None)

            # Access litellm_params passed to litellm.completion(), example access `metadata`
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})   # headers passed to LiteLLM proxy, can be found here

            # Acess Exceptions & Traceback
            exception_event = kwargs.get("exception", None)
            traceback_event = kwargs.get("traceback_exception", None)

            # Calculate cost using  litellm.completion_cost()
            cost = litellm.completion_cost(completion_response=response_obj)
            print("now checking response obj")
            
            print(
                f"""
                    Model: {model},
                    Messages: {messages},
                    User: {user},
                    Cost: {cost},
                    Response: {response_obj}
                    Proxy Metadata: {metadata}
                    Exception: {exception_event}
                    Traceback: {traceback_event}
                """
            )
        except Exception as e:
            print(f"Exception: {e}")

proxy_handler_instance = MyCustomHandler()

# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy
```

#### Step 2 - Pass your custom callback class in `config.yaml`
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

#### Step 3 - Start proxy + test request
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

#### Logging Proxy Request Object, Header, Url

Here's how you can access the `url`, `headers`, `request body` sent to the proxy for each request

```python
class MyCustomHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success!")

        litellm_params = kwargs.get("litellm_params", None)
        proxy_server_request = litellm_params.get("proxy_server_request")
        print(proxy_server_request)
```

**Expected Output**

```shell
{
  "url": "http://testserver/chat/completions",
  "method": "POST",
  "headers": {
    "host": "testserver",
    "accept": "*/*",
    "accept-encoding": "gzip, deflate",
    "connection": "keep-alive",
    "user-agent": "testclient",
    "authorization": "Bearer None",
    "content-length": "105",
    "content-type": "application/json"
  },
  "body": {
    "model": "Azure OpenAI GPT-4 Canada",
    "messages": [
      {
        "role": "user",
        "content": "hi"
      }
    ],
    "max_tokens": 10
  }
}

```

#### Logging `model_info` set in config.yaml 

Here is how to log the `model_info` set in your proxy `config.yaml`. Information on setting `model_info` on [config.yaml](https://docs.litellm.ai/docs/proxy/configs)

```python
class MyCustomHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success!")

        litellm_params = kwargs.get("litellm_params", None)
        model_info = litellm_params.get("model_info")
        print(model_info)
```

**Expected Output**
```json
{'mode': 'embedding', 'input_cost_per_token': 0.002}
```

### Logging responses from proxy
Both `/chat/completions` and `/embeddings` responses are available as `response_obj`

**Note: for `/chat/completions`, both `stream=True` and `non stream` responses are available as `response_obj`**

```python
class MyCustomHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success!")
        print(response_obj)

```

**Expected Output /chat/completion [for both `stream` and `non-stream` responses]**
```json
ModelResponse(
    id='chatcmpl-8Tfu8GoMElwOZuj2JlHBhNHG01PPo',
    choices=[
        Choices(
            finish_reason='stop',
            index=0,
            message=Message(
                content='As an AI language model, I do not have a physical body and therefore do not possess any degree or educational qualifications. My knowledge and abilities come from the programming and algorithms that have been developed by my creators.',
                role='assistant'
            )
        )
    ],
    created=1702083284,
    model='chatgpt-v-2',
    object='chat.completion',
    system_fingerprint=None,
    usage=Usage(
        completion_tokens=42,
        prompt_tokens=5,
        total_tokens=47
    )
)
```

**Expected Output /embeddings**
```json
{
    'model': 'ada',
    'data': [
        {
            'embedding': [
                -0.035126980394124985, -0.020624293014407158, -0.015343423001468182,
                -0.03980357199907303, -0.02750781551003456, 0.02111034281551838,
                -0.022069307044148445, -0.019442008808255196, -0.00955679826438427,
                -0.013143060728907585, 0.029583381488919258, -0.004725852981209755,
                -0.015198921784758568, -0.014069183729588985, 0.00897879246622324,
                0.01521205808967352,
                # ... (truncated for brevity)
            ]
        }
    ]
}
```


## Logging Proxy Input/Output - Langfuse
We will use the `--config` to set `litellm.success_callback = ["langfuse"]` this will log all successfull LLM calls to langfuse

**Step 1** Install langfuse

```shell
pip install langfuse>=2.0.0
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

### Logging Metadata to Langfuse


<Tabs>

<TabItem value="Curl" label="Curl Request">

Pass `metadata` as part of the request body

```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "metadata": {
        "generation_name": "ishaan-test-generation",
        "generation_id": "gen-id22",
        "trace_id": "trace-id22",
        "trace_user_id": "user-id2"
    }
}'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

Set `extra_body={"metadata": { }}` to `metadata` you want to pass

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:8000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "metadata": {
            "generation_name": "ishaan-generation-openai-client",
            "generation_id": "openai-client-gen-id22",
            "trace_id": "openai-client-trace-id22",
            "trace_user_id": "openai-client-user-id2"
        }
    }
)

print(response)
```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:8000",
    model = "gpt-3.5-turbo",
    temperature=0.1,
    extra_body={
        "metadata": {
            "generation_name": "ishaan-generation-langchain-client",
            "generation_id": "langchain-client-gen-id22",
            "trace_id": "langchain-client-trace-id22",
            "trace_user_id": "langchain-client-user-id2"
        }
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>

## Logging Proxy Input/Output - s3 Buckets

We will use the `--config` to set 
- `litellm.success_callback = ["s3"]` 

This will log all successfull LLM calls to s3 Bucket

**Step 1** Set AWS Credentials in .env

```shell
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION_NAME = ""
```

**Step 2**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`
```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["s3"]
  s3_callback_params:
    s3_bucket_name: logs-bucket-litellm   # AWS Bucket Name for S3
    s3_region_name: us-west-2              # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID  # us os.environ/<variable name> to pass environment variables. This is AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY  # AWS Secret Access Key for S3
    s3_endpoint_url: https://s3.amazonaws.com  # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 buckets
```

**Step 3**: Start the proxy, make a test request

Start proxy
```shell
litellm --config config.yaml --debug
```

Test Request
```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data ' {
    "model": "Azure OpenAI GPT-4 East",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
    }'
```

Your logs should be available on the specified s3 Bucket

## Logging Proxy Input/Output - DynamoDB

We will use the `--config` to set 
- `litellm.success_callback = ["dynamodb"]` 
- `litellm.dynamodb_table_name = "your-table-name"`

This will log all successfull LLM calls to DynamoDB

**Step 1** Set AWS Credentials in .env

```shell
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION_NAME = ""
```

**Step 2**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`
```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["dynamodb"]
  dynamodb_table_name: your-table-name
```

**Step 3**: Start the proxy, make a test request

Start proxy
```shell
litellm --config config.yaml --debug
```

Test Request
```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data ' {
    "model": "Azure OpenAI GPT-4 East",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
    }'
```

Your logs should be available on DynamoDB

#### Data Logged to DynamoDB /chat/completions

```json
{
  "id": {
    "S": "chatcmpl-8W15J4480a3fAQ1yQaMgtsKJAicen"
  },
  "call_type": {
    "S": "acompletion"
  },
  "endTime": {
    "S": "2023-12-15 17:25:58.424118"
  },
  "messages": {
    "S": "[{'role': 'user', 'content': 'This is a test'}]"
  },
  "metadata": {
    "S": "{}"
  },
  "model": {
    "S": "gpt-3.5-turbo"
  },
  "modelParameters": {
    "S": "{'temperature': 0.7, 'max_tokens': 100, 'user': 'ishaan-2'}"
  },
  "response": {
    "S": "ModelResponse(id='chatcmpl-8W15J4480a3fAQ1yQaMgtsKJAicen', choices=[Choices(finish_reason='stop', index=0, message=Message(content='Great! What can I assist you with?', role='assistant'))], created=1702641357, model='gpt-3.5-turbo-0613', object='chat.completion', system_fingerprint=None, usage=Usage(completion_tokens=9, prompt_tokens=11, total_tokens=20))"
  },
  "startTime": {
    "S": "2023-12-15 17:25:56.047035"
  },
  "usage": {
    "S": "Usage(completion_tokens=9, prompt_tokens=11, total_tokens=20)"
  },
  "user": {
    "S": "ishaan-2"
  }
}
```

#### Data logged to DynamoDB /embeddings

```json
{
  "id": {
    "S": "4dec8d4d-4817-472d-9fc6-c7a6153eb2ca"
  },
  "call_type": {
    "S": "aembedding"
  },
  "endTime": {
    "S": "2023-12-15 17:25:59.890261"
  },
  "messages": {
    "S": "['hi']"
  },
  "metadata": {
    "S": "{}"
  },
  "model": {
    "S": "text-embedding-ada-002"
  },
  "modelParameters": {
    "S": "{'user': 'ishaan-2'}"
  },
  "response": {
    "S": "EmbeddingResponse(model='text-embedding-ada-002-v2', data=[{'embedding': [-0.03503197431564331, -0.020601635798811913, -0.015375726856291294,
  }
}
```




## Logging Proxy Input/Output - Sentry

If api calls fail (llm/database) you can log those to Sentry: 

**Step 1** Install Sentry
```shell
pip install --upgrade sentry-sdk
```

**Step 2**: Save your Sentry_DSN and add `litellm_settings`: `failure_callback`
```shell
export SENTRY_DSN="your-sentry-dsn"
```

```yaml 
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  # other settings
  failure_callback: ["sentry"]
general_settings: 
  database_url: "my-bad-url" # set a fake url to trigger a sentry exception
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

## Logging Proxy Input/Output Traceloop (OpenTelemetry)

Traceloop allows you to log LLM Input/Output in the OpenTelemetry format

We will use the `--config` to set `litellm.success_callback = ["traceloop"]` this will log all successfull LLM calls to traceloop

**Step 1** Install traceloop-sdk and set Traceloop API key

```shell
pip install traceloop-sdk -U
```

Traceloop outputs standard OpenTelemetry data that can be connected to your observability stack. Send standard OpenTelemetry from LiteLLM Proxy to [Traceloop](https://www.traceloop.com/docs/openllmetry/integrations/traceloop), [Dynatrace](https://www.traceloop.com/docs/openllmetry/integrations/dynatrace), [Datadog](https://www.traceloop.com/docs/openllmetry/integrations/datadog)
, [New Relic](https://www.traceloop.com/docs/openllmetry/integrations/newrelic), [Honeycomb](https://www.traceloop.com/docs/openllmetry/integrations/honeycomb), [Grafana Tempo](https://www.traceloop.com/docs/openllmetry/integrations/grafana), [Splunk](https://www.traceloop.com/docs/openllmetry/integrations/splunk), [OpenTelemetry Collector](https://www.traceloop.com/docs/openllmetry/integrations/otel-collector)

**Step 2**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`
```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["traceloop"]
```

**Step 3**: Start the proxy, make a test request

Start proxy
```shell
litellm --config config.yaml --debug
```

Test Request
```
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data ' {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
    }'
```


