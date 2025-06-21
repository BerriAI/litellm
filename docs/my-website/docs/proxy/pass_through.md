import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create Pass Through Endpoints 

Route requests from your LiteLLM proxy to any external API. Perfect for custom models, image generation APIs, or any service you want to proxy through LiteLLM.

Onboard third-party endpoints like Bria API and Mistral OCR, set a cost per request, and give your developers access

## Usage 

In this example we will onboard the [Bria API](https://docs.bria.ai/image-generation/endpoints/text-to-image-base) and set a cost per request.

### 1. Create a pass through route on LiteLLM 

### Add route mappings 

`path`: This is the route clients shoudl use when calling LiteLLM Proxy.
`target`: This is the URL the request will be forwarded to.

<Image 
  img={require('../../img/pt_1.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

This allows for the following route mappings:

- `https://<litellm-proxy-base-url>/bria` will forward requests to `https://engine.prod.bria-api.com`
- `https://<litellm-proxy-base-url>/v1/text-to-image/base/model` will forward requests to `https://engine.prod.bria-api.com/v1/text-to-image/base/model`
- `https://<litellm-proxy-base-url>/v1/enhance_image` will forward requests to `https://engine.prod.bria-api.com/v1/enhance_image`


### 2. Add custom headers and cost per request

<Image 
  img={require('../../img/pt_2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

For making requests to the Bria API, we need to add the following headers:

- `'api_token: string'`

### 3. Test it! 

Make the following request to the Bria API through LiteLLM Proxy

```shell
curl -i -X POST \
  'http://localhost:4000/bria/v1/text-to-image/base/2.3' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your litellm api key>' \
  -d '{
    "prompt": "a book",
    "num_results": 2,
    "sync": true
  }'
```







### 4. View Request/Response Logs




**Example:** Add a route `/v1/rerank` that forwards requests to `https://api.cohere.com/v1/rerank` through LiteLLM Proxy


💡 This allows making the following Request to LiteLLM Proxy
```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": ["Carson City is the capital city of the American state of Nevada."]
  }'
```

## Tutorial - Create Pass Through on Proxy config.yaml

**Step 1** Define pass through routes on [litellm config.yaml](configs.md)

```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"                                  # route you want to add to LiteLLM Proxy Server
      target: "https://api.cohere.com/v1/rerank"          # URL this route should forward requests to
      headers:                                            # headers to forward to this URL
        Authorization: "bearer os.environ/COHERE_API_KEY" # (Optional) Auth Header to forward to your Endpoint
        content-type: application/json                    # (Optional) Extra Headers to pass to this endpoint 
        accept: application/json
      forward_headers: True                      # (Optional) Forward all headers from the incoming request to the target endpoint
```

**Step 2** Start Proxy Server in detailed_debug mode

```shell
litellm --config config.yaml --detailed_debug
```
**Step 3** Make Request to pass through endpoint

Here `http://localhost:4000` is your litellm proxy endpoint

```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": ["Carson City is the capital city of the American state of Nevada.",
                  "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                  "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                  "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
                  "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."]
  }'
```


🎉 **Expected Response**

This request got forwarded from LiteLLM Proxy -> Defined Target URL (with headers)

```shell
{
  "id": "37103a5b-8cfb-48d3-87c7-da288bedd429",
  "results": [
    {
      "index": 2,
      "relevance_score": 0.999071
    },
    {
      "index": 4,
      "relevance_score": 0.7867867
    },
    {
      "index": 0,
      "relevance_score": 0.32713068
    }
  ],
  "meta": {
    "api_version": {
      "version": "1"
    },
    "billed_units": {
      "search_units": 1
    }
  }
}
```


## ✨ [Enterprise] - Use LiteLLM keys/authentication on Pass Through Endpoints

Use this if you want the pass through endpoint to honour LiteLLM keys/authentication

This also enforces the key's rpm limits on pass-through endpoints.

Usage - set `auth: true` on the config
```yaml
general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      auth: true # 👈 Key change to use LiteLLM Auth / Keys
      headers:
        Authorization: "bearer os.environ/COHERE_API_KEY"
        content-type: application/json
        accept: application/json
```

Test Request with LiteLLM Key

```shell
curl --request POST \
  --url http://localhost:4000/v1/rerank \
  --header 'accept: application/json' \
  --header 'Authorization: Bearer sk-1234'\
  --header 'content-type: application/json' \
  --data '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": ["Carson City is the capital city of the American state of Nevada.",
                  "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                  "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                  "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
                  "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."]
  }'
```

## `pass_through_endpoints` Spec on config.yaml

All possible values for `pass_through_endpoints` and what they mean 

**Example config**
```yaml
general_settings:
  pass_through_endpoints:
    - path: "/v1/rerank"                                  # route you want to add to LiteLLM Proxy Server
      target: "https://api.cohere.com/v1/rerank"          # URL this route should forward requests to
      headers:                                            # headers to forward to this URL
        Authorization: "bearer os.environ/COHERE_API_KEY" # (Optional) Auth Header to forward to your Endpoint
        content-type: application/json                    # (Optional) Extra Headers to pass to this endpoint 
        accept: application/json
```

**Spec**

* `pass_through_endpoints` *list*: A collection of endpoint configurations for request forwarding.
  * `path` *string*: The route to be added to the LiteLLM Proxy Server.
  * `target` *string*: The URL to which requests for this path should be forwarded.
  * `headers` *object*: Key-value pairs of headers to be forwarded with the request. You can set any key value pair here and it will be forwarded to your target endpoint
    * `Authorization` *string*: The authentication header for the target API.
    * `content-type` *string*: The format specification for the request body.
    * `accept` *string*: The expected response format from the server.
    * `LANGFUSE_PUBLIC_KEY` *string*: Your Langfuse account public key - only set this when forwarding to Langfuse.
    * `LANGFUSE_SECRET_KEY` *string*: Your Langfuse account secret key - only set this when forwarding to Langfuse.
    * `<your-custom-header>` *string*: Pass any custom header key/value pair 
  * `forward_headers` *Optional(boolean)*: If true, all headers from the incoming request will be forwarded to the target endpoint. Default is `False`.


## Custom Chat Endpoints (Anthropic/Bedrock/Vertex)

Allow developers to call the proxy with Anthropic/boto3/etc. client sdk's.

Test our [Anthropic Adapter](../anthropic_completion.md) for reference [**Code**](https://github.com/BerriAI/litellm/blob/fd743aaefd23ae509d8ca64b0c232d25fe3e39ee/litellm/adapters/anthropic_adapter.py#L50)

### 1. Write an Adapter 

Translate the request/response from your custom API schema to the OpenAI schema (used by litellm.completion()) and back. 

For provider-specific params 👉 [**Provider-Specific Params**](../completion/provider_specific_params.md)

```python
from litellm import adapter_completion
import litellm 
from litellm import ChatCompletionRequest, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.llms.anthropic import AnthropicMessagesRequest, AnthropicResponse
import os

# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import os
import traceback
import uuid
from typing import Literal, Optional

import dotenv
import httpx
from pydantic import BaseModel


###################
# CUSTOM ADAPTER ##
###################
 
class AnthropicAdapter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()

    def translate_completion_input_params(
        self, kwargs
    ) -> Optional[ChatCompletionRequest]:
        """
        - translate params, where needed
        - pass rest, as is
        """
        request_body = AnthropicMessagesRequest(**kwargs)  # type: ignore

        translated_body = litellm.AnthropicConfig().translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )

        return translated_body

    def translate_completion_output_params(
        self, response: litellm.ModelResponse
    ) -> Optional[AnthropicResponse]:

        return litellm.AnthropicConfig().translate_openai_response_to_anthropic(
            response=response
        )

    def translate_completion_output_params_streaming(self) -> Optional[BaseModel]:
        return super().translate_completion_output_params_streaming()


anthropic_adapter = AnthropicAdapter()

###########
# TEST IT # 
###########

## register CUSTOM ADAPTER
litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["COHERE_API_KEY"] = "your-cohere-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = adapter_completion(model="gpt-3.5-turbo", messages=messages, adapter_id="anthropic")

# cohere call
response = adapter_completion(model="command-nightly", messages=messages, adapter_id="anthropic")
print(response)
```

### 2. Create new endpoint

We pass the custom callback class defined in Step1 to the config.yaml. Set callbacks to python_filename.logger_instance_name

In the config below, we pass

python_filename: `custom_callbacks.py`
logger_instance_name: `anthropic_adapter`. This is defined in Step 1

`target: custom_callbacks.proxy_handler_instance`

```yaml
model_list:
  - model_name: my-fake-claude-endpoint
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY


general_settings:
  master_key: sk-1234
  pass_through_endpoints:
    - path: "/v1/messages"                 # route you want to add to LiteLLM Proxy Server
      target: custom_callbacks.anthropic_adapter          # Adapter to use for this route
      headers:
        litellm_user_api_key: "x-api-key" # Field in headers, containing LiteLLM Key
```

### 3. Test it! 

**Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**Curl**

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
-H 'x-api-key: sk-1234' \
-H 'anthropic-version: 2023-06-01' \ # ignored
-H 'content-type: application/json' \
-D '{
    "model": "my-fake-claude-endpoint",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
}'
```

