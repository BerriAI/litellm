import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Guardrail

Use this is you want to write code to run a custom guardrail

## Quick Start 

### 1. Write a `CustomGuardrail` Class

A CustomGuardrail has 3 methods to enforce guardrails 
- `async_pre_call_hook` - (Optional) modify input or reject request before making LLM API call
- `async_moderation_hook` - (Optional) reject request, runs while making LLM API call (help to lower latency)
- `async_post_call_success_hook`- (Optional) apply guardrail on input/output, runs after making LLM API call

**[See detailed spec of methods here](#customguardrail-methods)**

**Example `CustomGuardrail` Class**

Create a new file called `custom_guardrail.py` and add this code to it
```python
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks


class myCustomGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank"
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        # In this guardrail, if a user inputs `litellm` we will mask it and then send it to the LLM
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        _content = _content.replace("litellm", "********")
                        message["content"] = _content

        verbose_proxy_logger.debug(
            "async_pre_call_hook: Message after masking %s", _messages
        )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """

        # this works the same as async_pre_call_hook, but just runs in parallel as the LLM API Call
        # In this guardrail, if a user inputs `litellm` we will mask it.
        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    if "litellm" in _content.lower():
                        raise ValueError("Guardrail failed words - `litellm` detected")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response

        If a response contains the word "coffee" -> we will raise an exception
        """
        verbose_proxy_logger.debug("async_pre_call_hook response: %s", response)
        if isinstance(response, litellm.ModelResponse):
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    verbose_proxy_logger.debug("async_pre_call_hook choice: %s", choice)
                    if (
                        choice.message.content
                        and isinstance(choice.message.content, str)
                        and "coffee" in choice.message.content
                    ):
                        raise ValueError("Guardrail failed Coffee Detected")


```

### 2. Pass your custom guardrail class in LiteLLM `config.yaml`

In the config below, we point the guardrail to our custom guardrail by setting `guardrail: custom_guardrail.myCustomGuardrail`

- Python Filename: `custom_guardrail.py`
- Guardrail class name : `myCustomGuardrail`. This is defined in Step 1

`guardrail: custom_guardrail.myCustomGuardrail`

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "custom-pre-guard"
    litellm_params:
      guardrail: custom_guardrail.myCustomGuardrail  # 👈 Key change
      mode: "pre_call"                  # runs async_pre_call_hook
  - guardrail_name: "custom-during-guard"
    litellm_params:
      guardrail: custom_guardrail.myCustomGuardrail  
      mode: "during_call"               # runs async_moderation_hook
  - guardrail_name: "custom-post-guard"
    litellm_params:
      guardrail: custom_guardrail.myCustomGuardrail
      mode: "post_call"                 # runs async_post_call_success_hook
```

### 3. Start LiteLLM Gateway 

<Tabs>
<TabItem value="docker" label="Docker Run">

Mount your `custom_guardrail.py` on the LiteLLM Docker container

This mounts your `custom_guardrail.py` file from your local directory to the `/app` directory in the Docker container, making it accessible to the LiteLLM Gateway.


```shell
docker run -d \
  -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --name my-app \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  -v $(pwd)/custom_guardrail.py:/app/custom_guardrail.py \
  my-app:latest \
  --config /app/config.yaml \
  --port 4000 \
  --detailed_debug \
```

</TabItem>

<TabItem value="py" label="litellm pip">


```shell
litellm --config config.yaml --detailed_debug
```

</TabItem>

</Tabs>

### 4. Test it 

#### Test `"custom-pre-guard"`


**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Modify input" value = "not-allowed">

Expect this to mask the word `litellm` before sending the request to the LLM API. [This runs the `async_pre_call_hook`](#1-write-a-customguardrail-class)

```shell
curl -i  -X POST http://localhost:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "say the word - `litellm`"
        }
    ],
   "guardrails": ["custom-pre-guard"]
}'
```

Expected response after pre-guard

```json
{
  "id": "chatcmpl-9zREDkBIG20RJB4pMlyutmi1hXQWc",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "It looks like you've chosen a string of asterisks. This could be a way to censor or hide certain text. However, without more context, I can't provide a specific word or phrase. If there's something specific you'd like me to say or if you need help with a topic, feel free to let me know!",
        "role": "assistant",
        "tool_calls": null,
        "function_call": null
      }
    }
  ],
  "created": 1724429701,
  "model": "gpt-4o-2024-05-13",
  "object": "chat.completion",
  "system_fingerprint": "fp_3aa7262c27",
  "usage": {
    "completion_tokens": 65,
    "prompt_tokens": 14,
    "total_tokens": 79
  },
  "service_tier": null
}

```

</TabItem>

<TabItem label="Successful Call " value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather"}
    ],
    "guardrails": ["custom-pre-guard"]
  }'
```

</TabItem>


</Tabs>


#### Test `"custom-during-guard"`


**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Unsuccessful call" value = "not-allowed">

Expect this to fail since since `litellm` is in the message content. [This runs the `async_moderation_hook`](#1-write-a-customguardrail-class)


```shell
curl -i  -X POST http://localhost:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "say the word - `litellm`"
        }
    ],
   "guardrails": ["custom-during-guard"]
}'
```

Expected response after running during-guard

```json
{
  "error": {
    "message": "Guardrail failed words - `litellm` detected",
    "type": "None",
    "param": "None",
    "code": "500"
  }
}
```

</TabItem>

<TabItem label="Successful Call " value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather"}
    ],
    "guardrails": ["custom-during-guard"]
  }'
```

</TabItem>


</Tabs>


#### Test `"custom-post-guard"`



**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Unsuccessful call" value = "not-allowed">

Expect this to fail since since `coffee` will be in the response content. [This runs the `async_post_call_success_hook`](#1-write-a-customguardrail-class)


```shell
curl -i  -X POST http://localhost:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "what is coffee"
        }
    ],
   "guardrails": ["custom-post-guard"]
}'
```

Expected response after running during-guard

```json
{
  "error": {
    "message": "Guardrail failed Coffee Detected",
    "type": "None",
    "param": "None",
    "code": "500"
  }
}
```

</TabItem>

<TabItem label="Successful Call " value = "allowed">

```shell
 curl -i  -X POST http://localhost:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "what is tea"
        }
    ],
   "guardrails": ["custom-post-guard"]
}'
```

</TabItem>


</Tabs>

## ✨ Pass additional parameters to guardrail

:::info

✨ This is an Enterprise only feature [Contact us to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::


Use this to pass additional parameters to the guardrail API call. e.g. things like success threshold

1. Use `get_guardrail_dynamic_request_body_params`

`get_guardrail_dynamic_request_body_params` is a method of the `litellm.integrations.custom_guardrail.CustomGuardrail` class that fetches the dynamic guardrail params passed in the request body.

```python
from typing import Any, Dict, List, Literal, Optional, Union
import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth

class myCustomGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank"
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        # Get dynamic params from request body
        params = self.get_guardrail_dynamic_request_body_params(request_data=data)
        # params will contain: {"success_threshold": 0.9}
        verbose_proxy_logger.debug("Guardrail params: %s", params)
        return data
```

2. Pass parameters in your API requests:

LiteLLM Proxy allows you to pass `guardrails` in the request body, following the [`guardrails` spec](quick_start#spec-guardrails-parameter).

<Tabs>
<TabItem value="openai" label="OpenAI Python">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a short poem"}],
    extra_body={
        "guardrails": [
            "custom-pre-guard": {
                "extra_body": {
                    "success_threshold": 0.9
                }
            }
        ]
    }
)
```
</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl 'http://0.0.0.0:4000/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "Write a short poem"
        }
    ],
    "guardrails": [
        "custom-pre-guard": {
            "extra_body": {
                "success_threshold": 0.9
            }
        }
    ]
}'
```
</TabItem>
</Tabs>

The `get_guardrail_dynamic_request_body_params` method will return:
```json
{
    "success_threshold": 0.9
}
```

## **CustomGuardrail methods**

| Component | Description | Optional | Checked Data | Can Modify Input | Can Modify Output | Can Fail Call |
|-----------|-------------|----------|--------------|------------------|-------------------|----------------|
| `async_pre_call_hook` | A hook that runs before the LLM API call | ✅ | INPUT | ✅ | ❌ | ✅ |
| `async_moderation_hook` | A hook that runs during the LLM API call| ✅ | INPUT | ❌ | ❌ | ✅ |
| `async_post_call_success_hook` | A hook that runs after a successful LLM API call| ✅ | INPUT, OUTPUT | ❌ | ✅ | ✅ |
