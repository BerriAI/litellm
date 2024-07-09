import Image from '@theme/IdealImage';

# 🔥 Langfuse - Logging LLM Input/Output

LangFuse is open Source Observability & Analytics for LLM Apps
Detailed production traces and a granular view on quality, cost and latency

<Image img={require('../../img/langfuse.png')} />

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
::: 

## Pre-Requisites
Ensure you have run `pip install langfuse` for this integration
```shell
pip install langfuse>=2.0.0 litellm
```

## Quick Start
Use just 2 lines of code, to instantly log your responses **across all providers** with Langfuse
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/logging_observability/LiteLLM_Langfuse.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Get your Langfuse API Keys from https://cloud.langfuse.com/
```python
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"] # logs errors to langfuse
```
```python
# pip install langfuse 
import litellm
import os

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
# Optional, defaults to https://cloud.langfuse.com
os.environ["LANGFUSE_HOST"] # optional

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

## Advanced
### Set Custom Generation names, pass metadata

Pass `generation_name` in `metadata`

```python
import litellm
from litellm import completion
import os

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""


# OpenAI and Cohere keys 
# You can use any of the litellm supported providers: https://docs.litellm.ai/docs/providers
os.environ['OPENAI_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 
 
# openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata = {
    "generation_name": "litellm-ishaan-gen", # set langfuse generation name
    # custom metadata fields
    "project": "litellm-proxy" 
  }
)
 
print(response)

```

### Set Custom Trace ID, Trace User ID, Trace Metadata, Trace Version, Trace Release and Tags

Pass `trace_id`, `trace_user_id`, `trace_metadata`, `trace_version`, `trace_release`, `tags` in `metadata`


```python
import litellm
from litellm import completion
import os

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

os.environ['OPENAI_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 

# set custom langfuse trace params and generation params
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata={
      "generation_name": "ishaan-test-generation",  # set langfuse Generation Name
      "generation_id": "gen-id22",                  # set langfuse Generation ID 
      "parent_observation_id": "obs-id9"            # set langfuse Parent Observation ID
      "version":  "test-generation-version"         # set langfuse Generation Version
      "trace_user_id": "user-id2",                  # set langfuse Trace User ID
      "session_id": "session-1",                    # set langfuse Session ID
      "tags": ["tag1", "tag2"],                     # set langfuse Tags
      "trace_name": "new-trace-name"                # set langfuse Trace Name
      "trace_id": "trace-id22",                     # set langfuse Trace ID
      "trace_metadata": {"key": "value"},           # set langfuse Trace Metadata
      "trace_version": "test-trace-version",        # set langfuse Trace Version (if not set, defaults to Generation Version)
      "trace_release": "test-trace-release",        # set langfuse Trace Release
      ### OR ### 
      "existing_trace_id": "trace-id22",            # if generation is continuation of past trace. This prevents default behaviour of setting a trace name
      ### OR enforce that certain fields are trace overwritten in the trace during the continuation ###
      "existing_trace_id": "trace-id22",
      "trace_metadata": {"key": "updated_trace_value"},            # The new value to use for the langfuse Trace Metadata
      "update_trace_keys": ["input", "output", "trace_metadata"],  # Updates the trace input & output to be this generations input & output also updates the Trace Metadata to match the passed in value
      "debug_langfuse": True,                                      # Will log the exact metadata sent to litellm for the trace/generation as `metadata_passed_to_litellm` 
  },
)

print(response)

```

You can also pass `metadata` as part of the request header with a `langfuse_*` prefix:

```shell
curl --location --request POST 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'langfuse_trace_id: trace-id2' \
    --header 'langfuse_trace_user_id: user-id2' \
    --header 'langfuse_trace_metadata: {"key":"value"}' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```


### Trace & Generation Parameters

#### Trace Specific Parameters

* `trace_id`       - Identifier for the trace, must use `existing_trace_id` instead of `trace_id` if this is an existing trace, auto-generated by default
* `trace_name`     - Name of the trace, auto-generated by default
* `session_id`     - Session identifier for the trace, defaults to `None`
* `trace_version`  - Version for the trace, defaults to value for `version`
* `trace_release`  - Release for the trace, defaults to `None`
* `trace_metadata` - Metadata for the trace, defaults to `None`
* `trace_user_id`  - User identifier for the trace, defaults to completion argument `user`
* `tags`           - Tags for the trace, defeaults to `None`

##### Updatable Parameters on Continuation

The following parameters can be updated on a continuation of a trace by passing in the following values into the `update_trace_keys` in the metadata of the completion.

* `input`          - Will set the traces input to be the input of this latest generation
* `output`         - Will set the traces output to be the output of this generation
* `trace_version`  - Will set the trace version to be the provided value (To use the latest generations version instead, use `version`)
* `trace_release`  - Will set the trace release to be the provided value
* `trace_metadata` - Will set the trace metadata to the provided value
* `trace_user_id`  - Will set the trace user id to the provided value

#### Generation Specific Parameters

* `generation_id`         - Identifier for the generation, auto-generated by default
* `generation_name`       - Identifier for the generation, auto-generated by default
* `parent_observation_id` - Identifier for the parent observation, defaults to `None`
* `prompt`                - Langfuse prompt object used for the generation, defaults to `None`

Any other key value pairs passed into the metadata not listed in the above spec for a `litellm` completion will be added as a metadata key value pair for the generation.

### Use LangChain ChatLiteLLM + Langfuse
Pass `trace_user_id`, `session_id` in model_kwargs
```python
import os
from langchain.chat_models import ChatLiteLLM
from langchain.schema import HumanMessage
import litellm

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

os.environ['OPENAI_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 

chat = ChatLiteLLM(
  model="gpt-3.5-turbo"
  model_kwargs={
      "metadata": {
        "trace_user_id": "user-id2", # set langfuse Trace User ID
        "session_id": "session-1" ,  # set langfuse Session ID
        "tags": ["tag1", "tag2"] 
      }
    }
  )
messages = [
    HumanMessage(
        content="what model are you"
    )
]
chat(messages)
```

## Redacting Messages, Response Content from Langfuse Logging 

### Redact Messages and Responses from all Langfuse Logging

Set `litellm.turn_off_message_logging=True` This will prevent the messages and responses from being logged to langfuse, but request metadata will still be logged.

### Redact Messages and Responses from specific Langfuse Logging

In the metadata typically passed for text completion or embedding calls you can set specific keys to mask the messages and responses for this call.

Setting `mask_input` to `True` will mask the input from being logged for this call 

Setting `mask_output` to `True` will make the output from being logged for this call.

Be aware that if you are continuing an existing trace, and you set `update_trace_keys` to include either `input` or `output` and you set the corresponding `mask_input` or `mask_output`, then that trace will have its existing input and/or output replaced with a redacted message.

## Troubleshooting & Errors
### Data not getting logged to Langfuse ? 
- Ensure you're on the latest version of langfuse `pip install langfuse -U`. The latest version allows litellm to log JSON input/outputs to langfuse

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
