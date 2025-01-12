import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Prompt Management

:::info

This feature is currently in beta, and might change unexpectedly. We expect this to be more stable by next month (February 2025).
 
:::

Run experiments or change the specific model (e.g. from gpt-4o to gpt4o-mini finetune) from your prompt management tool (e.g. Langfuse) instead of making changes in the application. 

Supported Integrations:
- [Langfuse](https://langfuse.com/docs/prompts/get-started)
- [Humanloop](../observability/humanloop)

## Quick Start


<Tabs>

<TabItem value="sdk" label="SDK">

```python
import os 
import litellm

os.environ["LANGFUSE_PUBLIC_KEY"] = "public_key" # [OPTIONAL] set here or in `.completion`
os.environ["LANGFUSE_SECRET_KEY"] = "secret_key" # [OPTIONAL] set here or in `.completion`

litellm.set_verbose = True # see raw request to provider

resp = litellm.completion(
    model="langfuse/gpt-3.5-turbo",
    prompt_id="test-chat-prompt",
    prompt_variables={"user_message": "this is used"}, # [OPTIONAL]
    messages=[{"role": "user", "content": "<IGNORED>"}],
)
```



</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: my-langfuse-model
    litellm_params:
      model: langfuse/openai-model
      prompt_id: "<langfuse_prompt_id>"
      api_key: os.environ/OPENAI_API_KEY
  - model_name: openai-model
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY
```

2. Start the proxy

```bash
litellm --config config.yaml --detailed_debug
```

3. Test it! 

<Tabs>
<TabItem value="curl" label="CURL">

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-langfuse-model",
    "messages": [
        {
            "role": "user",
            "content": "THIS WILL BE IGNORED"
        }
    ],
    "prompt_variables": {
        "key": "this is used"
    }
}'
```
</TabItem>
<TabItem value="OpenAI Python SDK" label="OpenAI Python SDK">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
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
        "prompt_variables": { # [OPTIONAL]
            "key": "this is used"
        }
    }
)

print(response)
```

</TabItem>
</Tabs>

</TabItem>
</Tabs>


**Expected Logs:**

```
POST Request Sent from LiteLLM:
curl -X POST \
https://api.openai.com/v1/ \
-d '{'model': 'gpt-3.5-turbo', 'messages': <YOUR LANGFUSE PROMPT TEMPLATE>}'
```

## How to set model 

### Set the model on LiteLLM 

You can do `langfuse/<litellm_model_name>`

<Tabs>
<TabItem value="sdk" label="SDK">

```python
litellm.completion(
    model="langfuse/gpt-3.5-turbo", # or `langfuse/anthropic/claude-3-5-sonnet`
    ...
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: langfuse/gpt-3.5-turbo # OR langfuse/anthropic/claude-3-5-sonnet
      prompt_id: <langfuse_prompt_id>
      api_key: os.environ/OPENAI_API_KEY
```

</TabItem>
</Tabs>

### Set the model in Langfuse

If the model is specified in the Langfuse config, it will be used.

<Image img={require('../../img/langfuse_prompt_management_model_config.png')} />

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
```

## What is 'prompt_variables'?

- `prompt_variables`: A dictionary of variables that will be used to replace parts of the prompt.


## What is 'prompt_id'?

- `prompt_id`: The ID of the prompt that will be used for the request.

<Image img={require('../../img/langfuse_prompt_id.png')} />

## What will the formatted prompt look like?

### `/chat/completions` messages

The `messages` field sent in by the client is ignored. 

The Langfuse prompt will replace the `messages` field.

To replace parts of the prompt, use the `prompt_variables` field. [See how prompt variables are used](https://github.com/BerriAI/litellm/blob/017f83d038f85f93202a083cf334de3544a3af01/litellm/integrations/langfuse/langfuse_prompt_management.py#L127)

If the Langfuse prompt is a string, it will be sent as a user message (not all providers support system messages).

If the Langfuse prompt is a list, it will be sent as is (Langfuse chat prompts are OpenAI compatible).

## Architectural Overview

<Image img={require('../../img/prompt_management_architecture_doc.png')} />

## API Reference

These are the params you can pass to the `litellm.completion` function in SDK and `litellm_params` in config.yaml

```
prompt_id: str # required
prompt_variables: Optional[dict] # optional
langfuse_public_key: Optional[str] # optional
langfuse_secret: Optional[str] # optional
langfuse_secret_key: Optional[str] # optional
langfuse_host: Optional[str] # optional
```
