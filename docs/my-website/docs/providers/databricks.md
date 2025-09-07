import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Databricks

LiteLLM supports all models on Databricks

:::tip

**We support ALL Databricks models, just set `model=databricks/<any-model-on-databricks>` as a prefix when sending litellm requests**

:::

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

### ENV VAR
```python
import os 
os.environ["DATABRICKS_API_KEY"] = ""
os.environ["DATABRICKS_API_BASE"] = ""
```

### Example Call

```python
from litellm import completion
import os
## set ENV variables
os.environ["DATABRICKS_API_KEY"] = "databricks key"
os.environ["DATABRICKS_API_BASE"] = "databricks base url" # e.g.: https://adb-3064715882934586.6.azuredatabricks.net/serving-endpoints

# Databricks dbrx-instruct call
response = completion(
    model="databricks/databricks-dbrx-instruct", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: dbrx-instruct
      litellm_params:
        model: databricks/databricks-dbrx-instruct
        api_key: os.environ/DATABRICKS_API_KEY
        api_base: os.environ/DATABRICKS_API_BASE
  ```



2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml --debug
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="dbrx-instruct",
      messages = [
        {
            "role": "system",
            "content": "Be a good human!"
        },
        {
            "role": "user",
            "content": "What do you know about earth?"
        }
    ]
  )

  print(response)
  ```

  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "dbrx-instruct",
      "messages": [
        {
            "role": "system",
            "content": "Be a good human!"
        },
        {
            "role": "user",
            "content": "What do you know about earth?"
        }
        ],
  }'
  ```
  </TabItem>

  </Tabs>


</TabItem>

</Tabs>

## Passing additional params - max_tokens, temperature 
See all litellm.completion supported params [here](../completion/input.md#translated-openai-params)

```python
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["DATABRICKS_API_KEY"] = "databricks key"
os.environ["DATABRICKS_API_BASE"] = "databricks api base"

# databricks dbrx call
response = completion(
    model="databricks/databricks-dbrx-instruct", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    max_tokens=20,
    temperature=0.5
)
```

**proxy**

```yaml
  model_list:
    - model_name: llama-3
      litellm_params:
        model: databricks/databricks-meta-llama-3-70b-instruct
        api_key: os.environ/DATABRICKS_API_KEY
        max_tokens: 20
        temperature: 0.5
```


## Usage - Thinking / `reasoning_content`

LiteLLM translates OpenAI's `reasoning_effort` to Anthropic's `thinking` parameter. [Code](https://github.com/BerriAI/litellm/blob/23051d89dd3611a81617d84277059cd88b2df511/litellm/llms/anthropic/chat/transformation.py#L298)

| reasoning_effort | thinking |
| ---------------- | -------- |
| "low"            | "budget_tokens": 1024 |
| "medium"         | "budget_tokens": 2048 |
| "high"           | "budget_tokens": 4096 |


Known Limitations:
- Support for passing thinking blocks back to Claude [Issue](https://github.com/BerriAI/litellm/issues/9790)
 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

# set ENV variables (can also be passed in to .completion() - e.g. `api_base`, `api_key`)
os.environ["DATABRICKS_API_KEY"] = "databricks key"
os.environ["DATABRICKS_API_BASE"] = "databricks base url"

resp = completion(
    model="databricks/databricks-claude-3-7-sonnet",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="low",
)

```

</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
- model_name: claude-3-7-sonnet
  litellm_params:
    model: databricks/databricks-claude-3-7-sonnet
    api_key: os.environ/DATABRICKS_API_KEY
    api_base: os.environ/DATABRICKS_API_BASE
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "claude-3-7-sonnet",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "low"
  }'
```

</TabItem>
</Tabs>


**Expected Response**

```python
ModelResponse(
    id='chatcmpl-c542d76d-f675-4e87-8e5f-05855f5d0f5e',
    created=1740470510,
    model='claude-3-7-sonnet-20250219',
    object='chat.completion',
    system_fingerprint=None,
    choices=[
        Choices(
            finish_reason='stop',
            index=0,
            message=Message(
                content="The capital of France is Paris.",
                role='assistant',
                tool_calls=None,
                function_call=None,
                provider_specific_fields={
                    'citations': None,
                    'thinking_blocks': [
                        {
                            'type': 'thinking',
                            'thinking': 'The capital of France is Paris. This is a very straightforward factual question.',
                            'signature': 'EuYBCkQYAiJAy6...'
                        }
                    ]
                }
            ),
            thinking_blocks=[
                {
                    'type': 'thinking',
                    'thinking': 'The capital of France is Paris. This is a very straightforward factual question.',
                    'signature': 'EuYBCkQYAiJAy6AGB...'
                }
            ],
            reasoning_content='The capital of France is Paris. This is a very straightforward factual question.'
        )
    ],
    usage=Usage(
        completion_tokens=68,
        prompt_tokens=42,
        total_tokens=110,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=0,
            text_tokens=None,
            image_tokens=None
        ),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0
    )
)
```

### Citations

Anthropic models served through Databricks can return citation metadata. LiteLLM
exposes these via `response.choices[0].message.provider_specific_fields["citations"]`.

### Pass `thinking` to Anthropic models

You can also pass the `thinking` parameter to Anthropic models.


You can also pass the `thinking` parameter to Anthropic models.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

# set ENV variables (can also be passed in to .completion() - e.g. `api_base`, `api_key`)
os.environ["DATABRICKS_API_KEY"] = "databricks key"
os.environ["DATABRICKS_API_BASE"] = "databricks base url"

response = litellm.completion(
  model="databricks/databricks-claude-3-7-sonnet",
  messages=[{"role": "user", "content": "What is the capital of France?"}],
  thinking={"type": "enabled", "budget_tokens": 1024},
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "databricks/databricks-claude-3-7-sonnet",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "thinking": {"type": "enabled", "budget_tokens": 1024}
  }'
```

</TabItem>
</Tabs>





## Supported Databricks Chat Completion Models 

:::tip

**We support ALL Databricks models, just set `model=databricks/<any-model-on-databricks>` as a prefix when sending litellm requests**

:::


| Model Name                 | Command                                                          |
|----------------------------|------------------------------------------------------------------|
| databricks/databricks-claude-3-7-sonnet    | `completion(model='databricks/databricks/databricks-claude-3-7-sonnet', messages=messages)`   | 
| databricks-meta-llama-3-1-70b-instruct    | `completion(model='databricks/databricks-meta-llama-3-1-70b-instruct', messages=messages)`   | 
| databricks-meta-llama-3-1-405b-instruct    | `completion(model='databricks/databricks-meta-llama-3-1-405b-instruct', messages=messages)`   | 
| databricks-dbrx-instruct    | `completion(model='databricks/databricks-dbrx-instruct', messages=messages)`   | 
| databricks-meta-llama-3-70b-instruct    | `completion(model='databricks/databricks-meta-llama-3-70b-instruct', messages=messages)`   | 
| databricks-llama-2-70b-chat    | `completion(model='databricks/databricks-llama-2-70b-chat', messages=messages)`   | 
| databricks-mixtral-8x7b-instruct    | `completion(model='databricks/databricks-mixtral-8x7b-instruct', messages=messages)`   | 
| databricks-mpt-30b-instruct    | `completion(model='databricks/databricks-mpt-30b-instruct', messages=messages)`   | 
| databricks-mpt-7b-instruct    | `completion(model='databricks/databricks-mpt-7b-instruct', messages=messages)`   | 


## Embedding Models

### Passing Databricks specific params - 'instruction'

For embedding models, databricks lets you pass in an additional param 'instruction'. [Full Spec](https://github.com/BerriAI/litellm/blob/43353c28b341df0d9992b45c6ce464222ebd7984/litellm/llms/databricks.py#L164)


```python
# !pip install litellm
from litellm import embedding
import os
## set ENV variables
os.environ["DATABRICKS_API_KEY"] = "databricks key"
os.environ["DATABRICKS_API_BASE"] = "databricks url"

# Databricks bge-large-en call
response = litellm.embedding(
      model="databricks/databricks-bge-large-en",
      input=["good morning from litellm"],
      instruction="Represent this sentence for searching relevant passages:",
  )
```

**proxy**

```yaml
  model_list:
    - model_name: bge-large
      litellm_params:
        model: databricks/databricks-bge-large-en
        api_key: os.environ/DATABRICKS_API_KEY
        api_base: os.environ/DATABRICKS_API_BASE
        instruction: "Represent this sentence for searching relevant passages:"
```

## Supported Databricks Embedding Models 

:::tip

**We support ALL Databricks models, just set `model=databricks/<any-model-on-databricks>` as a prefix when sending litellm requests**

:::


| Model Name                 | Command                                                          |
|----------------------------|------------------------------------------------------------------|
| databricks-bge-large-en    | `embedding(model='databricks/databricks-bge-large-en', messages=messages)`   |
| databricks-gte-large-en    | `embedding(model='databricks/databricks-gte-large-en', messages=messages)`   |
