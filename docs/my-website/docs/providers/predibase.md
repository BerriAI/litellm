import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Predibase

LiteLLM supports all models on Predibase


## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

### API KEYS
```python
import os 
os.environ["PREDIBASE_API_KEY"] = ""
```

### Example Call

```python
from litellm import completion
import os
## set ENV variables
os.environ["PREDIBASE_API_KEY"] = "predibase key"
os.environ["PREDIBASE_TENANT_ID"] = "predibase tenant id"

# predibase llama-3 call
response = completion(
    model="predibase/llama-3-8b-instruct", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: llama-3
      litellm_params:
        model: predibase/llama-3-8b-instruct
        api_key: os.environ/PREDIBASE_API_KEY
        tenant_id: os.environ/PREDIBASE_TENANT_ID
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
      model="llama-3",
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
      "model": "llama-3",
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

## Advanced Usage - Prompt Formatting 

LiteLLM has prompt template mappings for all `meta-llama` llama3 instruct models. [**See Code**](https://github.com/BerriAI/litellm/blob/4f46b4c3975cd0f72b8c5acb2cb429d23580c18a/litellm/llms/prompt_templates/factory.py#L1360)

To apply a custom prompt template: 

<Tabs>
<TabItem value="sdk" label="SDK">

```python 
import litellm

import os 
os.environ["PREDIBASE_API_KEY"] = ""

# Create your own custom prompt template 
litellm.register_prompt_template(
	    model="togethercomputer/LLaMA-2-7B-32K",
        initial_prompt_value="You are a good assistant" # [OPTIONAL]
	    roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n", # [OPTIONAL]
                "post_message": "\n<</SYS>>\n [/INST]\n" # [OPTIONAL]
            },
            "user": { 
                "pre_message": "[INST] ", # [OPTIONAL]
                "post_message": " [/INST]" # [OPTIONAL]
            }, 
            "assistant": {
                "pre_message": "\n" # [OPTIONAL]
                "post_message": "\n" # [OPTIONAL]
            }
        }
        final_prompt_value="Now answer as best you can:" # [OPTIONAL]
)

def predibase_custom_model():
    model = "predibase/togethercomputer/LLaMA-2-7B-32K"
    response = completion(model=model, messages=messages)
    print(response['choices'][0]['message']['content'])
    return response

predibase_custom_model()
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
# Model-specific parameters
model_list:
  - model_name: mistral-7b # model alias
    litellm_params: # actual params for litellm.completion()
      model: "predibase/mistralai/Mistral-7B-Instruct-v0.1" 
      api_key: os.environ/PREDIBASE_API_KEY
      initial_prompt_value: "\n"
      roles: {"system":{"pre_message":"<|im_start|>system\n", "post_message":"<|im_end|>"}, "assistant":{"pre_message":"<|im_start|>assistant\n","post_message":"<|im_end|>"}, "user":{"pre_message":"<|im_start|>user\n","post_message":"<|im_end|>"}}
      final_prompt_value: "\n"
      bos_token: "<s>"
      eos_token: "</s>"
      max_tokens: 4096
```

</TabItem>

</Tabs>

## Passing additional params - max_tokens, temperature 
See all litellm.completion supported params [here](https://docs.litellm.ai/docs/completion/input)

```python
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["PREDIBASE_API_KEY"] = "predibase key"

# predibae llama-3 call
response = completion(
    model="predibase/llama3-8b-instruct", 
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
        model: predibase/llama-3-8b-instruct
        api_key: os.environ/PREDIBASE_API_KEY
        max_tokens: 20
        temperature: 0.5
```

## Passings Predibase specific params - adapter_id, adapter_source, 
Send params [not supported by `litellm.completion()`](https://docs.litellm.ai/docs/completion/input) but supported by Predibase by passing them to `litellm.completion`

Example `adapter_id`, `adapter_source` are Predibase specific param - [See List](https://github.com/BerriAI/litellm/blob/8a35354dd6dbf4c2fcefcd6e877b980fcbd68c58/litellm/llms/predibase.py#L54)

```python
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["PREDIBASE_API_KEY"] = "predibase key"

# predibase llama3 call
response = completion(
    model="predibase/llama-3-8b-instruct", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    adapter_id="my_repo/3",
    adapter_source="pbase",
)
```

**proxy**

```yaml
  model_list:
    - model_name: llama-3
      litellm_params:
        model: predibase/llama-3-8b-instruct
        api_key: os.environ/PREDIBASE_API_KEY
        adapter_id: my_repo/3
        adapter_source: pbase
```
