import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Replicate

LiteLLM supports all models on Replicate


## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

### API KEYS
```python
import os 
os.environ["REPLICATE_API_KEY"] = ""
```

### Example Call

```python
from litellm import completion
import os
## set ENV variables
os.environ["REPLICATE_API_KEY"] = "replicate key"

# replicate llama-3 call
response = completion(
    model="replicate/meta/meta-llama-3-8b-instruct", 
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
        model: replicate/meta/meta-llama-3-8b-instruct
        api_key: os.environ/REPLICATE_API_KEY
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


### Expected Replicate Call 

This is the call litellm will make to replicate, from the above example: 

```bash

POST Request Sent from LiteLLM:
curl -X POST \
https://api.replicate.com/v1/models/meta/meta-llama-3-8b-instruct \
-H 'Authorization: Token your-api-key' -H 'Content-Type: application/json' \
-d '{'version': 'meta/meta-llama-3-8b-instruct', 'input': {'prompt': '<|start_header_id|>system<|end_header_id|>\n\nBe a good human!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat do you know about earth?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'}}'
```

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
os.environ["REPLICATE_API_KEY"] = ""

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

def test_replicate_custom_model():
    model = "replicate/togethercomputer/LLaMA-2-7B-32K"
    response = completion(model=model, messages=messages)
    print(response['choices'][0]['message']['content'])
    return response

test_replicate_custom_model()
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
# Model-specific parameters
model_list:
  - model_name: mistral-7b # model alias
    litellm_params: # actual params for litellm.completion()
      model: "replicate/mistralai/Mistral-7B-Instruct-v0.1" 
      api_key: os.environ/REPLICATE_API_KEY
      initial_prompt_value: "\n"
      roles: {"system":{"pre_message":"<|im_start|>system\n", "post_message":"<|im_end|>"}, "assistant":{"pre_message":"<|im_start|>assistant\n","post_message":"<|im_end|>"}, "user":{"pre_message":"<|im_start|>user\n","post_message":"<|im_end|>"}}
      final_prompt_value: "\n"
      bos_token: "<s>"
      eos_token: "</s>"
      max_tokens: 4096
```

</TabItem>

</Tabs>

## Advanced Usage - Calling Replicate Deployments
Calling a [deployed replicate LLM](https://replicate.com/deployments)
Add the `replicate/deployments/` prefix to your model, so litellm will call the `deployments` endpoint. This will call `ishaan-jaff/ishaan-mistral` deployment on replicate

```python
response = completion(
    model="replicate/deployments/ishaan-jaff/ishaan-mistral", 
    messages= [{ "content": "Hello, how are you?","role": "user"}]
)
```

:::warning Replicate Cold Boots

Replicate responses can take 3-5 mins due to replicate cold boots, if you're trying to debug try making the request with `litellm.set_verbose=True`. [More info on replicate cold boots](https://replicate.com/docs/how-does-replicate-work#cold-boots)

:::

## Replicate Models
liteLLM supports all replicate LLMs

For replicate models ensure to add a `replicate/` prefix to the `model` arg. liteLLM detects it using this arg. 

Below are examples on how to call replicate LLMs using liteLLM 

Model Name                  | Function Call                                                  | Required OS Variables                |
-----------------------------|----------------------------------------------------------------|--------------------------------------|
 replicate/llama-2-70b-chat | `completion(model='replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf', messages)` | `os.environ['REPLICATE_API_KEY']`    |
 a16z-infra/llama-2-13b-chat| `completion(model='replicate/a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52', messages)`| `os.environ['REPLICATE_API_KEY']`    |
 replicate/vicuna-13b  | `completion(model='replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b', messages)` | `os.environ['REPLICATE_API_KEY']` |
 daanelson/flan-t5-large    | `completion(model='replicate/daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f', messages)`    | `os.environ['REPLICATE_API_KEY']`    |
 custom-llm    | `completion(model='replicate/custom-llm-version-id', messages)`    | `os.environ['REPLICATE_API_KEY']`    |
  replicate deployment    | `completion(model='replicate/deployments/ishaan-jaff/ishaan-mistral', messages)`    | `os.environ['REPLICATE_API_KEY']`    |


## Passing additional params - max_tokens, temperature 
See all litellm.completion supported params [here](https://docs.litellm.ai/docs/completion/input)

```python
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["REPLICATE_API_KEY"] = "replicate key"

# replicate llama-2 call
response = completion(
    model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf", 
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
        model: replicate/meta/meta-llama-3-8b-instruct
        api_key: os.environ/REPLICATE_API_KEY
        max_tokens: 20
        temperature: 0.5
```

## Passings Replicate specific params
Send params [not supported by `litellm.completion()`](https://docs.litellm.ai/docs/completion/input) but supported by Replicate by passing them to `litellm.completion`

Example `seed`, `min_tokens` are Replicate specific param

```python
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["REPLICATE_API_KEY"] = "replicate key"

# replicate llama-2 call
response = completion(
    model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    seed=-1,
    min_tokens=2,
    top_k=20,
)
```

**proxy**

```yaml
  model_list:
    - model_name: llama-3
      litellm_params:
        model: replicate/meta/meta-llama-3-8b-instruct
        api_key: os.environ/REPLICATE_API_KEY
        min_tokens: 2
        top_k: 20
```
