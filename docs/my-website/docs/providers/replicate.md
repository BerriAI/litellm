# Replicate

LiteLLM supports all models on Replicate

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

# replicate llama-2 call
response = completion(
    model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

### Example - Calling Replicate Deployments
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

### Replicate Models
liteLLM supports all replicate LLMs

For replicate models ensure to add a `replicate/` prefix to the `model` arg. liteLLM detects it using this arg. 

Below are examples on how to call replicate LLMs using liteLLM 

Model Name                  | Function Call                                                  | Required OS Variables                |
-----------------------------|----------------------------------------------------------------|--------------------------------------|
 replicate/llama-2-70b-chat | `completion(model='replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf', messages, supports_system_prompt=True)` | `os.environ['REPLICATE_API_KEY']`    |
 a16z-infra/llama-2-13b-chat| `completion(model='replicate/a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52', messages, supports_system_prompt=True)`| `os.environ['REPLICATE_API_KEY']`    |
 replicate/vicuna-13b  | `completion(model='replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b', messages)` | `os.environ['REPLICATE_API_KEY']` |
 daanelson/flan-t5-large    | `completion(model='replicate/daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f', messages)`    | `os.environ['REPLICATE_API_KEY']`    |
 custom-llm    | `completion(model='replicate/custom-llm-version-id', messages)`    | `os.environ['REPLICATE_API_KEY']`    |
  replicate deployment    | `completion(model='replicate/deployments/ishaan-jaff/ishaan-mistral', messages)`    | `os.environ['REPLICATE_API_KEY']`    |


### Passing additional params - max_tokens, temperature 
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

### Passings Replicate specific params
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
