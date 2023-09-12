# Llama2 - Huggingface Tutorial 
[Huggingface](https://huggingface.co/) is an open source platform to deploy machine-learnings models. 

## Call Llama2 with Huggingface Inference Endpoints 
LiteLLM makes it easy to call your public, private or the default huggingface endpoints. 

In this case, let's try and call 3 models:  

| Model                                   | Type of Endpoint |
| --------------------------------------- | ---------------- |
| deepset/deberta-v3-large-squad2         | [Default Huggingface Endpoint](#case-1-call-default-huggingface-endpoint) |
| meta-llama/Llama-2-7b-hf                | [Public Endpoint](#case-2-call-llama2-public-huggingface-endpoint)              |
| meta-llama/Llama-2-7b-chat-hf           | [Private Endpoint](#case-3-call-llama2-private-huggingface-endpoint)             |

### Case 1: Call default huggingface endpoint

Here's the complete example:

```python
from litellm import completion 

model = "deepset/deberta-v3-large-squad2"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface")
```

What's happening? 
- model: This is the name of the deployed model on huggingface 
- messages: This is the input. We accept the OpenAI chat format. For huggingface, by default we iterate through the list and add the message["content"] to the prompt. [Relevant Code](https://github.com/BerriAI/litellm/blob/6aff47083be659b80e00cb81eb783cb24db2e183/litellm/llms/huggingface_restapi.py#L46)
- custom_llm_provider: Optional param. This is an optional flag, needed only for Azure, Replicate, Huggingface and Together-ai (platforms where you deploy your own models). This enables litellm to route to the right provider, for your model. 

### Case 2: Call Llama2 public Huggingface endpoint

We've deployed `meta-llama/Llama-2-7b-hf` behind a public endpoint - `https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud`.

Let's try it out: 
```python
from litellm import completion 

model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 
api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", api_base=api_base)
```

What's happening? 
- api_base: Optional param. Since this uses a deployed endpoint (not the [default huggingface inference endpoint](https://github.com/BerriAI/litellm/blob/6aff47083be659b80e00cb81eb783cb24db2e183/litellm/llms/huggingface_restapi.py#L35)), we pass that to LiteLLM. 

### Case 3: Call Llama2 private Huggingface endpoint

The only difference between this and the public endpoint, is that you need an `api_key` for this. 

On LiteLLM there's 3 ways you can pass in an api_key. 

Either via environment variables, by setting it as a package variable or when calling `completion()`. 

**Setting via environment variables**  
Here's the 1 line of code you need to add 
```python
os.environ["HF_TOKEN"] = "..."
```

Here's the full code: 
```python
from litellm import completion 

os.environ["HF_TOKEN"] = "..."

model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 
api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", api_base=api_base)
```

**Setting it as package variable**  
Here's the 1 line of code you need to add 
```python
litellm.huggingface_key = "..."
```

Here's the full code: 
```python
import litellm
from litellm import completion 

litellm.huggingface_key = "..."

model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 
api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", api_base=api_base)
```

**Passed in during completion call**  
```python
completion(..., api_key="...")
```

Here's the full code: 

```python
from litellm import completion 

model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 
api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", api_base=api_base, api_key="...")
```
