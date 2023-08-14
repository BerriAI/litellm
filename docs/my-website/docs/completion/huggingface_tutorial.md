# Llama2 - Huggingface Tutorial 
[Huggingface](https://huggingface.co/) is an open source platform to deploy machine-learnings models. 

## Call Llama2 with Huggingface Inference Endpoints 
LiteLLM makes it easy to call your public, private or the default huggingface endpoints. 

In this case, let's try and call 3 models: 
- `deepset/deberta-v3-large-squad2`: calls the default huggingface endpoint
- `meta-llama/Llama-2-7b-hf`: calls a public endpoint
- `meta-llama/Llama-2-7b-chat-hf`: call your privat endpoint

### Case 1: Call default huggingface endpoint

Here's the complete example:

```
from litellm import completion 

model = "deepset/deberta-v3-large-squad2"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface")
```

What's happening? 
- model - this is the name of the deployed model on huggingface 
- messages - this is the input. We accept the OpenAI chat format. For huggingface, by default we iterate through the list and add the message["content"] to the prompt.

### Case 2: Call Llama2 public endpoint

We've deployed `meta-llama/Llama-2-7b-hf` behind a public endpoint - `https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud`.

Let's try it out: 
```
from litellm import completion 

model = "meta-llama/Llama-2-7b-hf"
messages = [{"role": "user", "content": "Hey, how's it going?"}] # LiteLLM follows the OpenAI format 
custom_api_base = "https://ag3dkq4zui5nu8g3.us-east-1.aws.endpoints.huggingface.cloud"

### CALLING ENDPOINT
completion(model=model, messages=messages, custom_llm_provider="huggingface", custom_api_base=custom_api_base)
```

