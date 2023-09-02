import Image from '@theme/IdealImage';

# Replace GPT-4 with Llama2 in Production!
In this tutorial, we'll walk through replacing our GPT-4 endpoint with Llama2 in production. We'll assume you've deployed Llama2 on Huggingface Inference Endpoints (but any of TogetherAI, Baseten, Ollama, Petals, Openrouter should work as well).


# Relevant Links: 

* 🚀 [Your production dashboard!](https://admin.litellm.ai/)


* [Deploying models on Huggingface](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint)
* [All supported providers on LiteLLM](https://docs.litellm.ai/docs/completion/supported)

# Code Walkthrough

## 1. Replace GPT-4 with Llama2 
LiteLLM is a *drop-in replacement* for the OpenAI python sdk, so let's replace our openai ChatCompletion call with a LiteLLM completion call. 

### a) Replace Openai
Replace this
```python  
openai.ChatCompletion.create(model="gpt-4", messages=messages)
```

With this
```python  
from litellm import completion
completion(model="gpt-4", messages=messages)
```

### b) Replace GPT-4
Assume Llama2 is deployed at this endpoint: "https://my-unique-endpoint.us-east-1.aws.endpoints.huggingface.cloud" on Huggingface. 

```python  
from litellm import completion
completion(model="huggingface/https://my-unique-endpoint.us-east-1.aws.endpoints.huggingface.cloud", messages=messages)
```

## 2. 😱 But what if Llama2 isn't good enough?
In production, we don't know if Llama2 is going to provide:
* good results 
* quickly

### 💡 Split traffic b/w GPT-4 + Llama2
If Llama2 returns poor answers / is extremely slow, we want to roll-back this change, and use GPT-4 instead.

Instead of routing 100% of our traffic to Llama2, let's **start by just routing 20% traffic** to it and see how it does. 

```python 
## route 20% of responses to Llama2
split_per_model = {
	"gpt-4": 0.8, 
	"huggingface/https://my-unique-endpoint.us-east-1.aws.endpoints.huggingface.cloud": 0.2
}
```

## 3. Complete Code

### a) For Local
This is what our complete code looks like.
```python 
from litellm import completion_with_split_tests
import os 

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["HUGGINGFACE_API_KEY"] = "huggingface key"

## route 20% of responses to Llama2
split_per_model = {
	"gpt-4": 0.8, 
	"huggingface/https://my-unique-endpoint.us-east-1.aws.endpoints.huggingface.cloud": 0.2
}

messages = [{ "content": "Hello, how are you?","role": "user"}]

completion_with_split_tests(
  models=split_per_model, 
  messages=messages, 
)
```

### b) For Production

If we're in production, we don't want to keep going to code to change model/test details (prompt, split%, etc.) and redeploying changes. 

LiteLLM exposes a client dashboard to do this in a UI - and instantly updates your test config in prod.

#### Relevant Code 

```python
completion_with_split_tests(..., use_client=True, id="my-unique-id")
```

#### Complete Code

```python 
from litellm import completion_with_split_tests
import os 

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["HUGGINGFACE_API_KEY"] = "huggingface key"

## route 20% of responses to Llama2
split_per_model = {
	"gpt-4": 0.8, 
	"huggingface/https://my-unique-endpoint.us-east-1.aws.endpoints.huggingface.cloud": 0.2
}

messages = [{ "content": "Hello, how are you?","role": "user"}]

completion_with_split_tests(
  models=split_per_model, 
  messages=messages, 
  use_client=True, 
  id="my-unique-id" # Auto-create this @ https://admin.litellm.ai/
)
```

### A/B Testing Dashboard after running code - https://admin.litellm.ai/
<Image img={require('../../img/ab_test_logs.png')} />
