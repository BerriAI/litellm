# 🔑 LiteLLM Keys (Access Claude-2, Llama2-70b, etc.)

Use this if you're trying to add support for new LLMs and need access for testing. We provide a free $10 community-key for testing all providers on LiteLLM: 

## usage (community-key)

```python
import os
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
```

**Need a dedicated key?**
Email us @ krrish@berri.ai 

## Supported Models for LiteLLM Key
These are the models that currently work with the "sk-litellm-.." keys.

For a complete list of models/providers that you can call with LiteLLM, [check out our provider list](./providers/) or check out [models.litellm.ai](https://models.litellm.ai/)

* OpenAI models - [OpenAI docs](./providers/openai.md)
    * gpt-4
    * gpt-3.5-turbo
    * gpt-3.5-turbo-16k
* Llama2 models - [TogetherAI docs](./providers/togetherai.md)
    * togethercomputer/llama-2-70b-chat
    * togethercomputer/llama-2-70b
    * togethercomputer/LLaMA-2-7B-32K
    * togethercomputer/Llama-2-7B-32K-Instruct
    * togethercomputer/llama-2-7b
    * togethercomputer/CodeLlama-34b
    * WizardLM/WizardCoder-Python-34B-V1.0
    * NousResearch/Nous-Hermes-Llama2-13b
* Falcon models - [TogetherAI docs](./providers/togetherai.md)
    * togethercomputer/falcon-40b-instruct
    * togethercomputer/falcon-7b-instruct
* Jurassic/AI21 models - [AI21 docs](./providers/ai21.md)
    * j2-ultra
    * j2-mid
    * j2-light
* NLP Cloud models - [NLPCloud docs](./providers/nlp_cloud.md)
    * dolpin
    * chatdolphin 
* Anthropic models - [Anthropic docs](./providers/anthropic.md)
    * claude-2
    * claude-instant-v1


## For OpenInterpreter
This was initially built for the Open Interpreter community. If you're trying to use this feature in there, here's how you can do it:  
**Note**: You will need to clone and modify the Github repo, until [this PR is merged.](https://github.com/KillianLucas/open-interpreter/pull/288)

```
git clone https://github.com/krrishdholakia/open-interpreter-litellm-fork
```
To run it do: 
```
poetry build 

# call gpt-4 - always add 'litellm_proxy/' in front of the model name
poetry run interpreter --model litellm_proxy/gpt-4

# call llama-70b - always add 'litellm_proxy/' in front of the model name
poetry run interpreter --model litellm_proxy/togethercomputer/llama-2-70b-chat

# call claude-2 - always add 'litellm_proxy/' in front of the model name
poetry run interpreter --model litellm_proxy/claude-2
```

And that's it! 

Now you can call any model you like!


Want us to add more models? [Let us know!](https://github.com/BerriAI/litellm/issues/new/choose)