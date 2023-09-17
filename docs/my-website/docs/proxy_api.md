# 🚨 LITELLM API (Access Claude-2/Llama2-70b/etc.)

This is an api built for the Open Interpreter community. It provides access to: 
* OpenAI models 
    * gpt-4
    * gpt-3.5-turbo
    * gpt-3.5-turbo-16k
* Llama2 models
    * togethercomputer/llama-2-70b-chat
    * togethercomputer/llama-2-70b
    * togethercomputer/LLaMA-2-7B-32K
    * togethercomputer/Llama-2-7B-32K-Instruct
    * togethercomputer/llama-2-7b
    * togethercomputer/CodeLlama-34b
    * WizardLM/WizardCoder-Python-34B-V1.0
    * NousResearch/Nous-Hermes-Llama2-13b
* Falcon models
    * togethercomputer/falcon-40b-instruct
    * togethercomputer/falcon-7b-instruct
* Jurassic/AI21 models 
    * j2-ultra
    * j2-mid
    * j2-light
* NLP Cloud models 
    * dolpin
    * chatdolphin 
* Anthropic models 
    * claude-2
    * claude-instant-v1


Here's how to call it: 

## Through LiteLLM
```python
from litellm import completion
import os 
## set ENV variables
os.environ["OPENAI_API_KEY"] = "litellm-api-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion(
    model="command-nightly", 
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    api_base="https://proxy.litellm.ai",
    custom_llm_provider="openai",
    temperature=0.2,
    max_tokens=80,
)
print(response)
```

## In CodeInterpreter

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