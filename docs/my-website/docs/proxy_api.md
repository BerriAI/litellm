import TokenGen from '../src/components/TokenGen.js'

# 🚨 LITELLM API (Access Claude-2/Llama2-70b/etc.)

Use this if you're trying to add support for new LLMs and need access for testing: 
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

## Step 1: Save your LiteLLM API Key 

This is your unique LiteLLM API Key. Save this for later use. 
<TokenGen/>

## Step 2: Test a new LLM

Now let's test if claude-2 is working in our code
```
from litellm import completion 
import os 

# set env var
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-1234" # 👈 replace with your unique key

messages = [{"role": "user", "content": "Hey, how's it going?"}]

response = completion(model="claude-2", messages=messages) 

print(response) 
```

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