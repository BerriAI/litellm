# Open Interpreter LLM API 

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

**Note**: You will need to clone and modify the Github repo, until [this PR is merged.](https://github.com/KillianLucas/open-interpreter/pull/288)

In `interpreter.py` set,
```
os.environ["OPENAI_API_KEY"] = "openinterpreter-key"
litellm.api_base = "https://proxy.litellm.ai"
```

and change the model on [this line](https://github.com/KillianLucas/open-interpreter/blob/f803d0d7a545edabd541943145a2a60beaf604e4/interpreter/interpreter.py#L342C10-L342C10), to: 
```
self.model = "openai/gpt-4"  # 👈 always add 'openai/' in front of the model name
```

And that's it! 

Now you can call any model you like!


Want us to add more models? [Let us know!](https://github.com/BerriAI/litellm/issues/new/choose)