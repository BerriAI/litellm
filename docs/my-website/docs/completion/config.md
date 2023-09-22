# Model Config

Model-specific changes can make our code complicated, making it harder to debug errors. Use model configs to simplify this. 

### usage

E.g. If we want to implement: 
* Moderations check for Anthropic models (to avoid violating their safety policy)
* Model Fallbacks - specific + general

```python
from litellm import completion_with_config 
import os 

config = {
    "default_fallback_models": ["gpt-3.5-turbo", "claude-instant-1", "gpt-3.5-turbo-16k"],
    "model": {
        "claude-instant-1": {
            "needs_moderation": True
        },
        "gpt-3.5-turbo": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-3.5-turbo-16k"} 
            }
        },
    }
}

# set env var
os.environ["OPENAI_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your anthropic key


sample_text = "how does a court case get to the Supreme Court?" * 1000
messages = [{"content": sample_text, "role": "user"}]
response = completion_with_config(model="gpt-3.5-turbo", messages=messages, config=config)
```
[**See Code**](https://github.com/BerriAI/litellm/blob/30724d9e51cdc2c3e0eb063271b4f171bc01b382/litellm/utils.py#L2783)
### select model based on prompt size 

You can also use model configs to automatically select a model based on the prompt size. It checks the number of tokens in the prompt and max tokens for each model. It selects the model with max tokens > prompt tokens. 

```python
from litellm import completion_with_config 
import os 

config = {
    "available_models": ["gpt-3.5-turbo", "claude-instant-1", "gpt-3.5-turbo-16k"],
    "adapt_to_prompt_size": True, # 👈 key change
}

# set env var
os.environ["OPENAI_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your anthropic key


sample_text = "how does a court case get to the Supreme Court?" * 1000
messages = [{"content": sample_text, "role": "user"}]
response = completion_with_config(model="gpt-3.5-turbo", messages=messages, config=config)
```

### Complete Config Structure

```python
config = {
    "function": "completion", 
    "default_fallback_models": # [Optional] List of model names to try if a call fails
    "available_models": # [Optional] List of all possible models you could call 
    "adapt_to_prompt_size": # [Optional] True/False - if you want to select model based on prompt size (will pick from available_models)
    "model": {
        "model-name": {
            "needs_moderation": # [Optional] True/False - if you want to call openai moderations endpoint before making completion call. Will raise exception, if flagged. 
            "error_handling": {
                "error-type": { # One of the errors listed here - https://docs.litellm.ai/docs/exception_mapping#custom-mapping-list
                    "fallback_model": "" # str, name of the model it should try instead, when that error occurs 
                }
            }
        }
    }
}
```