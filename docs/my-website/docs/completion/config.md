# Model Config

Model-specific changes can make our code complicated, making it harder to debug errors. Use model configs to simplify this. 

### usage

Handling prompt logic. Different models have different context windows. Use `adapt_to_prompt_size` to select the right model for the prompt (in case the current model is too small).


```python
from litellm import completion_with_config 
import os 

config = {
    "available_models": ["gpt-3.5-turbo", "claude-instant-1", "gpt-3.5-turbo-16k"],
    "adapt_to_prompt_size": True, # 👈 key change
}

# set env var
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"


sample_text = "how does a court case get to the Supreme Court?" * 1000
messages = [{"content": sample_text, "role": "user"}]
response = completion_with_config(model="gpt-3.5-turbo", messages=messages, config=config)
```

[**See Code**](https://github.com/BerriAI/litellm/blob/30724d9e51cdc2c3e0eb063271b4f171bc01b382/litellm/utils.py#L2783)

### Complete Config Structure

```python
config = {
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