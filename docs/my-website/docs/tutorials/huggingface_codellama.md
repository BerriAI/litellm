# CodeLlama - Code Infilling 

This tutorial shows how you can call CodeLlama (hosted on Huggingface PRO Inference Endpoints), to fill code. 

This is a specialized task particular to code models. The model is trained to generate the code (including comments) that best matches an existing prefix and suffix. 

This task is available in the base and instruction variants of the **7B** and **13B** CodeLlama models. It is not available for any of the 34B models or the Python versions.

# usage

```python 
import os
from litellm import longer_context_model_fallback_dict, ContextWindowExceededError, completion

os.environ["HUGGINGFACE_API_KEY"] = "your-hf-token" # https://huggingface.co/docs/hub/security-tokens

## CREATE THE PROMPT
prompt_prefix = 'def remove_non_ascii(s: str) -> str:\n    """ '
prompt_suffix = "\n    return result"

### set <pre> <suf> to indicate the string before and after the part you want codellama to fill 
prompt = f"<PRE> {prompt_prefix} <SUF>{prompt_suffix} <MID>"

messages = [{"content": prompt, "role": "user"}]
model = "huggingface/codellama/CodeLlama-34b-Instruct-hf" # specify huggingface as the provider 'huggingface/'
response = completion(model=model, messages=messages, max_tokens=500)
```

# output 
```python
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s (str): The string to remove non-ASCII characters from.

    Returns:
        str: The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```