# Model Fallbacks w/ LiteLLM

Here's how you can implement model fallbacks across 3 LLM providers (OpenAI, Anthropic, Azure) using LiteLLM. 

## 1. Install LiteLLM
```python 
!pip install litellm
```

## 2. Basic Fallbacks Code 
```python 
import litellm
from litellm import embedding, completion

# set ENV variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

model_fallback_list = ["claude-instant-1", "gpt-3.5-turbo", "chatgpt-test"]

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]

for model in model_fallback_list:
  try:
      response = completion(model=model, messages=messages)
  except Exception as e:
      print(f"error occurred: {traceback.format_exc()}")
```

## 3. Context Window Exceptions 
LiteLLM provides a sub-class of the InvalidRequestError class for Context Window Exceeded errors ([docs](https://docs.litellm.ai/docs/exception_mapping)).

Implement model fallbacks based on context window exceptions. 

```python 
import litellm
from litellm import completion, ContextWindowExceededError, completion_with_fallbacks

# set ENV variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

context_window_fallback_list = ["gpt-3.5-turbo-16k", "gpt-4", "claude-instant-1"]

user_message = "Hello, how are you?"
messages = [{ "content": user_message,"role": "user"}]

for model in context_window_fallback_list:
  try:
      response = completion(model=model, messages=messages)
  except ContextWindowExceededError as e:
      completion_with_fallbacks(model=context_window_fallback_list[0], messages=messages, fallbacks=context_window_fallback_list[1:])
```