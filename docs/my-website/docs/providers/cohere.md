# Cohere

LiteLLM supports 'command', 'command-light', 'command-medium', 'command-medium-beta', 'command-xlarge-beta', 'command-nightly' models from [Cohere](https://cohere.com/). 

Like AI21, these models are available without a waitlist. 

### API KEYS

```python
import os 
os.environ["COHERE_API_KEY"] = ""
```

### Example Usage

```python

from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
```