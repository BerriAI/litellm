# Completion Token Usage & Cost
By default LiteLLM returns token usage in all completion requests ([See here](https://litellm.readthedocs.io/en/latest/output/))

However, we also expose 5 helper functions + **[NEW]** an API to calculate token usage across providers:

- `encode`: This encodes the text passed in, using the model-specific tokenizer. [**Jump to code**](#1-encode)

- `decode`: This decodes the tokens passed in, using the model-specific tokenizer. [**Jump to code**](#2-decode)

- `token_counter`: This returns the number of tokens for a given input - it uses the tokenizer based on the model, and defaults to tiktoken if no model-specific tokenizer is available. [**Jump to code**](#3-token_counter)

- `cost_per_token`: This returns the cost (in USD) for prompt (input) and completion (output) tokens. Uses the live list from `api.litellm.ai`. [**Jump to code**](#4-cost_per_token)

- `completion_cost`: This returns the overall cost (in USD) for a given LLM API Call. It combines `token_counter` and `cost_per_token` to return the cost for that query (counting both cost of input and output). [**Jump to code**](#5-completion_cost)

- `get_max_tokens`: This returns the maximum number of tokens allowed for the given model. [**Jump to code**](#6-get_max_tokens)

- `model_cost`: This returns a dictionary for all models, with their max_tokens, input_cost_per_token and output_cost_per_token. It uses the `api.litellm.ai` call shown below. [**Jump to code**](#7-model_cost)

- `register_model`: This registers new / overrides existing models (and their pricing details) in the model cost dictionary. [**Jump to code**](#8-register_model)

- `api.litellm.ai`: Live token + price count across [all supported models](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). [**Jump to code**](#9-apilitellmai)

📣 This is a community maintained list. Contributions are welcome! ❤️

## Example Usage 

### 1. `encode`
Encoding has model-specific tokenizers for anthropic, cohere, llama2 and openai. If an unsupported model is passed in, it'll default to using tiktoken (openai's tokenizer).

```python
from litellm import encode, decode

sample_text = "Hellö World, this is my input string!"
# openai encoding + decoding
openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
print(openai_tokens)
```

### 2. `decode`

Decoding is supported for anthropic, cohere, llama2 and openai.

```python
from litellm import encode, decode

sample_text = "Hellö World, this is my input string!"
# openai encoding + decoding
openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
openai_text = decode(model="gpt-3.5-turbo", tokens=openai_tokens)
print(openai_text)
```

### 3. `token_counter`

```python
from litellm import token_counter

messages = [{"user": "role", "content": "Hey, how's it going"}]
print(token_counter(model="gpt-3.5-turbo", messages=messages))
```

### 4. `cost_per_token`

```python
from litellm import cost_per_token

prompt_tokens =  5
completion_tokens = 10
prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(model="gpt-3.5-turbo", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

print(prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar)
```

### 5. `completion_cost`

* Input: Accepts a `litellm.completion()` response **OR** prompt + completion strings
* Output: Returns a `float` of cost for the `completion` call 

**litellm.completion()**
```python
from litellm import completion, completion_cost

response = completion(
            model="bedrock/anthropic.claude-v2",
            messages=messages,
            request_timeout=200,
        )
# pass your response from completion to completion_cost
cost = completion_cost(completion_response=response)
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
```

**prompt + completion string**
```python
from litellm import completion_cost
cost = completion_cost(model="bedrock/anthropic.claude-v2", prompt="Hey!", completion="How's it going?")
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
```
### 6. `get_max_tokens`

Input: Accepts a model name - e.g., gpt-3.5-turbo (to get a complete list, call litellm.model_list).
Output: Returns the maximum number of tokens allowed for the given model

```python 
from litellm import get_max_tokens 

model = "gpt-3.5-turbo"

print(get_max_tokens(model)) # Output: 4097
```

### 7. `model_cost`

* Output: Returns a dict object containing the max_tokens, input_cost_per_token, output_cost_per_token for all models on [community-maintained list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

```python 
from litellm import model_cost 

print(model_cost) # {'gpt-3.5-turbo': {'max_tokens': 4000, 'input_cost_per_token': 1.5e-06, 'output_cost_per_token': 2e-06}, ...}
```

### 8. `register_model`

* Input: Provide EITHER a model cost dictionary or a url to a hosted json blob
* Output: Returns updated model_cost dictionary + updates litellm.model_cost with model details.  

**Dictionary**
```python
from litellm import register_model

litellm.register_model({
        "gpt-4": {
        "max_tokens": 8192, 
        "input_cost_per_token": 0.00002, 
        "output_cost_per_token": 0.00006, 
        "litellm_provider": "openai", 
        "mode": "chat"
    },
})
```

**URL for json blob**
```python
import litellm

litellm.register_model(model_cost=
"https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json")
```



