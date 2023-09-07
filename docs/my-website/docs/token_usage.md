# Completion Token Usage & Cost
By default LiteLLM returns token usage in all completion requests ([See here](https://litellm.readthedocs.io/en/latest/output/))

However, we also expose 3 public helper functions to calculate token usage across providers:

- `token_counter`: This returns the number of tokens for a given input - it uses the tokenizer based on the model, and defaults to tiktoken if no model-specific tokenizer is available. 

- `cost_per_token`: This returns the cost (in USD) for prompt (input) and completion (output) tokens. It utilizes our model_cost map which can be found in `__init__.py` and also as a [community resource](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

- `completion_cost`: This returns the overall cost (in USD) for a given LLM API Call. It combines `token_counter` and `cost_per_token` to return the cost for that query (counting both cost of input and output). 

## Example Usage 

### 1. `token_counter`

```python
from litellm import token_counter

messages = [{"user": "role", "content": "Hey, how's it going"}]
print(token_counter(model="gpt-3.5-turbo", messages=messages))
```

### 2. `cost_per_token`

```python
from litellm import cost_per_token

prompt_tokens =  5
completion_tokens = 10
prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(model="gpt-3.5-turbo", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

print(prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar)
```

### 3. `completion_cost`

* Input: Accepts a `litellm.completion()` response
* Output: Returns a `float` of cost for the `completion` call 

```python
from litellm import completion, completion_cost

response = completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=messages,
            request_timeout=200,
        )
# pass your response from completion to completion_cost
cost = completion_cost(completion_response=response)
formatted_string = f"${float(cost):.10f}"
print(formatted_string)
```

