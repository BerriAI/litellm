# Completion Token Usage & Cost
By default LiteLLM returns token usage in all completion requests ([See here](https://litellm.readthedocs.io/en/latest/output/))

LiteLLM returns `response_cost` in all calls. 

```python
from litellm import completion 

response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_response="Hello world",
        )

print(response._hidden_params["response_cost"])
```

LiteLLM also exposes some helper functions:

- `encode`: This encodes the text passed in, using the model-specific tokenizer. [**Jump to code**](#1-encode)

- `decode`: This decodes the tokens passed in, using the model-specific tokenizer. [**Jump to code**](#2-decode)

- `token_counter`: This returns the number of tokens for a given input - it uses the tokenizer based on the model, and defaults to tiktoken if no model-specific tokenizer is available. [**Jump to code**](#3-token_counter)

- `create_pretrained_tokenizer` and `create_tokenizer`: LiteLLM provides default tokenizer support for OpenAI, Cohere, Anthropic, Llama2, and Llama3 models. If you are using a different model, you can create a custom tokenizer and pass it as `custom_tokenizer` to the `encode`, `decode`, and `token_counter` methods. [**Jump to code**](#4-create_pretrained_tokenizer-and-create_tokenizer)

- `cost_per_token`: This returns the cost (in USD) for prompt (input) and completion (output) tokens. Uses the live list from `api.litellm.ai`. [**Jump to code**](#5-cost_per_token)

- `completion_cost`: This returns the overall cost (in USD) for a given LLM API Call. It combines `token_counter` and `cost_per_token` to return the cost for that query (counting both cost of input and output). [**Jump to code**](#6-completion_cost)

- `get_max_tokens`: This returns the maximum number of tokens allowed for the given model. [**Jump to code**](#7-get_max_tokens)

- `model_cost`: This returns a dictionary for all models, with their max_tokens, input_cost_per_token and output_cost_per_token. It uses the `api.litellm.ai` call shown below. [**Jump to code**](#8-model_cost)

- `register_model`: This registers new / overrides existing models (and their pricing details) in the model cost dictionary. [**Jump to code**](#9-register_model)

- `api.litellm.ai`: Live token + price count across [all supported models](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). [**Jump to code**](#10-apilitellmai)

📣 [This is a community maintained list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). Contributions are welcome! ❤️

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

### 4. `create_pretrained_tokenizer` and `create_tokenizer`

```python
from litellm import create_pretrained_tokenizer, create_tokenizer

# get tokenizer from huggingface repo
custom_tokenizer_1 = create_pretrained_tokenizer("Xenova/llama-3-tokenizer")

# use tokenizer from json file
with open("tokenizer.json") as f:
    json_data = json.load(f)

json_str = json.dumps(json_data)

custom_tokenizer_2 = create_tokenizer(json_str)
```

### 5. `cost_per_token`

```python
from litellm import cost_per_token

prompt_tokens =  5
completion_tokens = 10
prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(model="gpt-3.5-turbo", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

print(prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar)
```

### 6. `completion_cost`

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
### 7. `get_max_tokens`

Input: Accepts a model name - e.g., gpt-3.5-turbo (to get a complete list, call litellm.model_list).
Output: Returns the maximum number of tokens allowed for the given model

```python 
from litellm import get_max_tokens 

model = "gpt-3.5-turbo"

print(get_max_tokens(model)) # Output: 4097
```

### 8. `model_cost`

* Output: Returns a dict object containing the max_tokens, input_cost_per_token, output_cost_per_token for all models on [community-maintained list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

```python 
from litellm import model_cost 

print(model_cost) # {'gpt-3.5-turbo': {'max_tokens': 4000, 'input_cost_per_token': 1.5e-06, 'output_cost_per_token': 2e-06}, ...}
```

### 9. `register_model`

* Input: Provide EITHER a model cost dictionary or a url to a hosted json blob
* Output: Returns updated model_cost dictionary + updates litellm.model_cost with model details.  

**Dictionary**
```python
import litellm

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

**Don't pull hosted model_cost_map**  
If you have firewalls, and want to just use the local copy of the model cost map, you can do so like this:
```bash
export LITELLM_LOCAL_MODEL_COST_MAP="True"
```

Note: this means you will need to upgrade to get updated pricing, and newer models.

## Performance: Disabling or Optimizing Tokenization

For high-throughput scenarios, tokenization can consume significant CPU resources. LiteLLM provides options to optimize or disable tokenization.

### Option 1: Disable Token Counter (Simple)

Completely disable tokenization for all requests:

```python
import litellm

# Disable all tokenization
litellm.disable_token_counter = True

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# response.usage will show 0 tokens (from provider if available, otherwise 0)
```

**Use when:**
- You don't need token counts or cost tracking
- Running in CPU-constrained environments
- Processing hundreds of concurrent requests

**Trade-off:** No client-side token counting, relies on provider-returned usage only.

### Option 2: Async Tokenization (Recommended)

Run tokenization in a threadpool for large inputs, preventing event loop blocking while keeping token counts for small requests:

```python
import litellm

# Configure async tokenization
litellm.async_tokenizer_threshold_bytes = 500_000  # 500KB threshold
litellm.tokenizer_threadpool_max_workers = 4       # Max 4 threads
litellm.tokenizer_timeout_seconds = 5.0            # 5s timeout

# Use async token counter
result = await litellm.async_token_counter(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": large_text}]
)
```

**How it works:**
- **Small inputs (< threshold):** Fast inline tokenization
- **Large inputs (≥ threshold):** Threadpool tokenization (non-blocking)
- **Pool saturated:** Falls back to inline (graceful degradation)
- **Timeout:** Returns 0 tokens and logs error (prevents hangs)

**Use when:**
- Processing mixed request sizes (small + large)
- Need token counts but want to prevent event loop blocking
- Running async web servers with LiteLLM

**Example: High-throughput server**

```python
import litellm
from fastapi import FastAPI

app = FastAPI()

# Configure on startup
litellm.async_tokenizer_threshold_bytes = 500_000  # 500KB
litellm.tokenizer_threadpool_max_workers = 8

@app.post("/chat")
async def chat(request):
    # Large inputs automatically use threadpool
    # Event loop stays responsive for other requests
    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=request.messages
    )
    return response
```

**Performance Impact:**

From our testing with 1MB inputs:
- **Without async tokenization:** Event loop blocked (0 Hz), sequential processing
- **With async tokenization:** Event loop responsive (88 Hz), concurrent processing
- **Speedup:** 225x faster for large batch processing
