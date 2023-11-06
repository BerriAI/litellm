# Custom API Server (OpenAI Format)

LiteLLM allows you to call your custom endpoint in the OpenAI ChatCompletion format

## API KEYS
No api keys required

## Set up your Custom API Server
Your server should have the following Endpoints:

Here's an example OpenAI proxy server with routes: https://replit.com/@BerriAI/openai-proxy#main.py

### Required Endpoints
- POST `/chat/completions` - chat completions endpoint 

### Optional Endpoints
- POST `/completions` - completions endpoint 
- Get `/models` - available models on server
- POST `/embeddings` - creates an embedding vector representing the input text.


## Example Usage

### Call `/chat/completions`
In order to use your custom OpenAI Chat Completion proxy with LiteLLM, ensure you set

* `api_base` to your proxy url, example "https://openai-proxy.berriai.repl.co"
* `custom_llm_provider` to `openai` this ensures litellm uses the `openai.ChatCompletion` to your api_base

```python
import os
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "anything" #key is not used for proxy

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion(
    model="command-nightly", 
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    api_base="https://openai-proxy.berriai.repl.co",
    custom_llm_provider="openai",
    temperature=0.2,
    max_tokens=80,
)
print(response)
```