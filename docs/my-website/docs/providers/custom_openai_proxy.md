# Custom API Server (OpenAI Format)

LiteLLM allows you to call your custom endpoint in the OpenAI ChatCompletion format

### API KEYS
No api keys required

### Example Usage

#### Pre-Requisites
Ensure your proxy server has the following route
Here's an example OpenAI proxy server with routes: https://replit.com/@BerriAI/openai-proxy#main.py

```python
@app.route('/chat/completions', methods=["POST"])
def chat_completion():
  print("got request for chat completion")

```

In order to use your custom OpenAI Chat Completion proxy with LiteLLM, ensure you set

* `api_base` to your proxy url, example "https://openai-proxy.berriai.repl.co"
* `custom_llm_provider` to `openai` this ensures litellm uses the `openai.ChatCompletion` to your api_base

```python
import os
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "set anything here - key is not used for proxy"

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