# liteLLM Proxy Server: 50+ LLM Models, Error Handling, Caching

### Azure, Llama2, OpenAI, Claude, Hugging Face, Replicate Models

[![PyPI Version](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)
[![PyPI Version](https://img.shields.io/badge/stable%20version-v0.1.345-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/)](https://pypi.org/project/litellm/0.1.1/)
![Downloads](https://img.shields.io/pypi/dm/litellm)
[![litellm](https://img.shields.io/badge/%20%F0%9F%9A%85%20liteLLM-OpenAI%7CAzure%7CAnthropic%7CPalm%7CCohere%7CReplicate%7CHugging%20Face-blue?color=green)](https://github.com/BerriAI/litellm)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/DYqQAW?referralCode=t3ukrU)

![4BC6491E-86D0-4833-B061-9F54524B2579](https://github.com/BerriAI/litellm/assets/17561003/f5dd237b-db5e-42e1-b1ac-f05683b1d724)

## What does liteLLM proxy do

- Make `/chat/completions` requests for 50+ LLM models **Azure, OpenAI, Replicate, Anthropic, Hugging Face**

  Example: for `model` use `claude-2`, `gpt-3.5`, `gpt-4`, `command-nightly`, `stabilityai/stablecode-completion-alpha-3b-4k`

  ```json
  {
    "model": "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    "messages": [
      {
        "content": "Hello, whats the weather in San Francisco??",
        "role": "user"
      }
    ]
  }
  ```

- **Consistent Input/Output** Format
  - Call all models using the OpenAI format - `completion(model, messages)`
  - Text responses will always be available at `['choices'][0]['message']['content']`
- **Error Handling** Using Model Fallbacks (if `GPT-4` fails, try `llama2`)
- **Logging** - Log Requests, Responses and Errors to `Supabase`, `Posthog`, `Mixpanel`, `Sentry`, `LLMonitor,` `Helicone` (Any of the supported providers here: https://litellm.readthedocs.io/en/latest/advanced/

  **Example: Logs sent to Supabase**
  <img width="1015" alt="Screenshot 2023-08-11 at 4 02 46 PM" src="https://github.com/ishaan-jaff/proxy-server/assets/29436595/237557b8-ba09-4917-982c-8f3e1b2c8d08">

- **Token Usage & Spend** - Track Input + Completion tokens used + Spend/model
- **Caching** - Implementation of Semantic Caching
- **Streaming & Async Support** - Return generators to stream text responses

## API Endpoints

### `/chat/completions` (POST)

This endpoint is used to generate chat completions for 50+ support LLM API Models. Use llama2, GPT-4, Claude2 etc

#### Input

This API endpoint accepts all inputs in raw JSON and expects the following inputs

- `model` (string, required): ID of the model to use for chat completions. See all supported models [here]: (https://litellm.readthedocs.io/en/latest/supported/):
  eg `gpt-3.5-turbo`, `gpt-4`, `claude-2`, `command-nightly`, `stabilityai/stablecode-completion-alpha-3b-4k`
- `messages` (array, required): A list of messages representing the conversation context. Each message should have a `role` (system, user, assistant, or function), `content` (message text), and `name` (for function role).
- Additional Optional parameters: `temperature`, `functions`, `function_call`, `top_p`, `n`, `stream`. See the full list of supported inputs here: https://litellm.readthedocs.io/en/latest/input/

#### Example JSON body

For claude-2

```json
{
  "model": "claude-2",
  "messages": [
    {
      "content": "Hello, whats the weather in San Francisco??",
      "role": "user"
    }
  ]
}
```

### Making an API request to the Proxy Server

```python
import requests
import json

# TODO: use your URL
url = "http://localhost:5000/chat/completions"

payload = json.dumps({
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "content": "Hello, whats the weather in San Francisco??",
      "role": "user"
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)

```

### Output [Response Format]

Responses from the server are given in the following format.
All responses from the server are returned in the following format (for all LLM models). More info on output here: https://litellm.readthedocs.io/en/latest/output/

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "I'm sorry, but I don't have the capability to provide real-time weather information. However, you can easily check the weather in San Francisco by searching online or using a weather app on your phone.",
        "role": "assistant"
      }
    }
  ],
  "created": 1691790381,
  "id": "chatcmpl-7mUFZlOEgdohHRDx2UpYPRTejirzb",
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 41,
    "prompt_tokens": 16,
    "total_tokens": 57
  }
}
```

## Installation & Usage

### Running Locally

1. Clone liteLLM repository to your local machine:
   ```
   git clone https://github.com/BerriAI/liteLLM-proxy
   ```
2. Install the required dependencies using pip
   ```
   pip install -r requirements.txt
   ```
3. Set your LLM API keys
   ```
   os.environ['OPENAI_API_KEY]` = "YOUR_API_KEY"
   or
   set OPENAI_API_KEY in your .env file
   ```
4. Run the server:
   ```
   python main.py
   ```

## Deploying

1. Quick Start: Deploy on Railway

   [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/DYqQAW?referralCode=t3ukrU)

2. `GCP`, `AWS`, `Azure`
   This project includes a `Dockerfile` allowing you to build and deploy a Docker Project on your providers

# Support / Talk with founders

- [Our calendar 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / +1 (412) 618-6238
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

## Roadmap

- [ ] Support hosted db (e.g. Supabase)
- [ ] Easily send data to places like posthog and sentry.
- [ ] Add a hot-cache for project spend logs - enables fast checks for user + project limitings
- [ ] Implement user-based rate-limiting
- [ ] Spending controls per project - expose key creation endpoint
- [ ] Need to store a keys db -> mapping created keys to their alias (i.e. project name)
- [ ] Easily add new models as backups / as the entry-point (add this to the available model list)
