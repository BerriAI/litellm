<h1 align="center">
        🚅 LiteLLM - A/B Testing LLMs in Production
    </h1>
    <p align="center">
        <p align="center">Call all LLM APIs using the OpenAI format [Anthropic, Huggingface, Cohere, Azure OpenAI etc.]</p>
    </p>

<h4 align="center">
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://pypi.org/project/litellm/0.1.1/" target="_blank">
        <img src="https://img.shields.io/badge/stable%20version-v0.1.424-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/" alt="Stable Version">
    </a>
    <a href="https://dl.circleci.com/status-badge/redirect/gh/BerriAI/litellm/tree/main" target="_blank">
        <img src="https://dl.circleci.com/status-badge/img/gh/BerriAI/litellm/tree/main.svg?style=svg" alt="CircleCI">
    </a>
    <img src="https://img.shields.io/pypi/dm/litellm" alt="Downloads">
    <a href="https://discord.gg/wuPM9dRgDw" target="_blank">
        <img src="https://dcbadge.vercel.app/api/server/wuPM9dRgDw?style=flat">
    </a>
</h4>

<h4 align="center">
    <a href="https://docs.litellm.ai/docs/completion/supported" target="_blank">100+ Supported Models</a> |
    <a href="https://docs.litellm.ai/docs/" target="_blank">Docs</a> |
    <a href="https://litellm.ai/playground" target="_blank">Demo Website</a>
</h4>

![pika-1693451518986-1x](https://github.com/BerriAI/litellm/assets/29436595/1cbf29a3-5313-4f61-ad6e-481dd8737309)

LiteLLM allows you to call 100+ LLMs using completion

## This template server allows you to define LLMs with their A/B test ratios

```python
llm_dict = {
    "gpt-4": 0.2,
    "together_ai/togethercomputer/llama-2-70b-chat": 0.4,
    "claude-2": 0.2,
    "claude-1.2": 0.2
}
```

All models defined can be called with the same Input/Output format using litellm `completion`
```python
from litellm import completion
# SET API KEYS in .env
# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)
# cohere call
response = completion(model="command-nightly", messages=messages)
# anthropic
response = completion(model="claude-2", messages=messages)
```

This server allows you to view responses, costs and latency on your LiteLLM dashboard


# Using LiteLLM A/B Testing Server
## Setup

### Install LiteLLM
```
pip install litellm
```

Stable version
```
pip install litellm==0.1.424
```

### Clone LiteLLM Git Repo
```
git clone https://github.com/BerriAI/litellm/
```

### Navigate to LiteLLM-A/B Test Server
```
cd litellm/cookbook/llm-ab-test-server
```

### Run the Server 
```
python3 main.py
```

### Set your LLM Configs
Set your LLMs and LLM weights you want to run A/B testing with 
In main.py set your selected LLMs you want to AB test in `llm_dict`
You can A/B test more than 100+ LLMs using LiteLLM https://docs.litellm.ai/docs/completion/supported
```python
llm_dict = {
    "gpt-4": 0.2,
    "together_ai/togethercomputer/llama-2-70b-chat": 0.4,
    "claude-2": 0.2,
    "claude-1.2": 0.2
}
```

#### Setting your API Keys 
Set your LLM API keys in a .env file in the directory or set them as `os.environ` variables.

See https://docs.litellm.ai/docs/completion/supported for the format of API keys 

LiteLLM generalizes api keys to follow the following format 
`PROVIDER_API_KEY`

## Making Requests to the LiteLLM Server Locally 
The server follows the Input/Output format set by the OpenAI Chat Completions API
Here is an example request made the LiteLLM Server

### Python
```python
import requests
import json

url = "http://localhost:5000/chat/completions"

payload = json.dumps({
  "messages": [
    {
      "content": "who is CTO of litellm",
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

### Curl Command
```
curl --location 'http://localhost:5000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
                    { 
                        "content": "who is CTO of litellm",
                        "role": "user"
                    }
                ]
    
}
'
```

## Viewing Logs
After running your first `completion()` call litellm autogenerates a new logs dashboard for you. Link to your Logs dashboard is generated in the terminal / console. 

Example Terminal Output with Log Dashboard

<img width="1280" alt="Screenshot 2023-08-25 at 8 53 27 PM" src="https://github.com/BerriAI/litellm/assets/29436595/8f4cc218-a991-4988-a05c-c8e508da5d18">



# support / talk with founders
- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

# why did we build this 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere
