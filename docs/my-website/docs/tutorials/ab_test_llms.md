# A/B Test LLMs

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

### LiteLLM Client UI
<Image img={require('../../img/ab_test_logs.png')} />



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
<Image img={require('../../img/term_output.png')} />






