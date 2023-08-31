import Image from '@theme/IdealImage';

# A/B Test LLMs - Tutorial!

Move freely from OpenAI to other models using LiteLLM. In this tutorial, we'll go through how to A/B test between OpenAI, Claude and TogetherAI using LiteLLM.

Resources: 
* [Code](https://github.com/BerriAI/litellm/tree/main/cookbook/llm-ab-test-server)
* [Sample Dashboard](https://lite-llm-abtest-ui.vercel.app/ishaan_discord@berri.ai)

# Code Walkthrough
## Main Code
This is the main piece of code that we'll write to handle our A/B test logic. We'll cover specific details in [Setup](#setup)
### Define LLMs with their A/B test ratios
In main.py set select the LLMs you want to AB test in `llm_dict` (and remember to set their API keys in the .env)!

We support 5+ providers and 100+ LLMs: https://docs.litellm.ai/docs/completion/supported

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
# SET API KEYS in .env - https://docs.litellm.ai/docs/completion/supported
os.environ["OPENAI_API_KEY"] = ""
os.environ["TOGETHERAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = "" 

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)
# cohere call
response = completion(model="together_ai/togethercomputer/llama-2-70b-chat", messages=messages)
# anthropic
response = completion(model="claude-2", messages=messages)
```

## Setup

### Install LiteLLM
```
pip install litellm
```

### Clone LiteLLM Git Repo
```
git clone https://github.com/BerriAI/litellm/
```

### Navigate to LiteLLM-A/B Test Server
```
cd litellm/cookbook/llm-ab-test-server
```

### Define LLMs with their A/B test ratios
In main.py set select the LLMs you want to AB test in `llm_dict` (and remember to set their API keys in the .env)!

We support 5+ providers and 100+ LLMs: https://docs.litellm.ai/docs/completion/supported

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
# SET API KEYS in .env - https://docs.litellm.ai/docs/completion/supported
os.environ["OPENAI_API_KEY"] = ""
os.environ["TOGETHERAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = "" 

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)
# cohere call
response = completion(model="together_ai/togethercomputer/llama-2-70b-chat", messages=messages)
# anthropic
response = completion(model="claude-2", messages=messages)
```

### Run the Server 
```
python3 main.py
```

## Testing our Server
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

View responses, costs and latency on your Log dashboard

<Image img={require('../../img/ab_test_logs.png')} />



**Note** You can turn this off by setting `litellm.use_client = False`





