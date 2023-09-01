import Image from '@theme/IdealImage';

# A/B Test LLMs - Tutorial!

Move freely from OpenAI to other models using LiteLLM. In this tutorial, we'll go through how to A/B test between OpenAI, Claude and TogetherAI using LiteLLM.

Resources: 
* [Code](https://github.com/BerriAI/litellm/tree/main/cookbook/llm-ab-test-server)
* [Sample Dashboard](https://lite-llm-abtest-ui.vercel.app/ishaan_discord@berri.ai)

# Code Walkthrough
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

### Select LLM + Make Completion call
Call the model using litellm.completion_with_split_tests, this uses the weights passed in to randomly select one of your provided models. [See implementation code](https://github.com/BerriAI/litellm/blob/9ccdbcbd6f14dd18827f59f8a1f9fd52d70443bb/litellm/utils.py#L1928)

```python
from litellm import completion_with_split_tests

response = completion_with_split_tests(model=llm_dict, messages=[{ "content": "Hello, how are you?","role": "user"}])

```

### Viewing Logs, Feedback 
In order to view logs set `litellm.token=<your-email>`
```python
import litellm
litellm.token='ishaan_discord@berri.ai'
```

Here is what your logs dashboard looks like:
<Image img={require('../../img/ab_test_logs.png')} />

Your logs will be available at: 
https://lite-llm-abtest-nckmhi7ue-clerkieai.vercel.app/your-token

### Live Demo UI
👉https://lite-llm-abtest-nckmhi7ue-clerkieai.vercel.app/ishaan_discord@berri.ai

## Viewing Responses + Custom Scores 
LiteLLM UI allows you to view responses and set custom scores for each response

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




