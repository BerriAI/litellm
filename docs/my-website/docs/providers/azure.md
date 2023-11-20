# Azure OpenAI
## API Keys, Params
api_key, api_base, api_version etc can be passed directly to `litellm.completion` - see here or set as `litellm.api_key` params see here
```python
import os
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# optional
os.environ["AZURE_AD_TOKEN"] = ""
os.environ["AZURE_API_TYPE"] = ""
```

## Usage
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_Azure_OpenAI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Completion - using .env variables

```python
from litellm import completion

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = completion(
    model = "azure/<your_deployment_name>", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

### Completion - using api_key, api_base, api_version

```python
import litellm

# azure call
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    api_key = "",                                       # azure api key
    messages = [{"role": "user", "content": "good morning"}],
)
```

### Completion - using azure_ad_token, api_base, api_version

```python
import litellm

# azure call
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    azure_ad_token="", 									# azure_ad_token 
    messages = [{"role": "user", "content": "good morning"}],
)
```

## Azure OpenAI Chat Completion Models

| Model Name       | Function Call                          |
|------------------|----------------------------------------|
| gpt-4            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4-0314            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-4-0613            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4-32k            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-4-32k-0314            | `completion('azure/<your deployment name>', messages)`         |
| gpt-4-32k-0613            | `completion('azure/<your deployment name>', messages)`         | 
| gpt-3.5-turbo    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-0301    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-0613    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-16k    | `completion('azure/<your deployment name>', messages)` |
| gpt-3.5-turbo-16k-0613    | `completion('azure/<your deployment name>', messages)`

## Advanced
### Azure API Load-Balancing

Use this if you're trying to load-balance across multiple Azure/OpenAI deployments. 

`Router` prevents failed requests, by picking the deployment which is below rate-limit and has the least amount of tokens used. 

In production, [Router connects to a Redis Cache](#redis-queue) to track usage across multiple deployments.

#### Quick Start

```python
pip install litellm
```

```python
from litellm import Router

model_list = [{ # list of model deployments 
	"model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-v-2", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "azure/chatgpt-functioncalling", 
		"api_key": os.getenv("AZURE_API_KEY"),
		"api_version": os.getenv("AZURE_API_VERSION"),
		"api_base": os.getenv("AZURE_API_BASE")
	},
	"tpm": 240000,
	"rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
	"litellm_params": { # params for litellm completion/embedding call 
		"model": "gpt-3.5-turbo", 
		"api_key": os.getenv("OPENAI_API_KEY"),
	},
	"tpm": 1000000,
	"rpm": 9000
}]

router = Router(model_list=model_list)

# openai.chat.completions.create replacement
response = router.completion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```

#### Redis Queue 

```python
router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=os.getenv("REDIS_PORT"))

print(response)
```


### Parallel Function calling
See a detailed walthrough of parallel function calling with litellm [here](https://docs.litellm.ai/docs/completion/function_call)
```python
# set Azure env variables
import os
os.environ['AZURE_API_KEY'] = "" # litellm reads AZURE_API_KEY from .env and sends the request
os.environ['AZURE_API_BASE'] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ['AZURE_API_VERSION'] = "2023-07-01-preview"

import litellm
import json
# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

## Step 1: send the conversation and available functions to the model
messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = litellm.completion(
    model="azure/chatgpt-functioncalling", # model = azure/<your-azure-deployment-name>
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nLLM Response1:\n", response)
response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
print("\nTool Choice:\n", tool_calls)
```


### Authentication with Azure Active Directory Tokens (Microsoft Entra ID)
This is a walkthrough on how to use Azure Active Directory Tokens - Microsoft Entra ID to make `litellm.completion()` calls 

Step 1 - Download Azure CLI 
Installation instructons: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
```shell
brew update && brew install azure-cli
```
Step 2 - Sign in using `az`
```shell
az login --output table
```

Step 3 - Generate azure ad token
```shell
az account get-access-token --resource https://cognitiveservices.azure.com
```

In this step you should see an `accessToken` generated
```shell
{
  "accessToken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IjlHbW55RlBraGMzaE91UjIybXZTdmduTG83WSIsImtpZCI6IjlHbW55RlBraGMzaE91UjIybXZTdmduTG83WSJ9",
  "expiresOn": "2023-11-14 15:50:46.000000",
  "expires_on": 1700005846,
  "subscription": "db38de1f-4bb3..",
  "tenant": "bdfd79b3-8401-47..",
  "tokenType": "Bearer"
}
```

Step 4 - Make litellm.completion call with Azure AD token

Set `azure_ad_token` = `accessToken` from step 3 or set `os.environ['AZURE_AD_TOKEN']`

```python
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    azure_ad_token="", 									# your accessToken from step 3 
    messages = [{"role": "user", "content": "good morning"}],
)

```
