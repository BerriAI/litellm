# Azure API Load-Balancing

Use this if you're trying to load-balance across multiple Azure/OpenAI deployments. 

`Router` prevents failed requests, by picking the deployment which is below rate-limit and has the least amount of tokens used. 

In production, [Router connects to a Redis Cache](#redis-queue) to track usage across multiple deployments.

## Quick Start

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

# openai.ChatCompletion.create replacement
response = router.completion(model="gpt-3.5-turbo", 
				messages=[{"role": "user", "content": "Hey, how's it going?"}]

print(response)
```

## Redis Queue 

In production, we use Redis to track usage across multiple Azure deployments.

```python
router = Router(model_list=model_list, 
                redis_host=os.getenv("REDIS_HOST"), 
                redis_password=os.getenv("REDIS_PASSWORD"), 
                redis_port=os.getenv("REDIS_PORT"))

print(response)
```

## Handle Multiple Azure Deployments via OpenAI Proxy Server

#### 1. Clone repo 
```shell
git clone https://github.com/BerriAI/litellm.git
```

#### 2. Add Azure/OpenAI deployments to `secrets_template.toml`
```python 
[model."gpt-3.5-turbo"] # model name passed in /chat/completion call or `litellm --model gpt-3.5-turbo`
model_list = [{ # list of model deployments 
    "model_name": "gpt-3.5-turbo", # openai model name 
    "litellm_params": { # params for litellm completion/embedding call 
        "model": "azure/chatgpt-v-2", 
        "api_key": "my-azure-api-key-1",
        "api_version": "my-azure-api-version-1",
        "api_base": "my-azure-api-base-1"
    },
    "tpm": 240000,
    "rpm": 1800
}, { # list of model deployments 
    "model_name": "gpt-3.5-turbo", # openai model name 
    "litellm_params": { # params for litellm completion/embedding call 
        "model": "azure/chatgpt-function-calling", 
        "api_key": "my-azure-api-key-2",
        "api_version": "my-azure-api-version-2",
        "api_base": "my-azure-api-base-2"
    },
    "tpm": 240000,
    "rpm": 1800
}, {
    "model_name": "gpt-3.5-turbo", # openai model name 
    "litellm_params": { # params for litellm completion/embedding call 
        "model": "gpt-3.5-turbo", 
        "api_key": "sk-...",
    },
    "tpm": 1000000,
    "rpm": 9000
}]
```

#### 3. Run with Docker Image
```shell
docker build -t litellm . && docker run -p 8000:8000 litellm
```


