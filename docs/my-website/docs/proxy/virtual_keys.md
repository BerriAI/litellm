# Key Management
Track Spend, Set budgets and create virtual keys for the proxy

Grant other's temporary access to your proxy, with keys that expire after a set duration.


:::info

- [Deploy LiteLLM Proxy with Key Management](https://docs.litellm.ai/docs/proxy/deploy#deploy-with-database)
- Dockerfile.database for LiteLLM Proxy + Key Management [here](https://github.com/BerriAI/litellm/blob/main/Dockerfile.database)


:::

## Quick Start

Requirements: 

- Need to a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)
- Set `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>` in your env 

(the proxy Dockerfile checks if the `DATABASE_URL` is set and then intializes the DB connection)

```shell
export DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>
```


You can then generate temporary keys by hitting the `/key/generate` endpoint.

[**See code**](https://github.com/BerriAI/litellm/blob/7a669a36d2689c7f7890bc9c93e04ff3c2641299/litellm/proxy/proxy_server.py#L672)

**Step 1: Save postgres db url**

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
        model: ollama/llama2
  - model_name: gpt-3.5-turbo
    litellm_params:
        model: ollama/llama2

general_settings: 
  master_key: sk-1234 # [OPTIONAL] if set all calls to proxy will require either this key or a valid generated token
  database_url: "postgresql://<user>:<password>@<host>:<port>/<dbname>"
```

**Step 2: Start litellm**

```shell
litellm --config /path/to/config.yaml
```

**Step 3: Generate temporary keys**

```shell 
curl 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
```

- `models`: *list or null (optional)* - Specify the models a token has access too. If null, then token has access to all models on server. 

- `duration`: *str or null (optional)* Specify the length of time the token is valid for. If null, default is set to 1 hour. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

- `metadata`: *dict or null (optional)* Pass metadata for the created token. If null defaults to {}

Expected response: 

```python
{
    "key": "sk-kdEXbIqZRwEeEiHwdg7sFA", # Bearer token
    "expires": "2023-11-19T01:38:25.838000+00:00" # datetime object
}
```

## Keys that don't expire

Just set duration to None. 

```bash
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "aliases": {"mistral-7b": "gpt-3.5-turbo"}, "duration": null}'
```

## Upgrade/Downgrade Models 

If a user is expected to use a given model (i.e. gpt3-5), and you want to:

- try to upgrade the request (i.e. GPT4)
- or downgrade it (i.e. Mistral)
- OR rotate the API KEY (i.e. open AI)
- OR access the same model through different end points (i.e. openAI vs openrouter vs Azure)

Here's how you can do that: 

**Step 1: Create a model group in config.yaml (save model name, api keys, etc.)**

```yaml
model_list:
  - model_name: my-free-tier
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8001
  - model_name: my-free-tier
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8002
  - model_name: my-free-tier
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8003
	- model_name: my-paid-tier
    litellm_params:
        model: gpt-4
        api_key: my-api-key
```

**Step 2: Generate a user key - enabling them access to specific models, custom model aliases, etc.**

```bash
curl -X POST "https://0.0.0.0:8000/key/generate" \
-H "Authorization: Bearer <your-master-key>" \
-H "Content-Type: application/json" \
-d '{
	"models": ["my-free-tier"], 
	"aliases": {"gpt-3.5-turbo": "my-free-tier"}, 
	"duration": "30min"
}'
```

- **How to upgrade / downgrade request?** Change the alias mapping
- **How are routing between diff keys/api bases done?** litellm handles this by shuffling between different models in the model list with the same model_name. [**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

## Tracking Spend 

You can get spend for a key by using the `/key/info` endpoint. 

```bash
curl 'http://0.0.0.0:8000/key/info?key=<user-key>' \
     -X GET \
     -H 'Authorization: Bearer <your-master-key>'
```

This is automatically updated (in USD) when calls are made to /completions, /chat/completions, /embeddings using litellm's completion_cost() function. [**See Code**](https://github.com/BerriAI/litellm/blob/1a6ea20a0bb66491968907c2bfaabb7fe45fc064/litellm/utils.py#L1654). 

**Sample response**

```python
{
    "key": "sk-tXL0wt5-lOOVK9sfY2UacA",
    "info": {
        "token": "sk-tXL0wt5-lOOVK9sfY2UacA",
        "spend": 0.0001065,
        "expires": "2023-11-24T23:19:11.131000Z",
        "models": [
            "gpt-3.5-turbo",
            "gpt-4",
            "claude-2"
        ],
        "aliases": {
            "mistral-7b": "gpt-3.5-turbo"
        },
        "config": {}
    }
}
```



## Set Budgets 

LiteLLM exposes a `/user/new` endpoint to create budgets for users, that persist across multiple keys. 

This is documented in the swagger (live on your server root endpoint - e.g. `http://0.0.0.0:8000/`). Here's an example request. 

```curl 
curl --location 'http://localhost:8000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["azure-models"], "max_budget": 0, "user_id": "krrish3@berri.ai"}' 
```
The request is a normal `/key/generate` request body + a `max_budget` field. 

**Sample Response**

```curl
{
    "key": "sk-YF2OxDbrgd1y2KgwxmEA2w",
    "expires": "2023-12-22T09:53:13.861000Z",
    "user_id": "krrish3@berri.ai",
    "max_budget": 0.0
}
```

## Custom Auth 

You can now override the default api key auth. 

Here's how: 

### 1. Create a custom auth file. 

Make sure the response type follows the `UserAPIKeyAuth` pydantic object. This is used by for logging usage specific to that user key.

```python
from litellm.proxy._types import UserAPIKeyAuth

async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth: 
    try: 
        modified_master_key = "sk-my-master-key"
        if api_key == modified_master_key:
            return UserAPIKeyAuth(api_key=api_key)
        raise Exception
    except: 
        raise Exception
```

### 2. Pass the filepath (relative to the config.yaml)

Pass the filepath to the config.yaml 

e.g. if they're both in the same dir - `./config.yaml` and `./custom_auth.py`, this is what it looks like:
```yaml 
model_list: 
  - model_name: "openai-model"
    litellm_params: 
      model: "gpt-3.5-turbo"

litellm_settings:
  drop_params: True
  set_verbose: True

general_settings:
  custom_auth: custom_auth.user_api_key_auth
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/caf2a6b279ddbe89ebd1d8f4499f65715d684851/litellm/proxy/utils.py#L122)

### 3. Start the proxy
```bash
$ litellm --config /path/to/config.yaml 
```


## [BETA] Dynamo DB 

Only live in `v1.16.21.dev1`. 

### Step 1. Save keys to env

```env
AWS_ACCESS_KEY_ID = "your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-access-key"
```

### Step 2. Add details to config 

```yaml
general_settings: 
  master_key: sk-1234
  database_type: "dynamo_db" 
  database_args: { # 👈  all args - https://github.com/BerriAI/litellm/blob/befbcbb7ac8f59835ce47415c128decf37aac328/litellm/proxy/_types.py#L190
    "billing_mode": "PAY_PER_REQUEST", 
    "region_name": "us-west-2" 
  }
```

### Step 3. Generate Key

```bash
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "aliases": {"mistral-7b": "gpt-3.5-turbo"}, "duration": null}'
```