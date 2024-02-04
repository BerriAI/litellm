# Virtual Keys, Users
Track Spend, Set budgets and create virtual keys for the proxy

Grant other's temporary access to your proxy, with keys that expire after a set duration.


:::info

- [Deploy LiteLLM Proxy with Key Management](https://docs.litellm.ai/docs/proxy/deploy#deploy-with-database)
- Dockerfile.database for LiteLLM Proxy + Key Management [here](https://github.com/BerriAI/litellm/blob/main/Dockerfile.database)


:::

## Setup

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
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai"}}'
```


## /key/generate

### Request
```shell
curl 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "models": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
  "duration": "20m",
  "metadata": {"user": "ishaan@berri.ai"},
  "team_id": "core-infra",
  "max_budget": 10,
}'
```


Request Params:

- `duration`: *Optional[str]* - Specify the length of time the token is valid for. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
- `key_alias`: *Optional[str]* - User defined key alias
- `team_id`: *Optional[str]* - The team id of the user
- `models`: *Optional[list]* - Model_name's a user is allowed to call. (if empty, key is allowed to call all models)
- `aliases`: *Optional[dict]* - Any alias mappings, on top of anything in the config.yaml model list. - https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---upgradedowngrade-models
- `config`: *Optional[dict]* - any key-specific configs, overrides config in config.yaml
- `spend`: *Optional[int]* - Amount spent by key. Default is 0. Will be updated by proxy whenever key is used. https://docs.litellm.ai/docs/proxy/virtual_keys#managing-auth---tracking-spend
- `max_budget`: *Optional[float]* - Specify max budget for a given key.
- `max_parallel_requests`: *Optional[int]* - Rate limit a user based on the number of parallel requests. Raises 429 error, if user's parallel requests > x.
- `metadata`: *Optional[dict]* - Metadata for key, store information for key. Example metadata = {"team": "core-infra", "app": "app2", "email": "ishaan@berri.ai" }


### Response

```python
{
    "key": "sk-kdEXbIqZRwEeEiHwdg7sFA", # Bearer token
    "expires": "2023-11-19T01:38:25.838000+00:00" # datetime object
    "key_name": "sk-...7sFA" # abbreviated key string, ONLY stored in db if `allow_user_auth: true` set - [see](./ui.md)
    ...
}
```

### Upgrade/Downgrade Models 

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


### Grant Access to new model 

Use model access groups to give users access to select models, and add new ones to it over time (e.g. mistral, llama-2, etc.)

**Step 1. Assign model, access group in config.yaml**

```yaml
model_list:
  - model_name: text-embedding-ada-002
    litellm_params:
      model: azure/azure-embedding-model
      api_base: "os.environ/AZURE_API_BASE"
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2023-07-01-preview"
    model_info:
      access_groups: ["beta-models"] # 👈 Model Access Group
```

**Step 2. Create key with access group**

```bash
curl --location 'http://localhost:8000/key/generate' \
-H 'Authorization: Bearer <your-master-key>' \
-H 'Content-Type: application/json' \
-d '{"models": ["beta-models"], # 👈 Model Access Group
			"max_budget": 0,}'
```


## /key/info

### Request
```shell
curl -X GET "http://0.0.0.0:8000/key/info?key=sk-02Wr4IAlN3NvPXvL5JVvDA" \
-H "Authorization: Bearer sk-1234"
```

Request Params:
- key: str - The key you want the info for

### Response

`token` is the hashed key (The DB stores the hashed key for security)
```json
{
  "key": "sk-02Wr4IAlN3NvPXvL5JVvDA",
  "info": {
    "token": "80321a12d03412c527f2bd9db5fabd746abead2e1d50b435a534432fbaca9ef5",
    "spend": 0.0,
    "expires": "2024-01-18T23:52:09.125000+00:00",
    "models": ["azure-gpt-3.5", "azure-embedding-model"],
    "aliases": {},
    "config": {},
    "user_id": "ishaan2@berri.ai",
    "team_id": "None",
    "max_parallel_requests": null,
    "metadata": {}
  }
}


```

## /key/update

### Request
```shell
curl 'http://0.0.0.0:8000/key/update' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "key": "sk-kdEXbIqZRwEeEiHwdg7sFA",
  "models": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
  "metadata": {"user": "ishaan@berri.ai"},
  "team_id": "core-infra"
}'
```

Request Params:
- key: str - The key that needs to be updated.

- models: list or null (optional) - Specify the models a token has access to. If null, then the token has access to all models on the server.

- metadata: dict or null (optional) - Pass metadata for the updated token. If null, defaults to an empty dictionary.

- team_id: str or null (optional) - Specify the team_id for the associated key.

### Response

```json
{
  "key": "sk-kdEXbIqZRwEeEiHwdg7sFA",
  "models": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
  "metadata": {
    "user": "ishaan@berri.ai"
  }
}

```


## /key/delete

### Request
```shell
curl 'http://0.0.0.0:8000/key/delete' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "keys": ["sk-kdEXbIqZRwEeEiHwdg7sFA"]
}'
```

Request Params:
- keys: List[str] - List of keys to delete

### Response

```json
{
  "deleted_keys": ["sk-kdEXbIqZRwEeEiHwdg7sFA"]
}
```

## /user/new

### Request

All [key/generate params supported](#keygenerate) for creating a user
```shell
curl 'http://0.0.0.0:4000/user/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{
  "user_id": "ishaan1",
  "user_email": "ishaan@litellm.ai",
  "user_role": "admin",
  "team_id": "cto-team",
  "max_budget": 20,
  "budget_duration": "1h"

}'
```

Request Params:

- user_id: str (optional - defaults to uuid)  - The unique identifier for the user.
- user_email: str (optional - defaults to "")  - The email address associated with the user.
- user_role: str (optional - defaults to "app_user") - The role assigned to the user. Can be "admin", "app_owner", "app_user"

**Possible `user_role` values**
```
"admin" - Maintaining the proxy and owning the overall budget
"app_owner" - employees maintaining the apps, each owner may own more than one app
"app_user" - users who know nothing about the proxy. These users get created when you pass `user` to /chat/completions
```
- team_id: str (optional - defaults to "") - The identifier for the team to which the user belongs.
- max_budget: float (optional - defaults to `null`) - The maximum budget allocated for the user. No budget checks done if `max_budget==null`
- budget_duration: str (optional - defaults to `null`) - The duration for which the budget is valid, e.g., "1h", "1d"

### Response
A key will be generated for the new user created

```shell
{
  "models": [],
  "spend": 0.0,
  "max_budget": null,
  "user_id": "ishaan1",
  "team_id": null,
  "max_parallel_requests": null,
  "metadata": {},
  "tpm_limit": null,
  "rpm_limit": null,
  "budget_duration": null,
  "allowed_cache_controls": [],
  "key_alias": null,
  "duration": null,
  "aliases": {},
  "config": {},
  "key": "sk-JflB33ucTqc2NYvNAgiBCA",
  "key_name": null,
  "expires": null
}

```

Request Params:
- keys: List[str] - List of keys to delete

### Response

```json
{
  "deleted_keys": ["sk-kdEXbIqZRwEeEiHwdg7sFA"]
}
```

## Default /key/generate params
Use this, if you need to control the default `max_budget` or any `key/generate` param per key. 

When a `/key/generate` request does not specify `max_budget`, it will use the `max_budget` specified in `default_key_generate_params`

Set `litellm_settings:default_key_generate_params`:
```yaml
litellm_settings:
  default_key_generate_params:
    max_budget: 1.5000
    models: ["azure-gpt-3.5"]
    duration:     # blank means `null`
    metadata: {"setting":"default"}
    team_id: "core-infra"
```
## Set Budgets - Per Key

Set `max_budget` in (USD $) param in the `key/generate` request. By default the `max_budget` is set to `null` and is not checked for keys

```shell
curl 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "metadata": {"user": "ishaan@berri.ai"},
  "team_id": "core-infra",
  "max_budget": 10,
}'
```

#### Expected Behaviour
- Costs Per key get auto-populated in `LiteLLM_VerificationToken` Table
- After the key crosses it's `max_budget`, requests fail

Example Request to `/chat/completions` when key has crossed budget

```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer sk-ULl_IKCVFy2EZRzQB16RUA' \
  --data ' {
  "model": "azure-gpt-3.5",
  "user": "e09b4da8-ed80-4b05-ac93-e16d9eb56fca",
  "messages": [
      {
      "role": "user",
      "content": "respond in 50 lines"
      }
  ],
}'
```


Expected Response from `/chat/completions` when key has crossed budget
```shell
{
  "detail":"Authentication Error, ExceededTokenBudget: Current spend for token: 7.2e-05; Max Budget for Token: 2e-07"
}   
```


## Set Budgets - Per User

LiteLLM exposes a `/user/new` endpoint to create budgets for users, that persist across multiple keys. 

This is documented in the swagger (live on your server root endpoint - e.g. `http://0.0.0.0:8000/`). Here's an example request. 

```shell 
curl --location 'http://localhost:8000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["azure-models"], "max_budget": 0, "user_id": "krrish3@berri.ai"}' 
```
The request is a normal `/key/generate` request body + a `max_budget` field. 

**Sample Response**

```shell
{
    "key": "sk-YF2OxDbrgd1y2KgwxmEA2w",
    "expires": "2023-12-22T09:53:13.861000Z",
    "user_id": "krrish3@berri.ai",
    "max_budget": 0.0
}
```

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
```shell
$ litellm --config /path/to/config.yaml 
```

## Custom /key/generate

If you need to add custom logic before generating a Proxy API Key (Example Validating `team_id`)

### 1. Write a custom `custom_generate_key_fn`


The input to the custom_generate_key_fn function is a single parameter: `data` [(Type: GenerateKeyRequest)](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/_types.py#L125)

The output of your `custom_generate_key_fn` should be a dictionary with the following structure
```python
{
    "decision": False,
    "message": "This violates LiteLLM Proxy Rules. No team id provided.",
}

```

- decision (Type: bool): A boolean value indicating whether the key generation is allowed (True) or not (False).

- message (Type: str, Optional): An optional message providing additional information about the decision. This field is included when the decision is False.


```python
async def custom_generate_key_fn(data: GenerateKeyRequest)-> dict:
        """
        Asynchronous function for generating a key based on the input data.

        Args:
            data (GenerateKeyRequest): The input data for key generation.

        Returns:
            dict: A dictionary containing the decision and an optional message.
            {
                "decision": False,
                "message": "This violates LiteLLM Proxy Rules. No team id provided.",
            }
        """
        
        # decide if a key should be generated or not
        print("using custom auth function!")
        data_json = data.json()  # type: ignore

        # Unpacking variables
        team_id = data_json.get("team_id")
        duration = data_json.get("duration")
        models = data_json.get("models")
        aliases = data_json.get("aliases")
        config = data_json.get("config")
        spend = data_json.get("spend")
        user_id = data_json.get("user_id")
        max_parallel_requests = data_json.get("max_parallel_requests")
        metadata = data_json.get("metadata")
        tpm_limit = data_json.get("tpm_limit")
        rpm_limit = data_json.get("rpm_limit")

        if team_id is not None and team_id == "litellm-core-infra@gmail.com":
            # only team_id="litellm-core-infra@gmail.com" can make keys
            return {
                "decision": True,
            }
        else:
            print("Failed custom auth")
            return {
                "decision": False,
                "message": "This violates LiteLLM Proxy Rules. No team id provided.",
            }
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
  custom_key_generate: custom_auth.custom_generate_key_fn
```




## [BETA] Dynamo DB 

Only live in `v1.16.21.dev1`. 

### Step 1. Save keys to env

```shell
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
    "user_table_name": "your-user-table",
    "key_table_name": "your-token-table",
    "config_table_name": "your-config-table"
  }
```

### Step 3. Generate Key

```bash
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "aliases": {"mistral-7b": "gpt-3.5-turbo"}, "duration": null}'
```