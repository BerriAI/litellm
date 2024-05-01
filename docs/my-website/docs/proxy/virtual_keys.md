import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🔑 Virtual Keys
Track Spend, and control model access via virtual keys for the proxy

:::info

- 🔑 [UI to Generate, Edit, Delete Keys (with SSO)](https://docs.litellm.ai/docs/proxy/ui)
- [Deploy LiteLLM Proxy with Key Management](https://docs.litellm.ai/docs/proxy/deploy#deploy-with-database)
- [Dockerfile.database for LiteLLM Proxy + Key Management](https://github.com/BerriAI/litellm/blob/main/Dockerfile.database)


:::

## Setup

Requirements: 

- Need a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)
- Set `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>` in your env 
- Set a `master key`, this is your Proxy Admin key - you can use this to create other keys (🚨 must start with `sk-`).
  - ** Set on config.yaml** set your master key under `general_settings:master_key`, example below
  - ** Set env variable** set `LITELLM_MASTER_KEY`

(the proxy Dockerfile checks if the `DATABASE_URL` is set and then intializes the DB connection)

```shell
export DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>
```


You can then generate keys by hitting the `/key/generate` endpoint.

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
  master_key: sk-1234 
  database_url: "postgresql://<user>:<password>@<host>:<port>/<dbname>" # 👈 KEY CHANGE
```

**Step 2: Start litellm**

```shell
litellm --config /path/to/config.yaml
```

**Step 3: Generate keys**

```shell 
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4"], "metadata": {"user": "ishaan@berri.ai"}}'
```

## Advanced - Spend Tracking 

Get spend per:
- key - via `/key/info` [Swagger](https://litellm-api.up.railway.app/#/key%20management/info_key_fn_key_info_get)
- user - via `/user/info` [Swagger](https://litellm-api.up.railway.app/#/user%20management/user_info_user_info_get)
- team - via `/team/info` [Swagger](https://litellm-api.up.railway.app/#/team%20management/team_info_team_info_get)  
- ⏳ end-users - via `/end_user/info` - [Comment on this issue for end-user cost tracking](https://github.com/BerriAI/litellm/issues/2633)

**How is it calculated?**

The cost per model is stored [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) and calculated by the [`completion_cost`](https://github.com/BerriAI/litellm/blob/db7974f9f216ee50b53c53120d1e3fc064173b60/litellm/utils.py#L3771) function.

**How is it tracking?**

Spend is automatically tracked for the key in the "LiteLLM_VerificationTokenTable". If the key has an attached 'user_id' or 'team_id', the spend for that user is tracked in the "LiteLLM_UserTable", and team in the "LiteLLM_TeamTable".

<Tabs>
<TabItem value="key-info" label="Key Spend">

You can get spend for a key by using the `/key/info` endpoint. 

```bash
curl 'http://0.0.0.0:4000/key/info?key=<user-key>' \
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
        "spend": 0.0001065, # 👈 SPEND
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

</TabItem>
<TabItem value="user-info" label="User Spend">

**1. Create a user**

```bash
curl --location 'http://localhost:4000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{user_email: "krrish@berri.ai"}' 
```

**Expected Response**

```bash
{
    ...
    "expires": "2023-12-22T09:53:13.861000Z",
    "user_id": "my-unique-id", # 👈 unique id
    "max_budget": 0.0
}
```

**2. Create a key for that user**

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4"], "user_id": "my-unique-id"}'
```

Returns a key - `sk-...`.

**3. See spend for user**

```bash
curl 'http://0.0.0.0:4000/user/info?user_id=my-unique-id' \
     -X GET \
     -H 'Authorization: Bearer <your-master-key>'
```

Expected Response

```bash
{
  ...
  "spend": 0 # 👈 SPEND
}
```

</TabItem>
<TabItem value="team-info" label="Team Spend">

Use teams, if you want keys to be owned by multiple people (e.g. for a production app).

**1. Create a team**

```bash
curl --location 'http://localhost:4000/team/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"team_alias": "my-awesome-team"}' 
```

**Expected Response**

```bash
{
    ...
    "expires": "2023-12-22T09:53:13.861000Z",
    "team_id": "my-unique-id", # 👈 unique id
    "max_budget": 0.0
}
```

**2. Create a key for that team**

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4"], "team_id": "my-unique-id"}'
```

Returns a key - `sk-...`.

**3. See spend for team**

```bash
curl 'http://0.0.0.0:4000/team/info?team_id=my-unique-id' \
     -X GET \
     -H 'Authorization: Bearer <your-master-key>'
```

Expected Response

```bash
{
  ...
  "spend": 0 # 👈 SPEND
}
```

</TabItem>
</Tabs>

## Advanced - Model Access

### Restrict models by `team_id`
`litellm-dev` can only access `azure-gpt-3.5`

**1. Create a team via `/team/new`**
```shell
curl --location 'http://localhost:4000/team/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_alias": "litellm-dev",
  "models": ["azure-gpt-3.5"]
}' 

# returns {...,"team_id": "my-unique-id"}
```

**2. Create a key for team**
```shell
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"team_id": "my-unique-id"}'
```

**3. Test it**
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-qo992IjKOC2CHKZGRoJIGA' \
    --data '{
        "model": "BEDROCK_GROUP",
        "messages": [
            {
                "role": "user",
                "content": "hi"
            }
        ]
    }'
```

```shell
{"error":{"message":"Invalid model for team litellm-dev: BEDROCK_GROUP.  Valid models for team are: ['azure-gpt-3.5']\n\n\nTraceback (most recent call last):\n  File \"/Users/ishaanjaffer/Github/litellm/litellm/proxy/proxy_server.py\", line 2298, in chat_completion\n    _is_valid_team_configs(\n  File \"/Users/ishaanjaffer/Github/litellm/litellm/proxy/utils.py\", line 1296, in _is_valid_team_configs\n    raise Exception(\nException: Invalid model for team litellm-dev: BEDROCK_GROUP.  Valid models for team are: ['azure-gpt-3.5']\n\n","type":"None","param":"None","code":500}}%            
```         

### Model Aliases

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
curl -X POST "https://0.0.0.0:4000/key/generate" \
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
curl --location 'http://localhost:4000/key/generate' \
-H 'Authorization: Bearer <your-master-key>' \
-H 'Content-Type: application/json' \
-d '{"models": ["beta-models"], # 👈 Model Access Group
			"max_budget": 0,}'
```

## Advanced - Custom Auth 

You can now override the default api key auth.

Here's how: 

#### 1. Create a custom auth file. 

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

#### 2. Pass the filepath (relative to the config.yaml)

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

#### 3. Start the proxy
```shell
$ litellm --config /path/to/config.yaml 
```

### Custom /key/generate

If you need to add custom logic before generating a Proxy API Key (Example Validating `team_id`)

#### 1. Write a custom `custom_generate_key_fn`


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


#### 2. Pass the filepath (relative to the config.yaml)

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


## Upperbound /key/generate params
Use this, if you need to set default upperbounds for `max_budget`, `budget_duration` or any `key/generate` param per key. 

Set `litellm_settings:upperbound_key_generate_params`:
```yaml
litellm_settings:
  upperbound_key_generate_params:
    max_budget: 100 # upperbound of $100, for all /key/generate requests
    duration: "30d" # upperbound of 30 days for all /key/generate requests
```

** Expected Behavior **

- Send a `/key/generate` request with `max_budget=200`
- Key will be created with `max_budget=100` since 100 is the upper bound

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

## Endpoints

### Keys 

#### [**👉 API REFERENCE DOCS**](https://litellm-api.up.railway.app/#/key%20management/)

### Users

#### [**👉 API REFERENCE DOCS**](https://litellm-api.up.railway.app/#/user%20management/)


### Teams

#### [**👉 API REFERENCE DOCS**](https://litellm-api.up.railway.app/#/team%20management)




