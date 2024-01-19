import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💰 Budgets, Rate Limits per user 

Requirements: 

- Need to a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)


## Set Budgets
LiteLLM exposes a `/user/new` endpoint to create budgets for users, that persist across multiple keys. 


```shell 
curl --location 'http://localhost:8000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["azure-models"], "max_budget": 0, "user_id": "krrish3@berri.ai"}' 
```
The request is a normal `/key/generate` request body + a `max_budget` field. 

[**See Swagger**](https://litellm-api.up.railway.app/#/user%20management/new_user_user_new_post)

**Sample Response**

```shell
{
    "key": "sk-YF2OxDbrgd1y2KgwxmEA2w",
    "expires": "2023-12-22T09:53:13.861000Z",
    "user_id": "krrish3@berri.ai",
    "max_budget": 0.0
}
```


## Set Rate Limits 

You can set: 
- max parallel requests
- tpm limits 
- rpm limits 

<Tabs>
<TabItem value="per-user" label="Per User">

Use `/user/new`, to persist rate limits across multiple keys.


```shell
curl --location 'http://0.0.0.0:8000/user/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"user_id": "krrish@berri.ai", "max_parallel_requests": 10, "tpm_limit": 20, "rpm_limit": 4}' 
```

[**See Swagger**](https://litellm-api.up.railway.app/#/user%20management/new_user_user_new_post)

**Expected Response**

```json
{
    "key": "sk-sA7VDkyhlQ7m8Gt77Mbt3Q",
    "expires": "2024-01-19T01:21:12.816168",
    "user_id": "krrish@berri.ai",
}
```

</TabItem>
<TabItem value="per-key" label="Per Key">

Use `/key/generate`, if you want them for just that key.

```shell
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"max_parallel_requests": 10, "tpm_limit": 20, "rpm_limit": 4}' 
```

**Expected Response**

```json
{
    "key": "sk-ulGNRXWtv7M0lFnnsQk0wQ",
    "expires": "2024-01-18T20:48:44.297973",
    "user_id": "78c2c8fc-c233-43b9-b0c3-eb931da27b84"  // 👈 auto-generated
}
```

</TabItem>
</Tabs>

## Grant Access to new model 

Use model access groups to give users access to select models, and add new ones to it over time (e.g. mistral, llama-2, etc.). 

Difference between doing this with `/key/generate` vs. `/user/new`? If you do it on `/user/new` it'll persist across multiple keys generated for that user.

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
curl --location 'http://localhost:8000/user/new' \
-H 'Authorization: Bearer <your-master-key>' \
-H 'Content-Type: application/json' \
-d '{"models": ["beta-models"], # 👈 Model Access Group
			"max_budget": 0}'
```


## Create new keys for existing user

Just include user_id in the `/key/generate` request.

```bash
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "user_id": "krrish@berri.ai"}'
```