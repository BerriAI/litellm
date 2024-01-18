# 💰 Budgets, Rate Limits per user 

Requirements: 

- Need to a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)


## Set Budgets
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


## Set Rate Limits 

Set max parallel requests a user can make, when you create user keys - `/key/generate`. 

```shell
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"duration": "20m", "max_parallel_requests": 1}' # 👈 max parallel requests = 1
```

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