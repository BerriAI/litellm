import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💰 Budgets, Rate Limits

Requirements: 

- Need to a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)


## Set Budgets

You can set budgets at 3 levels: 
- For the proxy 
- For a user 
- For a key


<Tabs>
<TabItem value="proxy" label="For Proxy">

Apply a budget across all calls on the proxy

**Step 1. Modify config.yaml**

```yaml
general_settings:
  master_key: sk-1234

litellm_settings:
  # other litellm settings
  max_budget: 0 # (float) sets max budget as $0 USD
  budget_duration: 30d # (str) frequency of reset - You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").
```

**Step 2. Start proxy**

```bash
litellm /path/to/config.yaml
```

**Step 3. Send test call**

```bash
curl --location 'http://0.0.0.0:8000/chat/completions' \
    --header 'Autherization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
}'
```
</TabItem>
<TabItem value="per-user" label="For User">

Apply a budget across multiple keys.

LiteLLM exposes a `/user/new` endpoint to create budgets for this.

You can:
- Add budgets to users [**Jump**](#add-budgets-to-users)
- Add budget durations, to reset spend [**Jump**](#add-budget-duration-to-users)

By default the `max_budget` is set to `null` and is not checked for keys

### **Add budgets to users**
```shell 
curl --location 'http://localhost:8000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["azure-models"], "max_budget": 0, "user_id": "krrish3@berri.ai"}' 
```

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

### **Add budget duration to users**

`budget_duration`: Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

```
curl 'http://0.0.0.0:8000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_id": "core-infra", # [OPTIONAL]
  "max_budget": 10,
  "budget_duration": 10s,
}'
```

### Create new keys for existing user

Now you can just call `/key/generate` with that user_id (i.e. krrish3@berri.ai) and:
- **Budget Check**: krrish3@berri.ai's budget (i.e. $10) will be checked for this key
- **Spend Tracking**: spend for this key will update krrish3@berri.ai's spend as well

```bash
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "user_id": "krrish3@berri.ai"}'
```

</TabItem>
<TabItem value="per-key" label="For Key">

Apply a budget on a key.

You can:
- Add budgets to keys [**Jump**](#add-budgets-to-keys)
- Add budget durations, to reset spend [**Jump**](#add-budget-duration-to-keys)

**Expected Behaviour**
- Costs Per key get auto-populated in `LiteLLM_VerificationToken` Table
- After the key crosses it's `max_budget`, requests fail
- If duration set, spend is reset at the end of the duration

By default the `max_budget` is set to `null` and is not checked for keys

### **Add budgets to keys**

```bash
curl 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_id": "core-infra", # [OPTIONAL]
  "max_budget": 10,
}'
```

Example Request to `/chat/completions` when key has crossed budget

```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer <generated-key>' \
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

### **Add budget duration to keys**

`budget_duration`: Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

```
curl 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_id": "core-infra", # [OPTIONAL]
  "max_budget": 10,
  "budget_duration": 10s,
}'
```

</TabItem>
</Tabs>

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