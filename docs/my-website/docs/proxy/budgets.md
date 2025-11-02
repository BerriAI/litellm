import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Budget Guide

Budgets in LiteLLM help you control spending on LLM API calls by setting spending limits at different levels. You can apply budgets to virtual keys, users, teams, team members, and end customers. When a budget is exceeded, requests can either fail (hard budget) or continue with warnings (soft budget). Budgets support automatic resets based on time periods (daily, weekly, monthly, etc.) and track spend in real-time.

# Budgets, Rate Limits

:::info **Budget Setup Options**
**Personal budgets**: Create virtual keys without team_id for individual spending limits

**Team budgets**: Add team_id to virtual keys to utilize a team's shared budget

**Team member budgets**: Set individual spending limits within the team's shared budget

***If a key belongs to a team, the team budget is applied, not the user's personal budget.***
:::

Requirements: 

- Need to a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc) [**See Setup**](./virtual_keys.md#setup)


## Set Budgets

### Global Proxy

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
curl --location 'http://0.0.0.0:4000/chat/completions' \
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

### Team

You can:
- Add budgets to Teams

:::info

**Step-by step tutorial on setting, resetting budgets on Teams here (API or using Admin UI)**

> **Prerequisite:**
> To enable team member rate limits, you must set the environment variable `EXPERIMENTAL_MULTI_INSTANCE_RATE_LIMITING=true` before starting the proxy server. Without this, team member rate limits will not be enforced.

👉 [https://docs.litellm.ai/docs/proxy/team_budgets](https://docs.litellm.ai/docs/proxy/team_budgets)

:::


#### **Add budgets to teams**
```shell 
curl --location 'http://localhost:4000/team/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_alias": "my-new-team_4",
  "members_with_roles": [{"role": "admin", "user_id": "5c4a0aa3-a1e1-43dc-bd87-3c2da8382a3a"}],
  "rpm_limit": 99
}' 
```

[**See Swagger**](https://litellm-api.up.railway.app/#/team%20management/new_team_team_new_post)

**Sample Response**

```shell
{
    "team_alias": "my-new-team_4",
    "team_id": "13e83b19-f851-43fe-8e93-f96e21033100",
    "admins": [],
    "members": [],
    "members_with_roles": [
        {
            "role": "admin",
            "user_id": "5c4a0aa3-a1e1-43dc-bd87-3c2da8382a3a"
        }
    ],
    "metadata": {},
    "tpm_limit": null,
    "rpm_limit": 99,
    "max_budget": null,
    "models": [],
    "spend": 0.0,
    "max_parallel_requests": null,
    "budget_duration": null,
    "budget_reset_at": null
}
```

#### **Add budget duration to teams**

`budget_duration`: Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

```
curl 'http://0.0.0.0:4000/team/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_alias": "my-new-team_4",
  "members_with_roles": [{"role": "admin", "user_id": "5c4a0aa3-a1e1-43dc-bd87-3c2da8382a3a"}],
  "budget_duration": 10s,
}'
```

### Team Members

Use this when you want to budget a users spend within a Team 


#### Step 1. Create User

Create a user with `user_id=ishaan`

```shell
curl --location 'http://0.0.0.0:4000/user/new' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "ishaan"
}'
```

#### Step 2. Add User to an existing Team - set `max_budget_in_team`

Set `max_budget_in_team` when adding a User to a team. We use the same `user_id` we set in Step 1

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_add' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{"team_id": "e8d1460f-846c-45d7-9b43-55f3cc52ac32", "max_budget_in_team": 0.000000000001, "member": {"role": "user", "user_id": "ishaan"}}'
```

#### Step 3. Create a Key for Team member from Step 1

Set `user_id=ishaan` from step 1

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "ishaan",
        "team_id": "e8d1460f-846c-45d7-9b43-55f3cc52ac32"
}'
```
Response from `/key/generate`

We use the `key` from this response in Step 4
```shell
{"key":"sk-RV-l2BJEZ_LYNChSx2EueQ", "models":[],"spend":0.0,"max_budget":null,"user_id":"ishaan","team_id":"e8d1460f-846c-45d7-9b43-55f3cc52ac32","max_parallel_requests":null,"metadata":{},"tpm_limit":null,"rpm_limit":null,"budget_duration":null,"allowed_cache_controls":[],"soft_budget":null,"key_alias":null,"duration":null,"aliases":{},"config":{},"permissions":{},"model_max_budget":{},"key_name":null,"expires":null,"token_id":null}% 
```

#### Step 4. Make /chat/completions requests for Team member

Use the key from step 3 for this request. After 2-3 requests expect to see The following error `ExceededBudget: Crossed spend within team` 


```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-RV-l2BJEZ_LYNChSx2EueQ' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "llama3",
    "messages": [
        {
        "role": "user",
        "content": "tes4"
        }
    ]
}'
```


### Internal User

Apply a budget across all calls an internal user (key owner) can make on the proxy. 

:::info

For keys, with a 'team_id' set, the team budget is used instead of the user's personal budget.

To apply a budget to a user within a team, use team member budgets.

:::

LiteLLM exposes a `/user/new` endpoint to create budgets for this.

You can:
- Add budgets to users [**Jump**](#add-budgets-to-users)
- Add budget durations, to reset spend [**Jump**](#add-budget-duration-to-users)

By default the `max_budget` is set to `null` and is not checked for keys

#### **Add budgets to users**
```shell 
curl --location 'http://localhost:4000/user/new' \
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

#### **Add budget duration to users**

`budget_duration`: Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

```
curl 'http://0.0.0.0:4000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_id": "core-infra", # [OPTIONAL]
  "max_budget": 10,
  "budget_duration": 10s,
}'
```

#### Create new keys for existing user

Now you can just call `/key/generate` with that user_id (i.e. krrish3@berri.ai) and:
- **Budget Check**: krrish3@berri.ai's budget (i.e. $10) will be checked for this key
- **Spend Tracking**: spend for this key will update krrish3@berri.ai's spend as well

```bash
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "user_id": "krrish3@berri.ai"}'
```

### Virtual Key

Apply a budget on a key.

You can:
- Add budgets to keys [**Jump**](#add-budgets-to-keys)
- Add budget durations, to reset spend [**Jump**](#add-budget-duration-to-keys)

**Expected Behaviour**
- Costs Per key get auto-populated in `LiteLLM_VerificationToken` Table
- After the key crosses it's `max_budget`, requests fail
- If duration set, spend is reset at the end of the duration

By default the `max_budget` is set to `null` and is not checked for keys

#### **Add budgets to keys**

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_id": "core-infra", # [OPTIONAL]
  "max_budget": 10,
}'
```

Example Request to `/chat/completions` when key has crossed budget

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
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

#### **Add budget duration to keys**

`budget_duration`: Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

```
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_id": "core-infra", # [OPTIONAL]
  "max_budget": 10,
  "budget_duration": 10s,
}'
```


### ✨ Virtual Key (Model Specific)

Apply model specific budgets on a key. Example: 
- Budget for `gpt-4o` is $0.0000001, for time period `1d` for `key = "sk-12345"`
- Budget for `gpt-4o-mini` is $10, for time period `30d` for `key = "sk-12345"`

:::info

✨ This is an Enterprise only feature [Get Started with Enterprise here](https://www.litellm.ai/#pricing)

:::


The spec for `model_max_budget` is **[`Dict[str, GenericBudgetInfo]`](#genericbudgetinfo)**

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model_max_budget": {"gpt-4o": {"budget_limit": "0.0000001", "time_period": "1d"}}
}'
```


#### Make a test request

We expect the first request to succeed, and the second request to fail since we cross the budget for `gpt-4o` on the Virtual Key

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Successful Call " value = "allowed">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <sk-generated-key>' \
--data ' {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": "testing request"
        }
      ]
    }
'
```

</TabItem>
<TabItem label="Unsuccessful call" value = "not-allowed">

Expect this to fail since since we cross the budget `model=gpt-4o` on the Virtual Key

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <sk-generated-key>' \
--data ' {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": "testing request"
        }
      ]
    }
'
```

Expected response on failure

```json
{
    "error": {
        "message": "LiteLLM Virtual Key: 9769f3f6768a199f76cc29xxxx, key_alias: None, exceeded budget for model=gpt-4o",
        "type": "budget_exceeded",
        "param": null,
        "code": "400"
    }
}
```

</TabItem>
</Tabs>


### Customers

Use this to budget `user` passed to `/chat/completions`, **without needing to create a key for every user**

**Step 1. Modify config.yaml**
Define `litellm.max_end_user_budget`
```yaml
general_settings:
  master_key: sk-1234

litellm_settings:
  max_end_user_budget: 0.0001 # budget for 'user' passed to /chat/completions
```

2. Make a /chat/completions call, pass 'user' - First call Works 
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-zi5onDRdHGD24v0Zdn7VBA' \
        --data ' {
        "model": "azure-gpt-3.5",
        "user": "ishaan3",
        "messages": [
            {
            "role": "user",
            "content": "what time is it"
            }
        ]
        }'
```

3. Make a /chat/completions call, pass 'user' - Call Fails, since 'ishaan3' over budget
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-zi5onDRdHGD24v0Zdn7VBA' \
        --data ' {
        "model": "azure-gpt-3.5",
        "user": "ishaan3",
        "messages": [
            {
            "role": "user",
            "content": "what time is it"
            }
        ]
        }'
```

Error
```shell
{"error":{"message":"Budget has been exceeded: User ishaan3 has exceeded their budget. Current spend: 0.0008869999999999999; Max Budget: 0.0001","type":"auth_error","param":"None","code":401}}%                
```

## Reset Budgets 

Reset budgets across keys/internal users/teams/customers

`budget_duration`: Budget is reset at the end of specified duration. If not set, budget is never reset. You can set duration as seconds ("30s"), minutes ("30m"), hours ("30h"), days ("30d").

<Tabs>
<TabItem value="users" label="Internal Users">

```bash
curl 'http://0.0.0.0:4000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "max_budget": 10,
  "budget_duration": 10s, # 👈 KEY CHANGE
}'
```
</TabItem>
<TabItem value="keys" label="Keys">

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "max_budget": 10,
  "budget_duration": 10s, # 👈 KEY CHANGE
}'
```

</TabItem>
<TabItem value="teams" label="Teams">

```bash
curl 'http://0.0.0.0:4000/team/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "max_budget": 10,
  "budget_duration": 10s, # 👈 KEY CHANGE
}'
```
</TabItem>
</Tabs>

**Note:** By default, the server checks for resets every 10 minutes, to minimize DB calls.

To change this, set `proxy_budget_rescheduler_min_time` and `proxy_budget_rescheduler_max_time`

E.g.: Check every 1 seconds
```yaml
general_settings: 
  proxy_budget_rescheduler_min_time: 1
  proxy_budget_rescheduler_max_time: 1
```

## Set Rate Limits 

You can set: 
- tpm limits (tokens per minute)
- rpm limits (requests per minute)
- max parallel requests
- rpm / tpm limits per model for a given key


<Tabs>
<TabItem value="per-team" label="Per Team">

Use `/team/new` or `/team/update`, to persist rate limits across multiple keys for a team.


```shell
curl --location 'http://0.0.0.0:4000/team/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"team_id": "my-prod-team", "max_parallel_requests": 10, "tpm_limit": 20, "rpm_limit": 4}' 
```

[**See Swagger**](https://litellm-api.up.railway.app/#/team%20management/new_team_team_new_post)

**Expected Response**

```json
{
    "key": "sk-sA7VDkyhlQ7m8Gt77Mbt3Q",
    "expires": "2024-01-19T01:21:12.816168",
    "team_id": "my-prod-team",
}
```

</TabItem>
<TabItem value="per-user" label="Per Internal User">

Use `/user/new` or `/user/update`, to persist rate limits across multiple keys for internal users.


```shell
curl --location 'http://0.0.0.0:4000/user/new' \
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
curl --location 'http://0.0.0.0:4000/key/generate' \
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
<TabItem value="per-key-model" label="Per API Key Per model">

**Set rate limits per model per api key**

Set `model_rpm_limit` and `model_tpm_limit` to set rate limits per model per api key

Here `gpt-4` is the `model_name` set on the [litellm config.yaml](configs.md)

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"model_rpm_limit": {"gpt-4": 2}, "model_tpm_limit": {"gpt-4":}}' 
```

**Expected Response**

```json
{
    "key": "sk-ulGNRXWtv7M0lFnnsQk0wQ",
    "expires": "2024-01-18T20:48:44.297973",
}
```

**Verify Model Rate Limits set correctly for this key**

**Make /chat/completions request check if `x-litellm-key-remaining-requests-gpt-4` returned**

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ulGNRXWtv7M0lFnnsQk0wQ" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude!ss eho ares"}
    ]
  }'
```


**Expected headers**

```shell
x-litellm-key-remaining-requests-gpt-4: 1
x-litellm-key-remaining-tokens-gpt-4: 179
```

These headers indicate:

- 1 request remaining for the GPT-4 model for key=`sk-ulGNRXWtv7M0lFnnsQk0wQ`
- 179 tokens remaining for the GPT-4 model for key=`sk-ulGNRXWtv7M0lFnnsQk0wQ`

</TabItem>
<TabItem value="per-end-user" label="For customers">

:::info 

You can also create a budget id for a customer on the UI, under the 'Rate Limits' tab.

:::

Use this to set rate limits for `user` passed to `/chat/completions`, without needing to create a key for every user

#### Step 1. Create Budget

Set a `tpm_limit` on the budget (You can also pass `rpm_limit` if needed)

```shell
curl --location 'http://0.0.0.0:4000/budget/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "budget_id" : "free-tier",
    "tpm_limit": 5
}'
```


#### Step 2. Create `Customer` with Budget

We use `budget_id="free-tier"` from Step 1 when creating this new customers

```shell
curl --location 'http://0.0.0.0:4000/customer/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "user_id" : "palantir",
    "budget_id": "free-tier"
}'
```


#### Step 3. Pass `user_id` id in `/chat/completions` requests

Pass the `user_id` from Step 2 as `user="palantir"` 

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "llama3",
    "user": "palantir",
    "messages": [
        {
        "role": "user",
        "content": "gm"
        }
    ]
}'
```


</TabItem>
</Tabs>

## Set default budget for ALL internal users 

Use this to set a default budget for users who you give keys to.

This will apply when a user has [`user_role="internal_user"`](./self_serve.md#available-roles) (set this via `/user/new` or `/user/update`). 

This will NOT apply if a key has a team_id (team budgets will apply then). [Tell us how we can improve this!](https://github.com/BerriAI/litellm/issues)

1. Define max budget in your config.yaml

```yaml
model_list: 
  - model_name: "gpt-3.5-turbo"
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  max_internal_user_budget: 0 # amount in USD
  internal_user_budget_duration: "1mo" # reset every month
```

2. Create key for user 

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{}'
```

Expected Response: 

```bash
{
  ...
  "key": "sk-X53RdxnDhzamRwjKXR4IHg"
}
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-X53RdxnDhzamRwjKXR4IHg' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hey, how's it going?"}]
}'
```

Expected Response: 

```bash
{
    "error": {
        "message": "ExceededBudget: User=<user_id> over budget. Spend=3.7e-05, Budget=0.0",
        "type": "budget_exceeded",
        "param": null,
        "code": "400"
    }
}
```

### [BETA] Multi-instance rate limiting

Enable multi-instance rate limiting with the env var `EXPERIMENTAL_MULTI_INSTANCE_RATE_LIMITING="True"`

**Important Notes:**
- Setting `EXPERIMENTAL_MULTI_INSTANCE_RATE_LIMITING="True"` is required for team member rate limits to function, not just for multi-instance scenarios.
- **Rate limits do not apply to proxy admin users.** 
- When testing rate limits, use internal user roles (non-admin) to ensure limits are enforced as expected.

Changes: 
- This moves to using async_increment instead of async_set_cache when updating current requests/tokens. 
- The in-memory cache is synced with redis every 0.01s, to avoid calling redis for every request. 
- In testing, this was found to be 2x faster than the previous implementation, and reduced drift between expected and actual fails to at most 10 requests at high-traffic (100 RPS across 3 instances). 


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
curl --location 'http://localhost:4000/user/new' \
-H 'Authorization: Bearer <your-master-key>' \
-H 'Content-Type: application/json' \
-d '{"models": ["beta-models"], # 👈 Model Access Group
			"max_budget": 0}'
```


## Create new keys for existing internal user

Just include user_id in the `/key/generate` request.

```bash
curl --location 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data '{"models": ["azure-models"], "user_id": "krrish@berri.ai"}'
```


## API Specification 

### `GenericBudgetInfo`

A Pydantic model that defines budget information with a time period and limit.

```python
class GenericBudgetInfo(BaseModel):
    budget_limit: float  # The maximum budget amount in USD
    time_period: str    # Duration string like "1d", "30d", etc.
```

#### Fields:
- `budget_limit` (float): The maximum budget amount in USD
- `time_period` (str): Duration string specifying the time period for the budget. Supported formats:
  - Seconds: "30s"
  - Minutes: "30m" 
  - Hours: "30h"
  - Days: "30d"

#### Example:
```json
{
  "budget_limit": "0.0001",
  "time_period": "1d"
}
```

## How Budgets Work Across All Levels

LiteLLM's budget system is hierarchical and flexible, allowing you to control spending at multiple organizational levels. Here's how budgets work from top to bottom:

### Budget Hierarchy

**1. Global Proxy Level**
- Applies to all requests across the entire proxy server
- Set in `config.yaml` using `max_budget` and `budget_duration`
- Every API call counts toward this global budget

**2. Organization/Team Level**
- Teams act as organizational containers with shared budgets
- Multiple users can belong to a team and share a team budget
- All keys assigned to a team use the team's budget, not individual user budgets
- Teams can have multiple members with different roles (admin, user)
- When a team budget is exceeded, all team members' requests fail

**3. Team Member Level**
- Individual users within a team can have their own spending limits (`max_budget_in_team`)
- A team member's budget is a subset of the team's total budget
- Useful for controlling individual spending within a shared team allocation
- Example: Team has $1,000/month budget, but a specific member has $100/month limit

**4. Internal User Level**
- Personal budgets for users who own API keys
- Applied when a key does NOT have a `team_id`
- Each user has their own spend tracking across all their keys
- If a user has multiple keys, the budget applies to all of them combined
- Example: User "krrish@berri.ai" has a $50/month personal budget

**5. Virtual Key Level**
- Individual API keys can have their own budgets
- Most granular level of control
- Useful for isolating spending per application or service
- Example: "Production key" has $500/month, "Testing key" has $10/month
- Response returns `x-litellm-key-remaining` headers showing remaining budget

**6. Model-Specific Budgets** ✨ (Enterprise)
- Set different budget limits per model on a single key
- Example: GPT-4o limited to $0.0001/day, GPT-4o-mini limited to $10/month
- Allows fine-grained control over expensive vs. budget-friendly models

**7. Customer/End-User Level**
- Budget customers who pass a `user` parameter in API calls
- No key creation needed - just pass `user="customer_id"` in requests
- Perfect for SaaS applications with multiple end customers
- Each customer gets their own spend tracking and budget limits
- Example: Free-tier customers limited to $0.0001/month

### Budget Precedence (Which Budget Applies?)

When multiple budgets exist, LiteLLM applies them in this order:

1. **Team Budget** (if key has `team_id`) - Takes precedence over user budget
2. **Team Member Budget** (if user is in a team with `max_budget_in_team`)
3. **Internal User Budget** (if key has `user_id` but no `team_id`)
4. **Virtual Key Budget** (if key has `max_budget`)
5. **Model-Specific Budget** (if set for that model)
6. **Global Proxy Budget** (applies to everything)

### Spend Tracking

- **Real-time Tracking**: Costs are calculated per API call and stored immediately in the database
- **Database Storage**: All spend is persisted in `LiteLLM_VerificationToken` table for auditing
- **Automatic Aggregation**: Spend automatically rolls up from keys → users → teams
- **No Double Counting**: If a key belongs to a team, team spend includes that key's spend (not counted twice)

### Budget Resets

- **Automatic Resets**: Set `budget_duration` (e.g., "10d", "30d", "1mo") and budgets reset automatically
- **Reset Timing**: By default, LiteLLM checks for resets every 10 minutes to minimize database calls
- **Custom Reset Frequency**: Configure with `proxy_budget_rescheduler_min_time` and `proxy_budget_rescheduler_max_time`
- **No Duration = No Reset**: If `budget_duration` is not set, the budget never resets (cumulative)

### Common Budget Scenarios

**Scenario 1: SaaS with Multiple Customers**
- Set `max_end_user_budget: 0.0001` globally
- Each customer gets their own budget tracked by the `user` field in requests
- No keys needed per customer

**Scenario 2: Multi-Team Organization**
- Create separate teams for different departments
- Set team budgets (e.g., Team A: $1000/mo, Team B: $500/mo)
- Team budgets override individual user budgets automatically

**Scenario 3: Development Environment Control**
- Create keys with low budgets for dev/testing
- Create separate keys with higher budgets for production
- Both keys can belong to the same user but have different limits

**Scenario 4: Expensive Model Gating**
- Use model-specific budgets to limit GPT-4 usage
- Set GPT-3.5-turbo with a higher budget for routine tasks
- Prevent accidental expensive calls from using all budget

### What Happens When Budget Exceeds?

When a budget is exceeded:
- **Hard Budget** (default): Request fails immediately with error `ExceededBudget`
- **Soft Budget** (if configured): Request succeeds but warning is logged
- **Error Response**: Returns `401` with message showing current spend vs. max budget
- **Example**: `"Budget has been exceeded: User ishaan3 has exceeded their budget. Current spend: 0.0008869999999999999; Max Budget: 0.0001"`

### Tips for Budget Management

1. **Start Conservative**: Set budgets lower than expected, then increase based on actual usage
2. **Monitor Regularly**: Use the UI dashboard to track spend across all levels
3. **Use Team Budgets**: For organizational costs, prefer team budgets over individual keys
4. **Set Durations**: Always set `budget_duration` for monthly/periodic resets (otherwise budgets accumulate forever)
5. **Test Before Production**: Create test keys with very low budgets to verify budget enforcement
6. **Combine with Rate Limits**: Use both budgets (spending) and rate limits (requests/tokens) for complete control
