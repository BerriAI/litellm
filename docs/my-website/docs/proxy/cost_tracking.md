import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# 💸 Spend Tracking

Track spend for keys, users, and teams across 100+ LLMs.

LiteLLM automatically tracks spend for all known models. See our [model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

### How to Track Spend with LiteLLM

**Step 1**

👉 [Setup LiteLLM with a Database](https://docs.litellm.ai/docs/proxy/virtual_keys#setup)

**Step2** Send `/chat/completions` request

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="llama3",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    user="palantir", # OPTIONAL: pass user to track spend by user
    extra_body={
        "metadata": {
            "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"] # ENTERPRISE: pass tags to track spend by tags
        }
    }
)

print(response)
```

</TabItem>

<TabItem value="Curl" label="Curl Request">

Pass `metadata` as part of the request body

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --data '{
    "model": "llama3",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "user": "palantir", # OPTIONAL: pass user to track spend by user
    "metadata": {
        "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"] # ENTERPRISE: pass tags to track spend by tags
    }
}'
```

</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os

os.environ["OPENAI_API_KEY"] = "sk-1234"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model = "llama3",
    user="palantir",
    extra_body={
        "metadata": {
            "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"] # ENTERPRISE: pass tags to track spend by tags
        }
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>

**Step3 - Verify Spend Tracked**
That's IT. Now Verify your spend was tracked

<Tabs>
<TabItem value="curl" label="Response Headers">

Expect to see `x-litellm-response-cost` in the response headers with calculated cost

<Image img={require('../../img/response_cost_img.png')} />

</TabItem>
<TabItem value="db" label="DB + UI">

The following spend gets tracked in Table `LiteLLM_SpendLogs`

```json
{
  "api_key": "fe6b0cab4ff5a5a8df823196cc8a450*****",                            # Hash of API Key used
  "user": "default_user",                                                       # Internal User (LiteLLM_UserTable) that owns `api_key=sk-1234`.
  "team_id": "e8d1460f-846c-45d7-9b43-55f3cc52ac32",                            # Team (LiteLLM_TeamTable) that owns `api_key=sk-1234`
  "request_tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"],# Tags sent in request
  "end_user": "palantir",                                                       # Customer - the `user` sent in the request
  "model_group": "llama3",                                                      # "model" passed to LiteLLM
  "api_base": "https://api.groq.com/openai/v1/",                                # "api_base" of model used by LiteLLM
  "spend": 0.000002,                                                            # Spend in $
  "total_tokens": 100,
  "completion_tokens": 80,
  "prompt_tokens": 20,

}
```

Navigate to the Usage Tab on the LiteLLM UI (found on https://your-proxy-endpoint/ui) and verify you see spend tracked under `Usage`

<Image img={require('../../img/admin_ui_spend.png')} />

</TabItem>
</Tabs>

### Allowing Non-Proxy Admins to access `/spend` endpoints

Use this when you want non-proxy admins to access `/spend` endpoints

:::info

Schedule a [meeting with us to get your Enterprise License](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

##### Create Key

Create Key with with `permissions={"get_spend_routes": true}`

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "permissions": {"get_spend_routes": true}
    }'
```

##### Use generated key on `/spend` endpoints

Access spend Routes with newly generate keys

```shell
curl -X GET 'http://localhost:4000/global/spend/report?start_date=2024-04-01&end_date=2024-06-30' \
  -H 'Authorization: Bearer sk-H16BKvrSNConSsBYLGc_7A'
```

#### Reset Team, API Key Spend - MASTER KEY ONLY

Use `/global/spend/reset` if you want to:

- Reset the Spend for all API Keys, Teams. The `spend` for ALL Teams and Keys in `LiteLLM_TeamTable` and `LiteLLM_VerificationToken` will be set to `spend=0`

- LiteLLM will maintain all the logs in `LiteLLMSpendLogs` for Auditing Purposes

##### Request

Only the `LITELLM_MASTER_KEY` you set can access this route

```shell
curl -X POST \
  'http://localhost:4000/global/spend/reset' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json'
```

##### Expected Responses

```shell
{"message":"Spend for all API Keys and Teams reset successfully","status":"success"}
```

## Total spend per user

Assuming you have been issuing keys for end users, and setting their `user_id` on the key, you can check their usage.

```shell title="Total for a user API" showLineNumbers
curl -L -X GET 'http://localhost:4000/user/info?user_id=jane_smith' \
-H 'Authorization: Bearer sk-...'
```

```json title="Total for a user API Response" showLineNumbers
{
  "user_id": "jane_smith",
  "user_info": {
    "spend": 0.1
  },
  "keys": [
    {
      "token": "6e952b0efcafbb6350240db25ed534b4ec6011b3e1ba1006eb4f903461fd36f6",
      "key_name": "sk-...KE_A",
      "key_alias": "user-01882d6b-e090-776a-a587-21c63e502670-01983ddb-872f-71a3-8b3a-f9452c705483",
      "soft_budget_cooldown": false,
      "spend": 0.1,
      "expires": "2025-07-31T19:14:13.968000+00:00",
      "models": [],
      "aliases": {},
      "config": {},
      "user_id": "01982d6b-e090-776a-a587-21c63e502660",
      "team_id": "f2044fde-2293-482f-bf35-a8dab4e85c5f",
      "permissions": {},
      "max_parallel_requests": null,
      "metadata": {},
      "blocked": null,
      "tpm_limit": null,
      "rpm_limit": null,
      "max_budget": null,
      "budget_duration": null,
      "budget_reset_at": null,
      "allowed_cache_controls": [],
      "allowed_routes": [],
      "model_spend": {},
      "model_max_budget": {},
      "budget_id": null,
      "organization_id": null,
      "object_permission_id": null,
      "created_at": "2025-07-24T19:14:13.970000Z",
      "created_by": "582b168f-fc11-4e14-ad6a-cf4bb3656ddc",
      "updated_at": "2025-07-24T19:14:13.970000Z",
      "updated_by": "582b168f-fc11-4e14-ad6a-cf4bb3656ddc",
      "litellm_budget_table": null,
      "litellm_organization_table": null,
      "object_permission": null,
      "team_alias": null
    }
  ],
  "teams": []
}
```

**Warning**
End users can provide the `user` parameter in their request bodies, doing this will increment the cost reported via `/customer/info?end_user_id=self-declared-user`, and not for the user that owns the key as reported by that API. This means users could "avoid" having their spend tracked, through their method.
This means if you need to track user spend, and are giving end users API keys, you must always set user_id when creating their api keys, and use keys issued for that user every time you're making LLM calls on their behalf in backend services. This will track their spend.

## Daily Spend Breakdown API

Retrieve granular daily usage data for a user (by model, provider, and API key) with a single endpoint.

Example Request:

```shell title="Daily Spend Breakdown API" showLineNumbers
curl -L -X GET 'http://localhost:4000/user/daily/activity?start_date=2025-03-20&end_date=2025-03-27' \
-H 'Authorization: Bearer sk-...'
```

```json title="Daily Spend Breakdown API Response" showLineNumbers
{
    "results": [
        {
            "date": "2025-03-27",
            "metrics": {
                "spend": 0.0177072,
                "prompt_tokens": 111,
                "completion_tokens": 1711,
                "total_tokens": 1822,
                "api_requests": 11
            },
            "breakdown": {
                "models": {
                    "gpt-4o-mini": {
                        "spend": 1.095e-05,
                        "prompt_tokens": 37,
                        "completion_tokens": 9,
                        "total_tokens": 46,
                        "api_requests": 1
                },
                "providers": { "openai": { ... }, "azure_ai": { ... } },
                "api_keys": { "3126b6eaf1...": { ... } }
            }
        }
    ],
    "metadata": {
        "total_spend": 0.7274667,
        "total_prompt_tokens": 280990,
        "total_completion_tokens": 376674,
        "total_api_requests": 14
    }
}
```

### API Reference

See our [Swagger API](https://litellm-api.up.railway.app/#/Budget%20%26%20Spend%20Tracking/get_user_daily_activity_user_daily_activity_get) for more details on the `/user/daily/activity` endpoint

## Custom Tags

Requirements:

- Virtual Keys & a database should be set up, see [virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys)

**Note:** By default, LiteLLM will track `User-Agent` as a custom tag for cost tracking. This enables viewing usage for tools like Claude Code, Gemini CLI, etc.

<Image img={require('../../img/claude_cli_tag_usage.png')} />

### Client-side spend tag

<Tabs>
<TabItem value="key" label="Set on Key">

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "metadata": {
        "tags": ["tag1", "tag2", "tag3"]
    }
}

'
```

</TabItem>
<TabItem value="team" label="Set on Team">

```bash
curl -L -X POST 'http://0.0.0.0:4000/team/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "metadata": {
        "tags": ["tag1", "tag2", "tag3"]
    }
}

'
```

</TabItem>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

Set `extra_body={"metadata": { }}` to `metadata` you want to pass

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "metadata": {
            "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"] # 👈 Key Change
        }
    }
)

print(response)
```

</TabItem>

<TabItem value="openai js" label="OpenAI JS">

```js
const openai = require("openai");

async function runOpenAI() {
  const client = new openai.OpenAI({
    apiKey: "sk-1234",
    baseURL: "http://0.0.0.0:4000",
  });

  try {
    const response = await client.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "user",
          content: "this is a test request, write a short poem",
        },
      ],
      metadata: {
        tags: ["model-anthropic-claude-v2.1", "app-ishaan-prod"], // 👈 Key Change
      },
    });
    console.log(response);
  } catch (error) {
    console.log("got this exception from server");
    console.error(error);
  }
}

// Call the asynchronous function
runOpenAI();
```

</TabItem>

<TabItem value="Curl" label="Curl Request">

Pass `metadata` as part of the request body

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "metadata": {"tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"]}
}'
```

</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model = "gpt-3.5-turbo",
    temperature=0.1,
    extra_body={
        "metadata": {
            "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"]
        }
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>

### Add custom headers to spend tracking

You can add custom headers to the request to track spend and usage.

```yaml
litellm_settings:
  extra_spend_tag_headers:
    - "x-custom-header"
```

### Disable user-agent tracking

You can disable user-agent tracking by setting `litellm_settings.disable_add_user_agent_to_request_tags` to `true`.

```yaml
litellm_settings:
  disable_add_user_agent_to_request_tags: true
```

## ✨ (Enterprise) Generate Spend Reports

Use this to charge other teams, customers, users

Use the `/global/spend/report` endpoint to get spend reports

<Tabs>

<TabItem value="per team" label="Spend Per Team">

#### Example Request

👉 Key Change: Specify `group_by=team`

```shell
curl -X GET 'http://localhost:4000/global/spend/report?start_date=2024-04-01&end_date=2024-06-30&group_by=team' \
  -H 'Authorization: Bearer sk-1234'
```

#### Example Response

<Tabs>

<TabItem value="response" label="Expected Response">

```shell
[
    {
        "group_by_day": "2024-04-30T00:00:00+00:00",
        "teams": [
            {
                "team_name": "Prod Team",
                "total_spend": 0.0015265,
                "metadata": [ # see the spend by unique(key + model)
                    {
                        "model": "gpt-4",
                        "spend": 0.00123,
                        "total_tokens": 28,
                        "api_key": "88dc28.." # the hashed api key
                    },
                    {
                        "model": "gpt-4",
                        "spend": 0.00123,
                        "total_tokens": 28,
                        "api_key": "a73dc2.." # the hashed api key
                    },
                    {
                        "model": "chatgpt-v-2",
                        "spend": 0.000214,
                        "total_tokens": 122,
                        "api_key": "898c28.." # the hashed api key
                    },
                    {
                        "model": "gpt-3.5-turbo",
                        "spend": 0.0000825,
                        "total_tokens": 85,
                        "api_key": "84dc28.." # the hashed api key
                    }
                ]
            }
        ]
    }
]
```

</TabItem>

<TabItem value="py-script" label="Script to Parse Response (Python)">

```python
import requests
url = 'http://localhost:4000/global/spend/report'
params = {
    'start_date': '2023-04-01',
    'end_date': '2024-06-30'
}

headers = {
    'Authorization': 'Bearer sk-1234'
}

# Make the GET request
response = requests.get(url, headers=headers, params=params)
spend_report = response.json()

for row in spend_report:
  date = row["group_by_day"]
  teams = row["teams"]
  for team in teams:
      team_name = team["team_name"]
      total_spend = team["total_spend"]
      metadata = team["metadata"]

      print(f"Date: {date}")
      print(f"Team: {team_name}")
      print(f"Total Spend: {total_spend}")
      print("Metadata: ", metadata)
      print()
```

Output from script

```shell
# Date: 2024-05-11T00:00:00+00:00
# Team: local_test_team
# Total Spend: 0.003675099999999999
# Metadata:  [{'model': 'gpt-3.5-turbo', 'spend': 0.003675099999999999, 'api_key': 'b94d5e0bc3a71a573917fe1335dc0c14728c7016337451af9714924ff3a729db', 'total_tokens': 3105}]

# Date: 2024-05-13T00:00:00+00:00
# Team: Unassigned Team
# Total Spend: 3.4e-05
# Metadata:  [{'model': 'gpt-3.5-turbo', 'spend': 3.4e-05, 'api_key': '9569d13c9777dba68096dea49b0b03e0aaf4d2b65d4030eda9e8a2733c3cd6e0', 'total_tokens': 50}]

# Date: 2024-05-13T00:00:00+00:00
# Team: central
# Total Spend: 0.000684
# Metadata:  [{'model': 'gpt-3.5-turbo', 'spend': 0.000684, 'api_key': '0323facdf3af551594017b9ef162434a9b9a8ca1bbd9ccbd9d6ce173b1015605', 'total_tokens': 498}]

# Date: 2024-05-13T00:00:00+00:00
# Team: local_test_team
# Total Spend: 0.0005715000000000001
# Metadata:  [{'model': 'gpt-3.5-turbo', 'spend': 0.0005715000000000001, 'api_key': 'b94d5e0bc3a71a573917fe1335dc0c14728c7016337451af9714924ff3a729db', 'total_tokens': 423}]
```

</TabItem>

</Tabs>

</TabItem>

<TabItem value="per customer" label="Spend Per Customer">

:::info

Customer [this is `user` passed to `/chat/completions` request](#how-to-track-spend-with-litellm)

- [LiteLLM API key](virtual_keys.md)

:::

#### Example Request

👉 Key Change: Specify `group_by=customer`

```shell
curl -X GET 'http://localhost:4000/global/spend/report?start_date=2024-04-01&end_date=2024-06-30&group_by=customer' \
  -H 'Authorization: Bearer sk-1234'
```

#### Example Response

```shell
[
    {
        "group_by_day": "2024-04-30T00:00:00+00:00",
        "customers": [
            {
                "customer": "palantir",
                "total_spend": 0.0015265,
                "metadata": [ # see the spend by unique(key + model)
                    {
                        "model": "gpt-4",
                        "spend": 0.00123,
                        "total_tokens": 28,
                        "api_key": "88dc28.." # the hashed api key
                    },
                    {
                        "model": "gpt-4",
                        "spend": 0.00123,
                        "total_tokens": 28,
                        "api_key": "a73dc2.." # the hashed api key
                    },
                    {
                        "model": "chatgpt-v-2",
                        "spend": 0.000214,
                        "total_tokens": 122,
                        "api_key": "898c28.." # the hashed api key
                    },
                    {
                        "model": "gpt-3.5-turbo",
                        "spend": 0.0000825,
                        "total_tokens": 85,
                        "api_key": "84dc28.." # the hashed api key
                    }
                ]
            }
        ]
    }
]
```

</TabItem>

<TabItem value="per key" label="Spend for Specific API Key">

👉 Key Change: Specify `api_key=sk-1234`

```shell
curl -X GET 'http://localhost:4000/global/spend/report?start_date=2024-04-01&end_date=2024-06-30&api_key=sk-1234' \
  -H 'Authorization: Bearer sk-1234'
```

#### Example Response

```shell
[
  {
    "api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
    "total_cost": 0.3201286305151999,
    "total_input_tokens": 36.0,
    "total_output_tokens": 1593.0,
    "model_details": [
      {
        "model": "dall-e-3",
        "total_cost": 0.31999939051519993,
        "total_input_tokens": 0,
        "total_output_tokens": 0
      },
      {
        "model": "llama3-8b-8192",
        "total_cost": 0.00012924,
        "total_input_tokens": 36,
        "total_output_tokens": 1593
      }
    ]
  }
]
```

</TabItem>

<TabItem value="per user" label="Spend for Internal User (Key Owner)">

:::info

Internal User (Key Owner): This is the value of `user_id` passed when calling [`/key/generate`](https://litellm-api.up.railway.app/#/key%20management/generate_key_fn_key_generate_post)

:::

👉 Key Change: Specify `internal_user_id=ishaan`

```shell
curl -X GET 'http://localhost:4000/global/spend/report?start_date=2024-04-01&end_date=2024-12-30&internal_user_id=ishaan' \
  -H 'Authorization: Bearer sk-1234'
```

#### Example Response

```shell
[
  {
    "api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
    "total_cost": 0.00013132,
    "total_input_tokens": 105.0,
    "total_output_tokens": 872.0,
    "model_details": [
      {
        "model": "gpt-3.5-turbo-instruct",
        "total_cost": 5.85e-05,
        "total_input_tokens": 15,
        "total_output_tokens": 18
      },
      {
        "model": "llama3-8b-8192",
        "total_cost": 7.282000000000001e-05,
        "total_input_tokens": 90,
        "total_output_tokens": 854
      }
    ]
  },
  {
    "api_key": "151e85e46ab8c9c7fad090793e3fe87940213f6ae665b543ca633b0b85ba6dc6",
    "total_cost": 5.2699999999999993e-05,
    "total_input_tokens": 26.0,
    "total_output_tokens": 27.0,
    "model_details": [
      {
        "model": "gpt-3.5-turbo",
        "total_cost": 5.2499999999999995e-05,
        "total_input_tokens": 24,
        "total_output_tokens": 27
      },
      {
        "model": "text-embedding-ada-002",
        "total_cost": 2e-07,
        "total_input_tokens": 2,
        "total_output_tokens": 0
      }
    ]
  },
  {
    "api_key": "60cb83a2dcbf13531bd27a25f83546ecdb25a1a6deebe62d007999dc00e1e32a",
    "total_cost": 9.42e-06,
    "total_input_tokens": 30.0,
    "total_output_tokens": 99.0,
    "model_details": [
      {
        "model": "llama3-8b-8192",
        "total_cost": 9.42e-06,
        "total_input_tokens": 30,
        "total_output_tokens": 99
      }
    ]
  }
]
```

</TabItem>

</Tabs>

## 📊 Spend Logs API - Individual Transaction Logs

The `/spend/logs` endpoint now supports a `summarize` parameter to control data format when using date filters.

### Key Parameters

| Parameter   | Description                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------- |
| `summarize` | **New parameter**: `true` (default) = aggregated data, `false` = individual transaction logs |

### Examples

**Get individual transaction logs:**

```bash
curl -X GET "http://localhost:4000/spend/logs?start_date=2024-01-01&end_date=2024-01-02&summarize=false" \
-H "Authorization: Bearer sk-1234"
```

**Get summarized data (default):**

```bash
curl -X GET "http://localhost:4000/spend/logs?start_date=2024-01-01&end_date=2024-01-02" \
-H "Authorization: Bearer sk-1234"
```

**Use Cases:**

- `summarize=false`: Analytics dashboards, ETL processes, detailed audit trails
- `summarize=true`: Daily spending reports, high-level cost tracking (legacy behavior)

## ✨ Custom Spend Log metadata

Log specific key,value pairs as part of the metadata for a spend log

:::info

Logging specific key,value pairs in spend logs metadata is an enterprise feature.

:::

Requirements: 

- Virtual Keys & a database should be set up, see [virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys)

#### Usage - /chat/completions requests with special spend logs metadata 


<Tabs>
<TabItem value="key" label="Set on Key">

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "metadata": {
      "spend_logs_metadata": {
          "hello": "world"
      }
    }
}

'
```

</TabItem>
<TabItem value="team" label="Set on Team">

```bash
curl -L -X POST 'http://0.0.0.0:4000/team/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "metadata": {
      "spend_logs_metadata": {
          "hello": "world"
      }
    }
}

'
```

</TabItem>

<TabItem value="openai" label="OpenAI Python v1.0.0+">

Set `extra_body={"metadata": { }}` to `metadata` you want to pass

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "metadata": {
            "spend_logs_metadata": {
                "hello": "world"
            }
        }
    }
)

print(response)
```

**Using Headers:**

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# Pass spend logs metadata via headers
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_headers={
        "x-litellm-spend-logs-metadata": '{"user_id": "12345", "project_id": "proj_abc", "request_type": "chat_completion"}'
    }
)

print(response)
```

</TabItem>


<TabItem value="openai js" label="OpenAI JS">

```js
const openai = require('openai');

async function runOpenAI() {
  const client = new openai.OpenAI({
    apiKey: 'sk-1234',
    baseURL: 'http://0.0.0.0:4000'
  });

  try {
    const response = await client.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'user',
          content: "this is a test request, write a short poem"
        },
      ],
      metadata: {
        spend_logs_metadata: { // 👈 Key Change
            hello: "world"
        }
      }
    });
    console.log(response);
  } catch (error) {
    console.log("got this exception from server");
    console.error(error);
  }
}

// Call the asynchronous function
runOpenAI();
```

**Using Headers:**

```js
const openai = require('openai');

async function runOpenAI() {
  const client = new openai.OpenAI({
    apiKey: 'sk-1234',
    baseURL: 'http://0.0.0.0:4000'
  });

  try {
    const response = await client.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'user',
          content: "this is a test request, write a short poem"
        },
      ]
    }, {
      headers: {
        'x-litellm-spend-logs-metadata': '{"user_id": "12345", "project_id": "proj_abc", "request_type": "chat_completion"}'
      }
    });
    console.log(response);
  } catch (error) {
    console.log("got this exception from server");
    console.error(error);
  }
}

// Call the asynchronous function
runOpenAI();
```

</TabItem>

<TabItem value="Curl" label="Curl Request">

Pass `metadata` as part of the request body

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "metadata": {
        "spend_logs_metadata": {
            "hello": "world"
        }
    }
}'
```

</TabItem>

<TabItem value="headers" label="Using Headers">

Pass `x-litellm-spend-logs-metadata` as a request header with JSON string

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'x-litellm-spend-logs-metadata: {"user_id": "12345", "project_id": "proj_abc", "request_type": "chat_completion"}' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model = "gpt-3.5-turbo",
    temperature=0.1,
    extra_body={
        "metadata": {
            "spend_logs_metadata": {
                "hello": "world"
            }
        }
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>


#### Viewing Spend w/ custom metadata

#### `/spend/logs` Request Format 

```bash
curl -X GET "http://0.0.0.0:4000/spend/logs?request_id=<your-call-id" \ # e.g.: chatcmpl-9ZKMURhVYSi9D6r6PJ9vLcayIK0Vm
-H "Authorization: Bearer sk-1234"
```

#### `/spend/logs` Response Format
```bash
[
    {
        "request_id": "chatcmpl-9ZKMURhVYSi9D6r6PJ9vLcayIK0Vm",
        "call_type": "acompletion",
        "metadata": {
            "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
            "user_api_key_alias": null,
            "spend_logs_metadata": { # 👈 LOGGED CUSTOM METADATA
                "hello": "world"
            },
            "user_api_key_team_id": null,
            "user_api_key_user_id": "116544810872468347480",
            "user_api_key_team_alias": null
        },
    }
]
```