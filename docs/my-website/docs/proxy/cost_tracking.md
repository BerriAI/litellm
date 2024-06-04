import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# 💸 Spend Tracking

Track spend for keys, users, and teams across 100+ LLMs.

### How to Track Spend with LiteLLM

**Step 1**

👉 [Setup LiteLLM with a Database](https://docs.litellm.ai/docs/proxy/deploy)


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
    user="palantir",
    extra_body={
        "metadata": {
            "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"]
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
    "user": "palantir",
    "metadata": {
        "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"]
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
            "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"]
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

## API Endpoints to get Spend
#### Getting Spend Reports - To Charge Other Teams, API Keys

Use the `/global/spend/report` endpoint to get daily spend per team, with a breakdown of spend per API Key, Model

##### Example Request

```shell
curl -X GET 'http://localhost:4000/global/spend/report?start_date=2024-04-01&end_date=2024-06-30' \
  -H 'Authorization: Bearer sk-1234'
```

##### Example Response
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

#### Allowing Non-Proxy Admins to access `/spend` endpoints 

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




## Spend Tracking for Azure OpenAI Models

Set base model for cost tracking azure image-gen call

#### Image Generation 

```yaml
model_list: 
  - model_name: dall-e-3
    litellm_params:
        model: azure/dall-e-3-test
        api_version: 2023-06-01-preview
        api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
        api_key: os.environ/AZURE_API_KEY
        base_model: dall-e-3 # 👈 set dall-e-3 as base model
    model_info:
        mode: image_generation
```

#### Chat Completions / Embeddings

**Problem**: Azure returns `gpt-4` in the response when `azure/gpt-4-1106-preview` is used. This leads to inaccurate cost tracking

**Solution** ✅ :  Set `base_model` on your config so litellm uses the correct model for calculating azure cost

Get the base model name from [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

Example config with `base_model`
```yaml
model_list:
  - model_name: azure-gpt-3.5
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
    model_info:
      base_model: azure/gpt-4-1106-preview
```

## Custom Input/Output Pricing

👉 Head to [Custom Input/Output Pricing](https://docs.litellm.ai/docs/proxy/custom_pricing) to setup custom pricing or your models