import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💰 Billing

Bill users for their usage.

Requirements:
- Setup a billing plan on Lago, for usage-based billing. We recommend following their Stripe tutorial - https://docs.getlago.com/templates/per-transaction/stripe#step-1-create-billable-metrics-for-transaction

Steps:
- Connect the proxy to Lago
- Set the id you want to bill for (customers, internal users, teams)
- Start! 

## 1. Connect proxy to Lago 

Add your Lago keys to the environment

```bash
export LAGO_API_BASE="http://localhost:3000" # self-host - https://docs.getlago.com/guide/self-hosted/docker#run-the-app
export LAGO_API_KEY="3e29d607-de54-49aa-a019-ecf585729070" # Get key - https://docs.getlago.com/guide/self-hosted/docker#find-your-api-key
export LAGO_API_EVENT_CODE="openai_tokens" # name of lago billing code
```

Set 'lago' as a callback on your proxy config.yaml

```yaml
...
litellm_settings:
  callbacks: ["lago"]
```

## 2. Set the id you want to bill for

For:
- Customers (id passed via 'user' param in /chat/completion call) = 'end_user_id'
- Internal Users (id set when [creating keys](https://docs.litellm.ai/docs/proxy/virtual_keys#advanced---spend-tracking)) = 'user_id' 
- Teams (id set when [creating keys](https://docs.litellm.ai/docs/proxy/virtual_keys#advanced---spend-tracking)) = 'team_id' 

```yaml
export LAGO_API_CHARGE_BY="end_user_id" # 👈 Charge 'Customers'. Default is 'end_user_id'.
```

## 3. Start billing! 


<Tabs>
<TabItem value="customers" label="Customer Billing">

### **Curl**
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
      "user": "my_customer_id" # 👈 whatever your customer id is
    }
'
```

### **OpenAI Python SDK**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
], user="my_customer_id") # 👈 whatever your customer id is

print(response)
```

### **Langchain**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os 

os.environ["OPENAI_API_KEY"] = "anything"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model = "gpt-3.5-turbo",
    temperature=0.1,
    extra_body={
        "user": "my_customer_id"  # 👈 whatever your customer id is
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
<TabItem value="internal_user" label="Internal User (Key Owner) Billing">

1. Create a key for that user 

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"user_id": "my-unique-id"}'
```

Response Object:

```bash
{
  "key": "sk-tXL0wt5-lOOVK9sfY2UacA",
}
```

2. Make API Calls with that Key 

```python
import openai
client = openai.OpenAI(
    api_key="sk-tXL0wt5-lOOVK9sfY2UacA", # 👈 Generated key
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
```

</TabItem>
<TabItem value="teams" label="Team Billing">

1. Create a key for that team 

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"team_id": "my-unique-id"}'
```

Response Object:

```bash
{
  "key": "sk-tXL0wt5-lOOVK9sfY2UacA",
}
```

2. Make API Calls with that Key 

```python
import openai
client = openai.OpenAI(
    api_key="sk-tXL0wt5-lOOVK9sfY2UacA", # 👈 Generated key
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
```
</TabItem>
</Tabs>

**See Results on Lago**


<Image img={require('../../img/lago_2.png')}  style={{ width: '500px', height: 'auto' }} />

## Advanced - Lago Logging object 

This is what LiteLLM will log to Lagos

```
{
    "event": {
      "transaction_id": "<generated_unique_id>",
      "external_customer_id": <selected_id>, # either 'end_user_id', 'user_id', or 'team_id'. Default 'end_user_id'. 
      "code": os.getenv("LAGO_API_EVENT_CODE"), 
      "properties": {
          "input_tokens": <number>,
          "output_tokens": <number>,
          "model": <string>,
          "response_cost": <number>, # 👈 LITELLM CALCULATED RESPONSE COST - https://github.com/BerriAI/litellm/blob/d43f75150a65f91f60dc2c0c9462ce3ffc713c1f/litellm/utils.py#L1473
      }
    }
}
```
