import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ✨ Enterprise Features - Content Mod

Features here are behind a commercial license in our `/enterprise` folder. [**See Code**](https://github.com/BerriAI/litellm/tree/main/enterprise)

:::info

[Get Started with Enterprise here](https://github.com/BerriAI/litellm/tree/main/enterprise)

:::

Features: 
- ✅ Content Moderation with LlamaGuard 
- ✅ Content Moderation with Google Text Moderations 
- ✅ Content Moderation with LLM Guard
- ✅ Reject calls from Blocked User list 
- ✅ Reject calls (incoming / outgoing) with Banned Keywords (e.g. competitors)
- ✅ Don't log/store specific requests (eg confidential LLM requests)
- ✅ Tracking Spend for Custom Tags



## Content Moderation
### Content Moderation with LlamaGuard 

Currently works with Sagemaker's LlamaGuard endpoint. 

How to enable this in your config.yaml: 

```yaml 
litellm_settings:
   callbacks: ["llamaguard_moderations"]
   llamaguard_model_name: "sagemaker/jumpstart-dft-meta-textgeneration-llama-guard-7b"
```

Make sure you have the relevant keys in your environment, eg.: 

```
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""
```

#### Customize LlamaGuard prompt 

To modify the unsafe categories llama guard evaluates against, just create your own version of [this category list](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/llamaguard_prompt.txt)

Point your proxy to it

```yaml
callbacks: ["llamaguard_moderations"]
  llamaguard_model_name: "sagemaker/jumpstart-dft-meta-textgeneration-llama-guard-7b"
  llamaguard_unsafe_content_categories: /path/to/llamaguard_prompt.txt
```

### Content Moderation with LLM Guard

Set the LLM Guard API Base in your environment 

```env
LLM_GUARD_API_BASE = "http://0.0.0.0:8192" # deployed llm guard api
```

Add `llmguard_moderations` as a callback 

```yaml
litellm_settings:
    callbacks: ["llmguard_moderations"]
```

Now you can easily test it

- Make a regular /chat/completion call 

- Check your proxy logs for any statement with `LLM Guard:`

Expected results: 

```
LLM Guard: Received response - {"sanitized_prompt": "hello world", "is_valid": true, "scanners": { "Regex": 0.0 }}
```

### Content Moderation with Google Text Moderation 

Requires your GOOGLE_APPLICATION_CREDENTIALS to be set in your .env (same as VertexAI).

How to enable this in your config.yaml: 

```yaml 
litellm_settings:
   callbacks: ["google_text_moderation"]
```

#### Set custom confidence thresholds

Google Moderations checks the test against several categories. [Source](https://cloud.google.com/natural-language/docs/moderating-text#safety_attribute_confidence_scores)

#### Set global default confidence threshold

By default this is set to 0.8. But you can override this in your config.yaml.

```yaml
litellm_settings: 
    google_moderation_confidence_threshold: 0.4 
```

#### Set category-specific confidence threshold

Set a category specific confidence threshold in your config.yaml. If none set, the global default will be used. 

```yaml
litellm_settings: 
    toxic_confidence_threshold: 0.1
```

Here are the category specific values: 

| Category | Setting |
| -------- | -------- |
| "toxic" | toxic_confidence_threshold: 0.1 | 
| "insult" | insult_confidence_threshold: 0.1 | 
| "profanity" | profanity_confidence_threshold: 0.1 | 
| "derogatory" | derogatory_confidence_threshold: 0.1 | 
| "sexual" | sexual_confidence_threshold: 0.1 | 
| "death_harm_and_tragedy" | death_harm_and_tragedy_threshold: 0.1 | 
| "violent" | violent_threshold: 0.1 | 
| "firearms_and_weapons" | firearms_and_weapons_threshold: 0.1 | 
| "public_safety" | public_safety_threshold: 0.1 | 
| "health" | health_threshold: 0.1 | 
| "religion_and_belief" | religion_and_belief_threshold: 0.1 | 
| "illicit_drugs" | illicit_drugs_threshold: 0.1 | 
| "war_and_conflict" | war_and_conflict_threshold: 0.1 | 
| "politics" | politics_threshold: 0.1 | 
| "finance" | finance_threshold: 0.1 | 
| "legal" | legal_threshold: 0.1 |


## Incognito Requests - Don't log anything

When `no-log=True`, the request will **not be logged on any callbacks** and there will be **no server logs on litellm**

```python
import openai
client = openai.OpenAI(
    api_key="anything",            # proxy api-key
    base_url="http://0.0.0.0:4000" # litellm proxy 
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
        "no-log": True
    }
)

print(response)
```


## Enable Blocked User Lists 
If any call is made to proxy with this user id, it'll be rejected - use this if you want to let users opt-out of ai features 

```yaml
litellm_settings: 
     callbacks: ["blocked_user_check"] 
     blocked_user_list: ["user_id_1", "user_id_2", ...]  # can also be a .txt filepath e.g. `/relative/path/blocked_list.txt` 
```

### How to test

<Tabs>


<TabItem value="openai" label="OpenAI Python v1.0.0+">

Set `user=<user_id>` to the user id of the user who might have opted out.

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
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
    user="user_id_1"
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl Request">

```bash 
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
      "user": "user_id_1" # this is also an openai supported param 
    }
'
```

</TabItem>
</Tabs>

:::info 

[Suggest a way to improve this](https://github.com/BerriAI/litellm/issues/new/choose)

:::

### Using via API


**Block all calls for a user id**

```
curl -X POST "http://0.0.0.0:4000/user/block" \
-H "Authorization: Bearer sk-1234" \ 
-D '{
"user_ids": [<user_id>, ...] 
}'
```

**Unblock calls for a user id**

```
curl -X POST "http://0.0.0.0:4000/user/unblock" \
-H "Authorization: Bearer sk-1234" \ 
-D '{
"user_ids": [<user_id>, ...] 
}'
```

## Enable Banned Keywords List

```yaml 
litellm_settings: 
     callbacks: ["banned_keywords"]
     banned_keywords_list: ["hello"] # can also be a .txt file - e.g.: `/relative/path/keywords.txt`
```

### Test this 

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Hello world!"
        }
      ]
    }
'
```
## Tracking Spend for Custom Tags

Requirements: 

- Virtual Keys & a database should be set up, see [virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys)

### Usage - /chat/completions requests with request tags 


<Tabs>


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
            "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"]
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


### Viewing Spend per tag

#### `/spend/tags` Request Format 
```shell
curl -X GET "http://0.0.0.0:4000/spend/tags" \
-H "Authorization: Bearer sk-1234"
```

#### `/spend/tags`Response Format
```shell
[
  {
    "individual_request_tag": "model-anthropic-claude-v2.1",
    "log_count": 6,
    "total_spend": 0.000672
  },
  {
    "individual_request_tag": "app-ishaan-local",
    "log_count": 4,
    "total_spend": 0.000448
  },
  {
    "individual_request_tag": "app-ishaan-prod",
    "log_count": 2,
    "total_spend": 0.000224
  }
]

```


<!-- ## Tracking Spend per Key

## Tracking Spend per User -->