import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Snowflake
| Property                   | Details                                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------------------|
| Description                | The Snowflake Cortex LLM REST API lets you access the COMPLETE and EMBED functions via HTTP POST requests |
| Provider Route on LiteLLM  | `snowflake/`                                                                                              |
| Link to Provider Doc       | [Snowflake ↗](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api)              |
| Base URLs                  | **Cortex LLM (v1):** `https://{account-id}.snowflakecomputing.com/api/v2/cortex/v1` <br/> **Cortex Search:** `https://{account-id}.snowflakecomputing.com/api/v2/databases/{db}/schemas/{schema}/cortex-search-services/{service}:query` <br/> **Cortex Agents:** `https://{account-id}.snowflakecomputing.com/api/v2/databases/{db}/schemas/{schema}/agents` <br/> **Cortex inference (COMPLETE/EMBED):** `https://{account-id}.snowflakecomputing.com/api/v2/cortex/inference:complete`, `https://{account-id}.snowflakecomputing.com/api/v2/cortex/inference:embed` |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions`, `/embeddings`                                                        |


## Available APIs

Snowflake Cortex exposes three REST API surfaces:

| API | Endpoint Pattern | Description | Reference |
|-----|-----------------|-------------|-----------|
| **Cortex LLM (v1)** | `POST /api/v2/cortex/v1/chat/completions` <br/> `POST /api/v2/cortex/v1/messages` | OpenAI-compatible chat completions and Anthropic-compatible messages. Supports all Cortex models. | [Cortex REST API ↗](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) |
| **Cortex Search** | `POST /api/v2/databases/{db}/schemas/{schema}/cortex-search-services/{service}:query` | Query a Cortex Search Service for low-latency semantic/hybrid search. | [Query Cortex Search ↗](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service) |
| **Cortex Agents** | `POST /api/v2/databases/{db}/schemas/{schema}/agents` <br/> `GET /api/v2/databases/{db}/schemas/{schema}/agents/{name}` <br/> `PUT /api/v2/databases/{db}/schemas/{schema}/agents/{name}` <br/> `DELETE /api/v2/databases/{db}/schemas/{schema}/agents/{name}` | Create, manage, and interact with Cortex Agent objects. | [Cortex Agents REST API ↗](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-rest-api) |

## Supported OpenAI Parameters
```
    "temperature",
    "max_tokens",
    "top_p",
    "response_format"
```

## API KEYS

Snowflake does have API keys. Instead, you access the Snowflake API with your JWT token and account identifier.

It is also possible to use [programmatic access tokens](https://docs.snowflake.com/en/user-guide/programmatic-access-tokens) (PAT). It can be defined by using 'pat/' prefix


```python
import os 
os.environ["SNOWFLAKE_JWT"] = "YOUR JWT"
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "YOUR ACCOUNT IDENTIFIER"
```
## Usage

```python
from litellm import completion, embedding

## set ENV variables
os.environ["SNOWFLAKE_JWT"] = "JWT_TOKEN"
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "YOUR ACCOUNT IDENTIFIER"

# Snowflake completion call
response = completion(
    model="snowflake/mistral-7b", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

# Snowflake embedding call
response = embedding(
    model="snowflake/mistral-7b", 
    input = ["My text"]
)

# Pass`api_key` and `account_id` as parameters
response = completion(
    model="snowflake/mistral-7b", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    account_id="AAAA-BBBB",
    api_key="JWT_TOKEN"
)

# using PAT
response = completion(
    model="snowflake/mistral-7b", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    api_key="pat/PAT_TOKEN"
)
```

## Usage with LiteLLM Proxy 

#### 1. Required env variables
```bash
export SNOWFLAKE_JWT=""
export SNOWFLAKE_ACCOUNT_ID = ""
```

#### 2. Start the proxy~
```yaml
model_list:
  - model_name: mistral-7b
    litellm_params:
        model: snowflake/mistral-7b
        api_key: YOUR_API_KEY
        api_base: https://YOUR-ACCOUNT-ID.snowflakecomputing.com/api/v2/cortex/inference:complete

```

```bash
litellm --config /path/to/config.yaml
```

#### 3. Test it
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "snowflake/mistral-7b",
      "messages": [
        {
          "role": "user",
          "content": "Hello, how are you?"
        }
      ]
    }
'
```
