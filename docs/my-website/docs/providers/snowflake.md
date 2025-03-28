import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Snowflake
| Property | Details |
|-------|-------|
| Description | The Snowflake Cortex LLM REST API lets you access the COMPLETE function via HTTP POST requests|
| Provider Route on LiteLLM | `snowflake/` |
| Link to Provider Doc | [Snowflake ↗](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api) |
| Base URL | [snowflakecomputing.com](snowflakecomputing.com) |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions` |



Currently, Snowflake's REST API does not have an endpoint for `snowflake-arctic-embed` embedding models. If you want to use these embedding models with Litellm, you can call them through our Hugging Face provider. 

Find the Arctic Embed models [here](https://huggingface.co/collections/Snowflake/arctic-embed-661fd57d50fab5fc314e4c18) on Hugging Face.

## Supported OpenAI Parameters
```
    "temperature",
    "max_tokens",
    "top_p",
    "response_format"
```

## API Keys

Snowflake does have API keys. Instead, you have to access the Snowflake API with your JWT token and account identifier.

```python
import os 
os.environ["SNOWFLAKE_JWT"] = "YOUR JWT"
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "YOUR ACCOUNT IDENTIFIER"
```
## Python SDK Usage 

```python
from litellm import completion

## set ENV variables
os.environ["SNOWFLAKE_JWT"] = "YOUR JWT"
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "YOUR ACCOUNT IDENTIFIER"

# Snowflake call
response = completion(
    model="snowflake/mistral-7b", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```
### Adding an api_base: 

If you want to specify the API base, you don't have to put the full snowflake url:

e.g `https://{account-id}.snowflakecomputing.com/api/v2/cortex/inference:complete/`

Instead, you can just pass
```python
response = completion(
    model="snowflake/mistral-7b", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    api_base = "snowflakecomputing.com"
)
```

## Usage with LiteLLM Proxy 


#### 1. Start the proxy~
```yaml
model_list:
  - model_name: mistral-7b
    litellm_params:
        model: snowflake/mistral-7b
        snowflake_jwt: <JWT>
        snowflake_account_id: <ACCOUNT-ID>
        api_base: snowflakecomputing.com

```

#### 2. Start Litellm Proxy Server
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
