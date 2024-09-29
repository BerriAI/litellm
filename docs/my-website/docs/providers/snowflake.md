import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Snowflake (Cortex Text Completion)

LiteLLM supports Snowflake Cortex LLM completion models.

### Required API Keys

```python
import os 
os.environ["SNOWFLAKE_API_KEY"] = "your-snowflake-connection-token"
```

### Usage
```python

from litellm import completion
from snowflake.snowpark import Session
import os

connection_parameters = {
    
    "account": os.getenv('SNOWFLAKE_ACCOUNT'),
    "user": os.getenv('SNOWFLAKE_USER'),
    "password": os.getenv('SNOWFLAKE_PASSWORD'),
    "role": os.getenv('SNOWFLAKE_ROLE'),
    "warehouse": os.getenv('SNOWFLAKE_WAREHOUSE'),
    "database": os.getenv('SNOWFLAKE_DATABASE'),
    "schema": os.getenv('SNOWFLAKE_SCHEMA'),
    "host":os.getenv('SNOWFLAKE_HOST')}  
      
# Authenticate to Snowflake + retrieve token
snowpark = Session.builder.configs(connection_parameters).create().connection.rest.token
os.environ["SNOWFLAKE_API_KEY"] = snowpark

# Snowflake Cortex call
response = completion(
    model = "snowflake/mistral-large2", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

### Usage - LiteLLM Proxy Server

Here's how to call Snowflake Cortex models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export SNOWFLAKE_API_KEY=""
```

### 2. Start the proxy 

```yaml
model_list:
  - model_name: mistral-large2
    litellm_params:
      model: snowflake/mistral-large2
      api_key: os.environ/SNOWFLAKE_API_KEY
```

### 3. Test it

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "mistral-large2",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```


## Cortex Complete Models

For a complete list of the latest models, visit the documentation (here)[https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api]

| Model Name          | Function Call                                      |
|---------------------|----------------------------------------------------|
| mistral-large2  | `response = completion(model="mistral-large2", messages=messages)` |
| mistral-large | `response = completion(model="mistral-large", messages=messages)` |
| mixtral-8x7b  | `response = completion(model="mixtral-8x7b", messages=messages)` |
| mistral-7b  | `response = completion(model="mistral-7b ", messages=messages)` |
| llama3.2-3b | `response = completion(model="llama3.2-3b", messages=messages)` |
| llama3.2-1b | `response = completion(model="llama3.2-1b", messages=messages)` |
| llama3.1-405b   | `response = completion(model="llama3.1-405b", messages=messages)` |
| llama3.1-70b   | `response = completion(model="llama3.1-70b", messages=messages)` |
| llama3.1-8b | `response = completion(model="llama3.1-8b", messages=messages)` |
| llama3-70b | `response = completion(model="llama3-70b", messages=messages)` |
| llama3-8b | `response = completion(model="llama3-8b", messages=messages)` |
| llama2-70b-chat | `response = completion(model="llama2-70b-chat", messages=messages)` |
| reka-core | `response = completion(model="reka-core", messages=messages)` |
| reka-flash | `response = completion(model="reka-flash", messages=messages)` |
| snowflake-arctic | `response = completion(model="snowflake-arctic", messages=messages)` |
| jamba-instruct | `response = completion(model="jamba-instruct", messages=messages)` |
| jamba-1.5-large | `response = completion(model="jamba-1.5-large", messages=messages)` |
| jamba-1.5-mini | `response = completion(model="jamba-1.5-mini", messages=messages)` |
