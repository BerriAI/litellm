
# Dynamic TPM/RPM Allocation 

Prevent projects from gobbling too much tpm/rpm. You should use this feature when you want to reserve tpm/rpm capacity for specific projects. For example, a realtime use case should get higher priority than a different use case.

Dynamically allocate TPM/RPM quota to api keys, based on active keys in that minute. [**See Code**](https://github.com/BerriAI/litellm/blob/9bffa9a48e610cc6886fc2dce5c1815aeae2ad46/litellm/proxy/hooks/dynamic_rate_limiter.py#L125)

1. Setup config.yaml 

```yaml showLineNumbers title="config.yaml"
model_list: 
  - model_name: my-fake-model
    litellm_params:
      model: gpt-3.5-turbo
      api_key: my-fake-key
      mock_response: hello-world
      tpm: 60

litellm_settings: 
  callbacks: ["dynamic_rate_limiter_v3"]

general_settings:
  master_key: sk-1234 # OR set `LITELLM_MASTER_KEY=".."` in your .env
  database_url: postgres://.. # OR set `DATABASE_URL=".."` in your .env
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python showLineNumbers title="test.py"
"""
- Run 2 concurrent teams calling same model
- model has 60 TPM
- Mock response returns 30 total tokens / request
- Each team will only be able to make 1 request per minute
"""

import requests
from openai import OpenAI, RateLimitError

def create_key(api_key: str, base_url: str): 
    response = requests.post(
        url="{}/key/generate".format(base_url), 
        json={},
        headers={
            "Authorization": "Bearer {}".format(api_key)
        }
    )

    _response = response.json()

    return _response["key"]

key_1 = create_key(api_key="sk-1234", base_url="http://0.0.0.0:4000")
key_2 = create_key(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# call proxy with key 1 - works
openai_client_1 = OpenAI(api_key=key_1, base_url="http://0.0.0.0:4000")

response = openai_client_1.chat.completions.with_raw_response.create(
    model="my-fake-model", messages=[{"role": "user", "content": "Hello world!"}],
)

print("Headers for call 1 - {}".format(response.headers))
_response = response.parse()
print("Total tokens for call - {}".format(_response.usage.total_tokens))


# call proxy with key 2 -  works 
openai_client_2 = OpenAI(api_key=key_2, base_url="http://0.0.0.0:4000")

response = openai_client_2.chat.completions.with_raw_response.create(
    model="my-fake-model", messages=[{"role": "user", "content": "Hello world!"}],
)

print("Headers for call 2 - {}".format(response.headers))
_response = response.parse()
print("Total tokens for call - {}".format(_response.usage.total_tokens))
# call proxy with key 2 -  fails
try:  
    openai_client_2.chat.completions.with_raw_response.create(model="my-fake-model", messages=[{"role": "user", "content": "Hey, how's it going?"}])
    raise Exception("This should have failed!")
except RateLimitError as e: 
    print("This was rate limited b/c - {}".format(str(e)))

```

**Expected Response**

```
This was rate limited b/c - Error code: 429 - {'error': {'message': {'error': 'Key=<hashed_token> over available TPM=0. Model TPM=0, Active keys=2'}, 'type': 'None', 'param': 'None', 'code': 429}}
```


#### ✨ [BETA] Set Priority / Reserve Quota

Reserve tpm/rpm capacity for projects in prod.

:::tip

Reserving tpm/rpm on keys based on priority is a premium feature. Please [get an enterprise license](./enterprise.md) for it. 
:::


1. Setup config.yaml

```yaml 
model_list:
  - model_name: gpt-3.5-turbo             
    litellm_params:
      model: "gpt-3.5-turbo"       
      api_key: os.environ/OPENAI_API_KEY 
      rpm: 100   

litellm_settings:
  callbacks: [""]
  priority_reservation: {"dev": 0, "prod": 1}

general_settings:
  master_key: sk-1234 # OR set `LITELLM_MASTER_KEY=".."` in your .env
  database_url: postgres://.. # OR set `DATABASE_URL=".."` in your .env
```


priority_reservation: 
- Dict[str, float]
  - str: can be any string
  - float: from 0 to 1. Specify the % of tpm/rpm to reserve for keys of this priority.

**Start Proxy**

```
litellm --config /path/to/config.yaml
```

2. Create a key with that priority

```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer <your-master-key>' \
-H 'Content-Type: application/json' \
-D '{
	"metadata": {"priority": "dev"} # 👈 KEY CHANGE
}'
```

**Expected Response**

```
{
  ...
  "key": "sk-.."
}
```


3. Test it!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: sk-...' \ # 👈 key from step 2.
  -d '{
  "model": "gpt-3.5-turbo",
  "messages": [
      {
      "role": "user",
      "content": "what llm are you"
      }
  ],
}'
```

**Expected Response**

```
Key=... over available RPM=0. Model RPM=100, Active keys=None
```

