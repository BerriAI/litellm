
# Dynamic TPM/RPM Allocation 

Prevent projects from gobbling too much tpm/rpm.

Dynamically allocate TPM/RPM quota to api keys, based on active keys in that minute. [**See Code**](https://github.com/BerriAI/litellm/blob/9bffa9a48e610cc6886fc2dce5c1815aeae2ad46/litellm/proxy/hooks/dynamic_rate_limiter.py#L125)

## Quick Start Usage

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


## [BETA] Set Priority / Reserve Quota

Reserve TPM/RPM capacity for different environments or use cases. This ensures critical production workloads always have guaranteed capacity, while development or lower-priority tasks use remaining quota.

**Use Cases:**
- Production vs Development environments
- Real-time applications vs batch processing
- Critical services vs experimental features

:::tip

Reserving TPM/RPM on keys based on priority is a premium feature. Please [get an enterprise license](./enterprise.md) for it. 
:::

### How Priority Reservation Works

Priority reservation allocates a percentage of your model's total TPM/RPM to specific priority levels. Keys with higher priority get guaranteed access to their reserved quota first.

**Example Scenario:**
- Model has 10 RPM total capacity
- Priority reservation: `{"prod": 0.9, "dev": 0.1}`
- Result: Production keys get 9 RPM guaranteed, Development keys get 1 RPM guaranteed

### Configuration

#### 1. Setup config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-3.5-turbo             
    litellm_params:
      model: "gpt-3.5-turbo"       
      api_key: os.environ/OPENAI_API_KEY 
      rpm: 10   # Total model capacity

litellm_settings:
  callbacks: ["dynamic_rate_limiter_v3"]
  priority_reservation:
    "prod": 0.9 # 90% reserved for production (9 RPM)
    "dev": 0.1 # 10% reserved for development (1 RPM)
    # Alternative format:
    # "prod":
    #   type: "rpm"    # Reserve based on requests per minute
    #   value: 9       # 9 RPM = 90% of 10 RPM capacity
    # "dev":
    #   type: "tpm"    # Reserve based on tokens per minute
    #   value: 100     # 100 TPM
  priority_reservation_settings:
    default_priority: 0  # Weight (0%) assigned to keys without explicit priority metadata
    saturation_threshold: 0.50 #  A model is saturated if it has hit 50% of its RPM limit

general_settings:
  master_key: sk-1234 # OR set `LITELLM_MASTER_KEY=".."` in your .env
  database_url: postgres://.. # OR set `DATABASE_URL=".."` in your.env
```

**Configuration Details:**

`priority_reservation`: Dict[str, Union[float, PriorityReservationDict]]
- **Key (str)**: Priority level name (can be any string like "prod", "dev", "critical", etc.)
- **Value**: Either a float (0.0-1.0) or dict with `type` and `value`
  - Float: `0.9` = 90% of capacity
  - Dict: `{"type": "rpm", "value": 9}` = 9 requests/min
  - Supported types: `"percent"`, `"rpm"`, `"tpm"`

`priority_reservation_settings`: Object (Optional)
- **default_priority (float)**: Weight/percentage (0.0 to 1.0) assigned to API keys that have no priority metadata set (defaults to 0.5)
- **saturation_threshold (float)**: Saturation level (0.0 to 1.0) at which strict priority enforcement begins for a model. Saturation is calculated as `max(current_rpm/max_rpm, current_tpm/max_tpm)`. Below this threshold, generous mode allows priority borrowing from unused capacity. Above this threshold, strict mode enforces normalized priority limits.
  - Example: When model usage is low, keys can use more than their allocated share. When model usage is high, keys are strictly limited to their allocated share.

**Start Proxy**

```bash
litellm --config /path/to/config.yaml
```

#### 2. Create Keys with Priority Levels

**Production Key:**
```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
  "metadata": {"priority": "prod"}
}'
```

**Development Key:**
```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
  "metadata": {"priority": "dev"}
}'
```

**Key Without Priority (uses default_priority weight):**
```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{}'
```

**Expected Response for both:**
```json
{
  "key": "sk-...",
  "metadata": {"priority": "prod"}, // or "dev"
  ...
}
```

#### 3. Test Priority Allocation

**Test Production Key (should get 9 RPM):**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-prod-key' \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello from prod"}]
  }'
```

**Test Development Key (should get 1 RPM):**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-dev-key' \
  -d '{
    "model": "gpt-3.5-turbo", 
    "messages": [{"role": "user", "content": "Hello from dev"}]
  }'
```

### Expected Behavior

With the configuration above:

1. **Production keys** can make up to 9 requests per minute (90% of 10 RPM)
2. **Development keys** can make up to 1 request per minute (10% of 10 RPM)
3. **Keys without explicit priority** get the default_priority weight (0 = 0%), which allocates 0 requests per minute (0% of 10 RPM)
4. Named priorities in `priority_reservation` and keys with `default_priority` operate independently

**Rate Limit Error Example:**
```json
{
  "error": {
    "message": "Key=sk-dev-... over available RPM=0. Model RPM=10, Reserved RPM for priority 'dev'=1, Active keys=1",
    "type": "rate_limit_exceeded",
    "code": 429
  }
}
```

### Demo Video

This video walks through setting up dynamic rate limiting with priority reservation and locust tests to validate the behavior.

<iframe width="840" height="500" src="https://www.loom.com/embed/1b54b93139ee415d959402cc0629f3f7
" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

