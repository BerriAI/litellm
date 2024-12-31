import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 💰 Setting Team Budgets

Track spend, set budgets for your Internal Team

## Setting Monthly Team Budgets

### 1. Create a team 
- Set `max_budget=000000001` ($ value the team is allowed to spend)
- Set `budget_duration="1d"` (How frequently the budget should update)

<Tabs>

<TabItem value="API" label="API">

Create a new team and set `max_budget` and `budget_duration`
```shell
curl -X POST 'http://0.0.0.0:4000/team/new' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{
            "team_alias": "QA Prod Bot", 
            "max_budget": 0.000000001, 
            "budget_duration": "1d"
        }' 
```

Response
```shell
{
 "team_alias": "QA Prod Bot",
 "team_id": "de35b29e-6ca8-4f47-b804-2b79d07aa99a",
 "max_budget": 0.0001,
 "budget_duration": "1d",
 "budget_reset_at": "2024-06-14T22:48:36.594000Z"
}  
```
</TabItem>

<TabItem value="UI" label="Admin UI">
<Image img={require('../../img/create_team_gif_good.gif')} />

</TabItem>


</Tabs>

Possible values for `budget_duration`

| `budget_duration` | When Budget will reset |
| --- | --- |
| `budget_duration="1s"` | every 1 second |
| `budget_duration="1m"` | every 1 min |
| `budget_duration="1h"` | every 1 hour |
| `budget_duration="1d"` | every 1 day |
| `budget_duration="30d"` | every 1 month |


### 2. Create a key for the `team`

Create a key for Team=`QA Prod Bot` and `team_id="de35b29e-6ca8-4f47-b804-2b79d07aa99a"` from Step 1 

<Tabs>

<TabItem value="api" label="API">

💡 **The Budget for Team="QA Prod Bot" budget will apply to this team**

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
     -H 'Authorization: Bearer sk-1234' \
     -H 'Content-Type: application/json' \
     -d '{"team_id": "de35b29e-6ca8-4f47-b804-2b79d07aa99a"}'
```

Response

```shell
{"team_id":"de35b29e-6ca8-4f47-b804-2b79d07aa99a", "key":"sk-5qtncoYjzRcxMM4bDRktNQ"}
```
</TabItem>

<TabItem value="UI" label="Admin UI">
<Image img={require('../../img/create_key_in_team.gif')} />
</TabItem>

</Tabs>

### 3. Test It

Use the key from step 2 and run this Request twice
<Tabs>

<TabItem value="api" label="API">

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
     -H 'Authorization: Bearer sk-mso-JSykEGri86KyOvgxBw' \
     -H 'Content-Type: application/json' \
     -d ' {
           "model": "llama3",
           "messages": [
             {
               "role": "user",
               "content": "hi"
             }
           ]
         }'
```

On the 2nd response - expect to see the following exception

```shell
{
 "error": {
   "message": "Budget has been exceeded! Current cost: 3.5e-06, Max budget: 1e-09",
   "type": "auth_error",
   "param": null,
   "code": 400
 }
}
```

</TabItem>

<TabItem value="UI" label="Admin UI">
<Image img={require('../../img/test_key_budget.gif')} />
</TabItem>
</Tabs>

## Advanced

### Prometheus metrics for `remaining_budget`

[More info about Prometheus metrics here](https://docs.litellm.ai/docs/proxy/prometheus)

You'll need the following in your proxy config.yaml

```yaml
litellm_settings:
  success_callback: ["prometheus"]
  failure_callback: ["prometheus"]
```

Expect to see this metric on prometheus to track the Remaining Budget for the team

```shell
litellm_remaining_team_budget_metric{team_alias="QA Prod Bot",team_id="de35b29e-6ca8-4f47-b804-2b79d07aa99a"} 9.699999999999992e-06
```


### Dynamic TPM/RPM Allocation 

Prevent projects from gobbling too much tpm/rpm.

Dynamically allocate TPM/RPM quota to api keys, based on active keys in that minute. [**See Code**](https://github.com/BerriAI/litellm/blob/9bffa9a48e610cc6886fc2dce5c1815aeae2ad46/litellm/proxy/hooks/dynamic_rate_limiter.py#L125)

1. Setup config.yaml 

```yaml 
model_list: 
  - model_name: my-fake-model
    litellm_params:
      model: gpt-3.5-turbo
      api_key: my-fake-key
      mock_response: hello-world
      tpm: 60

litellm_settings: 
  callbacks: ["dynamic_rate_limiter"]

general_settings:
  master_key: sk-1234 # OR set `LITELLM_MASTER_KEY=".."` in your .env
  database_url: postgres://.. # OR set `DATABASE_URL=".."` in your .env
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python
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
  callbacks: ["dynamic_rate_limiter"]
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
  -D '{
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

