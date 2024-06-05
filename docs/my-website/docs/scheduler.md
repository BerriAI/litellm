import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Request Prioritization

:::info 

Beta feature. Use for testing only. 

[Help us improve this](https://github.com/BerriAI/litellm/issues)
:::

Prioritize LLM API requests in high-traffic.

- Add request to priority queue
- Poll queue, to check if request can be made. Returns 'True':
    * if there's healthy deployments 
    * OR if request is at top of queue
- Priority - The lower the number, the higher the priority: 
    * e.g. `priority=0` > `priority=2000`

## Quick Start 

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "mock_response": "Hello world this is Macintosh!", # fakes the LLM API call
                "rpm": 1,
            },
        },
    ],
    timeout=2, # timeout request if takes > 2s
    routing_strategy="usage-based-routing-v2",
    polling_interval=0.03 # poll queue every 3ms if no healthy deployments
)

try:
    _response = await router.schedule_acompletion( # 👈 ADDS TO QUEUE + POLLS + MAKES CALL
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey!"}],
        priority=0, # 👈 LOWER IS BETTER
    )
except Exception as e:
    print("didn't make request")
```

## LiteLLM Proxy

To prioritize requests on LiteLLM Proxy call our beta openai-compatible `http://localhost:4000/queue` endpoint. 

<Tabs>
<TabItem value="curl" label="curl">

```curl 
curl -X POST 'http://localhost:4000/queue/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "model": "gpt-3.5-turbo-fake-model",
    "messages": [
        {
        "role": "user",
        "content": "what is the meaning of the universe? 1234"
        }],
    "priority": 0 👈 SET VALUE HERE
}'
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI SDK">

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
        "priority": 0 👈 SET VALUE HERE
    }
)

print(response)
```

</TabItem>
</Tabs>