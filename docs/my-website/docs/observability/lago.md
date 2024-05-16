import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Lago - Usage Based Billing

[Lago](https://www.getlago.com/) offers a self-hosted and cloud, metering and usage-based billing solution.

<Image img={require('../../img/lago.jpeg')} />

## Quick Start
Use just 1 lines of code, to instantly log your responses **across all providers** with Lago

Get your Lago [API Key](https://docs.getlago.com/guide/self-hosted/docker#find-your-api-key)

```python



litellm.callbacks = ["lago"] # logs cost + usage of successful calls to lago
```


<Tabs>
<TabItem value="sdk" label="SDK">

```python
# pip install lago 
import litellm
import os

os.environ["LAGO_API_BASE"] = "" # http://0.0.0.0:3000
os.environ["LAGO_API_KEY"] = ""
os.environ["LAGO_API_EVENT_CODE"] = "" # The billable metric's code - https://docs.getlago.com/guide/events/ingesting-usage#define-a-billable-metric

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set lago as a callback, litellm will send the data to lago
litellm.success_callback = ["lago"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add to Config.yaml
```yaml
model_list:
- litellm_params:
    api_base: https://openai-function-calling-workers.tasslexyz.workers.dev/
    api_key: my-fake-key
    model: openai/my-fake-model
  model_name: fake-openai-endpoint

litellm_settings:
  callbacks: ["lago"] # 👈 KEY CHANGE
```

2. Start Proxy

```
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "fake-openai-endpoint",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
    }
'
```

</TabItem>
</Tabs>


<Image img={require('../../img/lago_2.png')} />

## Advanced - Lagos Logging object 

This is what LiteLLM will log to Lagos

```
{
    "event": {
      "transaction_id": "<generated_unique_id>",
      "external_customer_id": <litellm_end_user_id>, # passed via `user` param in /chat/completion call - https://platform.openai.com/docs/api-reference/chat/create
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