import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Amberflo - Metering and Usage Based Billing

Amberflo is a cloud-based platform designed to help businesses implement and manage metering, usage-based billing, and analytics for their SaaS (Software as a Service) products and services.

Amberflo offers a robust platform for real-time tracking, metering and billing of usage events such as LLM tokens, API transactions, and consumption of compute, storage, and network resources, enabling precise billing based on actual usage. Whether you’re managing a subscription-based business model or billing customers per unique user, Amberflo’s billing cloud streamlines the invoicing process and integrates seamlessly with popular payment gateways like Stripe or ERP systems like Netsuite, ensuring smooth, automated customer invoicing and payment collection.

<Image img={require('../../img/amberflo.png')} />

## Quick Start
Use just 1 line of code, to instantly log your responses **across all providers** with Amberflo

Get your Amberflo [API Key](https://ui.amberflo.io/signup)

```python
litellm.callbacks = ["amberflo"] # logs usage of successful calls to amberflo
```

#### SDK


```python
import litellm
import os

# from https://ui.amberflo.io/
os.environ["AMBERFLO_API_KEY"] = ""

# LLM API Keys
os.environ['OPENAI_API_KEY'] = ""

# set amberflo as a callback, litellm will send the data to amberflo
litellm.callbacks = ["amberflo"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  user="your_customer_id" # 👈 SET YOUR CUSTOMER ID HERE
)
```

#### Proxy

1. Add to Config.yaml
```yaml
model_list:
- litellm_params:
    api_base: https://openai-function-calling-workers.tasslexyz.workers.dev/
    api_key: my-fake-key
    model: openai/my-fake-model
  model_name: fake-openai-endpoint

litellm_settings:
  callbacks: ["amberflo"] # 👈 KEY CHANGE
```

2. Start Proxy

```
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "fake-openai-endpoint",
      "user": "your_customer_id",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
    }
'
```