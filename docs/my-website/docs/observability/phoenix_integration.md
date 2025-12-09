import Image from '@theme/IdealImage';

# Arize Phoenix OSS

Open source tracing and evaluation platform

:::tip

This is community maintained, Please make an issue if you run into a bug
https://github.com/BerriAI/litellm

:::


## Pre-Requisites
Make an account on [Phoenix OSS](https://phoenix.arize.com)
OR self-host your own instance of [Phoenix](https://docs.arize.com/phoenix/deployment)

## Quick Start
Use just 2 lines of code, to instantly log your responses **across all providers** with Phoenix

You can also use the instrumentor option instead of the callback, which you can find [here](https://docs.arize.com/phoenix/tracing/integrations-tracing/litellm).

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp litellm[proxy]
```
```python
litellm.callbacks = ["arize_phoenix"]
```
```python
import litellm
import os

os.environ["PHOENIX_API_KEY"] = "" # Necessary only using Phoenix Cloud
os.environ["PHOENIX_COLLECTOR_HTTP_ENDPOINT"] = "" # The URL of your Phoenix OSS instance e.g. http://localhost:6006/v1/traces
os.environ["PHOENIX_PROJECT_NAME"]="litellm" # OPTIONAL: you can configure project names, otherwise traces would go to "default" project

# This defaults to https://app.phoenix.arize.com/v1/traces for Phoenix Cloud

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set arize as a callback, litellm will send the data to arize
litellm.callbacks = ["arize_phoenix"]
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

### Using with LiteLLM Proxy


```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

litellm_settings:
  callbacks: ["arize_phoenix"]

environment_variables:
    PHOENIX_API_KEY: "d0*****"
    PHOENIX_COLLECTOR_ENDPOINT: "https://app.phoenix.arize.com/v1/traces" # OPTIONAL, for setting the GRPC endpoint
    PHOENIX_COLLECTOR_HTTP_ENDPOINT: "https://app.phoenix.arize.com/v1/traces" # OPTIONAL, for setting the HTTP endpoint
```

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
