import Image from '@theme/IdealImage';

# Arize AI

AI Observability and Evaluation Platform

:::tip

This is community maintained, Please make an issue if you run into a bug
https://github.com/BerriAI/litellm

:::



## Pre-Requisites
Make an account on [Arize AI](https://app.arize.com/auth/login)

## Quick Start
Use just 2 lines of code, to instantly log your responses **across all providers** with arize


```python
litellm.callbacks = ["arize"]
```
```python
import litellm
import os

os.environ["ARIZE_SPACE_KEY"] = ""
os.environ["ARIZE_API_KEY"] = ""

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set arize as a callback, litellm will send the data to arize
litellm.callbacks = ["arize"]
 
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
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

litellm_settings:
  callbacks: ["arize"]

environment_variables:
    ARIZE_SPACE_KEY: "d0*****"
    ARIZE_API_KEY: "141a****"
    ARIZE_ENDPOINT: "https://otlp.arize.com/v1" # OPTIONAL - your custom arize GRPC api endpoint
    ARIZE_HTTP_ENDPOINT: "https://otlp.arize.com/v1" # OPTIONAL - your custom arize HTTP api endpoint. Set either this or ARIZE_ENDPOINT
```

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
