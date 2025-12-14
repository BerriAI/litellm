import Image from '@theme/IdealImage';

# Arize Phoenix OSS

Open source tracing and evaluation platform

:::tip

This is community maintained. Please make an issue if you run into a bug:
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

# Set env variables
os.environ["PHOENIX_API_KEY"] = "d0*****" # Set the Phoenix API key here. It is necessary only when using Phoenix Cloud.
os.environ["PHOENIX_COLLECTOR_HTTP_ENDPOINT"] = "https://app.phoenix.arize.com/s/<space-name>/v1/traces" # Set the URL of your Phoenix OSS instance, otherwise tracer would use https://app.phoenix.arize.com/v1/traces for Phoenix Cloud.
os.environ["PHOENIX_PROJECT_NAME"] = "litellm" # Configure the project name, otherwise traces would go to "default" project.
os.environ['OPENAI_API_KEY'] = "fake-key" # Set the OpenAI API key here.

# Set arize_phoenix as a callback & LiteLLM will send the data to Phoenix.
litellm.callbacks = ["arize_phoenix"]

# OpenAI call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

## Using with LiteLLM Proxy

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

litellm_settings:
  callbacks: ["arize_phoenix"]

general_settings:
  master_key: "sk-1234"

environment_variables:
    PHOENIX_API_KEY: "d0*****"
    PHOENIX_COLLECTOR_ENDPOINT: "https://app.phoenix.arize.com/s/<space-name>/v1/traces" # OPTIONAL - For setting the gRPC endpoint
    PHOENIX_COLLECTOR_HTTP_ENDPOINT: "https://app.phoenix.arize.com/s/<space-name>/v1/traces" # OPTIONAL - For setting the HTTP endpoint
```

2. Start the proxy

```bash
litellm --config config.yaml
```

3. Test it!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ "model": "gpt-4o", "messages": [{"role": "user", "content": "Hi 👋 - i'm openai"}]}'
```

## Supported Phoenix Endpoints
Phoenix now supports multiple deployment types. The correct endpoint depends on which version of Phoenix Cloud you are using.

**Phoenix Cloud (With Spaces - New Version)**
Use this if your Phoenix URL contains `/s/<space-name>` path.

```bash
https://app.phoenix.arize.com/s/<space-name>/v1/traces
```

**Phoenix Cloud (Legacy - Deprecated)**
Use this only if your deployment still shows the `/legacy` pattern.

```bash
https://app.phoenix.arize.com/legacy/v1/traces
```

**Phoenix Cloud (Without Spaces - Old Version)**
Use this if your Phoenix Cloud URL does not contain `/s/<space-name>` or `/legacy` path.

```bash
https://app.phoenix.arize.com/v1/traces
```

**Self-Hosted Phoenix (Local Instance)**
Use this when running Phoenix on your machine or a private server.

```bash
http://localhost:6006/v1/traces
```

Depending on which Phoenix Cloud version or deployment you are using, you should set the corresponding endpoint in `PHOENIX_COLLECTOR_HTTP_ENDPOINT` or `PHOENIX_COLLECTOR_ENDPOINT`.

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
