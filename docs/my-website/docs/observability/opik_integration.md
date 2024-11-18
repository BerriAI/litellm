import Image from '@theme/IdealImage';

# Comet Opik - Logging + Evals
Opik is an open source end-to-end [LLM Evaluation Platform](https://www.comet.com/site/products/opik/?utm_source=litelllm&utm_medium=docs&utm_content=intro_paragraph) that helps developers track their LLM prompts and responses during both development and production. Users can define and run evaluations to test their LLMs apps before deployment to check for hallucinations, accuracy, context retrevial, and more!


<Image img={require('../../img/opik.png')} />

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

## Pre-Requisites

You can learn more about setting up Opik in the [Opik quickstart guide](https://www.comet.com/docs/opik/quickstart/). You can also learn more about self-hosting Opik in our [self-hosting guide](https://www.comet.com/docs/opik/self-host/local_deployment).

## Quick Start
Use just 4 lines of code, to instantly log your responses **across all providers** with Opik

Get your Opik API Key by signing up [here](https://www.comet.com/signup?utm_source=litelllm&utm_medium=docs&utm_content=api_key_cell)!

```python
from litellm.integrations.opik.opik import OpikLogger
import litellm

opik_logger = OpikLogger()
litellm.callbacks = [opik_logger]
```

Full examples:

```python
from litellm.integrations.opik.opik import OpikLogger
import litellm
import os

# Configure the Opik API key or call opik.configure()
os.environ["OPIK_API_KEY"] = ""
os.environ["OPIK_WORKSPACE"] = ""

# LLM provider API Keys:
os.environ["OPENAI_API_KEY"] = ""

# set "opik" as a callback, litellm will send the data to an Opik server (such as comet.com)
opik_logger = OpikLogger()
litellm.callbacks = [opik_logger]

# openai call
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Why is tracking and evaluation of LLMs important?"}
    ]
)
```

If you are liteLLM within a function tracked using Opik's `@track` decorator,
you will need provide the `current_span_data` field in the metadata attribute
so that the LLM call is assigned to the correct trace:

```python
from opik import track
from opik.opik_context import get_current_span_data
from litellm.integrations.opik.opik import OpikLogger
import litellm

opik_logger = OpikLogger()
litellm.callbacks = [opik_logger]

@track()
def streaming_function(input):
    messages = [{"role": "user", "content": input}]
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        metadata = {
            "opik": {
                "current_span_data": get_current_span_data(),
                "tags": ["streaming-test"],
            },
        }
    )
    return response

response = streaming_function("Why is tracking and evaluation of LLMs important?")
chunks = list(response)
```

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
