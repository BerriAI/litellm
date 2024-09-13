import Image from '@theme/IdealImage';

# Comet Opik - Logging + Evals
Opik is an open source end-to-end [LLM Evaluation Platform](https://www.comet.com/site/products/opik/?utm_source=litelllm&utm_medium=docs&utm_content=intro_paragraph) that helps developers track their LLM prompts and responses during both development and production. Users can define and run evaluations to test their LLMs apps before deployment to check for hallucinations, accuracy, context retrevial, and more!


<Image img={require('../../img/opik.png')} />

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

## Pre-Requisites

Ensure you have run `pip install opik` for this integration

```shell
pip install opik litellm
```

## Quick Start
Use just 2 lines of code, to instantly log your responses **across all providers** with Opik

Get your Opik API Key by signing up [here](https://www.comet.com/signup?utm_source=litelllm&utm_medium=docs&utm_content=api_key_cell)!

```python
import litellm
litellm.success_callback = ["opik"]
```

Full examples:

```python
# pip install opik
import litellm
import os

os.environ["OPIK_API_KEY"] = ""

# LLM provider API Keys:
os.environ["OPENAI_API_KEY"] = ""



# set "opik" as a callback, litellm will send the data to an Opik server (such as comet.com)
litellm.success_callback = ["opik"]

# openai call
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Why is tracking and evaluation of LLMs important?"}
    ]
)
```

If you are using a streaming response, you need to surround the
call with Opik's `@track` and provide `current_span_data` and `current_trace_data`:

```python
from opik import track
from opik.opik_context import get_current_trace_data, get_current_span_data

litellm.success_callback = ["opik"]

@track()
def streaming_function(input):
    messages = [{"role": "user", "content": input}]
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        metadata = {
            "opik": {
                "current_span_data": get_current_span_data(),
                "current_trace_data": get_current_trace_data(),
                "tags": ["streaming-test"],
            },
        },
        stream=True,
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
