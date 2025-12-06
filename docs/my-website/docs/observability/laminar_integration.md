import Image from '@theme/IdealImage';

# Laminar - Observability for LiteLLM with Laminar

## What is Laminar

[Laminar](https://www.lmnr.ai) is a comprehensive open-source platform
for engineering AI agents.

Laminar tracks
- LLM call inputs and outputs,
- Request and response parameters, such as temperature and top p,
- Token counts and costs,
- LLM call latency.

Example complex LiteLLM trace on Laminar:

<Image img={require('../../img/laminar_complex_trace_example.png')} />

View a version of this guide in [Laminar docs](https://docs.lmnr.ai/tracing/integrations/litellm).

## Getting started

### Project API Key

Sign up on [Laminar cloud](https://www.lmnr.ai/sign-up) or spin up a
[local Laminar instance](https://www.github.com/lmnr-ai/lmnr), create a project,
and get an api key from project settings.

### Install the Laminar SDK

```shell
pip install 'lmnr[all]'
```

### Set the environment variables

```python
import os
os.environ["LMNR_PROJECT_API_KEY"] = "<YOUR_PROJECT_API_KEY>"
```

### Enable tracing in just 2 lines of code

```python
import litellm
from lmnr import Laminar, LaminarLiteLLMCallback

Laminar.initialize()
litellm.callbacks=[LaminarLiteLLMCallback()]
```

### Run your application

Once you've initialized Laminar at the start of your application,
you can make any calls to LiteLLM, and they all will be traced.

```python
import litellm
response = litellm.completion(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France? Tell me about the city."
        }
    ],
)
```

Here is what this trace may look like on Laminar.

<Image img={require('../../img/laminar_trace_example.png')} />

Direct
[link](https://www.lmnr.ai/shared/traces/caabbd7b-d2f5-b538-b93d-b4abafe05ef4)
to the trace.

## Support

For any question or issue with the integration you can reach out to the 
Laminar Team on [Discord](https://discord.gg/nNFUUDAKub) or
via [email](mailto:founders@lmnr.ai).
