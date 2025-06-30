import Image from '@theme/IdealImage';

# Laminar - Observability for LiteLLM with Laminar

## What is Laminar

[Laminar](https://www.lmnr.ai) is a comprehensive open-source platform
for engineering AI agents.

View a version of this guide in [Laminar docs](https://docs.lmnr.ai/tracing/integrations/litellm).

Example trace for a [Skyvern](https://www.skyvern.com/) browser agent powered by LiteLLM.

<Image img={require('../../img/laminar_trace_example.png')} />

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
from lmnr import Instruments, Laminar, LaminarLiteLLMCallback

Laminar.initialize(disabled_instruments=set([Instruments.OPENAI])
litellm.callbacks=[LaminarLiteLLMCallback()]
```

Laminar wraps every LiteLLM completion/acompletion call and
automatically instruments major provider SDKs. Since LiteLLM
uses OpenAI SDK for some of the LLM calls, we disable Laminar's
automatic instrumentations of OpenAI SDK to avoid double tracing.

### Run your application

Once you've initialized Laminar at the start of your application,
you can make any calls to LiteLLM, and they all will be traced.

```python
import litellm
response = litellm.completion(
    model="gpt-4.1-nano",
    messages=[
      {"role": "user", "content": "What is the capital of France?"}
    ],
)
```

## Support

For any question or issue with the integration you can reach out to the Laminar Team on [Discord](https://discord.gg/nNFUUDAKub) or via [email](mailto:founders@lmnr.ai).
