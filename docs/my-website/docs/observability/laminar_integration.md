import Image from '@theme/IdealImage';

# Laminar - Tracing LLM functions and program flow.

[Laminar](https://www.lmnr.ai) is the open-source platform for tracing and evaluating AI applications.

Laminar features:

- [Tracing compatible with AI SDK and more](https://docs.lmnr.ai/tracing/introduction),
- [Evaluations](https://docs.lmnr.ai/evaluations/introduction),
- [Browser agent observability](https://docs.lmnr.ai/tracing/browser-agent-observability)

<Image img={require('../../img/laminar_trace_example.png')} />

## Getting Started

### Project API key

Sign up on [Laminar cloud](https://www.lmnr.ai/sign-up) or spin up a
[local Laminar instance](https://www.github.com/lmnr-ai/lmnr), create a project,
and get an api key from project settings.

### Code

Install Laminar SDK. It will also install all the required OpenTelemetry packages.

```shell
pip install 'lmnr[all]'
```

Install the `proxy` extra.

```shell
pip install 'litellm[proxy]'
```

Set the environment variables (Note the lowercase 'a' for authorization):

```shell
OTEL_EXPORTER="otlp_grpc"
OTEL_ENDPOINT="https://api.lmnr.ai:8443"
OTEL_HEADERS="authorization=Bearer <project-api-key>"
```

Use just 1 line of code, to instantly log your LLM responses **across all providers** with OpenTelemetry:

```python
litellm.callbacks = ["otel"]
```

## Redacting Messages, Response Content from OpenTelemetry Logging

### Redact Messages and Responses from all OpenTelemetry Logging

Set `litellm.turn_off_message_logging=True` This will prevent the messages and responses from being logged to OpenTelemetry, but request metadata will still be logged.

### Redact Messages and Responses from specific OpenTelemetry Logging

In the metadata typically passed for text completion or embedding calls you can set specific keys to mask the messages and responses for this call.

Setting `mask_input` to `True` will mask the input from being logged for this call

Setting `mask_output` to `True` will make the output from being logged for this call.

Be aware that if you are continuing an existing trace, and you set `update_trace_keys` to include either `input` or `output` and you set the corresponding `mask_input` or `mask_output`, then that trace will have its existing input and/or output replaced with a redacted message.

## Groupping traces for more complex structure

If you want to use Laminar's features, such as sessions, manual spans, and the `observe` decorator,
you will need to install and initialize Laminar alongside setting LiteLLM's callback.

```python {1,4,7}
from lmnr import Laminar, observe
import litellm

Laminar.initialize(project_api_key="LMNR_PROJECT_API_KEY")
litellm.callbacks = ['otel']

@observe()
def completion(model, messages):
    response = litellm.completion(
        model=model,
        messages=messages,
    )
    return response

completion(
  "gpt-4.1-nano",
  [{"role": "user", "content": "What is the capital of France?"}]
)
```

This, however, will most likely result in your OpenAI calls being double-traced – once by LiteLLM and once by Laminar.
This is because LiteLLM uses OpenAI SDK under the hood to call some of the models and Laminar instruments OpenAI SDK.

To avoid this, you can disable OpenAI instrumentation at initialization.

```python {4}
from lmnr import Laminar, Instruments
Laminar.initialize(
  project_api_key="LMNR_PROJECT_API_KEY",
  disabled_instruments=set([Instruments.OPENAI])
)
```

## Support

For any question or issue with the integration you can reach out to the Laminar Team on [Discord](https://discord.gg/nNFUUDAKub) or via [email](mailto:founders@lmnr.ai).
