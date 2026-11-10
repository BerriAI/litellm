# Muse models on Vertex AI

Use `vertex_ai/meta/muse-wonder-1.0`, `vertex_ai/meta/muse-ember-1.0`, or
`vertex_ai/meta/muse-spark-1.3` with the Responses API. LiteLLM uses Google
Application Default Credentials and sends requests to the global Meta publisher
endpoint, with the bare model ID in the request body.

Google lists these models in Preview. Enable the model and obtain access in your
Google Cloud project before calling it. See [Google's Muse model guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/meta).

```python
import os

import litellm

response = litellm.responses(
    model="vertex_ai/meta/muse-wonder-1.0",
    vertex_project=os.environ["VERTEXAI_PROJECT"],
    input="Write a Python binary search function with type annotations.",
    max_output_tokens=4096,
    reasoning={"effort": "high"},
    store=False,
)
print(response.output_text)
```

Use `litellm.aresponses` for async calls and `stream=True` for streaming. Chat
Completions calls use LiteLLM's existing Responses bridge.

Requests are stateless. LiteLLM defaults to `store=False` and rejects
`store=True`. Send the conversation context in `input` on each request instead
of using `previous_response_id` or a stored conversation.

The catalog includes Google's documented input limits and capabilities. Google
does not list an output limit for Wonder or Spark on the linked model cards, so
those limits remain unspecified. Ember's documented output limit is included.

Google Marketplace publishes these default USD prices per million tokens, checked
November 9, 2026. The catalog includes uncached input, cache reads and output.
Ember uses the higher rates when input exceeds 200,000 tokens.

| Model | Input | Cached input | Output |
| --- | --- | --- | --- |
| Muse Wonder 1.0 | 4.00 | 0.20 | 20.00 |
| Muse Ember 1.0, up to 200,000 input tokens | 0.10 | 0.004 | 0.50 |
| Muse Ember 1.0, above 200,000 input tokens | 0.30 | 0.012 | 1.50 |
| Muse Spark 1.3 | 1.25 | 0.15 | 4.25 |

Sources are Google's Marketplace listings for
[Wonder](https://console.cloud.google.com/marketplace/product/meta/muse-wonder-1-0),
[Ember](https://console.cloud.google.com/marketplace/product/meta/muse-ember-1-0)
and [Spark](https://console.cloud.google.com/marketplace/product/meta/muse-spark-1-3).
These are list prices before offer discounts. Override the catalog rates when
your Google Cloud agreement supplies different prices.
