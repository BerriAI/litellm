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
    input_cost_per_token=float(os.environ["VERTEX_MUSE_INPUT_COST_PER_TOKEN"]),
    output_cost_per_token=float(os.environ["VERTEX_MUSE_OUTPUT_COST_PER_TOKEN"]),
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

Google's linked pricing page did not list Muse token prices when checked on
November 9, 2026. The Vertex entries therefore leave prices unspecified.
Configure the input and output rates from your Google Cloud agreement, in USD
per token, as shown above. Configure cache rates too if your agreement bills
cached tokens separately. Meta's direct API prices are not used as Vertex rates.
