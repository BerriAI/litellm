import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# A/B Testing - Traffic Mirroring

Traffic mirroring allows you to "mimic" production traffic to a secondary (silent) model for evaluation purposes. The silent model's response is gathered in the background and does not affect the latency or result of the primary request.

This is useful for:
- Testing a new model's performance on production prompts before switching.
- Comparing costs and latency between different providers.
- Debugging issues by mirroring traffic to a more verbose model.

## Quick Start

To enable traffic mirroring, add `silent_model` to the `litellm_params` of a deployment.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

model_list = [
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "azure/chatgpt-v-2",
            "api_key": "...",
            "silent_model": "gpt-4" # 👈 Mirror traffic to gpt-4
        },
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "openai/gpt-4",
            "api_key": "..."
        },
    }
]

router = Router(model_list=model_list)

# The request to "gpt-3.5-turbo" will trigger a background call to "gpt-4"
response = await router.acompletion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "How does traffic mirroring work?"}]
)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

Add `silent_model` to your `config.yaml`:

```yaml
model_list:
  - model_name: primary-model
    litellm_params:
      model: azure/gpt-35-turbo
      api_key: os.environ/AZURE_API_KEY
      silent_model: evaluation-model # 👈 Mirror traffic here
  - model_name: evaluation-model
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
```

</TabItem>
</Tabs>

## How it works
1. **Request Received**: A request is made to a model group (e.g. `primary-model`).
2. **Deployment Picked**: LiteLLM picks a deployment from the group.
3. **Primary Call**: LiteLLM makes the call to the primary deployment.
4. **Mirroring**: If `silent_model` is present, LiteLLM triggers a background call to that model. 
   - For **Sync** calls: Uses a shared thread pool.
   - For **Async** calls: Uses `asyncio.create_task`.
5. **Isolation**: The background call uses a `deepcopy` of the original request parameters and sets `metadata["is_silent_experiment"] = True`. It also strips out logging IDs to prevent collisions in usage tracking.

## Key Features
- **Latency Isolation**: The primary request returns as soon as it's ready. The background (silent) call does not block.
- **Unified Logging**: Background calls are processed via the Router, meaning they are automatically logged to your configured observability tools (Langfuse, S3, etc.).
- **Evaluation**: Use the `is_silent_experiment: True` flag in your logs to filter and compare results between the primary and mirrored calls.
