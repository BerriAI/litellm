import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] `/v1/messages`

LiteLLM provides a BETA endpoint in the spec of Anthropic's `/v1/messages` endpoint. 

This currently just supports the Anthropic API. 

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ |  |
| Logging | ✅ | works across all integrations |
| End-user Tracking | ✅ | |
| Streaming | ✅ | |
| Fallbacks | ✅ | between anthropic models |
| Loadbalancing | ✅ | between anthropic models |

Planned improvement:
- Vertex AI Anthropic support
- Bedrock Anthropic support

## Usage 

<Tabs>
<TabItem label="PROXY" value="proxy">

1. Setup config.yaml

```yaml
model_list:
    - model_name: anthropic-claude
      litellm_params:
        model: claude-3-7-sonnet-latest
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/messages' \
-H 'content-type: application/json' \
-H 'x-api-key: $LITELLM_API_KEY' \
-H 'anthropic-version: 2023-06-01' \
-d '{
  "model": "anthropic-claude",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "List 5 important events in the XIX century"
        }
      ]
    }
  ],
  "max_tokens": 4096
}'
```
</TabItem>
<TabItem value="sdk" label="SDK">

```python
from litellm.llms.anthropic.experimental_pass_through.messages.handler import anthropic_messages
import asyncio 
import os 

# set env 
os.environ["ANTHROPIC_API_KEY"] = "my-api-key"

messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

# Call the handler
async def call(): 
    response = await anthropic_messages(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
        max_tokens=100,
    )

asyncio.run(call())
```

</TabItem>
</Tabs>