import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Invoking A2A Agents

Learn how to invoke A2A agents through LiteLLM using different methods.

:::tip Deploy Your Own A2A Agent

Want to test with your own agent? Deploy this template A2A agent powered by Google Gemini:

[**shin-bot-litellm/a2a-gemini-agent**](https://github.com/shin-bot-litellm/a2a-gemini-agent) - Simple deployable A2A agent with streaming support

:::

## A2A SDK

Use the [A2A Python SDK](https://pypi.org/project/a2a-sdk) to invoke agents through LiteLLM using the A2A protocol.

### Non-Streaming

This example shows how to:
1. **List available agents** - Query `/v1/agents` to see which agents your key can access
2. **Select an agent** - Pick an agent from the list
3. **Invoke via A2A** - Use the A2A protocol to send messages to the agent

```python showLineNumbers title="invoke_a2a_agent.py"
from uuid import uuid4
import httpx
import asyncio
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# === CONFIGURE THESE ===
LITELLM_BASE_URL = "http://localhost:4000"  # Your LiteLLM proxy URL
LITELLM_VIRTUAL_KEY = "sk-1234"             # Your LiteLLM Virtual Key
# =======================

async def main():
    headers = {"Authorization": f"Bearer {LITELLM_VIRTUAL_KEY}"}
    
    async with httpx.AsyncClient(headers=headers) as client:
        # Step 1: List available agents
        response = await client.get(f"{LITELLM_BASE_URL}/v1/agents")
        agents = response.json()
        
        print("Available agents:")
        for agent in agents:
            print(f"  - {agent['agent_name']} (ID: {agent['agent_id']})")
        
        if not agents:
            print("No agents available for this key")
            return
        
        # Step 2: Select an agent and invoke it
        selected_agent = agents[0]
        agent_id = selected_agent["agent_id"]
        agent_name = selected_agent["agent_name"]
        print(f"\nInvoking: {agent_name}")
        
        # Step 3: Use A2A protocol to invoke the agent
        base_url = f"{LITELLM_BASE_URL}/a2a/{agent_id}"
        resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello, what can you do?"}],
                    "messageId": uuid4().hex,
                }
            ),
        )
        response = await a2a_client.send_message(request)
        print(f"Response: {response.model_dump(mode='json', exclude_none=True, indent=4)}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming

For streaming responses, use `send_message_streaming`:

```python showLineNumbers title="invoke_a2a_agent_streaming.py"
from uuid import uuid4
import httpx
import asyncio
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendStreamingMessageRequest

# === CONFIGURE THESE ===
LITELLM_BASE_URL = "http://localhost:4000"  # Your LiteLLM proxy URL
LITELLM_VIRTUAL_KEY = "sk-1234"             # Your LiteLLM Virtual Key
LITELLM_AGENT_NAME = "ij-local"             # Agent name registered in LiteLLM
# =======================

async def main():
    base_url = f"{LITELLM_BASE_URL}/a2a/{LITELLM_AGENT_NAME}"
    headers = {"Authorization": f"Bearer {LITELLM_VIRTUAL_KEY}"}
    
    async with httpx.AsyncClient(headers=headers) as httpx_client:
        # Resolve agent card and create client
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

        # Send a streaming message
        request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Tell me a long story"}],
                    "messageId": uuid4().hex,
                }
            ),
        )
        
        # Stream the response
        async for chunk in client.send_message_streaming(request):
            print(chunk.model_dump(mode="json", exclude_none=True))

if __name__ == "__main__":
    asyncio.run(main())
```

## /chat/completions API (OpenAI SDK)

You can also invoke A2A agents using the familiar OpenAI SDK by using the `a2a/` model prefix.

### Non-Streaming

<Tabs>
<TabItem value="python" label="Python" default>

```python showLineNumbers title="openai_non_streaming.py"
import openai

client = openai.OpenAI(
    api_key="sk-1234",  # Your LiteLLM Virtual Key
    base_url="http://localhost:4000"  # Your LiteLLM proxy URL
)

response = client.chat.completions.create(
    model="a2a/my-agent",  # Use a2a/ prefix with your agent name
    messages=[
        {"role": "user", "content": "Hello, what can you do?"}
    ]
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript showLineNumbers title="openai_non_streaming.ts"
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'sk-1234',  // Your LiteLLM Virtual Key
  baseURL: 'http://localhost:4000'  // Your LiteLLM proxy URL
});

const response = await client.chat.completions.create({
  model: 'a2a/my-agent',  // Use a2a/ prefix with your agent name
  messages: [
    { role: 'user', content: 'Hello, what can you do?' }
  ]
});

console.log(response.choices[0].message.content);
```

</TabItem>
<TabItem value="curl" label="cURL">

```bash showLineNumbers title="curl_non_streaming.sh"
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "a2a/my-agent",
    "messages": [
      {"role": "user", "content": "Hello, what can you do?"}
    ]
  }'
```

</TabItem>
</Tabs>

### Streaming

<Tabs>
<TabItem value="python" label="Python" default>

```python showLineNumbers title="openai_streaming.py"
import openai

client = openai.OpenAI(
    api_key="sk-1234",  # Your LiteLLM Virtual Key
    base_url="http://localhost:4000"  # Your LiteLLM proxy URL
)

stream = client.chat.completions.create(
    model="a2a/my-agent",  # Use a2a/ prefix with your agent name
    messages=[
        {"role": "user", "content": "Tell me a long story"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript showLineNumbers title="openai_streaming.ts"
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'sk-1234',  // Your LiteLLM Virtual Key
  baseURL: 'http://localhost:4000'  // Your LiteLLM proxy URL
});

const stream = await client.chat.completions.create({
  model: 'a2a/my-agent',  // Use a2a/ prefix with your agent name
  messages: [
    { role: 'user', content: 'Tell me a long story' }
  ],
  stream: true
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) {
    process.stdout.write(content);
  }
}
```

</TabItem>
<TabItem value="curl" label="cURL">

```bash showLineNumbers title="curl_streaming.sh"
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "a2a/my-agent",
    "messages": [
      {"role": "user", "content": "Tell me a long story"}
    ],
    "stream": true
  }'
```

</TabItem>
</Tabs>

## Key Differences

| Method | Use Case | Advantages |
|--------|----------|------------|
| **A2A SDK** | Native A2A protocol integration | • Full A2A protocol support<br/>• Access to task states and artifacts<br/>• Context management |
| **OpenAI SDK** | Familiar OpenAI-style interface | • Drop-in replacement for OpenAI calls<br/>• Easier migration from LLM to agent workflows<br/>• Works with existing OpenAI tooling |

:::tip Model Prefix

When using the OpenAI SDK, always prefix your agent name with `a2a/` (e.g., `a2a/my-agent`) to route requests to the A2A agent instead of an LLM provider.

:::
