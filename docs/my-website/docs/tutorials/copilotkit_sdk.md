import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# CopilotKit SDK with LiteLLM

Use CopilotKit SDK with any LLM provider through LiteLLM Proxy.

> **Note:** CopilotKit SDK integration with LiteLLM Proxy works with LiteLLM v1.81.7-nightly or higher.


## Quick Start

### 1. Add Model to Config

```yaml title="config.yaml"
model_list:
  - model_name: claude-sonnet-4-5
    litellm_params:
      model: "anthropic/claude-sonnet-4-5-20250514-v1:0"
      api_key: "os.environ/ANTHROPIC_API_KEY"
```

### 2. Start LiteLLM Proxy

```bash
litellm --config config.yaml
```

### 3. Use CopilotKit SDK

```typescript
import OpenAI from "openai";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const model = "claude-sonnet-4-5";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "sk-12345",
  baseURL: process.env.OPENAI_BASE_URL || "http://localhost:4000/v1",
});

const serviceAdapter = new OpenAIAdapter({ openai, model });
const runtime = new CopilotRuntime();

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
```

### 4. Test

```bash
curl -X POST http://localhost:3000/api/copilotkit \
  -H "Content-Type: application/json" \
  -d '{
    "method": "agent/run",
    "params": {
        "agentId": "default"
    },
        "runId": "your_run_id",
        "threadId": "your_thread_id",
        "runId": ""your_run_id"",
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "state": {},
        "messages": [
            {
                "id": "166e573e-f7c6-4c0f-8685-04dbefec18be",
                "content": "Hi",
                "role": "user"
            }
        ]
    }
}'
```

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `OPENAI_API_KEY` | `sk-12345` | Your LiteLLM API key |
| `OPENAI_BASE_URL` | `http://localhost:4000/v1` | LiteLLM proxy URL |


## Related Resources

- [CopilotKit Documentation](https://docs.copilotkit.ai)
- [LiteLLM Proxy Quick Start](../proxy/quick_start)
