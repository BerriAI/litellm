import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# MCP Guardrails

LiteLLM supports applying guardrails to MCP tool calls to ensure security and compliance. You can configure guardrails to run before or after MCP calls to validate inputs and block or mask sensitive information.

:::warning Deprecated: `pre_mcp_call` and `during_mcp_call` modes
The MCP-specific guardrail modes `pre_mcp_call` and `during_mcp_call` are **deprecated**. 

We are standardizing on the general-purpose modes:
- Use `pre_call` instead of `pre_mcp_call`
- Use `post_call` instead of `during_mcp_call`

The deprecated modes will continue to work for backwards compatibility, but we recommend migrating to `pre_call` and `post_call` for all new configurations.
:::

### Supported Guardrail Modes

Use the standard guardrail modes for MCP tool calls:

- `pre_call`: Run **before** the call, on **input**. Use this mode when you want to apply validation/masking/blocking for MCP requests
- `post_call`: Run **after** the call, on **output**. Use this mode for scanning responses

### Configuration Examples

Configure guardrails to run before MCP tool calls to validate and sanitize inputs:

```yaml title="config.yaml" showLineNumbers
guardrails:
  - guardrail_name: "mcp-input-validation"
    litellm_params:
      guardrail: presidio  # or other supported guardrails
      mode: "pre_call"  # Recommended: use pre_call instead of pre_mcp_call
      pii_entities_config:
        CREDIT_CARD: "BLOCK"  # Will block requests containing credit card numbers
        EMAIL_ADDRESS: "MASK"  # Will mask email addresses
        PHONE_NUMBER: "MASK"   # Will mask phone numbers
      default_on: true
```


### Usage Examples

#### Testing Pre-MCP Call Guardrails

Test your MCP guardrails with a request that includes sensitive information:

```bash title="Test MCP Guardrail" showLineNumbers
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is john@example.com"}
    ],
    "guardrails": ["mcp-input-validation"]
  }'
```

The request will be processed as follows:
1. Credit card number will be blocked (request rejected)
2. Email address will be masked (e.g., replaced with `<EMAIL_ADDRESS>`)

#### Using with MCP Tools

When using MCP tools, guardrails will be applied to the tool inputs:

```python title="Python Example with MCP Guardrails" showLineNumbers
import openai

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:4000"
)

# This request will trigger MCP guardrails
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Send an email to 555-123-4567 with my SSN 123-45-6789"}
    ],
    tools=[{"type": "mcp", "server_label": "litellm", "server_url": "litellm_proxy"}],
    guardrails=["mcp-input-validation"]
)
```

### Supported Guardrail Providers

MCP guardrails work with all LiteLLM-supported guardrail providers:

- **Presidio**: PII detection and masking
- **Bedrock**: AWS Bedrock guardrails
- **Lakera**: Content moderation
- **Aporia**: Custom guardrails
- **Noma**: Noma Security
- **PANW Prisma AIRS**: Prisma AIRS guardrails
- **Custom**: Your own guardrail implementations