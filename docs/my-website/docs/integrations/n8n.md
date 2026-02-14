# n8n Integration

Use LiteLLM as an AI provider in [n8n](https://n8n.io) workflows to access 100+ LLMs through a single, OpenAI-compatible interface.

## Quick Start

LiteLLM works with n8n's **OpenAI Chat Model** node using a custom baseURL pointing to your LiteLLM proxy.

### Prerequisites

- n8n v1.113.3 or later (earlier versions may work but are not tested)
- LiteLLM proxy running (v1.75.5+ recommended for full compatibility)
- API key for LiteLLM proxy

### Configuration Steps

#### 1. Start LiteLLM Proxy

```bash
# Using Docker
docker run -e STORE_MODEL_IN_DB=True -p 4000:4000 \
  docker.litellm.ai/berriai/litellm:main-latest

# Or via pip
pip install litellm
litellm --config config.yaml
```

#### 2. Configure n8n Credentials

1. In n8n, go to **Credentials** → **New Credential**
2. Select **OpenAI API**
3. Configure as follows:

| Field | Value |
|-------|-------|
| **API Key** | Your LiteLLM API key (or `sk-1234` for local testing) |
| **Base URL** | `http://localhost:4000` (or your LiteLLM proxy URL) |

:::tip
The Base URL field was added to OpenAI credentials in n8n via [PR #12634](https://github.com/n8n-io/n8n/pull/12634). If you don't see this option, update to a newer n8n version.
:::

#### 3. Use OpenAI Chat Model Node

In your workflow:

1. Add **OpenAI Chat Model** node (NOT the deprecated "OpenAI Model" node)
2. Select your configured OpenAI credential
3. Choose your model from the dropdown

The model list will populate automatically from your LiteLLM proxy's `/v1/models` endpoint.

## Supported Features

✅ **Working Features:**
- Chat completions (streaming and non-streaming)
- Model listing via `/v1/models`
- Tool/function calling
- Message history
- AI Agent node v1.7
- All LiteLLM supported models (Anthropic, Azure, Bedrock, etc.)

⚠️ **Known Limitations:**
- **AI Agent node v2.2** has compatibility issues ([n8n issue #19712](https://github.com/n8n-io/n8n/issues/19712))
  - **Workaround**: Use AI Agent node v1.7 instead

## Example Workflow

Here's a simple n8n workflow using LiteLLM:

```json
{
  "nodes": [
    {
      "name": "Chat Model",
      "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
      "parameters": {
        "model": "gpt-4o-mini",
        "options": {}
      },
      "credentials": {
        "openAiApi": {
          "id": "1",
          "name": "LiteLLM Proxy"
        }
      }
    }
  ]
}
```

## Advanced Configuration

### Using Multiple LLM Providers

Configure different models in your LiteLLM `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-deployment
      api_base: https://your-azure.openai.azure.com
      api_key: os.environ/AZURE_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: llama-3
    litellm_params:
      model: bedrock/meta.llama3-70b-instruct-v1:0
      aws_region_name: us-west-2
```

In n8n, you'll see all these models as `gpt-4`, `claude-3-opus`, and `llama-3` in the model dropdown.

### Metadata and Observability

To pass metadata through to LiteLLM for tracking and observability:

**Option 1: Use Community Package**

Install [@rlquilez/n8n-nodes-openai-litellm](https://www.npmjs.com/package/@rlquilez/n8n-nodes-openai-litellm):

```bash
cd ~/.n8n/nodes
npm install @rlquilez/n8n-nodes-openai-litellm
```

This package supports:
- Custom JSON metadata injection
- Langfuse session ID and user ID
- Proper metadata formatting for LiteLLM

**Option 2: Manual HTTP Request**

Use n8n's HTTP Request node for direct control:

```json
{
  "method": "POST",
  "url": "http://localhost:4000/v1/chat/completions",
  "headers": {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json"
  },
  "body": {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "metadata": {
      "user_id": "user-123",
      "session_id": "session-456",
      "environment": "production"
    }
  }
}
```

## Troubleshooting

### Models Not Loading

**Issue**: Model dropdown shows "[No data]"

**Solution**:
1. Ensure you're using **OpenAI Chat Model** node (not deprecated "OpenAI Model")
2. Verify your LiteLLM proxy is accessible from n8n
3. Check that `/v1/models` endpoint returns data:
   ```bash
   curl http://localhost:4000/v1/models
   ```

**Related**: [n8n issue #17001](https://github.com/n8n-io/n8n/issues/17001) (fixed in newer n8n versions)

### tool_calls Error

**Issue**: Error about `tool_calls` being null

**Solution**:
- Update LiteLLM to v1.75.5 or later
- This was fixed in [PR #13320](https://github.com/BerriAI/litellm/pull/13320)

**Related**: [LiteLLM issue #13055](https://github.com/BerriAI/litellm/issues/13055)

### AI Agent Node Errors

**Issue**: AI Agent v2.2 throws "Cannot read properties of null (reading 'length')"

**Solution**:
- Switch to AI Agent node v1.7 (fully compatible with LiteLLM)
- The v2.2 issue is tracked at [n8n issue #19712](https://github.com/n8n-io/n8n/issues/19712)

### 403 Forbidden on Model Listing

**Issue**: `/v1/models` returns 403 when end user is over budget

**Solution**:
- Fixed in LiteLLM v1.75.5+ ([PR #13320](https://github.com/BerriAI/litellm/pull/13320))
- Model listing is now allowed even when user is over budget

## Community Resources

- [n8n-nodes-openai-litellm](https://github.com/rlquilez/n8n-nodes-openai-litellm) - Enhanced metadata support
- [litellm-n8n-node](https://github.com/paulokuong/litellm-n8n-node) - Community integration package
- [litellm-n8n](https://github.com/sebastianrueckerai/litellm-n8n) - Integration examples

## Feature Requests

Want to see LiteLLM as an official n8n provider? Vote and comment on:
- [n8n issue #10065](https://github.com/n8n-io/n8n/issues/10065) - LiteLLM support feature request

## Support

- [LiteLLM Discord](https://discord.com/invite/wuPM9dRgDw)
- [n8n Community Forum](https://community.n8n.io/)
- [GitHub Issues](https://github.com/BerriAI/litellm/issues)

## Related Documentation

- [LiteLLM Proxy Server](/docs/simple_proxy)
- [Model Fallbacks & Retries](/docs/routing)
- [Load Balancing](/docs/load_balancing)
- [Logging & Observability](/docs/observability/callbacks)
