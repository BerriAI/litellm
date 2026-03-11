import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Claude Code Max Subscription

<div style={{ textAlign: 'center' }}>
<Image img={require('../../img/claude_code_max.png')} style={{ width: '100%', maxWidth: '800px', height: 'auto' }} />

Route Claude Code Max subscription traffic through LiteLLM AI Gateway.
</div>

**Why Claude Code Max over direct API?**
- **Lower costs** — Claude Code Max subscriptions are cheaper for Claude Code power users than per-token API pricing

**Why route through LiteLLM?**
- **Cost attribution** — Track spend per user, team, or key
- **Budgets & rate limits** — Set spending caps and request limits
- **Guardrails** — Apply content filtering and safety controls to all requests



## Quick Start Video

Watch the end-to-end walkthrough of setting up Claude Code with LiteLLM Gateway:

<iframe width="840" height="500" src="https://www.loom.com/embed/2d069b9e3bcc4cecaa5eb27a72ba7b3c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- Claude Max subscription
- LiteLLM Gateway running

## Step 1: Configure LiteLLM Proxy

Create a `config.yaml` with the critical `forward_client_headers_to_llm_api: true` setting:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514

  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022

  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022

general_settings:
  forward_client_headers_to_llm_api: true  # Required: forwards OAuth token to Anthropic

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

:::info Why `forward_client_headers_to_llm_api`?

This setting forwards the user's OAuth token (in the `Authorization` header) through LiteLLM to the Anthropic API, enabling per-user authentication with their Max subscription while LiteLLM handles tracking and controls.

:::

## Step 2: Start LiteLLM Proxy

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

## Walkthrough

### Part 1: Create a Virtual Key in LiteLLM

Navigate to the LiteLLM Dashboard and create a new virtual key for Claude Code usage.

#### 1.1 Open Virtual Keys Page

Navigate to the Virtual Keys section in the LiteLLM Dashboard.

<Image img={require('../../img/claude_code_max/step1.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 1.2 Click "Create New Key"

<Image img={require('../../img/claude_code_max/step2.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 1.3 Configure Key Details

Enter a key name (e.g., `claude-code-test`) and select the models you want to allow access to.

<Image img={require('../../img/claude_code_max/step3.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 1.4 Select Models

Choose the Anthropic models that should be accessible via this key (e.g., `anthropic-claude`, `claude-4.5-haiku`).

<Image img={require('../../img/claude_code_max/step5.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 1.5 Confirm Model Selection

<Image img={require('../../img/claude_code_max/step7.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 1.6 Create the Key

Click "Create Key" to generate your virtual key. Copy the generated key value (e.g., `sk-otsclFlEblQ-6D60ua2IZg`).

<Image img={require('../../img/claude_code_max/step8.jpeg')} style={{ width: '800px', height: 'auto' }} />

---

### Part 2: Sign into Claude Code Max Plan (Client Side)

Set up Claude Code environment variables and authenticate with your Max subscription.

#### 2.1 Set Environment Variables

Configure Claude Code to use LiteLLM Gateway with your virtual key:

```bash showLineNumbers title="Configure Claude Code Environment Variables"
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_MODEL="anthropic-claude"
export ANTHROPIC_CUSTOM_HEADERS="x-litellm-api-key: Bearer sk-otsclFlEblQ-6D60ua2IZg"
```

<Image img={require('../../img/claude_code_max/step15.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### Environment Variables Explained

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_BASE_URL` | Points Claude Code to your LiteLLM Gateway endpoint |
| `ANTHROPIC_MODEL` | The model name configured in your LiteLLM `config.yaml` |
| `ANTHROPIC_CUSTOM_HEADERS` | The `x-litellm-api-key` header for LiteLLM authentication |

#### 2.2 Launch Claude Code

Start Claude Code:

```bash showLineNumbers title="Launch Claude Code"
claude
```

<Image img={require('../../img/claude_code_max/step16.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 2.3 Select Login Method

Choose "Claude account with subscription" (Pro, Max, Team, or Enterprise).

<Image img={require('../../img/claude_code_max/step17.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 2.4 Authorize in Browser

Claude Code opens your browser to authenticate. Click "Authorize" to connect your Claude Max account.

<Image img={require('../../img/claude_code_max/step19.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 2.5 Login Successful

After authorization, you'll see the login success confirmation.

<Image img={require('../../img/claude_code_max/step20.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 2.6 Complete Setup

Press Enter to continue past the security notes and complete the setup.

<Image img={require('../../img/claude_code_max/step21.jpeg')} style={{ width: '800px', height: 'auto' }} />

---

### Part 3: Use Claude Code with LiteLLM

Now you can use Claude Code normally, and all requests will be tracked in LiteLLM.

#### 3.1 Make a Request in Claude Code

Start using Claude Code - requests will flow through LiteLLM Gateway.

<Image img={require('../../img/claude_code_max/step24.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 3.2 View Logs in LiteLLM Dashboard

Navigate to the Logs page in LiteLLM Dashboard to see all Claude Code requests.

<Image img={require('../../img/claude_code_max/step25.jpeg')} style={{ width: '800px', height: 'auto' }} />

#### 3.3 View Request Details

Click on a request to see detailed information including tokens, cost, duration, and model used.

<Image img={require('../../img/claude_code_max/step27.jpeg')} style={{ width: '800px', height: 'auto' }} />

The logs show:
- **Key Name**: `claude-code-test` (the virtual key you created)
- **Model**: `anthropic/claude-sonnet-4-20250514`
- **Tokens**: 65012 (64679 prompt + 333 completion)
- **Cost**: $0.249754
- **Status**: Success

<Image img={require('../../img/claude_code_max/step28.jpeg')} style={{ width: '800px', height: 'auto' }} />

---

## How It Works

LiteLLM Gateway handles two types of authentication:
1. **`x-litellm-api-key`**: Authenticates the request with LiteLLM (usage tracking, budgets, rate limits)
2. **OAuth Token (via `Authorization` header)**: Forwarded to Anthropic API for Claude Max authentication

```mermaid
sequenceDiagram
    participant User as Claude Code User
    participant LiteLLM as LiteLLM AI Gateway
    participant Anthropic as Anthropic API

    User->>LiteLLM: Request with:<br/>- x-litellm-api-key (LiteLLM auth)<br/>- Authorization: Bearer {oauth_token}

    Note over LiteLLM: 1. Validate x-litellm-api-key<br/>2. Check budgets/rate limits<br/>3. Log request for tracking

    LiteLLM->>Anthropic: Forward request with:<br/>- Authorization: Bearer {oauth_token}<br/>(User's Claude Max OAuth token)

    Note over Anthropic: Authenticate user via<br/>OAuth token from Max plan

    Anthropic-->>LiteLLM: Response

    Note over LiteLLM: Log usage, tokens, cost

    LiteLLM-->>User: Response
```

### Header Flow

| Header | Purpose | Handled By |
|--------|---------|------------|
| `x-litellm-api-key` | LiteLLM Gateway authentication, budget tracking, rate limits | LiteLLM |
| `Authorization: Bearer {oauth_token}` | Claude Max subscription authentication | Anthropic API |

### Complete Request Flow Example

Here's what a typical request looks like when Claude Code makes a call through LiteLLM:

```bash showLineNumbers title="Example Request from Claude Code to LiteLLM"
curl -X POST "http://localhost:4000/v1/messages" \
  -H "x-litellm-api-key: Bearer sk-otsclFlEblQ-6D60ua2IZg" \
  -H "Authorization: Bearer oauth_token_from_max_plan" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic-claude",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello, Claude!"}]
  }'
```

LiteLLM then:
1. Validates `x-litellm-api-key` for gateway access
2. Logs the request for usage tracking
3. Forwards the request to Anthropic with the OAuth `Authorization` header (because of `forward_client_headers_to_llm_api: true`)

## Advanced Configuration

### Per-Model Header Forwarding

For more granular control, you can enable header forwarding only for specific models:

```yaml showLineNumbers title="config.yaml - Per-Model Header Forwarding"
model_list:
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514

  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  model_group_settings:
    forward_client_headers_to_llm_api:
      - anthropic-claude
      - claude-3-5-haiku-20241022
```

### Budget Controls

Set up per-user budgets while using Max subscriptions:

```yaml showLineNumbers title="config.yaml - With Database for Budget Tracking"
model_list:
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514

general_settings:
  forward_client_headers_to_llm_api: true
  database_url: "postgresql://..."

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Then create virtual keys with budgets:

```bash showLineNumbers title="Create Virtual Key with Budget"
curl -X POST "http://localhost:4000/key/generate" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "developer-1",
    "max_budget": 100.00,
    "budget_duration": "monthly"
  }'
```

## Troubleshooting

### OAuth Token Not Being Forwarded

**Symptom**: Authentication errors from Anthropic API

**Solution**: Ensure `forward_client_headers_to_llm_api: true` is set in your config:

```yaml showLineNumbers title="config.yaml - Enable Header Forwarding"
general_settings:
  forward_client_headers_to_llm_api: true
```

### LiteLLM Authentication Failing

**Symptom**: 401 errors from LiteLLM Gateway

**Solution**: Verify `x-litellm-api-key` header is set correctly in `ANTHROPIC_CUSTOM_HEADERS`:

```bash showLineNumbers title="Verify Key Info"
curl -X GET "http://localhost:4000/key/info" \
  -H "Authorization: Bearer sk-otsclFlEblQ-6D60ua2IZg"
```

### Model Not Found

**Symptom**: Model not found errors

**Solution**: Ensure the `ANTHROPIC_MODEL` matches a model name in your config:

```bash showLineNumbers title="List Available Models"
curl "http://localhost:4000/v1/models" \
  -H "Authorization: Bearer sk-otsclFlEblQ-6D60ua2IZg"
```

## Related Documentation

- [Forward Client Headers](/docs/proxy/forward_client_headers) - Detailed header forwarding configuration
- [Claude Code Quickstart](/docs/tutorials/claude_responses_api) - Basic Claude Code + LiteLLM setup
- [Virtual Keys](/docs/proxy/virtual_keys) - Creating and managing API keys
- [Budgets & Rate Limits](/docs/proxy/users) - Setting up usage controls
