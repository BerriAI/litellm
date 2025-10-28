import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Code

This tutorial shows how to call Claude models through LiteLLM proxy from Claude Code.

:::info 

This tutorial is based on [Anthropic's official LiteLLM configuration documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration). This integration allows you to use any LiteLLM supported model through Claude Code with centralized authentication, usage tracking, and cost controls.

:::

<br />

### Video Walkthrough

<iframe width="840" height="500" src="https://www.loom.com/embed/3c17d683cdb74d36a3698763cc558f56" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- API keys for your chosen providers

## Installation

First, install LiteLLM with proxy support:

```bash
pip install 'litellm[proxy]'
```

### 1. Setup config.yaml

Create a secure configuration using environment variables:

```yaml
model_list:
  # Claude models
  - model_name: claude-3-5-sonnet-20241022    
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  
litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Set your environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

### 2. Start proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Verify Setup

Test that your proxy is working correctly:

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

### 4. Configure Claude Code

#### Method 1: Unified Endpoint (Recommended)

Configure Claude Code to use LiteLLM's unified endpoint:

Either a virtual key / master key can be used here

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

:::tip
LITELLM_MASTER_KEY gives claude access to all proxy models, whereas a virtual key would be limited to the models set in UI
:::

#### Method 2: Provider-specific Pass-through Endpoint

Alternatively, use the Anthropic pass-through endpoint:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

### 5. Use Claude Code

Start Claude Code and it will automatically use your configured models:

```bash
# Claude Code will use the models configured in your LiteLLM proxy
claude

# Or specify a model if you have multiple configured
claude --model claude-3-5-sonnet-20241022
claude --model claude-3-5-haiku-20241022
```

Example conversation:

## Troubleshooting

Common issues and solutions:

**Claude Code not connecting:**
- Verify your proxy is running: `curl http://0.0.0.0:4000/health`
- Check that `ANTHROPIC_BASE_URL` is set correctly
- Ensure your `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

**Authentication errors:**
- Verify your environment variables are set: `echo $LITELLM_MASTER_KEY`
- Check that your API keys are valid and have sufficient credits
- Ensure the `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

**Model not found:**
- Ensure the model name in Claude Code matches exactly with your `config.yaml`
- Check LiteLLM logs for detailed error messages

## Using Multiple Models

Expand your configuration to support multiple providers and models:

<Tabs>
<TabItem value="multi-provider" label="Multi-Provider Setup">

```yaml
model_list:
  # OpenAI models
  - model_name: codex-mini
    litellm_params:  
      model: openai/codex-mini
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1

  - model_name: o3-pro
    litellm_params:
      model: openai/o3-pro
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1

  # Anthropic models
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  # AWS Bedrock
  - model_name: claude-bedrock
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Switch between models seamlessly:

```bash
# Use Claude for complex reasoning
claude --model claude-3-5-sonnet-20241022

# Use Haiku for fast responses
claude --model claude-3-5-haiku-20241022

# Use Bedrock deployment
claude --model claude-bedrock
```

</TabItem>
</Tabs>

<Image img={require('../../img/release_notes/claude_code_demo.png')} style={{ width: '500px', height: 'auto' }} />


## Connecting MCP Servers

You can also connect MCP servers to Claude Code via LiteLLM Proxy.

:::note

Limitations:

- Currently, only HTTP MCP servers are supported
- Does not work in Cursor IDE yet.

:::

1. Add the MCP server to your `config.yaml`

In this example, we'll add the Github MCP server to our `config.yaml`

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  github_mcp:
    url: "https://api.githubcopilot.com/mcp"
    auth_type: oauth2
    authorization_url: https://github.com/login/oauth/authorize
    token_url: https://github.com/login/oauth/access_token
    client_id: os.environ/GITHUB_OAUTH_CLIENT_ID
    client_secret: os.environ/GITHUB_OAUTH_CLIENT_SECRET
    scopes: ["public_repo", "user:email"]
```

2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Use the MCP server in Claude Code

```bash
claude mcp add --transport http litellm_proxy http://0.0.0.0:4000/github_mcp/mcp --header "Authorization: Bearer sk-LITELLM_VIRTUAL_KEY"
```

4. Authenticate via Claude Code

a. Start Claude Code

```bash
claude
```

b. Authenticate via Claude Code

```bash
/mcp
```

c. Select the MCP server

```bash
> litellm_proxy
```

d. Start Oauth flow via Claude Code

```bash
> 1. Authenticate
 2. Reconnect
 3. Disable             
```

e. Once completed, you should see this success message:

<Image img={require('../../img/oauth_2_success.png')} style={{ width: '500px', height: 'auto' }} />

