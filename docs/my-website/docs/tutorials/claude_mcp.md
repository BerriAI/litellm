import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude MCP Quickstart

This tutorial shows how to connect MCP servers to Claude Code via LiteLLM Proxy.

:::info 

This tutorial is based on [Anthropic's official LiteLLM configuration documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration). This integration allows you to use any LiteLLM supported model through Claude Code with centralized authentication, usage tracking, and cost controls.

:::

## Connecting MCP Servers

You can also connect MCP servers to Claude Code via LiteLLM Proxy.


1. Add the MCP server to your `config.yaml`

<Tabs>
<TabItem value="github" label="GitHub MCP">

In this example, we'll add the Github MCP server to our `config.yaml`

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  github_mcp:
    url: "https://api.githubcopilot.com/mcp"
    auth_type: oauth2
    client_id: os.environ/GITHUB_OAUTH_CLIENT_ID
    client_secret: os.environ/GITHUB_OAUTH_CLIENT_SECRET
```

</TabItem>
<TabItem value="atlassian" label="Atlassian MCP">

In this example, we'll add the Atlassian MCP server to our `config.yaml`

```yaml title="config.yaml" showLineNumbers
atlassian_mcp:
  server_id: atlassian_mcp_id
  url: "https://mcp.atlassian.com/v1/sse"
  transport: "sse"
  auth_type: oauth2
```

</TabItem>
</Tabs>

2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Use the MCP server in Claude Code

```bash
claude mcp add --transport http litellm_proxy http://0.0.0.0:4000/github_mcp/mcp --header "Authorization: Bearer sk-LITELLM_VIRTUAL_KEY"
```

For MCP servers that require dynamic client registration (such as Atlassian), please set `x-litellm-api-key: Bearer sk-LITELLM_VIRTUAL_KEY` instead of using `Authorization: Bearer LITELLM_VIRTUAL_KEY`.

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

<img src={require('../../img/oauth_2_success.png').default} alt="OAuth 2.0 Success" style={{ width: '500px', height: 'auto' }} />

