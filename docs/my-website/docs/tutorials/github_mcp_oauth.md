import Image from '@theme/IdealImage';

# Call Github MCP via LiteLLM in Cursor

Pre-requisites:
- You have a LiteLLM Proxy running 

## Admin Flow

## 1. Pick an MCP Server ID

In this case, let's use `github_mcp_id`

This will be used for the dynamic url for MCP Oauth flow 

## 2. Create Oauth client on Github

:::info 

You may need your Github Admin to create the Oauth client for you.

:::

### 2.1 Access GitHub Settings
1. Log in to GitHub
2. Navigate to Settings (click your profile picture Settings)
3. Scroll down to Developer settings in the left sidebar
4. Click OAuth Apps

### 2.2 Register New OAuth App

1. Click New OAuth App (or Register a new application)

2. Fill in the application details:

- Application name: "LiteLLM MCP Integration"
- Homepage URL: https://litellm.ai
- Application description: "MCP server for GitHub integration"
- Authorization callback URL:
    - `{PROXY_BASE_URL}/oauth/{mcp_server_id}/callback` → in this case it would be `{PROXY_BASE_URL}/oauth/github_mcp_id/callback`


3. Click Register application

### 2.3 Get OAuth Credentials

After registration, you'll see your app details
Note down:
- Client ID
- Click **Generate a new client secret**
- Client Secret (copy immediately - shown only once)



## 3. Add Github MCP to LiteLLM Proxy (config.yaml)

:::info

Currently, Oauth2.0 authentication is only available for MCP servers via config.yaml.

We plan to have this on the UI soon. [Tell us](mailto:support@berri.ai) if you need it urgently.

:::

```yaml
mcp_servers:
  github_mcp:
    url: "https://api.githubcopilot.com/mcp"
    auth_type: oauth2
    authorization_url: https://github.com/login/oauth/authorize
    token_url: https://github.com/login/oauth/access_token
    client_id: os.environ/GITHUB_OAUTH_CLIENT_ID
    client_secret: os.environ/GITHUB_OAUTH_CLIENT_SECRET
    scopes: ["public_repo", "user:email"]
    server_id: github_mcp_id
```

Your MCP server is now available at:

- `{PROXY_BASE_URL}/mcp`

Oauth login is available at:
- `{PROXY_BASE_URL}/auth/{mcp_server_id}/login` → in this case it would be `{PROXY_BASE_URL}/auth/github_mcp_id/login`


## Developer Flow

## Login via Oauth2.0

1. Go to `{PROXY_BASE_URL}/auth/{mcp_server_id}/login`
2. Complete the Oauth2.0 flow to get PAT token
3. Save the PAT token for the next step

<Image
  img={require('../../img/github_code_pat_token.png')}
  alt="github_code_pat_token"
  style={{ maxWidth: '75%', height: 'auto' }}
/>

## 4. Add LiteLLM MCP to cursor 

1. Open Settings -> Cursor Settings

2. Navigate to MCP Integrations

3. Click Add Custom MCP Server

```json
{
  "mcpServers": {
    "LiteLLM": {
      "url": "{PROXY_BASE_URL}/mcp",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-{mcp_server_name}-authorization": "Bearer $GITHUB_PAT_TOKEN" → in this case it would be `x-mcp-github_mcp-authorization`
      }
    }
  }
}
```
