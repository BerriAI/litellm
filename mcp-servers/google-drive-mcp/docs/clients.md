# Client compatibility

The server works with MCP clients that support either local stdio servers or Streamable HTTP. Authentication requirements depend on whether the server represents one Google identity or many team members.

| Client scenario | Transport | Authentication | Server endpoint |
|---|---|---|---|
| Claude Desktop | stdio | Local OAuth, service account, or external token | Local command |
| Other local MCP clients | stdio | Local OAuth, service account, or external token | Local command |
| claude.ai custom connector | Streamable HTTP | Team mode | `https://<host>/mcp` |
| Other remote MCP clients | Streamable HTTP | Team mode when the client supports MCP OAuth; otherwise a protected single-identity deployment | `http(s)://<host>/mcp` |

The server exposes tools and, by default, the `gdrive:///` resource capability. For a tools-only client or a client that stalls while enumerating a large Drive, set `GOOGLE_DRIVE_MCP_DISABLE_RESOURCES=1` or add `--no-resources`.

## Client configuration

### Claude Desktop with npx

Claude Desktop configuration locations:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["-y", "@piotr-agier/google-drive-mcp"]
    }
  }
}
```

If the credentials file is not in the default config directory, pass its absolute path:

```json
{
  "mcpServers": {
    "google-drive": {
      "command": "npx",
      "args": ["-y", "@piotr-agier/google-drive-mcp"],
      "env": {
        "GOOGLE_DRIVE_OAUTH_CREDENTIALS": "/absolute/path/to/gcp-oauth.keys.json"
      }
    }
  }
}
```

Restart Claude Desktop after changing its configuration.

### Local build

```json
{
  "mcpServers": {
    "google-drive": {
      "command": "node",
      "args": ["/absolute/path/to/google-drive-mcp/dist/index.js"]
    }
  }
}
```

Run `npm install` and `npm run build` before using the local entry point.

### Generic stdio client

Configure the executable as `npx`, pass `-y` and `@piotr-agier/google-drive-mcp` as arguments, and pass any required variables in the client's server environment. Standard input and output must remain attached to the MCP process; logs are written to standard error.

### Streamable HTTP client

Start the server:

```bash
npx -y @piotr-agier/google-drive-mcp start --transport http --port 3100
```

Then configure the MCP endpoint:

```json
{
  "mcpServers": {
    "google-drive": {
      "url": "http://127.0.0.1:3100/mcp"
    }
  }
}
```

The default bind address is loopback-only. Read [Streamable HTTP transport](deployment.md#streamable-http-transport) before exposing it on another interface.

### claude.ai custom connector

Deploy [team mode](deployment.md#team-mode-multi-user-http-deployments) behind HTTPS, then add `https://<your-server>/mcp` as a custom connector. Each member completes Google consent and every call runs as that member.

### Docker

For local Docker-based stdio configuration, use the reusable wrapper described in [Docker usage](deployment.md#docker-usage).

