# Google Drive MCP Server

[![MCP Toplist](https://mcptoplist.com/badge/io.github.piotr-agier%2Fgoogle-drive-mcp.svg)](https://mcptoplist.com/server/io.github.piotr-agier%2Fgoogle-drive-mcp)

Connect an MCP client to Google Drive, Docs, Sheets, Slides, and Calendar through one self-hosted server. Search and organize files, create and edit Workspace content, manage sharing, and automate multi-step workflows while keeping control of the Google identity and credentials used for every call.

## Why this server

- **Drive-first workflows:** 130 tools cover file management, Shared Drives, permissions, revisions, rich Docs editing, Sheets formatting, Slides authoring, and Calendar events.
- **Local or hosted:** use stdio for a personal desktop client, Streamable HTTP for a hosted integration, or OAuth-protected team mode for a shared service.
- **Identity control:** local OAuth supports multiple Google accounts and per-tool account selection; service accounts and externally managed OAuth tokens are also supported.
- **Agent-friendly access:** tools expose targeted operations, while the optional `gdrive:///` resource interface supports direct reading and discovery.
- **Open and self-hosted:** credentials and tokens stay in the environment you operate.

This project remains focused on deep Drive and editor workflows rather than attempting to expose every Google Workspace API.

## Client compatibility

Compatibility is determined by the transport and authentication flow a client supports.

| Client type | Transport | Recommended mode |
|---|---|---|
| Claude Desktop | stdio | Local OAuth |
| Other local MCP clients, including Gemini CLI | stdio | Local OAuth |
| claude.ai custom connectors | Streamable HTTP | Team mode |
| Other remote MCP clients | Streamable HTTP | Single identity behind access control, or team mode with OAuth 2.1 |

See [Client configuration](docs/clients.md#client-configuration) for configuration examples and transport requirements.

## Quick start

### 1. Create Google OAuth credentials

In a Google Cloud project:

1. Enable the Drive, Docs, Sheets, Slides, and Calendar APIs.
2. Configure the OAuth consent screen and add your Google account as a test user when the app is in testing.
3. Create an OAuth client with application type **Desktop app**.
4. Download the JSON file and save it as:

```text
~/.config/google-drive-mcp/gcp-oauth.keys.json
```

The [setup guide](docs/setup.md) has the complete Google Cloud walkthrough and alternative credential locations.

### 2. Authenticate

```bash
npx -y @piotr-agier/google-drive-mcp auth
```

Complete the Google consent flow in the browser. Tokens are stored by default at `~/.config/google-drive-mcp/tokens.json`.

### 3. Add the server to your MCP client

For clients that use the common `mcpServers` configuration shape:

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

Restart the client after saving its configuration. Claude Desktop paths and HTTP examples are documented in [Client configuration](docs/clients.md#client-configuration).

### 4. Make a first tool call

Ask your client:

```text
Run authGetStatus and tell me which Google account is active.
```

Then try a read-only Drive request:

```text
Search my Google Drive for files modified in the last seven days.
```

If the identity is wrong or search returns no files, use the [troubleshooting guide](docs/troubleshooting.md).

## What you can do

### Organize Drive

```text
Find PDF files in /Reports, create an Archive folder there, and move files older
than one year into it.
```

### Build a report

```text
Create a Google Sheet for monthly results, summarize it in a Google Doc, and
create a short Google Slides presentation from the summary.
```

### Edit a document

```text
Find the "Project Plan" document, replace the old launch date, format the new
date in bold, and add a comment describing the change.
```

### Prepare a meeting

```text
Create a Calendar event with a Google Meet link and attach the project brief
from Drive.
```

<!--
  Anchors for sections that lived in this README before the guides moved to docs/.
  External deep links (directory listings, issue replies, blog posts) land on the
  table below, which names the guide each section moved to. Do not remove.
-->
<a id="features"></a><a id="example-usage"></a><a id="requirements"></a>
<a id="google-cloud-setup"></a><a id="installation"></a><a id="docker-usage"></a>
<a id="configuration"></a><a id="runtime-configuration-cli-args-or-env-vars"></a>
<a id="usage-with-claude-desktop"></a><a id="streamable-http-transport"></a>
<a id="team-mode-multi-user-http-deployments"></a><a id="available-tools"></a>
<a id="external-authentication"></a><a id="multi-account-support"></a>
<a id="authentication-flow"></a><a id="troubleshooting"></a><a id="development"></a>
<a id="advanced-configuration"></a><a id="contributing"></a><a id="support"></a>

## Documentation

Every section that used to live in this README is now in one of these guides.

| Guide | Contents |
|---|---|
| [Setup](docs/setup.md) | Requirements, Google Cloud APIs, OAuth credentials, and installation |
| [Client configuration](docs/clients.md) | Supported transports and client configuration |
| [Authentication](docs/authentication.md) | Local OAuth, multi-account, service accounts, external tokens, and scopes |
| [Configuration](docs/configuration.md) | CLI flags, environment variables, defaults, and precedence |
| [Deployment](docs/deployment.md) | Docker, Streamable HTTP, team mode, and reverse-proxy security |
| [Tool reference](docs/tools.md) | All Drive, Docs, Sheets, Slides, Calendar, and account tools |
| [Troubleshooting](docs/troubleshooting.md) | Authentication, API, identity, Docker, and rate-limit problems |
| [Development](docs/development.md) | Repository structure, build commands, tests, and contributions |

## Security

- Never commit OAuth credentials, service-account keys, access tokens, refresh tokens, or `tokens.json`.
- Use the narrowest OAuth scopes that support the tools you need.
- Keep the default HTTP bind address on `127.0.0.1` unless the server is protected by TLS and access control.
- Use [team mode](docs/deployment.md#team-mode-multi-user-http-deployments) for shared deployments so every request is authenticated as its caller.
- Treat `team-store.json` as a secret because it contains members' Google refresh tokens.

See [Authentication](docs/authentication.md) and [Deployment](docs/deployment.md) for the complete security and identity model.

## Development and support

See the [development guide](docs/development.md) to build and test the project.

- Report defects and request features in [GitHub Issues](https://github.com/piotr-agier/google-drive-mcp/issues).
- Review released changes in the [changelog](CHANGELOG.md).
- Contributions are welcome through pull requests.

## License

[MIT](LICENSE)

## Acknowledgments

- Built on the [Model Context Protocol](https://modelcontextprotocol.io).
- Uses the [Google APIs Node.js Client](https://github.com/googleapis/google-api-nodejs-client).
