# MCP connection evidence for #31318

Before: `47b15ffb677902fc4550a4471b82e09f77ec77d7`. After: `dbf94902294cf5fa4fe5af6681fcd83ffd743231`

Images were built from clean source archives using the root Dockerfile. Actual gateway backend hashes were checked against the source commit. Screenshots show the bundled dashboard, with no frontend changes

## Reproduce

1. Build each gateway from the corresponding Git commit using the root Dockerfile and tag it `litellm-local:<first 12 characters of commit>`
2. Copy compose.yaml into an isolated directory. Copy server.py and stdio_server.py into its diagnostics/ subdirectory. Supply your own LITELLM_MASTER_KEY, LITELLM_SALT_KEY and POSTGRES_PASSWORD through a local .env
3. Run `docker compose up -d`. Before is http://localhost:4001; after is http://localhost:4000. Both use PostgreSQL. The fixture serves real HTTP and SSE connections on the private Compose network. Stdio starts a real subprocess in the gateway container
4. Use the curl commands in PR #40359. Each matrix JSON contains the exact request payload, HTTP status, response and elapsed seconds

## Dashboard

Open `/ui/mcp-servers/`, select **+ Add New MCP Server**, then **+ Custom Server**. Set name diagnostic and authentication None. Leave source URL and static headers empty

For session failure, choose **Streamable HTTP (Recommended)** and URL `https://learn.microsoft.com/api/mcp/does-not-exist`. For success use `https://learn.microsoft.com/api/mcp`. For SSE closure, choose **Server-Sent Events (SSE)** and URL `http://mcp-diagnostics:8080/sse/closed/tools`. Scroll to Connection Status

The existing dashboard requires a URL for preview, so stdio verification uses the gateway API. No frontend changes were included

## Results

31 matching live preview cases passed at the final tip. Both revisions passed public Microsoft Learn tool listing and a read-only tool call, plus real SSE and stdio tool listing and calls returning pong. Unknown-error references matched diagnostic log entries containing exception types and stack locations without upstream exception text

Malformed JSON, empty JSON and interrupted connections now fail promptly where the SDK exposes the cause. Silent servers and empty HTTP event streams remain bounded by the configured timeout. One-second SDK deadlines now report one second, instead of the unrelated 30-second outer deadline

Public tool-list and tool-call checks passed on both revisions. Public service content and availability can vary

Screenshot revision and SHA-256 hashes are in screenshots.json. SHA256SUMS covers the fixture and compose files. No gateway credentials are included
