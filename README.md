Dashboard evidence for BerriAI/litellm#31318

Captured from source-built gateways at the commits in screenshots.json

Open /ui/mcp-servers/, select + Add New MCP Server, then + Custom Server. Use name diagnostic, Streamable HTTP (Recommended), and authentication None. Leave source URL and static headers empty

For the session failure, use https://learn.microsoft.com/api/mcp/does-not-exist and scroll to Connection Status

For the happy path, change MCP Server URL to https://learn.microsoft.com/api/mcp and observe Connection successful with three tools

Before uses localhost:4001. After uses localhost:4000
