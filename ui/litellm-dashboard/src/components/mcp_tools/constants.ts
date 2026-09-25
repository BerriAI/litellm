// Must match the backend SpecialMCPServerNames.no_mcp_servers enum value.
export const NO_MCP_SERVERS_SENTINEL = "no-mcp-servers";

export const ALL_PROXY_MCP_SERVERS_SENTINEL = "all-proxy-mcpservers";

// Must match the backend MCP_ALL_TOOLS_WILDCARD constant in litellm/types/mcp.py.
export const MCP_ALL_TOOLS_WILDCARD = "*";

export const MCP_TOOLS_PREVIEW_FORBIDDEN_MESSAGE =
  "Tool preview is not available for submissions. Tools will be verified by an admin during review.";
