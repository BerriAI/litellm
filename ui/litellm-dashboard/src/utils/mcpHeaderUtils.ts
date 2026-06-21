/**
 * Sanitize an MCP server alias for use in HTTP header names (x-mcp-{alias}-...).
 * RFC 7230 tchar allows token chars; aliases with spaces or hyphens break parsing
 * because the backend splits on the first dash after the x-mcp- prefix.
 * Keep in sync with litellm.proxy._experimental.mcp_server.utils.sanitize_mcp_alias_for_header.
 */
export function sanitizeMcpAliasForHeader(alias: string): string {
  return alias
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

/**
 * Build the passthrough auth header that forwards a browser-held PKCE token to
 * the upstream MCP server. The backend picks up the x-mcp-{alias}-authorization
 * pattern and forwards it; without a usable alias it falls back to the legacy
 * x-mcp-auth header.
 */
export function buildMcpPassthroughAuthHeader(
  serverAlias: string | null | undefined,
  token: string,
): Record<string, string> {
  const safeAlias = serverAlias ? sanitizeMcpAliasForHeader(serverAlias) : "";
  const headerName = safeAlias ? `x-mcp-${safeAlias}-authorization` : "x-mcp-auth";
  return { [headerName]: `Bearer ${token}` };
}
