import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";

export const ALL_MCP_SERVERS_SENTINEL = "__all__";
const TOOLSET_PREFIX = "toolset:";

export interface McpToolBlock {
  type: "mcp";
  server_label: string;
  server_url: string;
  require_approval: "never";
  allowed_tools?: string[];
}

export interface BuildMcpToolBlocksArgs {
  selectedMCPServers?: string[];
  mcpServers?: MCPServer[];
  mcpToolsets?: MCPToolset[];
  mcpServerToolRestrictions?: Record<string, string[]>;
}

/**
 * Build the litellm_proxy MCP reference blocks for a playground request.
 *
 * Every endpoint that supports MCP sends the same reference shape; the gateway
 * expands it server side and each endpoint's own transformation decides the
 * final tool shape. Keeping one builder here stops the endpoints from drifting
 * apart on routing name, label uniqueness, or escaping.
 *
 * server_name is used for both routing and labelling because it is the unique
 * registered identifier; aliases can collide across servers, and a duplicated
 * server_label causes silent tool-routing failures.
 *
 * The name is not percent-encoded: the gateway resolves it with a raw
 * `server_url.split("/")[-1]` and never url-decodes, so an encoded name would
 * fail server lookup rather than round-trip.
 */
export function buildMcpToolBlocks({
  selectedMCPServers,
  mcpServers,
  mcpToolsets,
  mcpServerToolRestrictions,
}: BuildMcpToolBlocksArgs): McpToolBlock[] {
  if (!selectedMCPServers || selectedMCPServers.length === 0) {
    return [];
  }

  if (selectedMCPServers.includes(ALL_MCP_SERVERS_SENTINEL)) {
    return [
      {
        type: "mcp",
        server_label: "litellm",
        server_url: "litellm_proxy/mcp",
        require_approval: "never",
      },
    ];
  }

  return selectedMCPServers.map((serverId) => {
    if (serverId.startsWith(TOOLSET_PREFIX)) {
      const toolsetId = serverId.slice(TOOLSET_PREFIX.length);
      const toolset = mcpToolsets?.find((t) => t.toolset_id === toolsetId);
      const toolsetName = toolset?.toolset_name || toolsetId;
      return {
        type: "mcp",
        server_label: toolsetName,
        server_url: `litellm_proxy/mcp/${toolsetName}`,
        require_approval: "never",
      };
    }

    const server = mcpServers?.find((s) => s.server_id === serverId);
    const routeName = server?.server_name || serverId;
    const allowedTools = mcpServerToolRestrictions?.[serverId] || [];

    return {
      type: "mcp",
      server_label: routeName,
      server_url: `litellm_proxy/mcp/${routeName}`,
      require_approval: "never",
      ...(allowedTools.length > 0 ? { allowed_tools: allowedTools } : {}),
    };
  });
}
