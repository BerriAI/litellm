import { ALL_PROXY_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import { MCPServer } from "@/components/mcp_tools/types";

export interface McpEntitlementUpdate {
  mcp_servers: string[];
  mcp_access_groups: string[];
  mcp_toolsets: string[];
  mcp_tool_permissions: Record<string, string[]>;
}

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];

const asToolPermissions = (value: unknown): Record<string, string[]> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([serverId, tools]) => [serverId, asStringArray(tools)]),
  );
};

const mcpServerMatchesIdentifier = (server: MCPServer, identifier: string): boolean =>
  server.server_id === identifier || server.server_name === identifier || server.alias === identifier;

/**
 * The `object_permission` a save sends, derived from what the editor currently shows.
 *
 * A tool allowlist is what narrows a grant and an absent one reads as no restriction, so dropping
 * an entry is the direction that widens. An entry is kept when an access group, a toolset, or the
 * all-proxy grant the admin retained could still supply its server, and dropped once nothing
 * indirect survives, which is what makes removing a grant actually remove it.
 *
 * A tool-permission key may be a server id, a name or an alias: the gateway normalizes all three
 * before looking up the allowlist, so an entry written by the API or by config can use any of them.
 * `allServers` is what resolves a key to its servers, plural: names and aliases are not unique, and
 * the gateway unions such a key into EVERY server answering to it, so the entry is kept while any
 * one of them is still granted. Resolving to the first match instead would make the outcome depend
 * on catalog order and could drop a restriction that was also covering a server still granted. A key
 * that resolves to nothing is kept too, since a server we cannot identify is one we cannot confirm
 * was deselected; that also covers a catalog that has not loaded or failed to load, where every key
 * is unresolvable and nothing is pruned.
 */
export const extractMcpEntitlement = (
  formValues: Record<string, unknown>,
  allServers: MCPServer[],
): McpEntitlementUpdate | null => {
  const selection = formValues.mcp_servers_and_groups;
  if (selection === null || typeof selection !== "object") return null;

  const { servers, accessGroups, toolsets } = selection as Record<string, unknown>;
  const mcpServers = asStringArray(servers);
  const mcpAccessGroups = asStringArray(accessGroups);
  const mcpToolsets = asStringArray(toolsets);
  const retainsIndirectGrant =
    mcpAccessGroups.length > 0 || mcpToolsets.length > 0 || mcpServers.includes(ALL_PROXY_MCP_SERVERS_SENTINEL);

  const grantsServerNamedBy = (permissionKey: string): boolean => {
    const named = allServers.filter((candidate) => mcpServerMatchesIdentifier(candidate, permissionKey));
    if (named.length === 0) return true;
    return named.some((server) => mcpServers.some((identifier) => mcpServerMatchesIdentifier(server, identifier)));
  };

  return {
    mcp_servers: mcpServers,
    mcp_access_groups: mcpAccessGroups,
    mcp_toolsets: mcpToolsets,
    mcp_tool_permissions: Object.fromEntries(
      Object.entries(asToolPermissions(formValues.mcp_tool_permissions)).filter(
        ([permissionKey]) => retainsIndirectGrant || grantsServerNamedBy(permissionKey),
      ),
    ),
  };
};
