import { ALL_PROXY_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";

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
 * an entry is the direction that widens. An entry is kept while its own server is still reachable,
 * directly or through a retained access group or toolset, and dropped once nothing reaches it, which
 * is what makes removing a grant actually remove it. Reachability is resolved per server rather than
 * per selection: the gateway treats every allowlist key as an independent server grant, so keeping
 * every key because some unrelated group survived would leave a deselected server callable.
 *
 * The catalog carries `mcp_access_groups` on each server and `tools[].server_id` on each toolset,
 * which is the same membership the gateway resolves against. A selected toolset missing from the
 * catalog is unresolvable, so nothing is pruned in that save.
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
  allToolsets: MCPToolset[] = [],
): McpEntitlementUpdate | null => {
  const selection = formValues.mcp_servers_and_groups;
  if (selection === null || typeof selection !== "object") return null;

  const { servers, accessGroups, toolsets } = selection as Record<string, unknown>;
  const mcpServers = asStringArray(servers);
  const mcpAccessGroups = asStringArray(accessGroups);
  const mcpToolsets = asStringArray(toolsets);
  const grantsEveryServer =
    mcpServers.includes(ALL_PROXY_MCP_SERVERS_SENTINEL) ||
    mcpToolsets.some((toolsetId) => !allToolsets.some((toolset) => toolset.toolset_id === toolsetId));

  const toolsetServerIds = new Set(
    allToolsets
      .filter((toolset) => mcpToolsets.includes(toolset.toolset_id))
      .flatMap((toolset) => toolset.tools.map((tool) => tool.server_id)),
  );

  const grants = (server: MCPServer): boolean =>
    mcpServers.some((identifier) => mcpServerMatchesIdentifier(server, identifier)) ||
    (server.mcp_access_groups ?? []).some((group) => mcpAccessGroups.includes(group)) ||
    toolsetServerIds.has(server.server_id);

  const grantsServerNamedBy = (permissionKey: string): boolean => {
    const named = allServers.filter((candidate) => mcpServerMatchesIdentifier(candidate, permissionKey));
    if (named.length === 0) return true;
    return named.some(grants);
  };

  return {
    mcp_servers: mcpServers,
    mcp_access_groups: mcpAccessGroups,
    mcp_toolsets: mcpToolsets,
    mcp_tool_permissions: Object.fromEntries(
      Object.entries(asToolPermissions(formValues.mcp_tool_permissions)).filter(
        ([permissionKey]) => grantsEveryServer || grantsServerNamedBy(permissionKey),
      ),
    ),
  };
};
