import { ALL_PROXY_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { mcpAllowedToolsFor, mcpServersForIdentifier } from "./effectiveMcpServers";

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
  const directServerIds = new Set(
    mcpServers.flatMap((identifier) =>
      mcpServersForIdentifier(allServers, identifier).map((server) => server.server_id),
    ),
  );

  const grants = (server: MCPServer): boolean =>
    directServerIds.has(server.server_id) ||
    (server.mcp_access_groups ?? []).some((group) => mcpAccessGroups.includes(group)) ||
    toolsetServerIds.has(server.server_id);

  const grantsServerNamedBy = (permissionKey: string): boolean => {
    const named = mcpServersForIdentifier(allServers, permissionKey);
    if (named.length === 0) return true;
    return named.some(grants);
  };

  const toolPermissions = asToolPermissions(formValues.mcp_tool_permissions);
  const splitKeys = new Set(
    Object.keys(toolPermissions).filter((key) => {
      const named = mcpServersForIdentifier(allServers, key);
      return !grantsEveryServer && named.length > 1 && named.some((server) => !grants(server));
    }),
  );
  const retainedServerIds = new Set(
    Array.from(splitKeys).flatMap((key) =>
      mcpServersForIdentifier(allServers, key)
        .filter(grants)
        .map((server) => server.server_id),
    ),
  );
  const keepPermissionKey = (key: string): boolean => {
    if (splitKeys.has(key)) return false;
    const named = mcpServersForIdentifier(allServers, key);
    if (named.length === 1 && retainedServerIds.has(named[0].server_id)) return false;
    return grantsEveryServer || grantsServerNamedBy(key);
  };
  const retainedPermissions = Object.fromEntries(
    Object.entries(toolPermissions).filter(([key]) => keepPermissionKey(key)),
  );
  const canonicalPermissions = Object.fromEntries(
    allServers
      .filter((server) => retainedServerIds.has(server.server_id))
      .map((server) => [server.server_id, [...(mcpAllowedToolsFor(server, toolPermissions, allServers) ?? [])]]),
  );

  return {
    mcp_servers: mcpServers,
    mcp_access_groups: mcpAccessGroups,
    mcp_toolsets: mcpToolsets,
    mcp_tool_permissions: { ...retainedPermissions, ...canonicalPermissions },
  };
};
