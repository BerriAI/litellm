import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import type { ObjectPermission } from "@/components/object_permission_types";
import {
  mcpAllowedToolsFor,
  mcpServersForIdentifier,
  mcpToolPermissionKeysFor,
  resolveEffectiveMcpServers,
} from "./effectiveMcpServers";

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

const formMcpGrant = (formValues: Record<string, unknown>): McpEntitlementUpdate | null => {
  const selection = formValues.mcp_servers_and_groups;
  if (selection === null || typeof selection !== "object") return null;
  const { servers, accessGroups, toolsets } = selection as Record<string, unknown>;
  return {
    mcp_servers: asStringArray(servers),
    mcp_access_groups: asStringArray(accessGroups),
    mcp_toolsets: asStringArray(toolsets),
    mcp_tool_permissions: asToolPermissions(formValues.mcp_tool_permissions),
  };
};

export const extractMcpEntitlement = (
  formValues: Record<string, unknown>,
  allServers: MCPServer[],
  allToolsets: MCPToolset[] = [],
): McpEntitlementUpdate | null => {
  const formGrant = formMcpGrant(formValues);
  if (!formGrant) return null;
  const {
    mcp_servers: mcpServers,
    mcp_access_groups: mcpAccessGroups,
    mcp_toolsets: mcpToolsets,
    mcp_tool_permissions: toolPermissions,
  } = formGrant;
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

const normalizedStrings = (values: string[]): string[] => Array.from(new Set(values)).toSorted();

const grantIdentity = (grant: McpEntitlementUpdate): string => {
  const normalized = {
    mcp_servers: normalizedStrings(grant.mcp_servers),
    mcp_access_groups: normalizedStrings(grant.mcp_access_groups),
    mcp_toolsets: normalizedStrings(grant.mcp_toolsets),
    mcp_tool_permissions: Object.fromEntries(
      Object.keys(grant.mcp_tool_permissions)
        .toSorted()
        .map((server) => [server, normalizedStrings(grant.mcp_tool_permissions[server])]),
    ),
  };
  return JSON.stringify(normalized);
};

const independentToolPermissions = (
  storedGrant: McpEntitlementUpdate,
  toolPermissions: Record<string, string[]>,
  allServers: MCPServer[],
  allToolsets: MCPToolset[],
): Record<string, string[]> => {
  const storedSelection = {
    allServers,
    selectedServers: storedGrant.mcp_servers,
    selectedAccessGroups: storedGrant.mcp_access_groups,
    selectedToolsets: storedGrant.mcp_toolsets,
    toolsets: allToolsets,
    toolPermissions: {},
  };
  const selectedServers = resolveEffectiveMcpServers(storedSelection);
  const selectedServerIds = new Set(selectedServers.map(({ server }) => server.server_id));
  const keyOnlyServers = Object.keys(storedGrant.mcp_tool_permissions)
    .flatMap((key) => mcpServersForIdentifier(allServers, key))
    .filter((server) => !selectedServerIds.has(server.server_id));
  const keyOnlyServerIds = new Set(keyOnlyServers.map((server) => server.server_id));
  const namesOnlyKeyOnlyServers = (key: string): boolean => {
    const named = mcpServersForIdentifier(allServers, key);
    return named.length > 0 && named.every((server) => keyOnlyServerIds.has(server.server_id));
  };
  const independentPermissions = Object.fromEntries(
    Object.entries(toolPermissions).filter(([key]) => namesOnlyKeyOnlyServers(key)),
  );
  const canonicalPermissions = Object.fromEntries(
    keyOnlyServers
      .filter((server) => mcpToolPermissionKeysFor(server, independentPermissions, allServers).length === 0)
      .flatMap((server) => {
        const allowed = mcpAllowedToolsFor(server, toolPermissions, allServers);
        return allowed ? [[server.server_id, [...allowed]]] : [];
      }),
  );
  return { ...independentPermissions, ...canonicalPermissions };
};

export const extractMcpEntitlementUpdate = (
  formValues: Record<string, unknown>,
  existingPermission: ObjectPermission | null | undefined,
  allServers: MCPServer[],
  allToolsets: MCPToolset[] = [],
): McpEntitlementUpdate | null => {
  const formGrant = formMcpGrant(formValues);
  if (!formGrant) return null;
  const storedGrant = {
    mcp_servers: existingPermission?.mcp_servers ?? [],
    mcp_access_groups: existingPermission?.mcp_access_groups ?? [],
    mcp_toolsets: existingPermission?.mcp_toolsets ?? [],
    mcp_tool_permissions: existingPermission?.mcp_tool_permissions ?? {},
  };
  const storedIdentity = grantIdentity(storedGrant);
  if (grantIdentity(formGrant) === storedIdentity) return null;
  const extracted = extractMcpEntitlement(formValues, allServers, allToolsets);
  if (!extracted) return null;
  const grant = {
    ...extracted,
    mcp_tool_permissions: formGrant.mcp_servers.includes(NO_MCP_SERVERS_SENTINEL)
      ? {}
      : {
          ...extracted.mcp_tool_permissions,
          ...independentToolPermissions(storedGrant, formGrant.mcp_tool_permissions, allServers, allToolsets),
        },
  };
  return grantIdentity(grant) === storedIdentity ? null : grant;
};
