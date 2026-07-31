import { z } from "zod/v4";
import { MCPServer, MCPToolset } from "../mcp_tools/types";

// Mirrors the backend resolver's union (direct + access_group + tool_perm + toolset), so the
// editor shows exactly the servers this permission level entitles.
export type McpGrantSource =
  | { readonly kind: "direct" }
  | { readonly kind: "accessGroup"; readonly name: string }
  | { readonly kind: "toolset"; readonly name: string }
  | { readonly kind: "toolPermission" };

export interface EffectiveMcpServer {
  readonly server: MCPServer;
  // The mcp_tool_permissions key an edit writes to. The backend accepts a server id, name or
  // alias interchangeably, so an API- or config-written entry may use any of them; writing the
  // key already in the map keeps an edit from leaving the original entry behind.
  readonly permissionKey: string;
  // Other keys in the map that name this same server. The backend unions every key's list, so a
  // write that touched only `permissionKey` would leave these still granting.
  readonly supersededKeys: readonly string[];
  // Keys naming this server that name another server too, which happens when two servers share a
  // name or alias. They are kept rather than collapsed, so their tools cannot be revoked here; the
  // editor has to say so rather than let an edit look like it narrowed the grant.
  readonly ambiguousKeys: readonly string[];
  // What this level currently allows on the server: the union across every equivalent key, which
  // is what the backend enforces. `undefined` means no entry at all, so no restriction from here.
  readonly allowedTools: readonly string[] | undefined;
  readonly source: McpGrantSource;
}

interface ResolveInput {
  readonly allServers: readonly MCPServer[];
  readonly selectedServers: readonly string[];
  readonly selectedAccessGroups: readonly string[];
  readonly selectedToolsets: readonly string[];
  readonly toolsets: readonly MCPToolset[];
  readonly toolPermissions: Readonly<Record<string, readonly string[]>>;
}

// Access groups come back as plain names, but older records carry `{ name }` objects.
const accessGroupRefSchema = z.union([z.string(), z.object({ name: z.string() })]);

const accessGroupNamesOf = (server: MCPServer): readonly string[] =>
  (server.mcp_access_groups ?? []).flatMap((group) => {
    const parsed = accessGroupRefSchema.safeParse(group);
    if (!parsed.success) return [];
    return [typeof parsed.data === "string" ? parsed.data : parsed.data.name];
  });

export const mcpServerMatchesIdentifier = (server: MCPServer, identifier: string): boolean =>
  server.server_id === identifier || server.server_name === identifier || server.alias === identifier;

// Every key in the map that names this server, id first so an id key stays the one an edit keeps.
export const mcpToolPermissionKeysFor = (
  server: MCPServer,
  toolPermissions: Readonly<Record<string, readonly string[]>>,
): readonly string[] =>
  [server.server_id, server.server_name, server.alias].filter(
    (identifier): identifier is string => typeof identifier === "string" && Object.hasOwn(toolPermissions, identifier),
  );

export const mcpToolPermissionKeyFor = (
  server: MCPServer,
  toolPermissions: Readonly<Record<string, readonly string[]>>,
): string => mcpToolPermissionKeysFor(server, toolPermissions)[0] ?? server.server_id;

// The union the backend enforces across equivalent keys, first-seen order preserved.
export const mcpAllowedToolsFor = (
  server: MCPServer,
  toolPermissions: Readonly<Record<string, readonly string[]>>,
): readonly string[] | undefined => {
  const keys = mcpToolPermissionKeysFor(server, toolPermissions);
  if (keys.length === 0) return undefined;
  return [...new Set(keys.flatMap((key) => toolPermissions[key] ?? []))];
};

// Collapse a server's grant onto one key: the kept key gets exactly what the admin sees, and the
// equivalent keys are dropped so nothing keeps granting under another spelling. A key that also
// names a DIFFERENT server (duplicate server names) is never dropped, since that would silently
// strip the other server's restriction.
export const applyToolPermissionWrite = ({
  toolPermissions,
  entry,
  allowed,
}: {
  readonly toolPermissions: Readonly<Record<string, readonly string[]>>;
  readonly entry: EffectiveMcpServer;
  readonly allowed: readonly string[];
}): Record<string, string[]> => {
  const kept: [string, string[]][] = Object.entries(toolPermissions)
    .filter(([key]) => !entry.supersededKeys.includes(key))
    .map(([key, tools]) => [key, key === entry.permissionKey ? [...allowed] : [...tools]]);
  const withWrite: [string, string[]][] = Object.hasOwn(toolPermissions, entry.permissionKey)
    ? kept
    : [...kept, [entry.permissionKey, [...allowed]]];
  return Object.fromEntries(withWrite);
};

export const resolveEffectiveMcpServers = ({
  allServers,
  selectedServers,
  selectedAccessGroups,
  selectedToolsets,
  toolsets,
  toolPermissions,
}: ResolveInput): readonly EffectiveMcpServer[] => {
  const namesOneServerOnly = (key: string): boolean =>
    allServers.filter((server) => mcpServerMatchesIdentifier(server, key)).length === 1;

  const entry = (server: MCPServer, source: McpGrantSource): EffectiveMcpServer => {
    const keys = mcpToolPermissionKeysFor(server, toolPermissions);
    const permissionKey = keys[0] ?? server.server_id;
    const editable = keys.filter((key) => key !== permissionKey);
    return {
      server,
      permissionKey,
      supersededKeys: editable.filter((key) => namesOneServerOnly(key)),
      ambiguousKeys: editable.filter((key) => !namesOneServerOnly(key)),
      allowedTools: mcpAllowedToolsFor(server, toolPermissions),
      source,
    };
  };

  const direct = selectedServers.flatMap((identifier) =>
    allServers
      .filter((server) => mcpServerMatchesIdentifier(server, identifier))
      .map((server) => entry(server, { kind: "direct" })),
  );

  const viaAccessGroups = selectedAccessGroups.flatMap((group) =>
    allServers
      .filter((server) => accessGroupNamesOf(server).includes(group))
      .map((server) => entry(server, { kind: "accessGroup", name: group })),
  );

  const viaToolsets = selectedToolsets.flatMap((toolsetId) => {
    const toolset = toolsets.find((candidate) => candidate.toolset_id === toolsetId);
    if (!toolset) return [];
    const toolsetServerIds = new Set(toolset.tools.map((tool) => tool.server_id));
    return allServers
      .filter((server) => toolsetServerIds.has(server.server_id))
      .map((server) => entry(server, { kind: "toolset", name: toolset.toolset_name }));
  });

  // A server named only under mcp_tool_permissions is entitled on purpose, so it belongs in the
  // editor: without it, an entry left over from a removed access group is invisible and unclearable.
  const viaToolPermissions = Object.keys(toolPermissions).flatMap((key) =>
    allServers
      .filter((server) => mcpServerMatchesIdentifier(server, key))
      .map((server) => entry(server, { kind: "toolPermission" })),
  );

  const candidates = [...direct, ...viaAccessGroups, ...viaToolsets, ...viaToolPermissions];
  return candidates.filter(
    (candidate, index) =>
      candidates.findIndex((other) => other.server.server_id === candidate.server.server_id) === index,
  );
};
