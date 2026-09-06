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
  // What the mcp_tool_permissions entries alone allow: the union across every equivalent key.
  // `undefined` means no entry at all, so those keys impose no restriction.
  readonly keyedTools: readonly string[] | undefined;
  // Tools a selected toolset grants on this server. The backend unions them with the keyed tools,
  // so they are allowed whatever this map holds and no edit here can revoke them; the editor shows
  // them allowed and locked rather than as tools an admin is free to turn off.
  readonly toolsetTools: readonly string[] | undefined;
  // What this level actually allows on the server, which is what the backend enforces: the keyed
  // union widened by the toolset grant. `undefined` means nothing restricts the server from here.
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

// Which servers an identifier names, with the same precedence the backend's expand_permission_list
// applies: a string that is a registry server id names exactly that server, and only a string that
// is not falls back to server_name/alias, which can name several. Matching all three fields at once
// would attach a grant to a server the backend never resolves the identifier to, so an edit made
// against that server would hand it access the original grant did not include.
export const mcpServersForIdentifier = (allServers: readonly MCPServer[], identifier: string): readonly MCPServer[] => {
  const byId = allServers.filter((server) => server.server_id === identifier);
  if (byId.length > 0) return byId;
  return allServers.filter((server) => server.server_name === identifier || server.alias === identifier);
};

// Every key in the map that names this server, id first so an id key stays the one an edit keeps.
// A key spelled like this server's name still belongs to another server when that string is that
// server's id, so the catalog decides membership rather than a field-by-field comparison.
export const mcpToolPermissionKeysFor = (
  server: MCPServer,
  toolPermissions: Readonly<Record<string, readonly string[]>>,
  allServers: readonly MCPServer[],
): readonly string[] =>
  [server.server_id, server.server_name, server.alias].filter(
    (identifier): identifier is string =>
      typeof identifier === "string" &&
      Object.hasOwn(toolPermissions, identifier) &&
      mcpServersForIdentifier(allServers, identifier).some((match) => match.server_id === server.server_id),
  );

// A key naming more than one server cannot be written on any one server's behalf: the backend
// unions it into every match, so an edit made here would move the other server's allowlist too.
const mcpKeyNamesOneServerOnly = (allServers: readonly MCPServer[], key: string): boolean =>
  mcpServersForIdentifier(allServers, key).length === 1;

// The key an edit writes: the first one that names this server and no other, falling back to the
// server's own id. When the only entry is a key several servers share, that fallback creates an
// id-keyed entry rather than rewriting the shared one, which would edit the other server too.
export const mcpToolPermissionKeyFor = (
  server: MCPServer,
  toolPermissions: Readonly<Record<string, readonly string[]>>,
  allServers: readonly MCPServer[],
): string =>
  mcpToolPermissionKeysFor(server, toolPermissions, allServers).find((key) =>
    mcpKeyNamesOneServerOnly(allServers, key),
  ) ?? server.server_id;

// The union the backend enforces across equivalent keys, first-seen order preserved.
export const mcpAllowedToolsFor = (
  server: MCPServer,
  toolPermissions: Readonly<Record<string, readonly string[]>>,
  allServers: readonly MCPServer[],
): readonly string[] | undefined => {
  const keys = mcpToolPermissionKeysFor(server, toolPermissions, allServers);
  if (keys.length === 0) return undefined;
  return [...new Set(keys.flatMap((key) => toolPermissions[key] ?? []))];
};

// Tool names the given toolsets grant on this server, `undefined` when they grant none.
const mcpToolsetToolsFor = (
  server: MCPServer,
  selectedToolsets: readonly string[],
  toolsets: readonly MCPToolset[],
): readonly string[] | undefined => {
  const granted = [
    ...new Set(
      toolsets
        .filter((toolset) => selectedToolsets.includes(toolset.toolset_id))
        .flatMap((toolset) =>
          toolset.tools.filter((tool) => tool.server_id === server.server_id).map((tool) => tool.tool_name),
        ),
    ),
  ];
  return granted.length > 0 ? granted : undefined;
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
  // A toolset grant is unioned in by the backend, so copying its tools into this entry would turn a
  // grant that ends with the toolset into a standing one. Leaving out a tool the entry already
  // holds would go the other way and drop a grant that survives the toolset, so only the tools the
  // toolset alone accounts for are withheld.
  const toolsetOnly = (entry.toolsetTools ?? []).filter((tool) => !(entry.keyedTools ?? []).includes(tool));
  const written = allowed.filter((tool) => !toolsetOnly.includes(tool));
  const kept: [string, string[]][] = Object.entries(toolPermissions)
    .filter(([key]) => !entry.supersededKeys.includes(key))
    .map(([key, tools]) => [key, key === entry.permissionKey ? [...written] : [...tools]]);
  const withWrite: [string, string[]][] = Object.hasOwn(toolPermissions, entry.permissionKey)
    ? kept
    : [...kept, [entry.permissionKey, [...written]]];
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
  const entry = (server: MCPServer, source: McpGrantSource): EffectiveMcpServer => {
    const keys = mcpToolPermissionKeysFor(server, toolPermissions, allServers);
    const permissionKey = mcpToolPermissionKeyFor(server, toolPermissions, allServers);
    const editable = keys.filter((key) => key !== permissionKey);
    const keyedTools = mcpAllowedToolsFor(server, toolPermissions, allServers);
    const toolsetTools = mcpToolsetToolsFor(server, selectedToolsets, toolsets);
    return {
      server,
      permissionKey,
      supersededKeys: editable.filter((key) => mcpKeyNamesOneServerOnly(allServers, key)),
      ambiguousKeys: editable.filter((key) => !mcpKeyNamesOneServerOnly(allServers, key)),
      keyedTools,
      toolsetTools,
      allowedTools:
        keyedTools === undefined && toolsetTools === undefined
          ? undefined
          : [...new Set([...(keyedTools ?? []), ...(toolsetTools ?? [])])],
      source,
    };
  };

  const direct = selectedServers.flatMap((identifier) =>
    mcpServersForIdentifier(allServers, identifier).map((server) => entry(server, { kind: "direct" })),
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
    mcpServersForIdentifier(allServers, key).map((server) => entry(server, { kind: "toolPermission" })),
  );

  const candidates = [...direct, ...viaAccessGroups, ...viaToolsets, ...viaToolPermissions];
  return candidates.filter(
    (candidate, index) =>
      candidates.findIndex((other) => other.server.server_id === candidate.server.server_id) === index,
  );
};
