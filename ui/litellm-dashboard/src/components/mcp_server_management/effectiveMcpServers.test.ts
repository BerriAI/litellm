import { describe, it, expect } from "vitest";
import { MCPServer, MCPToolset } from "../mcp_tools/types";
import {
  applyToolCheckboxWrite,
  applyToolDenyWrite,
  applyToolPermissionWrite,
  EffectiveMcpServer,
  emptyMcpAccessGroups,
  mcpAllowedToolsFor,
  mcpServersForIdentifier,
  mcpToolPermissionKeyFor,
  resolveEffectiveMcpServers,
} from "./effectiveMcpServers";

const server = (overrides: Partial<MCPServer> & { server_id: string }): MCPServer =>
  ({
    server_name: null,
    alias: null,
    created_at: "2026-01-01",
    created_by: "admin",
    updated_at: "2026-01-01",
    updated_by: "admin",
    ...overrides,
  }) as MCPServer;

const toolset = (overrides: Partial<MCPToolset> & { toolset_id: string }): MCPToolset =>
  ({ toolset_name: overrides.toolset_id, tools: [], ...overrides }) as MCPToolset;

const emptyInput = {
  allServers: [] as readonly MCPServer[],
  selectedServers: [] as readonly string[],
  selectedAccessGroups: [] as readonly string[],
  selectedToolsets: [] as readonly string[],
  toolsets: [] as readonly MCPToolset[],
  toolPermissions: {} as Readonly<Record<string, readonly string[]>>,
  deniedTools: {} as Readonly<Record<string, readonly string[]>>,
};

describe("mcpServersForIdentifier", () => {
  const target = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });

  it("matches on id, server name and alias alike", () => {
    expect(mcpServersForIdentifier([target], "uuid-1")).toEqual([target]);
    expect(mcpServersForIdentifier([target], "github_mcp")).toEqual([target]);
    expect(mcpServersForIdentifier([target], "GitHub")).toEqual([target]);
    expect(mcpServersForIdentifier([target], "other")).toEqual([]);
  });

  it("names every server sharing a duplicated name, as the backend does", () => {
    const twin = server({ server_id: "uuid-2", server_name: "github_mcp" });

    expect(mcpServersForIdentifier([target, twin], "github_mcp").map((match) => match.server_id)).toEqual([
      "uuid-1",
      "uuid-2",
    ]);
  });

  // The backend's expand_permission_list resolves a registry server id to that server and stops;
  // only a string that is no server's id falls through to the name/alias pass. An identifier can
  // name several servers, so this has to hold in either catalog order.
  it.each([
    { label: "id owner first", idOwnerFirst: true },
    { label: "name twin first", idOwnerFirst: false },
  ])("resolves an id to its own server even when another server is named after it ($label)", ({ idOwnerFirst }) => {
    const byId = server({ server_id: "collide", server_name: "Payments" });
    const byName = server({ server_id: "uuid-9", server_name: "collide" });
    const catalog = idOwnerFirst ? [byId, byName] : [byName, byId];

    expect(mcpServersForIdentifier(catalog, "collide").map((match) => match.server_id)).toEqual(["collide"]);
  });
});

describe("mcpToolPermissionKeyFor", () => {
  const target = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });

  it("returns the existing name key so an edit does not fork into a second entry", () => {
    expect(mcpToolPermissionKeyFor(target, { github_mcp: ["list_issues"] }, [target])).toBe("github_mcp");
  });

  // The map is a second collection of non-unique identifiers for one server, and this picks a
  // winner from it, so both write orders have to hold or a map-order winner would slip through.
  it.each([
    { label: "id key first", toolPermissions: { "uuid-1": ["list_prs"], github_mcp: ["list_issues"] } },
    { label: "name key first", toolPermissions: { github_mcp: ["list_issues"], "uuid-1": ["list_prs"] } },
  ])("prefers the id key when both an id and a name key exist ($label)", ({ toolPermissions }) => {
    expect(mcpToolPermissionKeyFor(target, toolPermissions, [target])).toBe("uuid-1");
  });

  it("falls back to the server id when no entry exists yet", () => {
    expect(mcpToolPermissionKeyFor(target, {}, [target])).toBe("uuid-1");
  });

  // Writing this key would hand the entry's tools to the server that owns the id, not to the one
  // being edited, so it is not this server's key however much its name looks like it.
  it.each([
    { label: "id owner first", idOwnerFirst: true },
    { label: "name twin first", idOwnerFirst: false },
  ])("ignores a key that is another server's id ($label)", ({ idOwnerFirst }) => {
    const byId = server({ server_id: "collide", server_name: "Payments" });
    const byName = server({ server_id: "uuid-9", server_name: "collide" });
    const catalog = idOwnerFirst ? [byId, byName] : [byName, byId];

    expect(mcpToolPermissionKeyFor(byName, { collide: ["list_issues"] }, catalog)).toBe("uuid-9");
    expect(mcpAllowedToolsFor(byName, { collide: ["list_issues"] }, catalog)).toBeUndefined();
    expect(mcpAllowedToolsFor(byId, { collide: ["list_issues"] }, catalog)).toEqual(["list_issues"]);
  });
});

describe("emptyMcpAccessGroups", () => {
  const grouped = server({ server_id: "srv-group", server_name: "Grouped", mcp_access_groups: ["prod"] });
  const objectGrouped = {
    ...grouped,
    server_id: "srv-obj",
    mcp_access_groups: [{ name: "legacy" }],
  } as unknown as MCPServer;

  it("names only the selected groups no loaded server belongs to", () => {
    expect(emptyMcpAccessGroups([grouped, objectGrouped], [], ["prod", "legacy", "ops_readonly"])).toEqual([
      "ops_readonly",
    ]);
  });

  it("names every selected group when no server is loaded and the registry is empty", () => {
    expect(emptyMcpAccessGroups([], [], ["prod"])).toEqual(["prod"]);
  });

  it("trusts the group registry when the caller's catalog hides the member servers", () => {
    expect(emptyMcpAccessGroups([], ["prod"], ["prod", "ops_readonly"])).toEqual(["ops_readonly"]);
  });
});

describe("resolveEffectiveMcpServers", () => {
  const direct = server({ server_id: "srv-direct", server_name: "Direct" });
  const grouped = server({ server_id: "srv-group", server_name: "Grouped", mcp_access_groups: ["prod"] });
  const inToolset = server({ server_id: "srv-toolset", server_name: "Toolsetted" });

  it("resolves a selected access group to its member servers", () => {
    const resolved = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: [direct, grouped],
      selectedAccessGroups: ["prod"],
    });

    expect(resolved).toEqual([
      {
        server: grouped,
        permissionKey: "srv-group",
        supersededKeys: [],
        ambiguousKeys: [],
        keyedTools: undefined,
        toolsetTools: undefined,
        allowedTools: undefined,
        source: { kind: "accessGroup", name: "prod" },
      },
    ]);
  });

  it("resolves access groups stored as objects rather than plain names", () => {
    const objectGrouped = { ...grouped, mcp_access_groups: [{ name: "prod" }] } as unknown as MCPServer;

    const resolved = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: [objectGrouped],
      selectedAccessGroups: ["prod"],
    });

    expect(resolved.map((entry) => entry.server.server_id)).toEqual(["srv-group"]);
  });

  it("resolves a selected toolset to the servers its tools live on", () => {
    const input = {
      ...emptyInput,
      allServers: [direct, inToolset],
      selectedToolsets: ["ts-1"],
      toolsets: [
        toolset({
          toolset_id: "ts-1",
          toolset_name: "Support",
          tools: [{ server_id: "srv-toolset", tool_name: "list_issues" }],
        }),
      ],
    };
    const resolved = resolveEffectiveMcpServers(input);

    expect(resolved).toEqual([
      {
        server: inToolset,
        permissionKey: "srv-toolset",
        supersededKeys: [],
        ambiguousKeys: [],
        keyedTools: undefined,
        toolsetTools: ["list_issues"],
        allowedTools: ["list_issues"],
        source: { kind: "toolset", name: "Support" },
      },
    ]);
  });

  it("yields nothing for a toolset that is not in the loaded list", () => {
    const input = {
      ...emptyInput,
      allServers: [inToolset],
      selectedToolsets: ["ts-missing"],
      toolsets: [],
    };
    const resolved = resolveEffectiveMcpServers(input);

    expect(resolved).toEqual([]);
  });

  it("includes a server that only a tool-permission entry names", () => {
    const resolved = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: [grouped],
      toolPermissions: { "srv-group": ["list_issues"] },
    });

    expect(resolved).toEqual([
      {
        server: grouped,
        permissionKey: "srv-group",
        supersededKeys: [],
        ambiguousKeys: [],
        keyedTools: ["list_issues"],
        toolsetTools: undefined,
        allowedTools: ["list_issues"],
        source: { kind: "toolPermission" },
      },
    ]);
  });

  it("reports a server once, attributing it to the strongest grant", () => {
    const input = {
      ...emptyInput,
      allServers: [grouped],
      selectedServers: ["srv-group"],
      selectedAccessGroups: ["prod"],
      toolPermissions: { "srv-group": ["list_issues"] },
    };
    const resolved = resolveEffectiveMcpServers(input);

    expect(resolved).toEqual([
      {
        server: grouped,
        permissionKey: "srv-group",
        supersededKeys: [],
        ambiguousKeys: [],
        keyedTools: ["list_issues"],
        toolsetTools: undefined,
        allowedTools: ["list_issues"],
        source: { kind: "direct" },
      },
    ]);
  });

  it("resolves a server selected by name", () => {
    const named = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });

    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["github_mcp"],
      toolPermissions: { github_mcp: ["list_issues"] },
    };
    const resolved = resolveEffectiveMcpServers(input);

    expect(resolved).toEqual([
      {
        server: named,
        permissionKey: "github_mcp",
        supersededKeys: [],
        ambiguousKeys: [],
        keyedTools: ["list_issues"],
        toolsetTools: undefined,
        allowedTools: ["list_issues"],
        source: { kind: "direct" },
      },
    ]);
  });

  // The backend resolves a selection that is a registry id to that server alone, so a server that
  // merely answers to the same string is not in the grant and must not become editable here: an
  // edit would write its own id into mcp_tool_permissions, which is itself a grant.
  it.each([
    { label: "id owner first", idOwnerFirst: true },
    { label: "name twin first", idOwnerFirst: false },
  ])("does not resolve a selected id to a server merely named after it ($label)", ({ idOwnerFirst }) => {
    const byId = server({ server_id: "collide", server_name: "Payments" });
    const byName = server({ server_id: "uuid-9", server_name: "collide" });

    const resolved = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: idOwnerFirst ? [byId, byName] : [byName, byId],
      selectedServers: ["collide"],
    });

    expect(resolved.map((entry) => entry.server.server_id)).toEqual(["collide"]);
  });

  it("resolves every server sharing a duplicated name, as the backend does", () => {
    const first = server({ server_id: "uuid-1", server_name: "shared" });
    const second = server({ server_id: "uuid-2", server_name: "shared" });

    const resolved = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: [first, second],
      selectedServers: ["shared"],
    });

    expect(resolved.map((entry) => entry.server.server_id)).toEqual(["uuid-1", "uuid-2"]);
  });
});

describe("equivalent permission keys for one server", () => {
  const named = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });
  const other = server({ server_id: "uuid-2", server_name: "other_mcp" });

  it("unions every equivalent key, which is what the backend enforces", () => {
    expect(mcpAllowedToolsFor(named, { "uuid-1": ["list_issues"], github_mcp: ["create_issue"] }, [named])).toEqual([
      "list_issues",
      "create_issue",
    ]);
  });

  it("reports the extra keys as superseded so a write can collapse them", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      toolPermissions: { "uuid-1": ["list_issues"], github_mcp: ["create_issue"], GitHub: ["delete_issue"] },
    };
    const resolved = resolveEffectiveMcpServers(input);

    expect(resolved).toHaveLength(1);
    expect(resolved[0].permissionKey).toBe("uuid-1");
    expect(resolved[0].supersededKeys).toEqual(["github_mcp", "GitHub"]);
    expect(resolved[0].ambiguousKeys).toEqual([]);
    expect(resolved[0].allowedTools).toEqual(["list_issues", "create_issue", "delete_issue"]);
  });

  it("collapses a write onto the kept key and drops the equivalents", () => {
    const toolPermissions = { "uuid-1": ["list_issues"], github_mcp: ["create_issue"], "uuid-2": ["ping"] };
    const input = {
      ...emptyInput,
      allServers: [named, other],
      selectedServers: ["uuid-1"],
      toolPermissions,
    };
    const [entry] = resolveEffectiveMcpServers(input);

    expect(applyToolPermissionWrite({ toolPermissions, entry, allowed: ["list_issues"] })).toEqual({
      "uuid-1": ["list_issues"],
      "uuid-2": ["ping"],
    });
  });

  it("leaves a single-key server, and every other server, untouched", () => {
    const toolPermissions = { github_mcp: ["list_issues"], "uuid-2": ["ping"] };
    const input = {
      ...emptyInput,
      allServers: [named, other],
      selectedServers: ["github_mcp"],
      toolPermissions,
    };
    const [entry] = resolveEffectiveMcpServers(input);

    expect(entry.supersededKeys).toEqual([]);
    expect(entry.ambiguousKeys).toEqual([]);
    expect(applyToolPermissionWrite({ toolPermissions, entry, allowed: [] })).toEqual({
      github_mcp: [],
      "uuid-2": ["ping"],
    });
  });

  // A name resolves to several servers, so a first-match implementation is right in one catalog
  // order and wrong in the other; both orders have to hold for this to pin anything.
  it.each([
    { label: "edited server first", editedFirst: true },
    { label: "other server first", editedFirst: false },
  ])("never drops a key that also names a different server ($label)", ({ editedFirst }) => {
    const firstShared = server({ server_id: "uuid-1", server_name: "shared" });
    const secondShared = server({ server_id: "uuid-2", server_name: "shared" });
    const toolPermissions = { "uuid-1": ["list_issues"], shared: ["create_issue"] };

    const input = {
      ...emptyInput,
      allServers: editedFirst ? [firstShared, secondShared] : [secondShared, firstShared],
      selectedServers: ["uuid-1"],
      toolPermissions,
    };
    const [entry] = resolveEffectiveMcpServers(input);

    expect(entry.supersededKeys).toEqual([]);
    expect(entry.ambiguousKeys).toEqual(["shared"]);
    expect(applyToolPermissionWrite({ toolPermissions, entry, allowed: ["list_issues"] })).toEqual({
      "uuid-1": ["list_issues"],
      shared: ["create_issue"],
    });
  });

  // The shared key is the only entry, so it would otherwise be the key an edit writes; writing it
  // moves the other server's allowlist too, which is the same widening the secondary-key guard
  // exists to prevent. Both catalog orders, since a shared name resolves to several servers.
  it.each([
    { label: "edited server first", editedFirst: true },
    { label: "other server first", editedFirst: false },
  ])("never writes through a shared key, even as a server's only entry ($label)", ({ editedFirst }) => {
    const firstShared = server({ server_id: "uuid-1", server_name: "shared" });
    const secondShared = server({ server_id: "uuid-2", server_name: "shared" });
    const toolPermissions = { shared: ["list_issues"] };

    const input = {
      ...emptyInput,
      allServers: editedFirst ? [firstShared, secondShared] : [secondShared, firstShared],
      selectedServers: ["uuid-1"],
      toolPermissions,
    };
    const resolved = resolveEffectiveMcpServers(input);

    const edited = resolved.find((entry) => entry.server.server_id === "uuid-1")!;
    expect(edited.permissionKey).toBe("uuid-1");
    expect(edited.supersededKeys).toEqual([]);
    expect(edited.ambiguousKeys).toEqual(["shared"]);
    // What the backend enforces on this server today, which is what the card has to show.
    expect(edited.allowedTools).toEqual(["list_issues"]);
    expect(
      applyToolPermissionWrite({ toolPermissions, entry: edited, allowed: ["list_issues", "create_issue"] }),
    ).toEqual({
      shared: ["list_issues"],
      "uuid-1": ["list_issues", "create_issue"],
    });
  });

  it("adds an entry for a server that had none", () => {
    const [entry] = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
    });

    expect(applyToolPermissionWrite({ toolPermissions: {}, entry, allowed: ["list_issues"] })).toEqual({
      "uuid-1": ["list_issues"],
    });
  });

  // A key that is another server's id belongs to that server, so reading it here would overstate
  // what this one allows and writing it would hand this server's tools to the other one.
  it.each([
    { label: "id owner first", idOwnerFirst: true },
    { label: "name twin first", idOwnerFirst: false },
  ])("does not treat another server's id as this server's key ($label)", ({ idOwnerFirst }) => {
    const byId = server({ server_id: "collide", server_name: "Payments" });
    const byName = server({ server_id: "uuid-9", server_name: "collide" });
    const toolPermissions = { collide: ["list_issues"] };

    const input = {
      ...emptyInput,
      allServers: idOwnerFirst ? [byId, byName] : [byName, byId],
      selectedServers: ["uuid-9"],
      toolPermissions,
    };
    const resolved = resolveEffectiveMcpServers(input);

    const edited = resolved.find((entry) => entry.server.server_id === "uuid-9")!;
    expect(edited.permissionKey).toBe("uuid-9");
    expect(edited.supersededKeys).toEqual([]);
    expect(edited.allowedTools).toBeUndefined();
    expect(applyToolPermissionWrite({ toolPermissions, entry: edited, allowed: ["ping"] })).toEqual({
      collide: ["list_issues"],
      "uuid-9": ["ping"],
    });
  });
});

// The backend adds a toolset's tools to whatever mcp_tool_permissions holds, so a toolset grant is
// part of what this level allows and none of it can be revoked by writing that map.
describe("tools a selected toolset grants", () => {
  const inToolset = server({ server_id: "srv-toolset", server_name: "Toolsetted" });
  const support = toolset({
    toolset_id: "ts-1",
    toolset_name: "Support",
    tools: [
      { server_id: "srv-toolset", tool_name: "list_issues" },
      { server_id: "srv-other", tool_name: "ignored" },
    ],
  });

  const resolveOne = (toolPermissions: Readonly<Record<string, readonly string[]>>) => {
    const input = {
      ...emptyInput,
      allServers: [inToolset],
      selectedToolsets: ["ts-1"],
      toolsets: [support],
      toolPermissions,
    };
    return resolveEffectiveMcpServers(input)[0];
  };

  it("reports them as allowed rather than leaving the server unrestricted", () => {
    const entry = resolveOne({});

    expect(entry.toolsetTools).toEqual(["list_issues"]);
    expect(entry.allowedTools).toEqual(["list_issues"]);
  });

  it("unions them with what the permission key allows", () => {
    const entry = resolveOne({ "srv-toolset": ["create_issue"] });

    expect(entry.keyedTools).toEqual(["create_issue"]);
    expect(entry.allowedTools).toEqual(["create_issue", "list_issues"]);
  });

  it("keeps a write from copying a toolset tool into the permission entry", () => {
    const entry = resolveOne({});

    expect(applyToolPermissionWrite({ toolPermissions: {}, entry, allowed: ["list_issues", "create_issue"] })).toEqual({
      "srv-toolset": ["create_issue"],
    });
  });

  it("keeps a tool the entry already holds even when the toolset grants it too", () => {
    const toolPermissions = { "srv-toolset": ["list_issues", "create_issue"] };
    const entry = resolveOne(toolPermissions);

    expect(applyToolPermissionWrite({ toolPermissions, entry, allowed: ["list_issues", "create_issue"] })).toEqual({
      "srv-toolset": ["list_issues", "create_issue"],
    });
  });

  it("still narrows a tool only the permission entry grants", () => {
    const toolPermissions = { "srv-toolset": ["create_issue", "delete_repo"] };
    const entry = resolveOne(toolPermissions);

    expect(applyToolPermissionWrite({ toolPermissions, entry, allowed: ["list_issues", "create_issue"] })).toEqual({
      "srv-toolset": ["create_issue"],
    });
  });

  it("leaves a server no selected toolset names unrestricted", () => {
    const input = {
      ...emptyInput,
      allServers: [inToolset],
      selectedServers: ["srv-toolset"],
      selectedToolsets: [],
      toolsets: [support],
    };
    const untouched = resolveEffectiveMcpServers(input)[0];

    expect(untouched.toolsetTools).toBeUndefined();
    expect(untouched.allowedTools).toBeUndefined();
  });
});

// The denylist write replaces the legacy allowlist write: an unchecked fetched tool lands under
// the server's key in mcp_tool_denied_tools, and selecting everything removes the entry entirely so
// a tool the upstream adds later is still allowed.
describe("applyToolDenyWrite", () => {
  const named = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });
  const other = server({ server_id: "uuid-2", server_name: "other_mcp" });
  const fetched = ["list_issues", "create_issue", "delete_issue"];

  const entryFor = (
    toolPermissions: Readonly<Record<string, readonly string[]>>,
    deniedTools: Readonly<Record<string, readonly string[]>> = {},
  ) => {
    const input = {
      ...emptyInput,
      allServers: [named, other],
      selectedServers: ["uuid-1"],
      toolPermissions,
      deniedTools,
    };
    return resolveEffectiveMcpServers(input)[0];
  };

  const writeInput = (
    entry: EffectiveMcpServer,
    checked: readonly string[],
    over: {
      toolPermissions?: Readonly<Record<string, readonly string[]>>;
      deniedTools?: Readonly<Record<string, readonly string[]>>;
      allServers?: readonly MCPServer[];
    } = {},
  ) => ({
    toolPermissions: over.toolPermissions ?? {},
    deniedTools: over.deniedTools ?? {},
    entry,
    allServers: over.allServers ?? [named, other],
    fetchedTools: fetched,
    checked,
  });

  it("writes the unchecked fetched tools as denied and clears the server's allowlist keys", () => {
    const toolPermissions = { "uuid-1": ["list_issues"], github_mcp: ["create_issue"], "uuid-2": ["ping"] };
    const entry = entryFor(toolPermissions);
    const input = writeInput(entry, ["list_issues"], { toolPermissions });

    expect(applyToolDenyWrite(input)).toEqual({
      toolPermissions: { "uuid-2": ["ping"] },
      deniedTools: { "uuid-1": ["create_issue", "delete_issue"] },
    });
  });

  it("removes the denylist entry entirely when everything is checked", () => {
    const deniedTools = { "uuid-1": ["delete_issue"], "uuid-2": ["other_tool"] };
    const entry = entryFor({}, deniedTools);
    const input = writeInput(entry, fetched, { deniedTools });

    expect(applyToolDenyWrite(input)).toEqual({ toolPermissions: {}, deniedTools: { "uuid-2": ["other_tool"] } });
  });

  it("denies every fetched tool when nothing is checked", () => {
    const entry = entryFor({}, {});
    const input = writeInput(entry, []);

    expect(applyToolDenyWrite(input)).toEqual({ toolPermissions: {}, deniedTools: { "uuid-1": fetched } });
  });

  it("keeps toolset-granted tools out of the denied write", () => {
    const support = toolset({
      toolset_id: "ts-1",
      toolset_name: "Support",
      tools: [{ server_id: "uuid-1", tool_name: "list_issues" }],
    });
    const toolsetResolve = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-1"],
      toolsets: [support],
    };
    const [entry] = resolveEffectiveMcpServers(toolsetResolve);
    const input = writeInput(entry, [], { allServers: [named] });

    expect(applyToolDenyWrite(input)).toEqual({
      toolPermissions: {},
      deniedTools: { "uuid-1": ["create_issue", "delete_issue"] },
    });
  });

  it("drops a denylist key under an equivalent alias when the server is edited", () => {
    const deniedTools = { github_mcp: ["delete_issue"], "uuid-2": ["other_tool"] };
    const entry = entryFor({}, deniedTools);
    const input = writeInput(entry, ["list_issues", "create_issue"], { deniedTools });

    expect(applyToolDenyWrite(input)).toEqual({
      toolPermissions: {},
      deniedTools: { "uuid-2": ["other_tool"], "uuid-1": ["delete_issue"] },
    });
  });

  it("leaves a denylist key that also names another server untouched", () => {
    const shared = server({ server_id: "uuid-9", server_name: "github_mcp" });
    const deniedTools = { github_mcp: ["delete_issue"] };
    const entry = entryFor({}, deniedTools);

    const written = applyToolDenyWrite(
      writeInput(entry, ["list_issues", "create_issue"], { deniedTools, allServers: [named, shared] }),
    );
    expect(written.deniedTools["github_mcp"]).toEqual(["delete_issue"]);
    expect(written.deniedTools["uuid-1"]).toEqual(["delete_issue"]);
  });

  it("keeps a denied tool the upstream is not currently listing", () => {
    const deniedTools = { "uuid-1": ["gone_tool", "list_issues"] };
    const entry = entryFor({}, deniedTools);
    const input = writeInput(entry, ["list_issues", "create_issue"], { deniedTools });

    expect(applyToolDenyWrite(input).deniedTools["uuid-1"]).toEqual(["delete_issue", "gone_tool"]);
  });

  it("keeps the denylist entry when a stale denied name survives a Select All", () => {
    const deniedTools = { "uuid-1": ["gone_tool"] };
    const entry = entryFor({}, deniedTools);
    const input = writeInput(entry, fetched, { deniedTools });

    expect(applyToolDenyWrite(input).deniedTools).toEqual({ "uuid-1": ["gone_tool"] });
  });

  it("removes the denylist entry when nothing remains denied", () => {
    const deniedTools = { "uuid-1": ["list_issues"] };
    const entry = entryFor({}, deniedTools);
    const input = writeInput(entry, fetched, { deniedTools });

    expect(applyToolDenyWrite(input).deniedTools).toEqual({});
  });

  // Codex repro: a toolset tool renders as a locked checkbox, so unchecking a VISIBLE box cannot
  // re-decide it — a deny already written against it must survive the rebuild from fetchedTools.
  it("keeps a deny on a toolset tool the locked checkbox cannot re-decide", () => {
    const support = toolset({
      toolset_id: "ts-1",
      toolset_name: "Support",
      tools: [{ server_id: "uuid-1", tool_name: "read" }],
    });
    const resolveInput = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      selectedToolsets: ["ts-1"],
      toolsets: [support],
      deniedTools: { "uuid-1": ["read"] },
    };
    const [entry] = resolveEffectiveMcpServers(resolveInput);
    const input = {
      toolPermissions: {},
      deniedTools: { "uuid-1": ["read"] },
      entry,
      allServers: [named],
      fetchedTools: ["read", "write"],
      checked: ["write"],
    };

    expect(applyToolDenyWrite(input)).toEqual({ toolPermissions: {}, deniedTools: { "uuid-1": ["read"] } });
  });
});

// A denylist cannot GRANT a tool, so the deny write is only valid for a server granted
// independently of any tool list (direct or access-group source) that no selected toolset feeds.
// Every other shape keeps the standing-grant write: a server reached only through a toolset or by
// a tool_permissions key would otherwise no-op or lose its sole grant, and a server with a toolset
// plus an allowlist keeps the key because the backend unions the toolset into the allowlist.
describe("applyToolCheckboxWrite", () => {
  const named = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });
  const support = toolset({
    toolset_id: "ts-1",
    toolset_name: "Support",
    tools: [{ server_id: "uuid-1", tool_name: "a" }],
  });
  const reader = toolset({
    toolset_id: "ts-2",
    toolset_name: "Reader",
    tools: [{ server_id: "uuid-1", tool_name: "read" }],
  });

  const toolsetOnlyEntry = (deniedTools: Readonly<Record<string, readonly string[]>> = {}) => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-1"],
      toolsets: [support],
      deniedTools,
    };
    return resolveEffectiveMcpServers(input)[0];
  };

  const checkboxInput = (
    entry: EffectiveMcpServer,
    checked: readonly string[],
    over: {
      toolPermissions?: Readonly<Record<string, readonly string[]>>;
      deniedTools?: Readonly<Record<string, readonly string[]>>;
      fetchedTools?: readonly string[];
    } = {},
  ) => ({
    toolPermissions: over.toolPermissions ?? {},
    deniedTools: over.deniedTools ?? {},
    entry,
    allServers: [named],
    fetchedTools: over.fetchedTools ?? ["a", "x", "y"],
    checked,
  });

  const resolvedAfterWrite = (written: {
    toolPermissions: Record<string, string[]>;
    deniedTools: Record<string, string[]>;
  }) => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      toolPermissions: written.toolPermissions,
      deniedTools: written.deniedTools,
    };
    return resolveEffectiveMcpServers(input)[0];
  };

  it("writes an allowlist grant for a toolset-only server, leaving the denylist untouched", () => {
    const deniedTools = { "uuid-9": ["unrelated"] };
    const entry = toolsetOnlyEntry(deniedTools);
    const input = checkboxInput(entry, ["a", "x"], { deniedTools });

    expect(applyToolCheckboxWrite(input)).toEqual({
      toolPermissions: { "uuid-1": ["x"] },
      deniedTools,
    });
  });

  it("writes every fetched tool outside the toolset on Select All", () => {
    const entry = toolsetOnlyEntry();
    const input = checkboxInput(entry, ["a", "x", "y"]);

    expect(applyToolCheckboxWrite(input).toolPermissions).toEqual({ "uuid-1": ["x", "y"] });
  });

  it("keeps a toolPermission-only server granting on Select All", () => {
    const input = { ...emptyInput, allServers: [named], toolPermissions: { "uuid-1": ["read"] } };
    const [entry] = resolveEffectiveMcpServers(input);
    expect(entry.source.kind).toBe("toolPermission");

    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["read", "write"], {
        toolPermissions: { "uuid-1": ["read"] },
        fetchedTools: ["read", "write"],
      }),
    );
    expect(written).toEqual({ toolPermissions: { "uuid-1": ["read", "write"] }, deniedTools: {} });

    // The allowlist stays the grant, so the server must keep resolving rather than vanish.
    const afterWrite = { ...emptyInput, allServers: [named], toolPermissions: written.toolPermissions };
    const [resolved] = resolveEffectiveMcpServers(afterWrite);
    expect(resolved.server.server_id).toBe("uuid-1");
  });

  it("leaves the key present but empty when the last tool is unchecked on a toolPermission entry", () => {
    const input = { ...emptyInput, allServers: [named], toolPermissions: { "uuid-1": ["read"] } };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, [], { toolPermissions: { "uuid-1": ["read"] }, fetchedTools: ["read"] }),
    );

    expect(written.toolPermissions).toEqual({ "uuid-1": [] });
  });

  it("takes the deny write for a direct server no toolset feeds", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      toolPermissions: { "uuid-1": ["a"] },
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(checkboxInput(entry, ["a", "x"], { toolPermissions: { "uuid-1": ["a"] } }));

    expect(written).toEqual({ toolPermissions: {}, deniedTools: { "uuid-1": ["y"] } });
  });

  // A direct server a selected toolset also feeds keeps the allowlist write: the backend unions
  // the toolset's tools into key_tools, so the keyed extras must stay under toolPermissions.
  it("allowlist-writes a direct server a selected toolset also feeds", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      toolPermissions: { "uuid-1": ["write"] },
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["read", "write"], {
        toolPermissions: { "uuid-1": ["write"] },
        fetchedTools: ["read", "write", "delete"],
      }),
    );

    expect(written).toEqual({ toolPermissions: { "uuid-1": ["write"] }, deniedTools: {} });
    expect(resolvedAfterWrite(written).allowedTools).toEqual(expect.arrayContaining(["read", "write"]));
    expect(resolvedAfterWrite(written).allowedTools).toHaveLength(2);
  });

  it("trims the allowlist when a toolset-fed direct server has a keyed tool unchecked", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      toolPermissions: { "uuid-1": ["write"] },
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["read"], {
        toolPermissions: { "uuid-1": ["write"] },
        fetchedTools: ["read", "write", "delete"],
      }),
    );

    expect(written).toEqual({ toolPermissions: { "uuid-1": [] }, deniedTools: {} });
    expect(resolvedAfterWrite(written).allowedTools).toEqual(["read"]);
  });

  // The allowlist write lifts this entry's deny on a re-checked tool: otherwise the backend's
  // deny-wins rule would keep the tool blocked and the admin could never re-enable it here.
  it("lifts the entry's own deny on a checked editable tool", () => {
    const deniedTools = { "uuid-1": ["write"] };
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      deniedTools,
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["write"], { deniedTools, fetchedTools: ["read", "write"] }),
    );

    expect(written).toEqual({ toolPermissions: { "uuid-1": ["write"] }, deniedTools: {} });

    const afterWrite = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      toolPermissions: written.toolPermissions,
      deniedTools: written.deniedTools,
    };
    const [resolved] = resolveEffectiveMcpServers(afterWrite);
    expect(resolved.allowedTools).toEqual(expect.arrayContaining(["write"]));
    expect(resolved.deniedTools ?? []).not.toContain("write");
  });

  it("keeps other denied names on the same server when only some are re-checked", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      toolPermissions: { "uuid-1": ["read"] },
      deniedTools: { "uuid-1": ["write", "delete"] },
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["read", "write"], {
        toolPermissions: { "uuid-1": ["read"] },
        deniedTools: { "uuid-1": ["write", "delete"] },
        fetchedTools: ["read", "write", "delete"],
      }),
    );

    expect(written).toEqual({
      toolPermissions: { "uuid-1": ["read", "write"] },
      deniedTools: { "uuid-1": ["delete"] },
    });
  });

  it("leaves existing denies untouched for tools the admin did not check", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      toolPermissions: { "uuid-1": ["read"] },
      deniedTools: { "uuid-1": ["write"] },
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, [], {
        toolPermissions: { "uuid-1": ["read"] },
        deniedTools: { "uuid-1": ["write"] },
        fetchedTools: ["read", "write"],
      }),
    );

    expect(written).toEqual({ toolPermissions: { "uuid-1": [] }, deniedTools: { "uuid-1": ["write"] } });
  });

  it("keeps a denied locked toolset tool out of a Select All write", () => {
    const deniedTools = { "uuid-1": ["read"] };
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      deniedTools,
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["read", "write"], { deniedTools, fetchedTools: ["read", "write"] }),
    );

    expect(written).toEqual({ toolPermissions: { "uuid-1": ["write"] }, deniedTools: { "uuid-1": ["read"] } });

    const afterWrite = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      toolPermissions: written.toolPermissions,
      deniedTools: written.deniedTools,
    };
    const [resolved] = resolveEffectiveMcpServers(afterWrite);
    expect(resolved.deniedTools).toContain("read");
    expect((resolved.allowedTools ?? []).filter((name) => !(resolved.deniedTools ?? []).includes(name))).toEqual([
      "write",
    ]);
  });

  it("keeps a locked toolset tool's deny since its checkbox can never re-decide it", () => {
    const deniedTools = { "uuid-1": ["read"] };
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedToolsets: ["ts-2"],
      toolsets: [reader],
      deniedTools,
    };
    const [entry] = resolveEffectiveMcpServers(input);
    const written = applyToolCheckboxWrite(
      checkboxInput(entry, ["write"], { deniedTools, fetchedTools: ["read", "write"] }),
    );

    expect(written).toEqual({
      toolPermissions: { "uuid-1": ["write"] },
      deniedTools: { "uuid-1": ["read"] },
    });
  });
});

describe("resolveEffectiveMcpServers denylist", () => {
  const named = server({ server_id: "uuid-1", server_name: "github_mcp", alias: "GitHub" });

  it("unions denied tools across equivalent keys", () => {
    const input = {
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
      deniedTools: { "uuid-1": ["a"], github_mcp: ["b"] },
    };
    const [entry] = resolveEffectiveMcpServers(input);

    expect(entry.deniedTools).toEqual(["a", "b"]);
  });

  it("reports no denylist as undefined", () => {
    const [entry] = resolveEffectiveMcpServers({
      ...emptyInput,
      allServers: [named],
      selectedServers: ["uuid-1"],
    });

    expect(entry.deniedTools).toBeUndefined();
  });
});
