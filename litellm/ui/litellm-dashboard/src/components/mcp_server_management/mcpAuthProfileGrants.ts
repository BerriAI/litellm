import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";

const profileIdsOf = (server: MCPServer): readonly string[] => (server.additional_auth ?? []).map((p) => p.id);

const sameId = (a: string, b: string): boolean => a.toLowerCase() === b.toLowerCase();

export const profileServers = (allServers: readonly MCPServer[]): readonly MCPServer[] =>
  allServers.filter((server) => profileIdsOf(server).length > 0);

export const grantedProfileFor = (grants: readonly string[], server: MCPServer): string | undefined =>
  grants.find((grant) => profileIdsOf(server).some((id) => sameId(id, grant)));

interface Reach {
  readonly scoped: readonly MCPServer[];
  readonly all: readonly MCPServer[];
  readonly serverIds: readonly string[];
  readonly toolsets: readonly MCPToolset[];
  readonly toolsetIds: readonly string[];
}

export const reachableServers = ({ scoped, all, serverIds, toolsets, toolsetIds }: Reach): readonly MCPServer[] => {
  const toolsetServerIds = toolsets
    .filter((toolset) => toolsetIds.includes(toolset.toolset_id))
    .flatMap((toolset) => toolset.tools.map((tool) => tool.server_id));
  const reached = new Set([...serverIds, ...toolsetServerIds]);
  const outsideScope = all.filter(
    (server) => reached.has(server.server_id) && !scoped.some((s) => s.server_id === server.server_id),
  );
  return [...scoped, ...outsideScope];
};

export const withServerProfile = (
  grants: readonly string[],
  server: MCPServer,
  profileId: string | null,
): readonly string[] => {
  const others = grants.filter((grant) => !profileIdsOf(server).some((id) => sameId(id, grant)));
  return profileId === null ? others : [...others, profileId];
};

interface GrantEntryInput {
  readonly grants: readonly string[] | undefined;
  readonly servers: readonly MCPServer[] | undefined;
  readonly loadedGrants?: readonly string[];
}

const sameGrants = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((grant) => b.some((other) => sameId(grant, other)));

export const authProfileGrantEntry = ({
  grants,
  servers,
  loadedGrants,
}: GrantEntryInput): { readonly mcp_auth_profiles?: readonly string[] } => {
  if (servers === undefined || grants === undefined) return {};
  const loaded = loadedGrants ?? [];
  const unseen = loaded.filter(
    (grant) =>
      servers.every((server) => grantedProfileFor([grant], server) === undefined) &&
      !grants.some((chosen) => sameId(chosen, grant)),
  );
  const kept = [...grants, ...unseen];
  return sameGrants(kept, loaded) ? {} : { mcp_auth_profiles: kept };
};

export const grantChangeNeedsServers = ({ grants, servers, loadedGrants }: GrantEntryInput): boolean =>
  servers === undefined && grants !== undefined && !sameGrants(grants, loadedGrants ?? []);
