import { useQuery } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { mcpAllowedToolsFor, mcpServersForIdentifier } from "@/components/mcp_server_management/effectiveMcpServers";
import { getAgentsList, vectorStoreListCall } from "@/components/networking";
import { useControlPlane } from "@/contexts/ControlPlaneContext";

import type { KeyAccess } from "./keyOverviewSections";

interface NamedRecord {
  readonly id: string;
  readonly name: string | null;
}

export interface ResolvedMcpServer {
  readonly identifier: string;
  readonly name: string;
  readonly allowedTools: readonly string[] | undefined;
}

export interface KeyAccessNames {
  readonly mcpServers: readonly ResolvedMcpServer[];
  readonly toolsetName: (id: string) => string;
  readonly vectorStoreName: (id: string) => string;
  readonly agentName: (id: string) => string;
}

const toNamedRecords = (rows: unknown, idKey: string, nameKey: string): readonly NamedRecord[] =>
  Array.isArray(rows)
    ? rows.flatMap((row: unknown) => {
        if (typeof row !== "object" || row === null) return [];
        const record = row as Readonly<Record<string, unknown>>;
        const id = record[idKey];
        const name = record[nameKey];
        return typeof id === "string" ? [{ id, name: typeof name === "string" && name ? name : null }] : [];
      })
    : [];

const nameLookup =
  (records: readonly NamedRecord[]) =>
  (id: string): string =>
    records.find((record) => record.id === id)?.name ?? id;

export function useKeyAccessNames(access: KeyAccess): KeyAccessNames {
  const controlPlane = useControlPlane();
  const { accessToken } = useAuthorized();
  const { data: servers } = useMCPServers();
  const { data: toolsets } = useMCPToolsets();
  const vectorStores = useQuery({
    queryKey: ["keyDetail", "vectorStoreNames"],
    queryFn: async () =>
      toNamedRecords((await vectorStoreListCall(accessToken!)).data, "vector_store_id", "vector_store_name"),
    enabled: Boolean(accessToken) && access.vectorStores.length > 0,
  });
  const agents = useQuery({
    queryKey: ["keyDetail", "agentNames"],
    queryFn: async () => toNamedRecords((await getAgentsList(accessToken!)).agents, "agent_id", "agent_name"),
    enabled: !controlPlane && Boolean(accessToken) && access.agents.length > 0,
  });

  const serverCatalog = servers ?? [];
  const mcpServers =
    access.mcp.kind === "list"
      ? access.mcp.servers.map((identifier) => {
          const [detail] = mcpServersForIdentifier(serverCatalog, identifier);
          return {
            identifier,
            name: detail ? detail.alias || detail.server_name || detail.server_id : identifier,
            allowedTools: detail
              ? mcpAllowedToolsFor(detail, access.mcpToolPermissions, serverCatalog)
              : access.mcpToolPermissions[identifier],
          };
        })
      : [];

  return {
    mcpServers,
    toolsetName: (id) => toolsets?.find((toolset) => toolset.toolset_id === id)?.toolset_name ?? id,
    vectorStoreName: nameLookup(vectorStores.data ?? []),
    agentName: nameLookup(agents.data ?? []),
  };
}
