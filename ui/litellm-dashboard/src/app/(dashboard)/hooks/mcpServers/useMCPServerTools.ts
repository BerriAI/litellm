import { useQueries, type QueryObserverResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { listMCPTools } from "@/components/networking";
import useAuthorized from "../useAuthorized";

export const mcpServerToolKeys = createQueryKeys("mcpServerTools");

export interface MCPServerToolEntry {
  name: string;
  description: string;
}

// "idle" means the server has not been fetched yet, so callers can tell it apart from
// a server that loaded and has no tools.
export type MCPServerToolsStatus = "idle" | "loading" | "error" | "success";

export interface MCPServerToolsState {
  tools: MCPServerToolEntry[];
  status: MCPServerToolsStatus;
}

type RawTool = string | { name?: string; tool_name?: string; description?: string };

const toEntry = (tool: RawTool): MCPServerToolEntry =>
  typeof tool === "string"
    ? { name: tool, description: "" }
    : { name: tool.name ?? tool.tool_name ?? "", description: tool.description ?? "" };

// listMCPTools never throws: it reports failures as `{ tools: [], error }`.
// Throw here so a broken server surfaces as a query error instead of an empty tool list.
export const fetchServerTools = async (accessToken: string, serverId: string): Promise<MCPServerToolEntry[]> => {
  const result = await listMCPTools(accessToken, serverId);
  if (!Array.isArray(result) && result?.error) {
    throw new Error(result.message || result.error);
  }
  const toolList: RawTool[] = Array.isArray(result) ? result : result?.tools ?? [];
  return toolList.map(toEntry);
};

const toStatus = ({
  status,
  fetchStatus,
}: Pick<QueryObserverResult, "status" | "fetchStatus">): MCPServerToolsStatus => {
  if (status !== "pending") return status;
  return fetchStatus === "idle" ? "idle" : "loading";
};

/**
 * Tool catalogs for a set of MCP servers, one query per server so a slow or failing
 * server never blocks the others. Only servers in `enabledServerIds` are fetched;
 * the rest stay "idle" until they are enabled.
 */
export const useMCPServerTools = (
  serverIds: readonly string[],
  enabledServerIds: ReadonlySet<string>,
): ReadonlyMap<string, MCPServerToolsState> => {
  const { accessToken } = useAuthorized();
  return useQueries({
    queries: serverIds.map((serverId) => ({
      queryKey: mcpServerToolKeys.detail(serverId),
      queryFn: () => fetchServerTools(accessToken!, serverId),
      enabled: !!accessToken && enabledServerIds.has(serverId),
      staleTime: 5 * 60 * 1000,
      retry: false,
    })),
    combine: (results) =>
      new Map(
        results.map((result, idx): [string, MCPServerToolsState] => [
          serverIds[idx],
          { tools: result.data ?? [], status: toStatus(result) },
        ]),
      ),
  });
};
