import { MCPServer } from "@/components/mcp_tools/types";
import { ApiError, deriveErrorMessage, unwrapProxyErrorMessage } from "@/lib/http/client";

export type McpIdentifierField = "server_name" | "alias";

export interface McpIdentifierDuplicate {
  field: McpIdentifierField;
  serverId: string;
}

export const normalizeMcpIdentifier = (value: string | null | undefined): string =>
  (value ?? "").trim().replace(/\s+/g, "_").toLowerCase();

export function findDuplicateMcpServer(
  servers: readonly Pick<MCPServer, "server_id" | "server_name" | "alias">[] | undefined,
  serverName: string | null | undefined,
  alias: string | null | undefined,
  excludeServerId?: string,
): McpIdentifierDuplicate | null {
  const candidates: ReadonlyArray<readonly [McpIdentifierField, string | null | undefined]> = [
    ["alias", alias],
    ["server_name", serverName],
  ];
  for (const [field, value] of candidates) {
    const normalized = normalizeMcpIdentifier(value);
    if (!normalized) {
      continue;
    }
    const hit = (servers ?? []).find(
      (server) =>
        server.server_id !== excludeServerId &&
        [server.server_name, server.alias].some((existing) => normalizeMcpIdentifier(existing) === normalized),
    );
    if (hit) {
      return { field, serverId: hit.server_id };
    }
  }
  return null;
}

export const DUPLICATE_IDENTIFIER_MESSAGE = "An MCP server with this name/alias already exists.";

export const mcpSubmitErrorReason = (error: unknown): string => {
  if (error instanceof ApiError) {
    return deriveErrorMessage(error.body);
  }
  return error instanceof Error ? unwrapProxyErrorMessage(error.message) : deriveErrorMessage(error);
};
