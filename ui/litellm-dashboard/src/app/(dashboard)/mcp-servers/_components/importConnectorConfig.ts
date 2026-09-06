export interface MCPConnectorImportResult {
  name: string;
  server_id: string;
  alias: string;
}

export interface MCPConnectorImportSkipped {
  name: string;
  reason: string;
}

export interface MCPConnectorImportFailure {
  name: string;
  error: string;
}

export interface MCPConnectorImportResponse {
  imported: MCPConnectorImportResult[];
  skipped: MCPConnectorImportSkipped[];
  errors: MCPConnectorImportFailure[];
}

export type ParsedConnectorConfig =
  | { ok: true; payload: Record<string, unknown>; connectorCount: number }
  | { ok: false; error: string };

export const parseConnectorConfig = (text: string): ParsedConnectorConfig => {
  const trimmed = text.trim();
  if (!trimmed) {
    return { ok: false, error: "Paste your connector JSON before importing." };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { ok: false, error: "Invalid JSON. Check for missing quotes, commas, or brackets." };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: "Expected a JSON object with an mcpServers or mcp_servers key." };
  }
  const record = parsed as Record<string, unknown>;
  const mapping = record.mcpServers;
  if (mapping !== undefined) {
    if (typeof mapping !== "object" || mapping === null || Array.isArray(mapping)) {
      return { ok: false, error: "mcpServers must be an object mapping connector names to definitions." };
    }
    const connectorCount = Object.keys(mapping).length;
    if (connectorCount === 0) {
      return { ok: false, error: "mcpServers contains no connectors." };
    }
    return { ok: true, payload: { mcpServers: mapping }, connectorCount };
  }
  const list = record.mcp_servers;
  if (list !== undefined) {
    if (!Array.isArray(list)) {
      return { ok: false, error: "mcp_servers must be an array of connector definitions." };
    }
    if (list.length === 0) {
      return { ok: false, error: "mcp_servers contains no connectors." };
    }
    return { ok: true, payload: { mcp_servers: list }, connectorCount: list.length };
  }
  return { ok: false, error: "Expected a JSON object with an mcpServers or mcp_servers key." };
};
