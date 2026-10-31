import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";

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

export const parseConnectorConfig = (text: string, t: TFunction = i18n.t.bind(i18n)): ParsedConnectorConfig => {
  const trimmed = text.trim();
  if (!trimmed) {
    return { ok: false, error: t("mcpServers.importConnectorConfig.errors.emptyInput") };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { ok: false, error: t("mcpServers.importConnectorConfig.errors.invalidJson") };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { ok: false, error: t("mcpServers.importConnectorConfig.errors.unrecognizedShape") };
  }
  const record = parsed as Record<string, unknown>;
  const mapping = record.mcpServers;
  if (mapping !== undefined) {
    if (typeof mapping !== "object" || mapping === null || Array.isArray(mapping)) {
      return { ok: false, error: t("mcpServers.importConnectorConfig.errors.mappingNotObject") };
    }
    const connectorCount = Object.keys(mapping).length;
    if (connectorCount === 0) {
      return { ok: false, error: t("mcpServers.importConnectorConfig.errors.mappingEmpty") };
    }
    return { ok: true, payload: { mcpServers: mapping }, connectorCount };
  }
  const list = record.mcp_servers;
  if (list !== undefined) {
    if (!Array.isArray(list)) {
      return { ok: false, error: t("mcpServers.importConnectorConfig.errors.listNotArray") };
    }
    if (list.length === 0) {
      return { ok: false, error: t("mcpServers.importConnectorConfig.errors.listEmpty") };
    }
    return { ok: true, payload: { mcp_servers: list }, connectorCount: list.length };
  }
  return { ok: false, error: t("mcpServers.importConnectorConfig.errors.unrecognizedShape") };
};
