import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import { MCPEnvVar, MCPEnvVarScope, type MCPServer } from "@/components/mcp_tools/types";

export const getMCPNetworkAccess = (
  server: Pick<MCPServer, "available_on_public_internet" | "mcp_info">,
  t: TFunction = i18n.t.bind(i18n),
): {
  readonly label: string;
  readonly dotClassName: string;
  readonly description: string;
} => {
  const explicitlyPublished = server.mcp_info?.is_public_explicit;
  if (server.available_on_public_internet === true || explicitlyPublished === true) {
    return {
      label: t("mcpServers.utils.networkAccess.allNetworks.label"),
      dotClassName: "bg-success",
      description:
        server.available_on_public_internet === true
          ? t("mcpServers.utils.networkAccess.allNetworks.publicDescription")
          : t("mcpServers.utils.networkAccess.allNetworks.hubDescription"),
    };
  }
  if (server.available_on_public_internet === false && explicitlyPublished === false) {
    return {
      label: t("mcpServers.utils.networkAccess.internalOnly.label"),
      dotClassName: "bg-warning",
      description: t("mcpServers.utils.networkAccess.internalOnly.description"),
    };
  }
  return {
    label: t("mcpServers.utils.networkAccess.unknown.label"),
    dotClassName: "bg-border",
    description: t("mcpServers.utils.networkAccess.unknown.description"),
  };
};

export const extractMCPToken = (url: string): { token: string | null; baseUrl: string } => {
  try {
    const mcpIndex = url.indexOf("/mcp/");
    if (mcpIndex === -1) return { token: null, baseUrl: url };

    const parts = url.split("/mcp/");
    if (parts.length !== 2) return { token: null, baseUrl: url };

    const baseUrl = parts[0] + "/mcp/";
    const afterMcp = parts[1];

    // If there's no content after /mcp/, return null token
    if (!afterMcp) return { token: null, baseUrl: url };

    return {
      token: afterMcp,
      baseUrl: baseUrl,
    };
  } catch (error) {
    console.error("Error parsing MCP URL:", error);
    return { token: null, baseUrl: url };
  }
};

export const maskUrl = (url: string): string => {
  const { token, baseUrl } = extractMCPToken(url);
  if (!token) return url;
  return baseUrl + "...";
};

export const getMaskedAndFullUrl = (url: string): { maskedUrl: string; hasToken: boolean } => {
  const { token } = extractMCPToken(url);
  return {
    maskedUrl: maskUrl(url),
    hasToken: !!token,
  };
};

// Validation utilities for MCP server forms
export const validateMCPServerUrl = (value: string, t: TFunction = i18n.t.bind(i18n)) => {
  if (!value) return Promise.resolve();
  // More flexible URL validation that allows Kubernetes service names and various URL formats
  const urlPattern = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;
  return urlPattern.test(value) ? Promise.resolve() : Promise.reject(t("mcpServers.utils.validation.invalidUrl"));
};

export const validateMCPServerName = (value: string, t: TFunction = i18n.t.bind(i18n)) => {
  return value && (value.includes("-") || value.includes(" "))
    ? Promise.reject(t("mcpServers.utils.validation.invalidServerName"))
    : Promise.resolve();
};

export const TOOL_DISPLAY_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

export const validateToolDisplayName = (value: string, t: TFunction = i18n.t.bind(i18n)) => {
  return value && !TOOL_DISPLAY_NAME_PATTERN.test(value)
    ? Promise.reject(t("mcpServers.utils.validation.invalidToolDisplayName"))
    : Promise.resolve();
};

// Normalize the env_vars form list into the payload shape the backend expects.
// Drops empty rows, invalid identifiers, and duplicate names; user-scoped entries never carry a value.
export const normalizeEnvVars = (list: unknown): MCPEnvVar[] => {
  if (!Array.isArray(list)) return [];
  const seen = new Set<string>();
  const out: MCPEnvVar[] = [];
  for (const entry of list) {
    if (!entry || typeof entry !== "object") continue;
    const record = entry as Record<string, unknown>;
    const name = String(record.name ?? "").trim();
    if (!name || seen.has(name)) continue;
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) continue;
    const scope: MCPEnvVarScope = record.scope === "user" ? "user" : "global";
    out.push({
      name,
      value: scope === "user" ? "" : String(record.value ?? ""),
      scope,
      description: (record.description as string | undefined) || undefined,
    });
    seen.add(name);
  }
  return out;
};

/** Normalize tool override maps from API/DB (dict or JSON string) for form state. */
export const normalizeToolOverrideMap = (
  value: Record<string, string> | string | null | undefined,
): Record<string, string> => {
  if (!value) return {};
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, string>;
      }
    } catch {
      return {};
    }
    return {};
  }
  return value;
};
