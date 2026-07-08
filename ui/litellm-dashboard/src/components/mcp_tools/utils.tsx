import { MCPEnvVar, MCPEnvVarScope } from "./types";

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
export const validateMCPServerUrl = (value: string) => {
  if (!value) return Promise.resolve();
  // More flexible URL validation that allows Kubernetes service names and various URL formats
  const urlPattern = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;
  return urlPattern.test(value)
    ? Promise.resolve()
    : Promise.reject("Please enter a valid URL (e.g., http://service-name.domain:1234/path or https://example.com)");
};

export const validateMCPServerName = (value: string) => {
  return value && (value.includes("-") || value.includes(" "))
    ? Promise.reject("Cannot contain '-' (hyphen) or spaces. Please use '_' (underscore) instead.")
    : Promise.resolve();
};

export const TOOL_DISPLAY_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

export const validateToolDisplayName = (value: string) => {
  return value && !TOOL_DISPLAY_NAME_PATTERN.test(value)
    ? Promise.reject("Only letters, digits, underscores, and hyphens are allowed (no spaces).")
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
