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

// A tool's display name replaces the tool name sent to the LLM provider. Providers
// such as AWS Bedrock require tool names to match [a-zA-Z0-9_-]+, so spaces and
// special characters must be rejected before they are persisted.
const TOOL_DISPLAY_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

export const TOOL_DISPLAY_NAME_ERROR =
  "Display name may only contain letters, numbers, underscores, and hyphens (a-z, A-Z, 0-9, _, -). It replaces the tool name sent to the provider; spaces or special characters fail with providers like AWS Bedrock.";

export const isValidToolDisplayName = (value: string): boolean =>
  value === "" || TOOL_DISPLAY_NAME_PATTERN.test(value);

export const findInvalidToolDisplayName = (
  map: Record<string, string>,
): { toolName: string; displayName: string } | null => {
  for (const [toolName, displayName] of Object.entries(map || {})) {
    if (!isValidToolDisplayName(displayName)) {
      return { toolName, displayName };
    }
  }
  return null;
};
