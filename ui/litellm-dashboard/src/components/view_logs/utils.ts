import { MCP_CALL_TYPES } from "./constants";

/**
 * Derive a short, human-readable display name for a log entry.
 * Strips provider prefixes, date suffixes, and version tags.
 */
export function getEventDisplayName(callType: string, model: string): string {
  const raw = (model || "").trim();
  const isMcp = MCP_CALL_TYPES.includes(callType);

  if (isMcp) {
    return raw.replace(/^mcp:\s*/i, "").split("/").pop() || raw || "mcp_tool";
  }

  const lastSegment = raw.split("/").pop() || raw;
  const noSuffix = lastSegment.replace(/-20\d{6}.*$/i, "").replace(/:.*$/, "");
  const claudeMatch = noSuffix.match(/claude-[a-z0-9-]+/i);
  if (claudeMatch) return claudeMatch[0];
  return noSuffix || "llm_call";
}
