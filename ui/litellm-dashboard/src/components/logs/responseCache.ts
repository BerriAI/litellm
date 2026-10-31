import type { LogEntry } from "./types";

export function getResponseCacheStatus(log: Pick<LogEntry, "cache_hit" | "cache_key">): "hit" | "miss" | undefined {
  const value = String(log.cache_hit ?? "").toLowerCase();
  if (value === "true") return "hit";
  if (value === "false" || (log.cache_key && log.cache_key !== "Cache OFF")) return "miss";
  return undefined;
}
