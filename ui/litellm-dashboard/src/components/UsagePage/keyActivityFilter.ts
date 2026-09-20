import type { ModelActivityData } from "./types";

export function keyActivityMatches(apiKey: string, data: ModelActivityData, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle === "") return true;
  const meta = data.key_metadata;
  return [apiKey, data.label, meta?.key_alias, meta?.user_id, meta?.user_email].some(
    (field) => field?.toLowerCase().includes(needle) ?? false,
  );
}

export function filterKeyActivity(
  keyMetrics: Record<string, ModelActivityData>,
  query: string,
): Record<string, ModelActivityData> {
  if (query.trim() === "") return keyMetrics;
  return Object.fromEntries(
    Object.entries(keyMetrics).filter(([apiKey, data]) => keyActivityMatches(apiKey, data, query)),
  );
}
