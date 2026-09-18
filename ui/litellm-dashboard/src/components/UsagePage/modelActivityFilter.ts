import type { ModelActivityData } from "./types";

export function modelActivityMatches(modelKey: string, data: ModelActivityData, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (needle === "") return true;
  return [modelKey, data.label].some((field) => field?.toLowerCase().includes(needle) ?? false);
}

export function filterModelActivity(
  modelMetrics: Record<string, ModelActivityData>,
  query: string,
): Record<string, ModelActivityData> {
  if (query.trim() === "") return modelMetrics;
  return Object.fromEntries(
    Object.entries(modelMetrics).filter(([modelKey, data]) => modelActivityMatches(modelKey, data, query)),
  );
}
