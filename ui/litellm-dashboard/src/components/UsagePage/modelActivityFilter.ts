import type { ModelActivityData } from "./types";

export function filterModelActivity(
  modelMetrics: Record<string, ModelActivityData>,
  query: string,
): Record<string, ModelActivityData> {
  const needle = query.trim().toLowerCase();
  if (needle === "") return modelMetrics;
  return Object.fromEntries(
    Object.entries(modelMetrics).filter(
      ([model, data]) => model.toLowerCase().includes(needle) || data.label.toLowerCase().includes(needle),
    ),
  );
}
