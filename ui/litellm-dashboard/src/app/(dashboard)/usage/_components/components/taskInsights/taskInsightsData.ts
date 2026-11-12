import type { components } from "@/lib/http/schema";
import type { TaskInsightsMetric } from "../../usageUrlState";

export type TaskInsightsOverview = components["schemas"]["TaskInsightsOverviewResponse"];
export type TaskMatrix = components["schemas"]["TaskMatrixResponse"];
export type TaskMatrixEntry = components["schemas"]["TaskMatrixEntry"];
export type TaskDetail = components["schemas"]["TaskDetailResponse"];
export type TaskSummary = components["schemas"]["TaskSummary"];
export type TaskTaxonomy = components["schemas"]["TaskTaxonomyResponse"];
export type TaskDefinition = components["schemas"]["TaskDefinition"];
export type TaskMapPoint = components["schemas"]["TaskMapPoint"];
export type ModelConfigSummary = components["schemas"]["ModelConfigSummary"];
export type TaskCompatibility = components["schemas"]["TaskCompatibility"];
export type DraftTaskEdit = components["schemas"]["DraftTaskEdit"];

export const TASK_COLORS = [
  "#2b3fd6",
  "#0ea5e9",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ef4444",
  "#14b8a6",
  "#ec4899",
  "#84cc16",
  "#6366f1",
  "#f97316",
  "#06b6d4",
] as const;
export const UNASSIGNED_COLOR = "#9ca3af";

export const isoDay = (date: Date): string => {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
};

/** Colors follow the ranked task order, so the map, the breakdown and the heatmap agree on every task. */
export const taskColors = (tasks: readonly Pick<TaskSummary, "task_id" | "kind">[]): ReadonlyMap<string, string> =>
  new Map(
    tasks.map((task, index) => [
      task.task_id,
      task.kind === "task" ? TASK_COLORS[index % TASK_COLORS.length] : UNASSIGNED_COLOR,
    ]),
  );

const cellKey = (modelConfigId: string, taskId: string) => `${modelConfigId}\u0000${taskId}`;

export const entryLookup = (
  entries: readonly TaskMatrixEntry[],
): ((modelConfigId: string, taskId: string) => TaskMatrixEntry | undefined) => {
  const byCell = new Map(entries.map((entry) => [cellKey(entry.model_config_id, entry.task_id), entry]));
  return (modelConfigId, taskId) => byCell.get(cellKey(modelConfigId, taskId));
};

export interface CellReading {
  /** The value the cell is shaded by, or null when the cell has no evidence for the metric. */
  value: number | null;
  observations: number;
}

/** A missing cost or latency stays missing, never zero, so the heatmap greys it out instead of shading it light. */
export const cellReading = (entry: TaskMatrixEntry | undefined, metric: TaskInsightsMetric): CellReading => {
  if (!entry) return { value: null, observations: 0 };
  if (metric === "share") return { value: entry.request_count, observations: entry.request_count };
  if (metric === "cost") {
    return { value: entry.cost_observations > 0 ? entry.avg_cost_usd : null, observations: entry.cost_observations };
  }
  return {
    value: entry.latency_observations > 0 ? entry.p95_latency_ms : null,
    observations: entry.latency_observations,
  };
};

/** Share of a task's requests that one deployment served, the row-wise reading of the count metric. */
export const shareOfTask = (
  entries: readonly TaskMatrixEntry[],
  modelConfigId: string,
  taskId: string,
): number | null => {
  const total = entries
    .filter((entry) => entry.task_id === taskId)
    .reduce((sum, entry) => sum + entry.request_count, 0);
  const served = entries.find((entry) => entry.task_id === taskId && entry.model_config_id === modelConfigId);
  return total > 0 && served ? served.request_count / total : null;
};

/** Intensity from 0.12 to 1 within the metric's observed range, so the lightest shaded cell stays visible. */
export const heatIntensity = (value: number | null, values: readonly (number | null)[]): number | null => {
  if (value === null) return null;
  const known = values.filter((item): item is number => item !== null);
  const min = Math.min(...known);
  const max = Math.max(...known);
  if (max === min) return 1;
  return 0.12 + (0.88 * (value - min)) / (max - min);
};

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 });
const usdTotal = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const count = new Intl.NumberFormat("en-US");
const percent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });

export const formatUsd = (value: number | null | undefined): string | null =>
  value === null || value === undefined ? null : usd.format(value);
export const formatUsdTotal = (value: number | null | undefined): string | null =>
  value === null || value === undefined ? null : usdTotal.format(value);
export const formatCount = (value: number): string => count.format(value);
export const formatShare = (value: number | null | undefined): string | null =>
  value === null || value === undefined ? null : percent.format(value);
export const formatLatency = (value: number | null | undefined): string | null => {
  if (value === null || value === undefined) return null;
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
};

export const formatReading = (reading: CellReading, metric: TaskInsightsMetric): string | null => {
  if (reading.value === null) return null;
  if (metric === "share") return formatCount(reading.value);
  if (metric === "cost") return formatUsd(reading.value);
  return formatLatency(reading.value);
};

export const compatibilityFor = (
  compatibility: readonly TaskCompatibility[],
  modelConfigId: string,
  taskId: string,
): TaskCompatibility | undefined =>
  compatibility.find((item) => item.model_config_id === modelConfigId && item.task_id === taskId);

export const modelLabel = (config: Pick<ModelConfigSummary, "model_name" | "model_config_id">): string =>
  config.model_name ?? config.model_config_id;

/** Share of all requests in the window that landed in the frozen taxonomy. */
export const coverage = (totals: TaskInsightsOverview["totals"]): number | null =>
  totals.request_count > 0 ? totals.classified_count / totals.request_count : null;

export const mapBubbles = (tasks: readonly TaskSummary[]) =>
  tasks
    .filter((task) => task.kind === "task" && task.map_x !== null && task.map_y !== null)
    .map((task) => ({ task_id: task.task_id, name: task.name, x: task.map_x, y: task.map_y, z: task.request_count }));

export const isDiscoveryBusy = (status: TaskInsightsOverview["status"] | undefined): boolean =>
  status?.discovery_status === "requested" || status?.discovery_status === "running";
