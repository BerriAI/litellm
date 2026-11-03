import { z } from "zod";
import type { DailyData, MetricWithMetadata, SpendMetrics } from "@/components/UsagePage/types";
import type { AgentRow } from "../components/overview/agentCatalog";
import { clientAgentDetail } from "../components/overview/clientAgentDetail";

const count = z.number().finite().nonnegative();
const exportRowFields = {
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  entity_id: z.string(),
  model: z.string().nullable(),
  spend: z.number().finite(),
  flat_cost: z.number().finite(),
  prompt_tokens: count,
  completion_tokens: count,
  api_requests: count,
  successful_requests: count,
  failed_requests: count,
  cache_read_input_tokens: count,
  cache_creation_input_tokens: count,
};
const exportRow = z.object(exportRowFields);
type ExportRow = z.infer<typeof exportRow>;
type MetricKey = Exclude<keyof ExportRow, "date" | "entity_id" | "model">;

export interface ClientAgentModelsResult {
  results: DailyData[];
  error: string | null;
}

const grouped = (rows: readonly ExportRow[], key: (row: ExportRow) => string) => {
  const sorted = [...rows].sort((a, b) => key(a).localeCompare(key(b)));
  const starts = sorted.flatMap((row, index) => (index === 0 || key(row) !== key(sorted[index - 1]) ? [index] : []));
  return starts.map((start, index) => [key(sorted[start]), sorted.slice(start, starts[index + 1])] as const);
};

const sum = (rows: readonly ExportRow[], key: MetricKey) => rows.reduce((total, row) => total + row[key], 0);

const metricsFor = (rows: readonly ExportRow[]): SpendMetrics => ({
  spend: sum(rows, "spend"),
  flat_cost: sum(rows, "flat_cost"),
  prompt_tokens: sum(rows, "prompt_tokens"),
  completion_tokens: sum(rows, "completion_tokens"),
  total_tokens: sum(rows, "prompt_tokens") + sum(rows, "completion_tokens"),
  api_requests: sum(rows, "api_requests"),
  successful_requests: sum(rows, "successful_requests"),
  failed_requests: sum(rows, "failed_requests"),
  cache_read_input_tokens: sum(rows, "cache_read_input_tokens"),
  cache_creation_input_tokens: sum(rows, "cache_creation_input_tokens"),
});

const breakdownFor = (
  rows: readonly ExportRow[],
  key: (row: ExportRow) => string,
): Record<string, MetricWithMetadata> =>
  Object.fromEntries(
    grouped(rows, key).map(([name, members]) => [
      name,
      { metrics: metricsFor(members), metadata: {}, api_key_breakdown: {} },
    ]),
  );

export const clientAgentModelsFromExport = (payload: unknown, agent: AgentRow): ClientAgentModelsResult => {
  const parsed = z.array(exportRow).safeParse(payload);
  if (!parsed.success) return { results: [], error: "The proxy returned incomplete model usage data" };
  const tags = new Set(agent.tags);
  const rows = parsed.data.filter((row) => tags.has(row.entity_id));
  const identities = rows.map((row) => JSON.stringify([row.date, row.entity_id, row.model]));
  if (new Set(identities).size !== identities.length) {
    return { results: [], error: "The proxy returned duplicate model usage data. Try again." };
  }
  const results = grouped(rows, (row) => row.date).map(([date, members]) => ({
    date,
    metrics: metricsFor(members),
    breakdown: {
      models: breakdownFor(members, (row) => row.model || "Unknown model"),
      model_groups: {},
      entities: breakdownFor(members, (row) => row.entity_id),
      providers: {},
      api_keys: {},
      mcp_servers: {},
    },
  }));
  if (!clientAgentDetail(agent, results).dailyMatchesSummary) {
    return {
      results: [],
      error: "Model usage does not match these client totals yet. Refresh or choose a shorter range.",
    };
  }
  return { results, error: null };
};
