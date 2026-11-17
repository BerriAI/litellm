import type { DailyData, SpendMetrics, BreakdownMetrics, MetricWithMetadata } from "@/components/UsagePage/types";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  toDailyData,
  type DailyActivityAggregatedResponse,
  type DailyActivityMetadata,
} from "@/components/UsagePage/dailyActivityApi";
import {
  rollUpBreakdown,
  type UsageMetric,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
export interface CustomerUsageTotals {
  spend: number;
  tokens: number;
  requests: number;
}

export interface CustomerUsageRow extends CustomerUsageTotals {
  id: string;
  label: string;
  share: number;
}

export const rankCustomerUsage = (
  rows: readonly Omit<CustomerUsageRow, "share">[],
  metric: UsageMetric,
  totals: CustomerUsageTotals,
): CustomerUsageRow[] =>
  rows
    .filter((row) => row.spend > 0 || row.tokens > 0 || row.requests > 0)
    .map((row) => ({ ...row, share: totals[metric] > 0 ? row[metric] / totals[metric] : 0 }))
    .sort((a, b) => b[metric] - a[metric] || a.label.localeCompare(b.label) || a.id.localeCompare(b.id));

export const rankCustomerModels = (
  days: readonly DailyData[],
  metric: UsageMetric,
  totals: CustomerUsageTotals,
): CustomerUsageRow[] => {
  const withModelGroups = days.map((day) => ({
    ...day,
    breakdown: {
      ...day.breakdown,
      model_groups:
        Object.keys(day.breakdown.model_groups).length > 0 ? day.breakdown.model_groups : day.breakdown.models,
    },
  }));
  return rankCustomerUsage(
    rollUpBreakdown(withModelGroups, "model_groups").map((model) => ({
      id: model.key,
      label: model.key,
      spend: model.spend,
      tokens: model.tokens,
      requests: model.requests,
    })),
    metric,
    totals,
  );
};

const ZERO_METRICS: SpendMetrics = {
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

const sumMetrics = (values: readonly SpendMetrics[]): SpendMetrics =>
  values.reduce(
    (total, value) => ({
      ...total,
      ...Object.fromEntries(
        (Object.keys(value) as (keyof SpendMetrics)[]).map((field) => [
          field,
          (total[field] ?? 0) + (value[field] ?? 0),
        ]),
      ),
    }),
    ZERO_METRICS,
  );

const mergeKeys = (
  left: BreakdownMetrics["api_keys"],
  right: BreakdownMetrics["api_keys"],
): BreakdownMetrics["api_keys"] => {
  const result = { ...left };
  for (const [id, entry] of Object.entries(right)) {
    const previous = result[id];
    result[id] = {
      ...entry,
      metrics: sumMetrics([...(previous ? [previous.metrics] : []), entry.metrics]),
      metadata: {
        ...previous?.metadata,
        ...entry.metadata,
        key_alias: entry.metadata.key_alias ?? previous?.metadata.key_alias ?? null,
      },
    };
  }
  return result;
};

const mergeMetrics = (
  left: Record<string, MetricWithMetadata>,
  right: Record<string, MetricWithMetadata>,
): Record<string, MetricWithMetadata> => {
  const result = { ...left };
  for (const [id, entry] of Object.entries(right)) {
    const previous = result[id];
    result[id] = {
      ...entry,
      metrics: sumMetrics([...(previous ? [previous.metrics] : []), entry.metrics]),
      metadata: { ...previous?.metadata, ...entry.metadata },
      api_key_breakdown: mergeKeys(previous?.api_key_breakdown ?? {}, entry.api_key_breakdown),
    };
  }
  return result;
};

const mergeBreakdowns = (left: BreakdownMetrics, right: BreakdownMetrics): BreakdownMetrics => ({
  models: mergeMetrics(left.models, right.models),
  model_groups: mergeMetrics(left.model_groups, right.model_groups),
  providers: mergeMetrics(left.providers, right.providers),
  mcp_servers: mergeMetrics(left.mcp_servers, right.mcp_servers),
  endpoints: mergeMetrics(left.endpoints ?? {}, right.endpoints ?? {}),
  entities: mergeMetrics(left.entities, right.entities),
  api_keys: mergeKeys(left.api_keys, right.api_keys),
});

export const mergeCustomerActivityPages = (
  pages: readonly DailyActivityAggregatedResponse[],
): { results: DailyData[]; metadata: DailyActivityMetadata } => {
  const days = new Map<string, DailyData>();
  for (const day of pages.flatMap(toDailyData)) {
    const previous = days.get(day.date);
    days.set(
      day.date,
      previous
        ? {
            date: day.date,
            metrics: sumMetrics([previous.metrics, day.metrics]),
            breakdown: mergeBreakdowns(previous.breakdown, day.breakdown),
          }
        : day,
    );
  }
  const results = [...days.values()].sort((a, b) => a.date.localeCompare(b.date));
  const total = sumMetrics(results.map((day) => day.metrics));
  return {
    results,
    metadata: {
      ...EMPTY_DAILY_ACTIVITY_METADATA,
      total_spend: total.spend,
      total_flat_cost: total.flat_cost ?? 0,
      total_ptu_hours: total.ptu_hours ?? 0,
      total_prompt_tokens: total.prompt_tokens,
      total_completion_tokens: total.completion_tokens,
      total_tokens: total.total_tokens,
      total_api_requests: total.api_requests,
      total_successful_requests: total.successful_requests,
      total_failed_requests: total.failed_requests,
      total_cache_read_input_tokens: total.cache_read_input_tokens,
      total_cache_creation_input_tokens: total.cache_creation_input_tokens,
      total_compression_saved_tokens: total.compression_saved_tokens ?? 0,
      total_compression_savings_spend: total.compression_savings_spend ?? 0,
      total_prompt_caching_savings_spend: total.prompt_caching_savings_spend ?? 0,
      total_gateway_injected_caching_savings_spend: total.gateway_injected_caching_savings_spend ?? 0,
      total_autorouter_savings_spend: total.autorouter_savings_spend ?? 0,
      total_response_time_ms: total.total_response_time_ms ?? 0,
      total_timed_requests: total.timed_requests ?? 0,
      total_api_keys: new Set(results.flatMap((day) => Object.keys(day.breakdown.api_keys))).size,
    },
  };
};

const usageTotals = (value: SpendMetrics): CustomerUsageTotals => ({
  spend: value.spend,
  tokens: value.total_tokens,
  requests: value.api_requests,
});
const addTotals = (left: CustomerUsageTotals, right: CustomerUsageTotals): CustomerUsageTotals => ({
  spend: left.spend + right.spend,
  tokens: left.tokens + right.tokens,
  requests: left.requests + right.requests,
});

export const customerKeyRankings = (
  days: readonly DailyData[],
  unnamedKey: string,
): Omit<CustomerUsageRow, "share">[] => {
  const keys = new Map<string, Omit<CustomerUsageRow, "share">>();
  for (const [id, entry] of days.flatMap((day) => Object.entries(day.breakdown.api_keys))) {
    const previous = keys.get(id);
    keys.set(id, {
      id,
      label: entry.metadata.key_alias?.trim() || previous?.label || unnamedKey,
      ...addTotals(previous ?? { spend: 0, tokens: 0, requests: 0 }, usageTotals(entry.metrics)),
    });
  }
  return [...keys.values()];
};

interface EntityRankingOptions {
  unassignedLabel: string;
  onlyEntityId?: string;
}

export const customerEntityDailyData = (
  primary: readonly DailyData[],
  days: readonly DailyData[],
  labels: ReadonlyMap<string, string>,
  onlyEntityId?: string,
): DailyData[] => {
  const byDate = new Map(days.map((day) => [day.date, day]));
  return primary.map((day) => {
    const entities = onlyEntityId
      ? {
          [onlyEntityId]: {
            metrics: day.metrics,
            metadata: { alias: labels.get(onlyEntityId) },
            api_key_breakdown: day.breakdown.api_keys,
          },
        }
      : Object.fromEntries(
          Object.entries(byDate.get(day.date)?.breakdown.entities ?? {})
            .filter(([id]) => labels.has(id))
            .map(([id, entry]) => [id, { ...entry, metadata: { ...entry.metadata, alias: labels.get(id) } }]),
        );
    return { ...day, breakdown: { ...day.breakdown, entities } };
  });
};

export const customerEntityRankings = (
  days: readonly DailyData[],
  labels: ReadonlyMap<string, string>,
  metadata: DailyActivityMetadata,
  options: EntityRankingOptions,
): Omit<CustomerUsageRow, "share">[] => {
  const entries = new Map<string, CustomerUsageTotals>();
  for (const [id, entry] of days.flatMap((day) => Object.entries(day.breakdown.entities))) {
    const identity = options.onlyEntityId ?? id;
    if (labels.has(identity))
      entries.set(
        identity,
        addTotals(entries.get(identity) ?? { spend: 0, tokens: 0, requests: 0 }, usageTotals(entry.metrics)),
      );
  }
  const rows = [...entries].map(([id, total]) => ({ id, label: labels.get(id)!, ...total }));
  const assigned = rows.reduce(addTotals, { spend: 0, tokens: 0, requests: 0 });
  const unassigned = {
    spend: Math.max(0, metadata.total_spend - assigned.spend),
    tokens: Math.max(0, metadata.total_tokens - assigned.tokens),
    requests: Math.max(0, metadata.total_api_requests - assigned.requests),
  };
  return unassigned.spend > 0.000001 || unassigned.tokens > 0 || unassigned.requests > 0
    ? [...rows, { id: "", label: options.unassignedLabel, ...unassigned }]
    : rows;
};
