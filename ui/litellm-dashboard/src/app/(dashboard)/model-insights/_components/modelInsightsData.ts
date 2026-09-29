export type Metric = "requests" | "spend" | "tokens";

export type ModelMetric = {
  model_group: string;
  model: string;
  provider: string;
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
};
export type DailyMetric = ModelMetric & { date: string };
export type ModelInsightsResponse = {
  start_date: string;
  end_date: string;
  daily: DailyMetric[];
  top_models: ModelMetric[];
};
export type TaskSummary = {
  task_type: string;
  label: string;
  category: string;
  value: number;
  share: number;
  leader: string;
  provider: string;
};
export type ModelInsightTasksResponse = { start_date: string; end_date: string; tasks: TaskSummary[] };

export type RankedModel = { model_group: string; provider: string; share: number; delta: number };
const DAY_MS = 86_400_000;
const WEEK_DAYS = 7;

export const metricValue = (row: ModelMetric, metric: Metric) => {
  if (metric === "requests") return row.requests;
  if (metric === "spend") return row.spend;
  return row.prompt_tokens + row.completion_tokens;
};

const COMPACT_SPEND_FROM = 10_000;

export const formatMetric = (value: number, metric: Metric) => {
  if (metric === "spend") {
    const compact = value >= COMPACT_SPEND_FROM;
    const options: Intl.NumberFormatOptions = {
      style: "currency",
      currency: "USD",
      notation: compact ? "compact" : "standard",
      maximumFractionDigits: compact ? 1 : 2,
    };
    return new Intl.NumberFormat("en-US", options).format(value);
  }
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
};

const toDay = (date: string) => Date.parse(`${date}T00:00:00Z`);
const isoDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);

export type DateRange = { start: string; end: string };

export const modelOrder = (rows: DailyMetric[], metric: Metric) => {
  const totals = new Map<string, number>();
  for (const row of rows) totals.set(row.model_group, (totals.get(row.model_group) ?? 0) + metricValue(row, metric));
  return [...totals.entries()].sort((a, b) => b[1] - a[1]).map(([model]) => model);
};

export const buildWeeklySeries = (rows: DailyMetric[], models: string[], metric: Metric, range: DateRange) => {
  const weekMs = WEEK_DAYS * DAY_MS;
  const origin = toDay(range.start);
  const weekCount = Math.floor((toDay(range.end) - origin) / weekMs) + 1;
  const buckets = Array.from({ length: weekCount }, (_, week) => ({
    date: isoDay(origin + week * weekMs),
    ...Object.fromEntries(models.map((model) => [model, 0])),
  })) as Record<string, number | string>[];
  for (const row of rows) {
    const bucket = buckets[Math.floor((toDay(row.date) - origin) / weekMs)];
    if (bucket) bucket[row.model_group] = Number(bucket[row.model_group] ?? 0) + metricValue(row, metric);
  }
  return buckets;
};

const shareByModel = (rows: { model_group: string; provider: string }[], values: number[]) => {
  const totals = new Map<string, { provider: string; value: number }>();
  rows.forEach((row, index) => {
    const current = totals.get(row.model_group) ?? { provider: row.provider, value: 0 };
    totals.set(row.model_group, { provider: row.provider, value: current.value + values[index] });
  });
  const grand = [...totals.values()].reduce((sum, entry) => sum + entry.value, 0);
  return { totals, grand };
};

const halfShares = (daily: DailyMetric[], metric: Metric, range: DateRange) => {
  const midpoint = isoDay(toDay(range.start) + Math.floor((toDay(range.end) - toDay(range.start)) / 2 + DAY_MS / 2));
  const share = (rows: DailyMetric[]) => {
    const { totals, grand } = shareByModel(
      rows,
      rows.map((row) => metricValue(row, metric)),
    );
    return {
      hasUsage: grand > 0,
      of: (model: string) => (grand === 0 ? 0 : ((totals.get(model)?.value ?? 0) / grand) * 100),
    };
  };
  return {
    earlier: share(daily.filter((row) => row.date < midpoint)),
    later: share(daily.filter((row) => row.date >= midpoint)),
  };
};

export const rankModels = (
  rows: ModelMetric[],
  daily: DailyMetric[],
  metric: Metric,
  range: DateRange,
): RankedModel[] => {
  const { totals, grand } = shareByModel(
    rows,
    rows.map((row) => metricValue(row, metric)),
  );
  const { earlier, later } = halfShares(daily, metric, range);
  const comparable = earlier.hasUsage && later.hasUsage;
  return [...totals.entries()]
    .sort((a, b) => b[1].value - a[1].value)
    .map(([model_group, entry]) => ({
      model_group,
      provider: entry.provider,
      share: grand === 0 ? 0 : (entry.value / grand) * 100,
      delta: comparable ? later.of(model_group) - earlier.of(model_group) : 0,
    }));
};
