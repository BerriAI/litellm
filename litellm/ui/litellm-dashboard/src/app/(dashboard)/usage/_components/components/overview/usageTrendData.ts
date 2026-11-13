import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import i18n from "@/i18n/i18n";
import { colorForModel, OTHER_COLOR, type Series } from "./overviewData";

export type TrendGranularity = "day" | "week" | "month";
export type TrendMetric = "tokens" | "spend" | "requests";

/** Inclusive range of the `YYYY-MM-DD` day strings the daily activity endpoints filter on. */
export interface DayRange {
  readonly start: string;
  readonly end: string;
}

export interface TrendBucket {
  readonly key: string;
  readonly start: string;
  readonly end: string;
  readonly partial: boolean;
}

interface ModelTotals {
  readonly key: string;
  readonly spend: number;
  readonly tokens: number;
  readonly requests: number;
}

export type ObservedRate =
  | { readonly kind: "rate"; readonly usdPerMillion: number }
  | { readonly kind: "noTokens" }
  | { readonly kind: "noSpend" }
  | { readonly kind: "none" };

export interface ContributionRow extends ModelTotals {
  readonly tokenShare: number | null;
  readonly spendShare: number | null;
  readonly rate: ObservedRate;
  readonly color: string;
  readonly grouped: boolean;
}

interface UsageTrend {
  readonly buckets: readonly TrendBucket[];
  readonly series: Readonly<Record<TrendMetric, Series>>;
  readonly totals: Readonly<Record<TrendMetric, ReadonlyMap<string, number>>>;
  readonly rows: readonly ContributionRow[];
  readonly total: ContributionRow;
  readonly unattributed: Readonly<Record<TrendMetric, number>>;
}

const LEADING_MODEL_LIMIT = 8;
const TREND_METRICS: readonly TrendMetric[] = ["tokens", "spend", "requests"];
const DAY_MS = 86_400_000;
const MONTHLY_FROM_DAYS = 60;
const WEEKLY_FROM_DAYS = 32;
const RECONCILE_TOLERANCE: Readonly<Record<TrendMetric, number>> = { spend: 0.005, tokens: 0.5, requests: 0.5 };

const dayIndex = (date: string): number => Math.round(Date.parse(`${date}T00:00:00Z`) / DAY_MS);
const dayAt = (index: number): string => new Date(index * DAY_MS).toISOString().slice(0, 10);
const monthKey = (date: string): string => date.slice(0, 7);
const monthParts = (key: string): readonly [number, number] => [Number(key.slice(0, 4)), Number(key.slice(5, 7))];
const monthOrdinal = (key: string): number => {
  const [year, month] = monthParts(key);
  return year * 12 + (month - 1);
};
const monthAt = (ordinal: number): string =>
  `${Math.floor(ordinal / 12)}-${String((ordinal % 12) + 1).padStart(2, "0")}`;
const lastDayOfMonth = (key: string): string => {
  const [year, month] = monthParts(key);
  return dayAt(Math.round(Date.UTC(year, month, 0) / DAY_MS));
};
const later = (a: string, b: string): string => (a > b ? a : b);
const earlier = (a: string, b: string): string => (a < b ? a : b);

const daysInRange = (range: DayRange): number => dayIndex(range.end) - dayIndex(range.start) + 1;

export const defaultGranularity = (range: DayRange): TrendGranularity => {
  const days = daysInRange(range);
  if (days >= MONTHLY_FROM_DAYS) return "month";
  if (days >= WEEKLY_FROM_DAYS) return "week";
  return "day";
};

/**
 * The window right before `range`. A range starting on the 1st is compared month for month, so
 * month-to-date Nov 1-12 meets Oct 1-12 and full months meet full months. Any other range meets
 * the same number of days immediately before it.
 */
export const previousPeriod = (range: DayRange): DayRange => {
  if (range.start.endsWith("-01")) {
    const startMonth = monthOrdinal(monthKey(range.start));
    const endMonth = monthOrdinal(monthKey(range.end));
    const months = endMonth - startMonth + 1;
    const previousEndMonth = monthAt(endMonth - months);
    const previousMonthEnd = lastDayOfMonth(previousEndMonth);
    const endsOnMonthEnd = range.end === lastDayOfMonth(monthKey(range.end));
    const sameDay = earlier(`${previousEndMonth}-${range.end.slice(8, 10)}`, previousMonthEnd);
    return { start: `${monthAt(startMonth - months)}-01`, end: endsOnMonthEnd ? previousMonthEnd : sameDay };
  }
  const days = daysInRange(range);
  const start = dayIndex(range.start);
  return { start: dayAt(start - days), end: dayAt(start - 1) };
};

/** `range` widened to any returned day outside it, so no row the server sent is left off a chart. */
export const coveringRange = (range: DayRange, results: readonly DailyData[]): DayRange =>
  results.reduce<DayRange>(
    (covered, day) => ({ start: earlier(covered.start, day.date), end: later(covered.end, day.date) }),
    range,
  );

export const trendBuckets = (range: DayRange, granularity: TrendGranularity): readonly TrendBucket[] => {
  const first = dayIndex(range.start);
  const days = daysInRange(range);
  if (days <= 0) return [];
  if (granularity === "day") {
    return Array.from({ length: days }, (_, offset) => {
      const date = dayAt(first + offset);
      return { key: date, start: date, end: date, partial: false };
    });
  }
  if (granularity === "week") {
    return Array.from({ length: Math.ceil(days / 7) }, (_, week) => {
      const start = dayAt(first + week * 7);
      const end = earlier(dayAt(first + week * 7 + 6), range.end);
      return { key: start, start, end, partial: daysInRange({ start, end }) < 7 };
    });
  }
  const firstMonth = monthOrdinal(monthKey(range.start));
  const months = monthOrdinal(monthKey(range.end)) - firstMonth + 1;
  return Array.from({ length: months }, (_, offset) => {
    const key = monthAt(firstMonth + offset);
    const start = later(`${key}-01`, range.start);
    const end = earlier(lastDayOfMonth(key), range.end);
    return { key, start, end, partial: start !== `${key}-01` || end !== lastDayOfMonth(key) };
  });
};

const bucketKeyOf = (date: string, granularity: TrendGranularity, range: DayRange): string => {
  if (granularity === "day") return date;
  if (granularity === "month") return monthKey(date);
  const first = dayIndex(range.start);
  return dayAt(first + Math.floor((dayIndex(date) - first) / 7) * 7);
};

const metricOf = (metrics: SpendMetrics, metric: TrendMetric): number => {
  if (metric === "spend") return metrics.spend;
  if (metric === "tokens") return metrics.total_tokens;
  return metrics.api_requests;
};

const byKey = (a: string, b: string): number => Number(a > b) - Number(a < b);

const byContribution = (a: ModelTotals, b: ModelTotals): number =>
  b.spend - a.spend || b.tokens - a.tokens || b.requests - a.requests || byKey(a.key, b.key);

/** Every model group in the range with its summed spend, tokens and requests, largest contribution first. */
const modelTotals = (results: readonly DailyData[]): readonly ModelTotals[] => {
  const totals = new Map<string, ModelTotals>();
  for (const [key, entry] of results.flatMap((day) => Object.entries(day.breakdown.model_groups ?? {}))) {
    const row = totals.get(key) ?? { key, spend: 0, tokens: 0, requests: 0 };
    totals.set(key, {
      key,
      spend: row.spend + entry.metrics.spend,
      tokens: row.tokens + entry.metrics.total_tokens,
      requests: row.requests + entry.metrics.api_requests,
    });
  }
  return [...totals.values()].sort(byContribution);
};

const hasUsage = (row: ModelTotals): boolean => row.spend !== 0 || row.tokens !== 0 || row.requests !== 0;

const share = (part: number, whole: number): number | null => (whole > 0 ? (part / whole) * 100 : null);

export const observedRate = (spend: number, tokens: number): ObservedRate => {
  if (tokens > 0 && spend === 0) return { kind: "noSpend" };
  if (tokens > 0) return { kind: "rate", usdPerMillion: (spend / tokens) * 1_000_000 };
  if (spend !== 0) return { kind: "noTokens" };
  return { kind: "none" };
};

const UTC_MONTH_DAY = { month: "short", day: "numeric", timeZone: "UTC" } as const;
const UTC_FULL_DAY = { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" } as const;
const formatDay = (date: string, options: Intl.DateTimeFormatOptions): string =>
  new Date(`${date}T00:00:00Z`).toLocaleDateString("en-US", options);

/** "Oct 1 – Oct 12", with years once the range leaves a single year. */
export const formatDayRange = (range: DayRange): string => {
  const options = range.start.slice(0, 4) === range.end.slice(0, 4) ? UTC_MONTH_DAY : UTC_FULL_DAY;
  if (range.start === range.end) return formatDay(range.start, UTC_FULL_DAY);
  return `${formatDay(range.start, options)} – ${formatDay(range.end, options)}`;
};

export const bucketTickLabel = (bucket: TrendBucket, granularity: TrendGranularity, isFirst: boolean): string => {
  if (granularity !== "month") return formatDay(bucket.start, UTC_MONTH_DAY);
  const month = formatDay(bucket.start, { month: "short", timeZone: "UTC" });
  if (bucket.partial) return `${month} ${Number(bucket.start.slice(8, 10))}–${Number(bucket.end.slice(8, 10))}`;
  return isFirst || bucket.key.endsWith("-01") ? `${month} ${bucket.key.slice(0, 4)}` : month;
};

export const bucketTooltipLabel = (bucket: TrendBucket, granularity: TrendGranularity): string => {
  if (granularity === "day") return formatDay(bucket.start, { ...UTC_FULL_DAY, weekday: "short" });
  if (granularity === "week") {
    return i18n.t("usage.trend.weekTooltip", { range: formatDayRange(bucket), days: daysInRange(bucket) });
  }
  const month = formatDay(bucket.start, { month: "long", year: "numeric", timeZone: "UTC" });
  return bucket.partial ? i18n.t("usage.trend.partialMonthTooltip", { month, range: formatDayRange(bucket) }) : month;
};

const seriesKey = (index: number): string => `s${index}`;

/**
 * Tokens, spend and requests per bucket, stacked by the same leading model groups in the same order
 * and colors, with every other group summed into "Other". Bars and table rows are both sums of the
 * response's own model-group breakdown, so they reconcile. Any gap between that breakdown and the
 * day totals is reported in `unattributed`, never folded into a segment.
 */
export const usageTrend = ({
  results,
  range,
  granularity,
  colorByModel,
  leaderLimit = LEADING_MODEL_LIMIT,
}: {
  results: readonly DailyData[];
  range: DayRange;
  granularity: TrendGranularity;
  colorByModel: ReadonlyMap<string, string>;
  leaderLimit?: number;
}): UsageTrend => {
  const covered = coveringRange(range, results);
  const buckets = trendBuckets(covered, granularity);
  const models = modelTotals(results);
  const leaders = models.filter(hasUsage).slice(0, leaderLimit);
  const leaderIndex = new Map(leaders.map((row, index) => [row.key, index]));
  const hasOther = models.some((row) => !leaderIndex.has(row.key) && hasUsage(row));
  const labels = [...leaders.map((row) => row.key), ...(hasOther ? [i18n.t("usage.overview.otherSeries")] : [])];
  const keys = labels.map((_, index) => seriesKey(index));
  const colors = labels.map((label, index) =>
    index < leaders.length ? colorForModel(label, colorByModel) : OTHER_COLOR,
  );
  const otherIndex = leaders.length;

  const cells = new Map(
    buckets.map((bucket) => [bucket.key, TREND_METRICS.map(() => new Array<number>(labels.length).fill(0))]),
  );
  for (const day of results) {
    const row = cells.get(bucketKeyOf(day.date, granularity, covered));
    if (row === undefined) continue;
    for (const [key, entry] of Object.entries(day.breakdown.model_groups ?? {})) {
      const index = leaderIndex.get(key) ?? (hasOther ? otherIndex : -1);
      if (index === -1) continue;
      TREND_METRICS.forEach((metric, metricIndex) => {
        row[metricIndex][index] += metricOf(entry.metrics, metric);
      });
    }
  }

  const tickLabels = buckets.map((bucket, index) => bucketTickLabel(bucket, granularity, index === 0));
  const seriesFor = (metricIndex: number): Series => ({
    data: buckets.map((bucket, bucketIndex) => ({
      date: bucket.key,
      label: tickLabels[bucketIndex],
      ...Object.fromEntries(keys.map((key, index) => [key, cells.get(bucket.key)?.[metricIndex][index] ?? 0])),
    })),
    keys,
    labels,
    colors,
  });
  const series = Object.fromEntries(TREND_METRICS.map((metric, index) => [metric, seriesFor(index)])) as Record<
    TrendMetric,
    Series
  >;
  const totals = Object.fromEntries(
    TREND_METRICS.map((metric, metricIndex) => [
      metric,
      new Map(
        buckets.map((bucket) => [
          bucket.key,
          (cells.get(bucket.key)?.[metricIndex] ?? []).reduce((sum, value) => sum + value, 0),
        ]),
      ),
    ]),
  ) as Record<TrendMetric, ReadonlyMap<string, number>>;

  const sum = (metric: TrendMetric): number => models.reduce((acc, row) => acc + row[metric], 0);
  const grand = { spend: sum("spend"), tokens: sum("tokens"), requests: sum("requests") };
  const rows = models.map((row) => ({
    ...row,
    tokenShare: share(row.tokens, grand.tokens),
    spendShare: share(row.spend, grand.spend),
    rate: observedRate(row.spend, row.tokens),
    color: leaderIndex.has(row.key) ? colorForModel(row.key, colorByModel) : OTHER_COLOR,
    grouped: !leaderIndex.has(row.key),
  }));
  const total: ContributionRow = {
    key: i18n.t("usage.trend.table.total"),
    ...grand,
    tokenShare: grand.tokens > 0 ? 100 : null,
    spendShare: grand.spend > 0 ? 100 : null,
    rate: observedRate(grand.spend, grand.tokens),
    color: OTHER_COLOR,
    grouped: false,
  };
  const unattributed = Object.fromEntries(
    TREND_METRICS.map((metric) => {
      const dayTotal = results.reduce((acc, day) => acc + metricOf(day.metrics, metric), 0);
      const gap = dayTotal - grand[metric];
      return [metric, Math.abs(gap) > RECONCILE_TOLERANCE[metric] ? gap : 0];
    }),
  ) as Record<TrendMetric, number>;

  return { buckets, series, totals, rows, total, unattributed };
};
