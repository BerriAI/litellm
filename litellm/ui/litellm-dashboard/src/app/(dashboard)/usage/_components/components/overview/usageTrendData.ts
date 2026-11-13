import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import i18n from "@/i18n/i18n";
import { colorForModel, OTHER_COLOR, type Series } from "./overviewData";

export type TrendGranularity = "day" | "week" | "month";
export type TrendMetric = "tokens" | "spend" | "requests";
export type TrendDimension = "model_groups" | "models";

export interface DayRange {
  readonly start: string;
  readonly end: string;
}

export interface TrendBucket extends DayRange {
  readonly key: string;
  readonly partial: boolean;
}

interface Totals {
  readonly spend: number;
  readonly tokens: number;
  readonly requests: number;
}

interface KeyedTotals extends Totals {
  readonly key: string;
}

export type ObservedRate =
  | { readonly kind: "rate"; readonly usdPerMillion: number }
  | { readonly kind: "noTokens" }
  | { readonly kind: "noSpend" }
  | { readonly kind: "none" };

export interface ContributionRow extends KeyedTotals {
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
  readonly unattributed: Totals;
}

const LEADING_MODEL_LIMIT = 8;
const DAY_MS = 86_400_000;
const MONTHLY_FROM_DAYS = 60;
const WEEKLY_FROM_DAYS = 32;
const ZERO: Totals = { spend: 0, tokens: 0, requests: 0 };
const RECONCILE_TOLERANCE: Totals = { spend: 0.005, tokens: 0.5, requests: 0.5 };

const dayIndex = (date: string): number => Math.round(Date.parse(`${date}T00:00:00Z`) / DAY_MS);
const dayAt = (index: number): string => new Date(index * DAY_MS).toISOString().slice(0, 10);
const monthKey = (date: string): string => date.slice(0, 7);
const monthOrdinal = (key: string): number => Number(key.slice(0, 4)) * 12 + Number(key.slice(5, 7)) - 1;
const monthAt = (ordinal: number): string =>
  `${Math.floor(ordinal / 12)}-${String((ordinal % 12) + 1).padStart(2, "0")}`;
const lastDayOfMonth = (key: string): string => dayAt(dayIndex(`${monthAt(monthOrdinal(key) + 1)}-01`) - 1);
const later = (a: string, b: string): string => (a > b ? a : b);
const earlier = (a: string, b: string): string => (a < b ? a : b);
const byKey = (a: string, b: string): number => Number(a > b) - Number(a < b);

const daysInRange = (range: DayRange): number => dayIndex(range.end) - dayIndex(range.start) + 1;

export const defaultGranularity = (range: DayRange): TrendGranularity => {
  const days = daysInRange(range);
  if (days >= MONTHLY_FROM_DAYS) return "month";
  if (days >= WEEKLY_FROM_DAYS) return "week";
  return "day";
};

/**
 * A range starting on the 1st is compared month for month with the same elapsed days in its last
 * month, so month-to-date Nov 1-12 meets Oct 1-12. Any other range meets the same number of days before it.
 */
export const previousPeriod = (range: DayRange): DayRange => {
  if (!range.start.endsWith("-01")) {
    const start = dayIndex(range.start);
    return { start: dayAt(start - daysInRange(range)), end: dayAt(start - 1) };
  }
  const startMonth = monthOrdinal(monthKey(range.start));
  const endMonth = monthOrdinal(monthKey(range.end));
  const months = endMonth - startMonth + 1;
  const previousEndMonth = monthAt(endMonth - months);
  const previousMonthEnd = lastDayOfMonth(previousEndMonth);
  const endsOnMonthEnd = range.end === lastDayOfMonth(monthKey(range.end));
  const sameDay = earlier(`${previousEndMonth}-${range.end.slice(8, 10)}`, previousMonthEnd);
  return { start: `${monthAt(startMonth - months)}-01`, end: endsOnMonthEnd ? previousMonthEnd : sameDay };
};

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

const totalsOf = (metrics: SpendMetrics): Totals => ({
  spend: metrics.spend,
  tokens: metrics.total_tokens,
  requests: metrics.api_requests,
});
const add = (a: Totals, b: Totals): Totals => ({
  spend: a.spend + b.spend,
  tokens: a.tokens + b.tokens,
  requests: a.requests + b.requests,
});
const sumTotals = (items: readonly Totals[]): Totals => items.reduce<Totals>(add, ZERO);

const sumByKey = (items: readonly KeyedTotals[]): ReadonlyMap<string, Totals> => {
  const sorted = [...items].sort((a, b) => byKey(a.key, b.key));
  const starts = sorted.flatMap((item, index) => (index === 0 || item.key !== sorted[index - 1].key ? [index] : []));
  return new Map(
    starts.map((start, index) => [sorted[start].key, sumTotals(sorted.slice(start, starts[index + 1]))] as const),
  );
};

const entriesOf = (day: DailyData, dimension: TrendDimension) => Object.entries(day.breakdown[dimension] ?? {});

const byContribution =
  (rankBy: TrendMetric) =>
  (a: KeyedTotals, b: KeyedTotals): number => {
    const larger = [b[rankBy] - a[rankBy], b.spend - a.spend, b.tokens - a.tokens, b.requests - a.requests].find(
      (difference) => difference !== 0,
    );
    return larger ?? byKey(a.key, b.key);
  };

const modelTotals = (
  results: readonly DailyData[],
  dimension: TrendDimension,
  rankBy: TrendMetric,
): readonly KeyedTotals[] =>
  [
    ...sumByKey(
      results.flatMap((day) => entriesOf(day, dimension).map(([key, entry]) => ({ key, ...totalsOf(entry.metrics) }))),
    ),
  ]
    .map(([key, totals]) => ({ key, ...totals }))
    .sort(byContribution(rankBy));

const share = (part: number, whole: number): number | null => (whole > 0 ? (part / whole) * 100 : null);

export const observedRate = (spend: number, tokens: number): ObservedRate => {
  if (tokens > 0 && spend === 0) return { kind: "noSpend" };
  if (tokens > 0) return { kind: "rate", usdPerMillion: (spend / tokens) * 1_000_000 };
  if (spend !== 0) return { kind: "noTokens" };
  return { kind: "none" };
};

const formatDay = (date: string, options: Intl.DateTimeFormatOptions): string =>
  new Intl.DateTimeFormat(i18n.language, { ...options, timeZone: "UTC" }).format(new Date(`${date}T00:00:00Z`));
const MONTH_DAY = { month: "short", day: "numeric" } as const;
const FULL_DAY = { ...MONTH_DAY, year: "numeric" } as const;

export const formatDayRange = (range: DayRange): string => {
  if (range.start === range.end) return formatDay(range.start, FULL_DAY);
  const options = range.start.slice(0, 4) === range.end.slice(0, 4) ? MONTH_DAY : FULL_DAY;
  return i18n.t("usage.trend.dayRange", {
    start: formatDay(range.start, options),
    end: formatDay(range.end, options),
  });
};

export const bucketTickLabel = (bucket: TrendBucket, granularity: TrendGranularity, isFirst: boolean): string => {
  if (granularity !== "month") return formatDay(bucket.start, MONTH_DAY);
  if (bucket.partial) {
    return i18n.t("usage.trend.partialMonthTick", {
      month: formatDay(bucket.start, { month: "short" }),
      from: formatDay(bucket.start, { day: "numeric" }),
      to: formatDay(bucket.end, { day: "numeric" }),
    });
  }
  const withYear = isFirst || bucket.key.endsWith("-01");
  return formatDay(bucket.start, withYear ? { month: "short", year: "numeric" } : { month: "short" });
};

export const bucketTooltipLabel = (bucket: TrendBucket, granularity: TrendGranularity): string => {
  if (granularity === "day") return formatDay(bucket.start, { ...FULL_DAY, weekday: "short" });
  if (granularity === "week") {
    return i18n.t("usage.trend.weekTooltip", { range: formatDayRange(bucket), days: daysInRange(bucket) });
  }
  const month = formatDay(bucket.start, { month: "long", year: "numeric" });
  return bucket.partial ? i18n.t("usage.trend.partialMonthTooltip", { month, range: formatDayRange(bucket) }) : month;
};

const cellKey = (bucket: string, series: number): string => `${bucket}\u0000${series}`;

/**
 * Bars and table rows are both sums of the response's own breakdown, so they reconcile. A gap between
 * that breakdown and the day totals is reported in `unattributed`, never drawn as a segment.
 */
export const usageTrend = ({
  results,
  range,
  granularity,
  colorByModel,
  dimension = "model_groups",
  rankBy = "spend",
}: {
  results: readonly DailyData[];
  range: DayRange;
  granularity: TrendGranularity;
  colorByModel: ReadonlyMap<string, string>;
  dimension?: TrendDimension;
  rankBy?: TrendMetric;
}): UsageTrend => {
  const covered = coveringRange(range, results);
  const buckets = trendBuckets(covered, granularity);
  const models = modelTotals(results, dimension, rankBy);
  const contributes = (row: Totals): boolean => row[rankBy] !== 0;
  const leaders = models.filter(contributes).slice(0, LEADING_MODEL_LIMIT);
  const leaderIndex = new Map(leaders.map((row, index) => [row.key, index]));
  const hasOther = models.some((row) => !leaderIndex.has(row.key) && contributes(row));
  const labels = [...leaders.map((row) => row.key), ...(hasOther ? [i18n.t("usage.overview.otherSeries")] : [])];
  const keys = labels.map((_, index) => `s${index}`);
  const colors = labels.map((label, index) =>
    index < leaders.length ? colorForModel(label, colorByModel) : OTHER_COLOR,
  );

  const cells = sumByKey(
    results.flatMap((day) => {
      const bucket = bucketKeyOf(day.date, granularity, covered);
      return entriesOf(day, dimension)
        .map(([key, entry]) => ({ index: leaderIndex.get(key) ?? leaders.length, metrics: entry.metrics }))
        .filter(({ index }) => index < labels.length)
        .map(({ index, metrics }) => ({ key: cellKey(bucket, index), ...totalsOf(metrics) }));
    }),
  );
  const cell = (bucket: string, index: number): Totals => cells.get(cellKey(bucket, index)) ?? ZERO;
  const tickLabels = buckets.map((bucket, index) => bucketTickLabel(bucket, granularity, index === 0));
  const seriesFor = (metric: TrendMetric): Series => ({
    data: buckets.map((bucket, bucketIndex) => ({
      date: bucket.key,
      label: tickLabels[bucketIndex],
      ...Object.fromEntries(keys.map((key, index) => [key, cell(bucket.key, index)[metric]])),
    })),
    keys,
    labels,
    colors,
  });
  const totalsFor = (metric: TrendMetric): ReadonlyMap<string, number> =>
    new Map(
      buckets.map((bucket) => [bucket.key, keys.reduce((sum, _, index) => sum + cell(bucket.key, index)[metric], 0)]),
    );

  const grand = sumTotals(models);
  const dayTotals = sumTotals(results.map((day) => totalsOf(day.metrics)));
  const gap = (metric: TrendMetric): number => {
    const difference = dayTotals[metric] - grand[metric];
    return Math.abs(difference) > RECONCILE_TOLERANCE[metric] ? difference : 0;
  };
  const rows = models.map((row) => ({
    ...row,
    tokenShare: share(row.tokens, grand.tokens),
    spendShare: share(row.spend, grand.spend),
    rate: observedRate(row.spend, row.tokens),
    color: leaderIndex.has(row.key) ? colorForModel(row.key, colorByModel) : OTHER_COLOR,
    grouped: !leaderIndex.has(row.key),
  }));

  return {
    buckets,
    series: { tokens: seriesFor("tokens"), spend: seriesFor("spend"), requests: seriesFor("requests") },
    totals: { tokens: totalsFor("tokens"), spend: totalsFor("spend"), requests: totalsFor("requests") },
    rows,
    total: {
      key: i18n.t("usage.trend.table.total"),
      ...grand,
      tokenShare: grand.tokens > 0 ? 100 : null,
      spendShare: grand.spend > 0 ? 100 : null,
      rate: observedRate(grand.spend, grand.tokens),
      color: OTHER_COLOR,
      grouped: false,
    },
    unattributed: { spend: gap("spend"), tokens: gap("tokens"), requests: gap("requests") },
  };
};
