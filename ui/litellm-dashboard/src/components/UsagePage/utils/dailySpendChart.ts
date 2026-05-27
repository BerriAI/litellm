import { DateRangePickerValue } from "@tremor/react";

import { BreakdownMetrics, DailyData, SpendMetrics } from "../types";

/**
 * Format a Date as a local-time YYYY-MM-DD string.
 *
 * Mirrors `formatDate` in `networking.tsx` so that comparisons between the
 * picker range and the API-returned `date` strings use the same calendar.
 * Using local-time components matches what the daily activity calls send to
 * the backend in `start_date`/`end_date`.
 */
export function formatLocalDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

const SPEND_METRIC_KEYS: (keyof SpendMetrics)[] = [
  "spend",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "api_requests",
  "successful_requests",
  "failed_requests",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
];

function zeroMetrics(): SpendMetrics {
  return {
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
}

function addMetrics(a: SpendMetrics, b: SpendMetrics): SpendMetrics {
  const out = { ...a };
  for (const key of SPEND_METRIC_KEYS) {
    out[key] = (a[key] || 0) + (b[key] || 0);
  }
  return out;
}

function emptyBreakdown(): BreakdownMetrics {
  return {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
    endpoints: {},
  };
}

type BreakdownObjectKey = Exclude<keyof BreakdownMetrics, "api_keys">;
const BREAKDOWN_OBJECT_KEYS: BreakdownObjectKey[] = [
  "models",
  "model_groups",
  "mcp_servers",
  "providers",
  "entities",
  "endpoints",
];

function mergeMetricWithMetadata<T extends { metrics: SpendMetrics; metadata?: object }>(
  a: T,
  b: T,
): T {
  return {
    ...a,
    ...b,
    metrics: addMetrics(a.metrics, b.metrics),
    metadata: { ...(a.metadata || {}), ...(b.metadata || {}) },
  };
}

function mergeBreakdown(
  a: BreakdownMetrics | undefined,
  b: BreakdownMetrics | undefined,
): BreakdownMetrics {
  const merged = emptyBreakdown();
  for (const src of [a, b]) {
    if (!src) continue;
    for (const k of BREAKDOWN_OBJECT_KEYS) {
      const fromSrc = src[k] || {};
      const target = merged[k] as { [key: string]: any };
      for (const [id, value] of Object.entries(fromSrc)) {
        if (target[id]) {
          target[id] = mergeMetricWithMetadata(target[id], value);
        } else {
          target[id] = value;
        }
      }
    }
    // api_keys has a different value type (KeyMetricWithMetadata) but the
    // merge shape is the same — sum metrics, prefer the latter metadata.
    const fromSrcKeys = src.api_keys || {};
    for (const [id, value] of Object.entries(fromSrcKeys)) {
      if (merged.api_keys[id]) {
        merged.api_keys[id] = mergeMetricWithMetadata(merged.api_keys[id], value);
      } else {
        merged.api_keys[id] = value;
      }
    }
  }
  return merged;
}

/**
 * Inclusive YYYY-MM-DD comparison.
 *
 * Both inputs are calendar date strings — no timezones involved.
 */
function dateInRange(date: string, fromYmd: string | null, toYmd: string | null): boolean {
  if (fromYmd && date < fromYmd) return false;
  if (toYmd && date > toYmd) return false;
  return true;
}

/**
 * Collapse the paginated daily activity `results` into one entry per calendar
 * date, summing metrics and merging breakdowns, then clamp to the picker
 * range and sort ascending.
 *
 * Fixes LIT-3383: when the Usage time range is "Today" (or any single-day
 * range), `usePaginatedDailyActivity` concatenates per-page results without
 * collapsing duplicate dates, so the Daily Spend bar chart renders one bar
 * per page — each labelled with the same date. Additionally, the backend
 * timezone padding in `_adjust_dates_for_timezone` can return rows for the
 * day *after* the picker range (e.g. tomorrow UTC for a PST user), giving
 * a phantom second bar. Clamping by the picker range strips those.
 */
export function getDailySpendChartData(
  results: DailyData[],
  dateValue?: DateRangePickerValue,
): DailyData[] {
  const fromYmd = dateValue?.from ? formatLocalDate(dateValue.from) : null;
  const toYmd = dateValue?.to ? formatLocalDate(dateValue.to) : null;

  const byDate = new Map<string, DailyData>();
  for (const row of results) {
    if (!row || !row.date) continue;
    if (!dateInRange(row.date, fromYmd, toYmd)) continue;

    const existing = byDate.get(row.date);
    if (!existing) {
      byDate.set(row.date, {
        date: row.date,
        metrics: addMetrics(zeroMetrics(), row.metrics),
        breakdown: mergeBreakdown(emptyBreakdown(), row.breakdown),
      });
    } else {
      existing.metrics = addMetrics(existing.metrics, row.metrics);
      existing.breakdown = mergeBreakdown(existing.breakdown, row.breakdown);
    }
  }

  return Array.from(byDate.values()).sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime(),
  );
}
