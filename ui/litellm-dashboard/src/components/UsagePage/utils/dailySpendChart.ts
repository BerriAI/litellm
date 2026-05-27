/**
 * Helpers for the "Daily Spend" bar chart on the Usage page.
 *
 * Bug: when the user picks a single-day range like "Today", the underlying
 * `/user/daily/activity` endpoint returns one row per UTC calendar day that
 * overlaps the requested range. Because the UI sends the range using the
 * browser's local timezone (start-of-day -> end-of-day local) and the server
 * groups by UTC day, that range can span 1-2 UTC days and produce 1-2 bars
 * whose X-axis ticks are calendar dates ("2025-05-26", "2026-05-27") even
 * though the user only selected a single day.
 *
 * Fix: when the selected date range is a single local calendar day, collapse
 * those rows into a single bar and replace the X-axis label with a time-of-day
 * string ("12 AM") -- Tremor's BarChart has no `xAxisFormatter` prop, so we
 * mutate the `index` value itself. All other ranges are returned unchanged
 * (sorted ascending by date).
 */

import type { DateRangePickerValue } from "@tremor/react";

import type { BreakdownMetrics, DailyData, SpendMetrics } from "../types";

export function isSingleDayRange(value: DateRangePickerValue | undefined | null): boolean {
  if (!value || !value.from || !value.to) return false;
  const from = new Date(value.from);
  const to = new Date(value.to);
  return (
    from.getFullYear() === to.getFullYear() &&
    from.getMonth() === to.getMonth() &&
    from.getDate() === to.getDate()
  );
}

export const SINGLE_DAY_TIME_LABEL = "12 AM";

function emptyMetrics(): SpendMetrics {
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

function emptyBreakdown(): BreakdownMetrics {
  return {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  };
}

function addMetrics(into: SpendMetrics, from: SpendMetrics): void {
  into.spend += from.spend || 0;
  into.prompt_tokens += from.prompt_tokens || 0;
  into.completion_tokens += from.completion_tokens || 0;
  into.total_tokens += from.total_tokens || 0;
  into.api_requests += from.api_requests || 0;
  into.successful_requests += from.successful_requests || 0;
  into.failed_requests += from.failed_requests || 0;
  into.cache_read_input_tokens += from.cache_read_input_tokens || 0;
  into.cache_creation_input_tokens += from.cache_creation_input_tokens || 0;
}

export function collapseDailyResults(results: DailyData[], label: string): DailyData {
  const metrics = emptyMetrics();
  for (const row of results) {
    if (row && row.metrics) addMetrics(metrics, row.metrics);
  }
  return {
    date: label,
    metrics,
    breakdown: emptyBreakdown(),
  };
}

export function getDailySpendChartData(
  results: DailyData[],
  dateValue: DateRangePickerValue | undefined | null,
): DailyData[] {
  if (!results || results.length === 0) return [];
  if (isSingleDayRange(dateValue)) {
    return [collapseDailyResults(results, SINGLE_DAY_TIME_LABEL)];
  }
  return [...results].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}
