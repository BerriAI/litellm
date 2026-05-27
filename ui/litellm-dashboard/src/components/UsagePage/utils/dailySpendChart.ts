import type { DateRangePickerValue } from "@tremor/react";
import type { DailyData, SpendMetrics } from "../types";
export const SINGLE_DAY_TIME_LABEL = "Today";
const ZERO: SpendMetrics = { spend:0, prompt_tokens:0, completion_tokens:0, total_tokens:0, api_requests:0, successful_requests:0, failed_requests:0, cache_read_input_tokens:0, cache_creation_input_tokens:0 };
export function isSingleDayRange(d: DateRangePickerValue | null | undefined): boolean {
  if (!d) return false;
  const { from, to } = d;
  if (!from || !to) return false;
  return from.getFullYear()===to.getFullYear() && from.getMonth()===to.getMonth() && from.getDate()===to.getDate();
}
export function collapseDailyResults<T extends DailyData>(rows: T[], label: string): T {
  const m: SpendMetrics = { ...ZERO };
  for (const r of rows) { const x: any = r.metrics ?? {};
    m.spend += x.spend ?? 0; m.prompt_tokens += x.prompt_tokens ?? 0; m.completion_tokens += x.completion_tokens ?? 0;
    m.total_tokens += x.total_tokens ?? 0; m.api_requests += x.api_requests ?? 0;
    m.successful_requests += x.successful_requests ?? 0; m.failed_requests += x.failed_requests ?? 0;
    m.cache_read_input_tokens += x.cache_read_input_tokens ?? 0; m.cache_creation_input_tokens += x.cache_creation_input_tokens ?? 0;
  }
  const breakdown = rows[0]?.breakdown ?? { models:{}, model_groups:{}, mcp_servers:{}, providers:{}, api_keys:{}, entities:{} };
  return { ...(rows[0] ?? {}), date: label, metrics: m, breakdown } as T;
}
export function getDailySpendChartData<T extends DailyData>(rows: T[], dv: DateRangePickerValue | null | undefined): T[] {
  if (rows.length === 0) return [];
  if (isSingleDayRange(dv)) return [collapseDailyResults(rows, SINGLE_DAY_TIME_LABEL)];
  return [...rows].sort((a,b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}
