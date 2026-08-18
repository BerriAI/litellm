import { SEQUENTIAL_COLOR_RAMP, type ChartColor } from "@/components/shared/charts";
import type { ActiveRequest } from "./activeRequestsApi";

export const GROUP_BY = [
  { value: "model", label: "Model" },
  { value: "end_user_id", label: "End User" },
  { value: "user_id", label: "User" },
  { value: "key_alias", label: "Key" },
  { value: "pod", label: "Pod" },
] as const;

export type GroupBy = (typeof GROUP_BY)[number]["value"];

export const AGE_BINS = [
  { label: "< 10s", limit: 10 },
  { label: "10-30s", limit: 30 },
  { label: "30-60s", limit: 60 },
  { label: "1-5m", limit: 300 },
  { label: "> 5m", limit: Number.POSITIVE_INFINITY },
] as const;

export const TOP_N = 8;
export const UNATTRIBUTED = "unattributed";

export type GroupedCount = {
  label: string;
  requests: number;
} & Record<string, unknown>;

export const magnitudeFills = (
  count: number,
  options: { dark: boolean; ascending: boolean },
): readonly ChartColor[] => {
  const darkestFirst = options.ascending !== options.dark;
  const ramp = darkestFirst ? [...SEQUENTIAL_COLOR_RAMP] : [...SEQUENTIAL_COLOR_RAMP].reverse();
  return Array.from({ length: count }, (_, index) => ramp[Math.min(index, ramp.length - 1)]);
};

export const chartHeight = (rows: number): number => Math.max(200, Math.min(rows * 42 + 48, 440));

export const countBy = (items: readonly ActiveRequest[], groupBy: GroupBy): GroupedCount[] => {
  const counts = items.reduce<Map<string, number>>((acc, item) => {
    const raw = item[groupBy];
    const key = typeof raw === "string" && raw.trim() !== "" ? raw : UNATTRIBUTED;
    return new Map(acc).set(key, (acc.get(key) ?? 0) + 1);
  }, new Map());

  const sorted = [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
  const head = sorted.slice(0, TOP_N);
  const tail = sorted.slice(TOP_N);
  const other = tail.reduce((sum, [, value]) => sum + value, 0);

  return [
    ...head.map(([label, requests]) => ({ label, requests })),
    ...(other > 0 ? [{ label: `Other (${tail.length})`, requests: other }] : []),
  ];
};

export const countByAge = (items: readonly ActiveRequest[], nowMs: number): GroupedCount[] => {
  const nowSeconds = nowMs / 1000;
  return AGE_BINS.map(({ label, limit }, index) => {
    const lower = index === 0 ? 0 : AGE_BINS[index - 1].limit;
    const requests = items.filter((item) => {
      const age = nowSeconds - item.started_at;
      return age >= lower && age < limit;
    }).length;
    return { label, requests };
  });
};
