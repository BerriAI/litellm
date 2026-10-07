import type { Sample } from "./types";

type Execution = Sample["executions"][number];

export interface DayCount {
  readonly day: string;
  readonly affected: number;
  readonly unaffected: number;
}

export interface Reach {
  readonly affected: number;
  readonly total: number;
}

export interface Frequency extends Reach {
  readonly days: readonly DayCount[];
}

const DAY_MS = 86_400_000;
export const FREQUENCY_WINDOW_DAYS = 14;
export const FREQUENCY_MAX_DAYS = 90;

const dayKey = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

function uniqueRuns(executions: readonly Execution[]): readonly Execution[] {
  return [...new Map(executions.map((run) => [run.id, run] as const)).values()];
}

/** How many distinct sampled traces a finding hit, without building the per-day histogram. */
export function findingReach(occurrences: readonly string[], executions: readonly Execution[]): Reach {
  const hit = new Set(occurrences);
  const sampled = uniqueRuns(executions);
  return { affected: sampled.filter((run) => hit.has(run.id)).length, total: sampled.length };
}

/**
 * Per UTC day counts ending on the latest sampled day. The axis spans at least two weeks so bars sit on a stable
 * scale, and at most FREQUENCY_MAX_DAYS so a trace with a bogus ancient timestamp cannot stretch it unbounded.
 */
export function findingFrequency(occurrences: readonly string[], executions: readonly Execution[]): Frequency {
  const hit = new Set(occurrences);
  const sampled = uniqueRuns(executions);
  const dated = sampled
    .map((run) => ({ affected: hit.has(run.id), ms: Date.parse(run.start_time) }))
    .filter(({ ms }) => Number.isFinite(ms));
  const reach = { affected: sampled.filter((run) => hit.has(run.id)).length, total: sampled.length };
  if (dated.length === 0) return { ...reach, days: [] };
  const last = Date.parse(dayKey(dated.reduce((max, { ms }) => Math.max(max, ms), -Infinity)));
  const floor = last - (FREQUENCY_MAX_DAYS - 1) * DAY_MS;
  const inWindow = dated.reduce((min, { ms }) => (ms >= floor ? Math.min(min, ms) : min), Infinity);
  const first = Date.parse(dayKey(inWindow));
  const start = Math.min(first, last - (FREQUENCY_WINDOW_DAYS - 1) * DAY_MS);
  const length = Math.round((last - start) / DAY_MS) + 1;
  const counts = dated.reduce((acc, { ms, affected }) => {
    const index = Math.floor((ms - start) / DAY_MS);
    if (index < 0 || index >= length) return acc;
    const [a, u] = acc.get(index) ?? [0, 0];
    return acc.set(index, affected ? [a + 1, u] : [a, u + 1]);
  }, new Map<number, readonly [number, number]>());
  return {
    ...reach,
    days: Array.from({ length }, (_, i) => {
      const [affected, unaffected] = counts.get(i) ?? [0, 0];
      return { day: dayKey(start + i * DAY_MS), affected, unaffected };
    }),
  };
}

export function percentLabel(affected: number, total: number): string | null {
  if (total <= 0) return null;
  return `${Number(((affected / total) * 100).toFixed(1))}%`;
}

export const dayLabel = (day: string): string =>
  new Date(`${day}T00:00:00Z`).toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
