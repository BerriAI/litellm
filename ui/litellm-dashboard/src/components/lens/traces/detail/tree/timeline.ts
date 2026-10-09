const MIN_BAR_PERCENT = 0.6;
const TICK_STEPS_MS = [
  1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000,
];

export interface BarGeometry {
  readonly left: number;
  readonly width: number;
}

/** Where a span sits on the run's timeline, in percent of the run; tiny spans keep a visible sliver. */
export function barGeometry(startMs: number, durationMs: number, totalMs: number): BarGeometry {
  if (totalMs <= 0) return { left: 0, width: MIN_BAR_PERCENT };
  const left = Math.min(100 - MIN_BAR_PERCENT, Math.max(0, (startMs / totalMs) * 100));
  const width = Math.max(MIN_BAR_PERCENT, Math.min(100 - left, (durationMs / totalMs) * 100));
  return { left, width };
}

export const BAR_TRACK = "w-1/3";

/** Compact label for a round tick: `0`, `500ms`, `2s`, `1.5m`. */
export function tickLabel(ms: number): string {
  if (ms === 0) return "0";
  if (ms >= 60_000) return `${+(ms / 60_000).toFixed(1)}m`;
  if (ms >= 1000) return `${+(ms / 1000).toFixed(1)}s`;
  return `${ms}ms`;
}

/** Round tick offsets across the run: the smallest step that fits at most `maxTicks` labels. */
export function timeTicks(totalMs: number, maxTicks = 6): readonly number[] {
  if (totalMs <= 0) return [0];
  const step = TICK_STEPS_MS.find((candidate) => totalMs / candidate <= maxTicks) ?? Math.ceil(totalMs / maxTicks);
  return Array.from({ length: Math.floor(totalMs / step) + 1 }, (_, i) => i * step);
}
