const HOUR_MS = 60 * 60 * 1000;

export interface TimeWindow {
  readonly startMs: number;
  readonly endMs: number;
}

export const RANGE_PRESETS = [
  { hours: 1, label: "Last hour" },
  { hours: 6, label: "Last 6 hours" },
  { hours: 24, label: "Last 24 hours" },
  { hours: 168, label: "Last 7 days" },
  { hours: 720, label: "Last 30 days" },
] as const;

export type RangeHours = (typeof RANGE_PRESETS)[number]["hours"];
export const RANGE_HOURS: readonly RangeHours[] = RANGE_PRESETS.map((preset) => preset.hours);
export const DEFAULT_RANGE_HOURS: RangeHours = 24;
export const isRangeHours = (hours: number): hours is RangeHours => RANGE_HOURS.includes(hours as RangeHours);

export const presetLabel = (hours: number): string =>
  RANGE_PRESETS.find((preset) => preset.hours === hours)?.label ?? `Last ${hours} hours`;

/** A preset length that ends now while live, or at the moment live was paused. */
export interface RelativeRange {
  readonly hours: number;
  readonly anchorMs: number | null;
}

export const isLive = (range: RelativeRange): boolean => range.anchorMs === null;

/** The window a range covers; a live range rolls with `nowMs`, a paused one ignores it. */
export const timeWindow = (range: RelativeRange, nowMs: number): TimeWindow => {
  const endMs = range.anchorMs ?? nowMs;
  return { startMs: endMs - range.hours * HOUR_MS, endMs };
};

export const LIVE_TAIL_INTERVAL_MS = 15000;
