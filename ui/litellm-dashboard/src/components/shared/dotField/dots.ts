export const SERIES_COLORS = ["#0011b3", "#3b5bfd", "#7c93ff", "#22b3e8"] as const;
export const NEUTRAL_COLOR = "#94a3b8";
export const FAILED_COLOR = "#e5484d";
export const DOT_ROWS = 12;
export const DOT_COLS = 2;
export const DOT_PITCH = 6;
export const FIELD_HEIGHT = DOT_ROWS * DOT_PITCH;

export type Dot = { kind: "series"; color: string } | { kind: "failed" } | { kind: "grid" };

/** One column of the field: how many items it holds, how many failed, and the series name of each success. */
export interface DotColumn {
  total: number;
  failed: number;
  series: readonly string[];
  opacity?: number;
}

/** Column-index band [lo, hi], inclusive. Columns outside it are dimmed. */
export interface DotBand {
  lo: number;
  hi: number;
}

const hash = (seed: number, value: number): number => Math.imul(seed ^ Math.imul(value, 0x9e3779b1), 0x85ebca6b) >>> 0;

export function seriesColor(name: string): string {
  if (!name) return NEUTRAL_COLOR;
  const code = Array.from(name).reduce((total, char) => hash(total, char.charCodeAt(0)), 7);
  return SERIES_COLORS[code % SERIES_COLORS.length];
}

/** Lit dot count for a column: square-root scaled against the busiest column so sparse traffic still reads. */
export function litDots(total: number, max: number, capacity = DOT_ROWS * DOT_COLS): number {
  if (total === 0 || max === 0) return 0;
  return Math.max(DOT_COLS, Math.round(Math.sqrt(total / max) * capacity));
}

/**
 * Bottom-up dots for one column. Successes fill from the bottom in series-colored bands sized by each series'
 * share; failures sit on top. A stable sprinkle of gaps (seeded per column) keeps the field organic.
 */
export function columnDots(column: DotColumn, max: number, seed: number): readonly Dot[] {
  const capacity = DOT_ROWS * DOT_COLS;
  const lit = litDots(column.total, max, capacity);
  const failed =
    column.failed === 0 ? 0 : Math.min(lit, Math.max(DOT_COLS, Math.round((column.failed / column.total) * lit)));
  const ok = lit - failed;
  const series = column.series.length ? column.series : [""];
  return Array.from({ length: capacity }, (_, index): Dot => {
    if (index >= lit) return { kind: "grid" };
    const gap = index >= DOT_COLS && hash(seed + 1, index) % 100 < 12;
    if (gap) return { kind: "grid" };
    if (index >= ok) return { kind: "failed" };
    return { kind: "series", color: seriesColor(series[Math.floor((index / ok) * series.length)]) };
  });
}
