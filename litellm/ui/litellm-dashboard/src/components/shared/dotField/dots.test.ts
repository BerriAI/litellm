import { describe, expect, it } from "vitest";

import { DOT_COLS, DOT_ROWS, NEUTRAL_COLOR, SERIES_COLORS, columnDots, litDots, seriesColor } from "./dots";

const capacity = DOT_ROWS * DOT_COLS;
const column = (total: number, failed = 0, series: string[] = []) => ({ total, failed, series });
const count = (dots: ReturnType<typeof columnDots>, kind: string) => dots.filter((dot) => dot.kind === kind).length;

describe("litDots", () => {
  it("lights nothing for an empty column and at least one full row for any items", () => {
    expect(litDots(0, 10)).toBe(0);
    expect(litDots(1, 1000)).toBe(DOT_COLS);
  });

  it("fills the busiest column and keeps quieter columns visible with square-root scaling", () => {
    expect(litDots(4, 4)).toBe(capacity);
    expect(litDots(1, 4)).toBe(capacity / 2);
  });
});

describe("columnDots", () => {
  it("always returns one dot per slot in the column", () => {
    expect(columnDots(column(0), 4, 1)).toHaveLength(capacity);
    expect(count(columnDots(column(0), 4, 1), "grid")).toBe(capacity);
  });

  it("puts failures above the successes", () => {
    const dots = columnDots(column(4, 1, ["a", "a", "a"]), 4, 3);
    const lastSeries = dots.findLastIndex((dot) => dot.kind === "series");
    const firstFailed = dots.findIndex((dot) => dot.kind === "failed");
    expect(firstFailed).toBeGreaterThan(lastSeries);
    expect(count(dots, "failed")).toBeGreaterThan(0);
  });

  it("colors successes by series in bands sized by each series' share", () => {
    const other = ["beta", "gamma", "delta", "omega", "zeta"].find(
      (name) => seriesColor(name) !== seriesColor("alpha"),
    );
    if (!other) throw new Error("palette maps every candidate to one color");
    const dots = columnDots(column(4, 0, ["alpha", "alpha", "alpha", other]), 4, 5);
    const colors = dots.flatMap((dot) => (dot.kind === "series" ? [dot.color] : []));
    const alpha = colors.filter((color) => color === seriesColor("alpha")).length;
    const beta = colors.filter((color) => color === seriesColor(other)).length;
    expect(alpha).toBeGreaterThan(beta * 2);
    expect(beta).toBeGreaterThan(0);
  });

  it("keeps the bottom row solid so even a single item reads as a mark", () => {
    const dots = columnDots(column(1, 0, ["a"]), 1000, 9);
    expect(dots.slice(0, DOT_COLS).every((dot) => dot.kind === "series")).toBe(true);
  });

  it("is deterministic for the same column and seed so the field does not flicker on re-render", () => {
    expect(columnDots(column(3, 1, ["a", "b"]), 4, 11)).toEqual(columnDots(column(3, 1, ["a", "b"]), 4, 11));
  });
});

describe("seriesColor", () => {
  it("gives each series a stable color from the palette", () => {
    expect(seriesColor("claude-code")).toBe(seriesColor("claude-code"));
    expect(SERIES_COLORS).toContain(seriesColor("claude-code"));
  });

  it("keeps an unnamed series neutral so it never looks like a named one", () => {
    expect(seriesColor("")).toBe(NEUTRAL_COLOR);
    expect(SERIES_COLORS).not.toContain(seriesColor(""));
  });
});
