import { describe, expect, it } from "vitest";

import {
  AGENT_DOT_COLORS,
  FIELD_COLS,
  FIELD_ROWS,
  UNNAMED_AGENT_COLOR,
  agoLabel,
  isReceiving,
  agentDotColor,
  columnDots,
  columnTop,
  litDots,
} from "./lensField";

const capacity = FIELD_ROWS * FIELD_COLS;
const bucket = (runs: number, failed = 0, agents: string[] = []) => ({ runs, failed, agents });
const count = (dots: ReturnType<typeof columnDots>, kind: string) => dots.filter((dot) => dot.kind === kind).length;

describe("litDots", () => {
  it("lights nothing for an empty bucket and at least one full row for any runs", () => {
    expect(litDots(0, 10)).toBe(0);
    expect(litDots(1, 1000)).toBe(FIELD_COLS);
  });

  it("fills the busiest bucket and keeps quieter buckets visible with square-root scaling", () => {
    expect(litDots(4, 4)).toBe(capacity);
    expect(litDots(1, 4)).toBe(capacity / 2);
  });
});

describe("columnDots", () => {
  it("always returns one dot per slot in the column", () => {
    expect(columnDots(bucket(0), 4, 1)).toHaveLength(capacity);
    expect(count(columnDots(bucket(0), 4, 1), "grid")).toBe(capacity);
  });

  it("puts failures above the successful runs", () => {
    const dots = columnDots(bucket(4, 1, ["a", "a", "a"]), 4, 3);
    const lastRun = dots.findLastIndex((dot) => dot.kind === "run");
    const firstFailed = dots.findIndex((dot) => dot.kind === "failed");
    expect(firstFailed).toBeGreaterThan(lastRun);
    expect(count(dots, "failed")).toBeGreaterThan(0);
  });

  it("colors successful runs by agent in bands sized by each agent's share", () => {
    const other = ["beta", "gamma", "delta", "omega", "zeta"].find(
      (name) => agentDotColor(name) !== agentDotColor("alpha"),
    );
    if (!other) throw new Error("palette maps every candidate to one color");
    const dots = columnDots(bucket(4, 0, ["alpha", "alpha", "alpha", other]), 4, 5);
    const colors = dots.flatMap((dot) => (dot.kind === "run" ? [dot.color] : []));
    const alpha = colors.filter((color) => color === agentDotColor("alpha")).length;
    const beta = colors.filter((color) => color === agentDotColor(other)).length;
    expect(alpha).toBeGreaterThan(beta * 2);
    expect(beta).toBeGreaterThan(0);
  });

  it("keeps the bottom row solid so even a single run reads as a mark", () => {
    const dots = columnDots(bucket(1, 0, ["a"]), 1000, 9);
    expect(dots.slice(0, FIELD_COLS).every((dot) => dot.kind === "run")).toBe(true);
  });

  it("is deterministic for the same bucket and seed so the field does not flicker on re-render", () => {
    expect(columnDots(bucket(3, 1, ["a", "b"]), 4, 11)).toEqual(columnDots(bucket(3, 1, ["a", "b"]), 4, 11));
  });
});

describe("agentDotColor", () => {
  it("gives each agent a stable color from the palette", () => {
    expect(agentDotColor("claude-code")).toBe(agentDotColor("claude-code"));
    expect(AGENT_DOT_COLORS).toContain(agentDotColor("claude-code"));
  });

  it("keeps runs with no agent name neutral so they never look like a named agent", () => {
    expect(agentDotColor("")).toBe(UNNAMED_AGENT_COLOR);
    expect(AGENT_DOT_COLORS).not.toContain(agentDotColor(""));
  });
});

describe("columnTop", () => {
  it("reaches the top of the field only for the busiest bucket", () => {
    expect(columnTop(4, 4)).toBe(1);
    expect(columnTop(0, 4)).toBe(0);
    expect(columnTop(1, 4)).toBeGreaterThan(0);
    expect(columnTop(1, 4)).toBeLessThan(columnTop(2, 4));
  });
});

describe("agoLabel", () => {
  const now = Date.UTC(2026, 9, 3, 12, 0, 0);

  it("rounds down to the largest whole unit", () => {
    expect(agoLabel(now - 2_000, now)).toBe("just now");
    expect(agoLabel(now - 45_000, now)).toBe("45s ago");
    expect(agoLabel(now - 12 * 60_000 - 59_000, now)).toBe("12m ago");
    expect(agoLabel(now - 3 * 3_600_000, now)).toBe("3h ago");
    expect(agoLabel(now - 2 * 86_400_000, now)).toBe("2d ago");
  });

  it("treats a timestamp slightly in the future as just now instead of negative", () => {
    expect(agoLabel(now + 10_000, now)).toBe("just now");
  });
});

describe("isReceiving", () => {
  const now = Date.UTC(2026, 9, 3, 12, 0, 0);

  it("is live only when a trace landed in the last five minutes", () => {
    expect(isReceiving(now - 4 * 60_000, now)).toBe(true);
    expect(isReceiving(now - 6 * 60_000, now)).toBe(false);
    expect(isReceiving(null, now)).toBe(false);
  });
});
