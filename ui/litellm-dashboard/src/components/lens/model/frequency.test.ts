import { describe, expect, it } from "vitest";
import { FREQUENCY_WINDOW_DAYS, findingFrequency, percentLabel } from "./frequency";
import type { Sample } from "./types";

const run = (id: string, start_time: string): Sample["executions"][number] => ({
  id,
  start_time,
  name: id,
  metadata: [],
  root_seen: true,
  service: "agent",
  source: "traces",
  span_count: 1,
  team_id: "",
  trace_id: id,
  trace_ref: "",
});

describe("findingFrequency", () => {
  it("counts affected sampled traces per day and keeps empty days between them", () => {
    const runs = [
      run("a", "2026-10-01T09:00:00Z"),
      run("b", "2026-10-01T23:59:00Z"),
      run("c", "2026-10-03T00:01:00Z"),
      run("c", "2026-10-03T00:01:00Z"),
    ];
    const frequency = findingFrequency(["b", "c", "not-sampled"], runs);
    expect(frequency.affected).toBe(2);
    expect(frequency.total).toBe(3);
    expect(frequency.days.slice(-3)).toEqual([
      { day: "2026-10-01", affected: 1, unaffected: 1 },
      { day: "2026-10-02", affected: 0, unaffected: 0 },
      { day: "2026-10-03", affected: 1, unaffected: 0 },
    ]);
  });

  it("pads a short sample back to a two-week axis ending on the latest sampled day", () => {
    const { days } = findingFrequency(["a"], [run("a", "2026-10-04T12:00:00Z")]);
    expect(days).toHaveLength(FREQUENCY_WINDOW_DAYS);
    expect(days[0]).toEqual({ day: "2026-09-21", affected: 0, unaffected: 0 });
    expect(days.at(-1)).toEqual({ day: "2026-10-04", affected: 1, unaffected: 0 });
  });

  it("keeps every sampled day when the sample spans longer than two weeks", () => {
    const { days } = findingFrequency([], [run("a", "2026-09-01T00:00:00Z"), run("b", "2026-10-01T00:00:00Z")]);
    expect(days[0].day).toBe("2026-09-01");
    expect(days).toHaveLength(31);
  });

  it("returns no days when nothing was sampled", () => {
    expect(findingFrequency(["a"], [])).toEqual({ affected: 0, total: 0, days: [] });
  });
});

describe("percentLabel", () => {
  it("rounds to one decimal and drops a trailing zero", () => {
    expect(percentLabel(5, 54)).toBe("9.3%");
    expect(percentLabel(1, 2)).toBe("50%");
  });

  it("has no percentage without a denominator", () => {
    expect(percentLabel(0, 0)).toBeNull();
  });
});
