import { describe, expect, it } from "vitest";
import { FREQUENCY_MAX_DAYS, FREQUENCY_WINDOW_DAYS, findingFrequency, findingReach, percentLabel } from "./frequency";
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

  it("never draws more than the max window, even when real samples spread wider", () => {
    const runs = Array.from({ length: 200 }, (_, i) =>
      run(`d${i}`, new Date(Date.UTC(2026, 9, 4) - i * 86_400_000).toISOString()),
    );
    const { days, total } = findingFrequency([], runs);
    expect(total).toBe(200);
    expect(days).toHaveLength(FREQUENCY_MAX_DAYS);
    expect(days.at(-1)?.day).toBe("2026-10-04");
  });

  it("keeps every sampled day when the sample spans longer than two weeks", () => {
    const { days } = findingFrequency([], [run("a", "2026-09-01T00:00:00Z"), run("b", "2026-10-01T00:00:00Z")]);
    expect(days[0].day).toBe("2026-09-01");
    expect(days).toHaveLength(31);
  });

  it("ignores a trace with an ancient timestamp when sizing the axis but still counts it", () => {
    const runs = [run("ancient", "1970-01-01T00:00:00.000000001Z"), run("a", "2026-10-04T12:00:00Z")];
    const frequency = findingFrequency(["ancient", "a"], runs);
    expect(frequency.days).toHaveLength(FREQUENCY_WINDOW_DAYS);
    expect(frequency.days.at(-1)).toEqual({ day: "2026-10-04", affected: 1, unaffected: 0 });
    expect(frequency.days.reduce((sum, d) => sum + d.affected + d.unaffected, 0)).toBe(1);
    expect(frequency).toMatchObject({ affected: 2, total: 2 });
  });

  it("keeps every in-window sample when one crafted 1970 span joins a full sample", () => {
    const recent = Array.from({ length: 250 }, (_, i) =>
      run(`r${i}`, new Date(Date.UTC(2026, 9, 4) - (i % 30) * 86_400_000).toISOString()),
    );
    const { days, total } = findingFrequency(["r0"], [...recent, run("bogus", "1970-01-01T00:00:00Z")]);
    expect(total).toBe(251);
    expect(days).toHaveLength(30);
    expect(days.reduce((sum, d) => sum + d.affected + d.unaffected, 0)).toBe(250);
  });

  it("returns no days when nothing was sampled", () => {
    expect(findingFrequency(["a"], [])).toEqual({ affected: 0, total: 0, days: [] });
  });
});

describe("findingReach", () => {
  it("counts distinct sampled traces the finding hit, ignoring ids that were never sampled", () => {
    const runs = [run("a", "2026-10-01T00:00:00Z"), run("a", "2026-10-01T00:00:00Z"), run("b", "1970-01-01T00:00:00Z")];
    expect(findingReach(["a", "missing"], runs)).toEqual({ affected: 1, total: 2 });
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
