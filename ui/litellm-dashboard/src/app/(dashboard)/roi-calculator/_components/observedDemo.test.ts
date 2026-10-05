import { describe, expect, it } from "vitest";
import { observedSnapshotSchema, weeklyMerges, type Period } from "./observedData";
import { createObservedDemo } from "./observedDemo";

describe("observed sample report", () => {
  it("clamps the year-ago comparison on leap day while keeping the selected length", () => {
    const report = createObservedDemo(28, new Date("2024-03-01T12:00:00Z"));
    expect(report.periods.current.window.end).toBe("2024-02-29");
    expect(report.periods.last_year.window).toEqual({ start: "2023-02-01", end: "2023-02-28" });
  });

  it.each([7, 28, 90])("keeps totals, attribution, costs, and comparison windows consistent for %i days", (days) => {
    const report = createObservedDemo(days, new Date("2026-10-03T14:00:00Z"));
    expect(observedSnapshotSchema.safeParse(report).success).toBe(true);
    for (const period of ["current", "previous", "last_year"] satisfies Period[]) {
      const metrics = report.periods[period];
      const pulls = report.pulls[period];
      const people = report.people.map((person) => person.periods[period]);
      const start = Date.parse(metrics.window.start);
      const end = Date.parse(metrics.window.end) + 86_400_000;
      expect((end - start) / 86_400_000).toBe(days);
      expect(metrics.merged_prs).toBe(pulls.length);
      expect(new Set(pulls.map((pull) => pull.url)).size).toBe(pulls.length);
      expect(weeklyMerges(report, period).reduce((sum, value) => sum + value, 0)).toBe(pulls.length);
      expect(pulls.every((pull) => Date.parse(pull.merged_at) >= start && Date.parse(pull.merged_at) < end)).toBe(true);
      expect(metrics.matched_users_recorded_spend).toBe(
        people.reduce((sum, person) => sum + person.gateway_recorded_spend, 0),
      );
      expect(people.reduce((sum, person) => sum + person.merged_prs, 0)).toBe(metrics.matched_internal_prs);
      for (const person of people) {
        const attributed = pulls.filter((pull) => person.pr_urls.includes(pull.url));
        expect(attributed).toHaveLength(person.merged_prs);
        expect(person.direct_authored).toBe(attributed.filter((pull) => !pull.agent).length);
        expect(person.declared_agent_owned).toBe(attributed.filter((pull) => pull.agent).length);
        expect(person.recorded_spend_per_attributed_pr).toBe(person.gateway_recorded_spend / person.merged_prs);
      }
    }
    expect(Date.parse(report.periods.previous.window.end) + 86_400_000).toBe(
      Date.parse(report.periods.current.window.start),
    );
  });
});
