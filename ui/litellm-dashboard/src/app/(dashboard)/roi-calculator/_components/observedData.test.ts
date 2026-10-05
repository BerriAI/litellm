import { describe, expect, it, vi } from "vitest";
import {
  change,
  duration,
  money,
  visiblePeople,
  weeklyMerges,
  filterObservedPulls,
  reportPeople,
  recordedBranches,
  type ObservedPerson,
  type ObservedSnapshot,
} from "./observedData";
import { createObservedDemo } from "./observedDemo";

const period = (merged: number, spend: number | null) => ({
  merged_prs: merged,
  prs_per_week: merged / 4,
  median_merge_hours: null,
  direct_authored: merged,
  declared_agent_owned: 0,
  gateway_recorded_spend: spend ?? 0,
  recorded_spend_per_attributed_pr: spend !== null && merged > 0 ? spend / merged : null,
  spend_observation: spend === null ? ("no_records" as const) : ("records_present" as const),
  pr_urls: [],
});
const person = (name: string, merged: number, spend: number | null): ObservedPerson => ({
  name,
  email: `${name}@example.test`,
  logins: [`old-${name}`],
  periods: { current: period(merged, spend), previous: period(0, null), last_year: period(0, null) },
});

describe("observed ROI metrics", () => {
  it("shows the same contributors when the browser has no Map.groupBy", () => {
    const sample = { ...createObservedDemo(7), people: [] };
    const expected = reportPeople(sample, false);
    const legacyMap = new Proxy(Map, {
      get: (target, key, receiver) => (key === "groupBy" ? undefined : Reflect.get(target, key, receiver)),
    });
    vi.stubGlobal("Map", legacyMap);
    try {
      expect(reportPeople(sample, false)).toEqual(expected);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("filters by attributed changes across providers, independently of spend or author names", () => {
    const sample = createObservedDemo(7);
    const external = {
      ...sample.pulls.current[0],
      url: "https://gitlab.com/demo/api/-/merge_requests/999",
      author: "alex-demo",
      agent: false,
      connection_id: "demo-gitlab",
    };
    const report = { ...sample, pulls: { ...sample.pulls, current: [...sample.pulls.current, external] } };
    const noSpend = {
      ...report,
      people: report.people.map((row) => ({
        ...row,
        periods: {
          ...row.periods,
          current: { ...row.periods.current, gateway_recorded_spend: 0, spend_observation: "no_records" as const },
        },
      })),
    };
    expect(filterObservedPulls(noSpend, "current", true)).toEqual(sample.pulls.current);
    expect(filterObservedPulls(noSpend, "current", false)).toEqual(report.pulls.current);
    expect(filterObservedPulls(noSpend, "current", true).some((pull) => pull.agent)).toBe(true);
    expect(reportPeople(noSpend, true).map((row) => row.email)).toEqual(sample.people.map((row) => row.email));
  });

  it("keeps unmatched identities separate by host, with unknown spend and accurate periods", () => {
    const sample = createObservedDemo(7);
    const external = {
      ...sample.pulls.current[0],
      author: "contributor",
      agent: false,
      connection_id: "gitlab-public",
      url: "https://gitlab.com/demo/api/-/merge_requests/999",
      merge_hours: 4,
    };
    const otherHost = {
      ...external,
      connection_id: "gitlab-private",
      url: "https://git.example.test/demo/api/-/merge_requests/999",
      merge_hours: null,
    };
    const requested = {
      ...external,
      url: "https://gitlab.com/demo/api/-/merge_requests/1000",
      author: "devin-ai",
      agent: true,
      requester: "contributor",
      merge_hours: 8,
    };
    const unassigned = { ...requested, url: "https://gitlab.com/demo/api/-/merge_requests/1001", requester: "" };
    const report = {
      ...sample,
      pulls: {
        ...sample.pulls,
        current: [...sample.pulls.current, external, otherHost, requested, unassigned],
        previous: [external],
      },
    };
    const outsiders = reportPeople(report, false).filter((row) => !row.matched);
    expect(outsiders).toHaveLength(2);
    expect(new Set(outsiders.map((row) => row.id)).size).toBe(2);
    const publicPerson = outsiders.find((row) => row.host === "gitlab.com")!;
    const expectedCurrent = {
      merged_prs: 2,
      prs_per_week: 2,
      median_merge_hours: 6,
      direct_authored: 1,
      declared_agent_owned: 1,
      spend_observation: "no_records",
      recorded_spend_per_attributed_pr: null,
    };
    expect(publicPerson.periods.current).toMatchObject(expectedCurrent);
    expect(publicPerson.periods.previous.merged_prs).toBe(1);
    expect(outsiders.find((row) => row.host === "git.example.test")!.periods.current.median_merge_hours).toBeNull();
    expect(filterObservedPulls(report, "current", false)).toContain(unassigned);
  });

  it("filters branch spend using matched change ownership and the full repository and branch key", () => {
    const sample = createObservedDemo(7);
    const own = sample.pulls.current[0];
    const externalCost = { ...own.branch_cost, repo: "gitlab.com/outside/service", spend: 99 };
    const external = {
      ...own,
      url: "https://gitlab.com/outside/service/-/merge_requests/999",
      branch_cost: externalCost,
    };
    const shared = { repo: own.branch_cost.repo, branch: "shared", spend: 12, requests: 4 };
    const ambiguous = {
      ...own,
      source_branch: shared.branch,
      branch_cost: { ...shared, status: "ambiguous" as const },
    };
    const report = {
      ...sample,
      pulls: { ...sample.pulls, current: [ambiguous, external] },
      unlinked_branches: [shared, { ...shared, branch: "unowned" }],
    };
    expect(recordedBranches(report, true)).toEqual([shared]);
    expect(recordedBranches(report, false).map((row) => row.branch)).toEqual([
      externalCost.branch,
      "shared",
      "unowned",
    ]);
  });

  it("does not claim infinite growth when the baseline is missing", () => {
    expect(change(12, 0)).toBeNull();
    expect(change(0, 0)).toBeNull();
    expect(change(15, 10)).toBe(50);
    expect(change(0, 10)).toBe(-100);
  });

  it("distinguishes missing cost, measured zero, and small nonzero spend", () => {
    expect(money(null)).toBe("Unavailable");
    expect(money(0)).toBe("$0.00");
    expect(money(0.001)).toBe("<$0.01");
    expect(money(1.235)).toBe("$1.24");
    expect(duration(null)).toBe("Unavailable");
  });

  it("searches historical identities and sorts without changing the source", () => {
    const people = [person("Ari", 2, 8), person("Bea", 10, 0), person("Cam", 0, null)];
    expect(visiblePeople(people, " OLD-ARI ", "merged").map((row) => row.name)).toEqual(["Ari"]);
    expect(visiblePeople(people, "", "merged").map((row) => row.name)).toEqual(["Bea", "Ari", "Cam"]);
    expect(visiblePeople(people, "", "cost").map((row) => row.name)).toEqual(["Ari", "Bea", "Cam"]);
    expect(people.map((row) => row.name)).toEqual(["Ari", "Bea", "Cam"]);
    expect(visiblePeople(people, "missing", "name")).toEqual([]);
  });

  it("keeps short elapsed merge times from rounding to zero hours", () => {
    expect(duration(null)).toBe("Unavailable");
    expect(duration(0)).toBe("0m");
    expect(duration(16 / 3600)).toBe("<1m");
    expect(duration(59 / 3600)).toBe("<1m");
    expect(duration(1 / 60)).toBe("1m");
    expect(duration(79 / 3600)).toBe("1.3m");
    expect(duration(140 / 3600)).toBe("2.3m");
    expect(duration(0.5)).toBe("30m");
    expect(duration(1)).toBe("1h");
    expect(duration(3.82)).toBe("3.8h");
  });

  it.each([7, 28, 90])("includes every day of a %s-day range in its weekly chart", (days) => {
    const start = Date.parse("2026-01-01T00:00:00Z");
    const atDay = (day: number) => new Date(start + day * 86400000).toISOString();
    const pulls = Array.from({ length: days + 1 }, (_, day) => ({ merged_at: atDay(day) }));
    const snapshot = {
      periods: { current: { window: { start: "2026-01-01", end: atDay(days - 1).slice(0, 10) } } },
      pulls: { current: pulls },
    } as ObservedSnapshot;
    const weeks = weeklyMerges(snapshot, "current");
    expect(weeks).toHaveLength(Math.ceil(days / 7));
    expect(weeks.reduce((sum, count) => sum + count, 0)).toBe(days);
    expect(weeks.at(-1)).toBe(days % 7 || 7);
  });

  it("aligns comparisons to their own UTC windows and counts each boundary once", () => {
    const currentStart = "2026-01-01";
    const previousStart = "2025-12-04";
    const pulls = [
      "2026-01-01T00:00:00Z",
      "2026-01-07T23:59:59Z",
      "2026-01-08T00:00:00Z",
      "2026-01-28T23:59:59Z",
      "2026-01-29T00:00:00Z",
    ].map((merged_at, number) => ({
      number,
      merged_at,
      title: "Example",
      url: "https://github.com/example/repo/pull/1",
      author: "ari",
      agent: false,
      merge_hours: 1,
    }));
    const snapshot = {
      periods: {
        current: { window: { start: currentStart, end: "2026-01-28" } },
        previous: { window: { start: previousStart, end: "2025-12-31" } },
      },
      pulls: { current: pulls, previous: [{ ...pulls[0], merged_at: `${previousStart}T00:00:00Z` }] },
    } as ObservedSnapshot;
    expect(weeklyMerges(snapshot, "current")).toEqual([2, 1, 0, 1]);
    expect(weeklyMerges(snapshot, "previous")).toEqual([1, 0, 0, 0]);
  });
});
