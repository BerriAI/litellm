import { describe, expect, it } from "vitest";

import type { ProjectDailySpendRow } from "@/components/networking";

import { buildDailySpendSeries, buildProjectSpendBreakdown, summarizeProjectUsage } from "./projectUsageAggregations";

const row = (overrides: Partial<ProjectDailySpendRow> = {}): ProjectDailySpendRow => ({
  date: "2026-09-01",
  project_id: "project-alpha",
  project_alias: "Project Alpha",
  spend: 1,
  prompt_tokens: 10,
  completion_tokens: 5,
  total_tokens: 15,
  api_requests: 3,
  successful_requests: 2,
  failed_requests: 1,
  ...overrides,
});

describe("summarizeProjectUsage", () => {
  it("returns all-zero totals for no rows", () => {
    const zeroTotals = {
      total_spend: 0,
      total_api_requests: 0,
      total_successful_requests: 0,
      total_failed_requests: 0,
      total_tokens: 0,
    };

    expect(summarizeProjectUsage([])).toEqual(zeroTotals);
  });

  it("sums spend, requests, and tokens across every row regardless of project or date", () => {
    const projectBetaRow: Partial<ProjectDailySpendRow> = {
      project_id: "project-beta",
      date: "2026-09-02",
      spend: 2.25,
      api_requests: 4,
      successful_requests: 4,
      failed_requests: 0,
      total_tokens: 40,
    };
    const rows = [row({ spend: 1.5 }), row(projectBetaRow)];
    const expectedTotals = {
      total_spend: 3.75,
      total_api_requests: 7,
      total_successful_requests: 6,
      total_failed_requests: 1,
      total_tokens: 55,
    };

    expect(summarizeProjectUsage(rows)).toEqual(expectedTotals);
  });
});

describe("buildDailySpendSeries", () => {
  it("sums same-day spend across projects into a single point", () => {
    const rows = [row(), row({ project_id: "project-beta", spend: 2 })];

    expect(buildDailySpendSeries(rows)).toEqual([{ date: "2026-09-01", spend: 3 }]);
  });

  it("sorts distinct dates oldest first regardless of input order", () => {
    const rows = [row({ date: "2026-09-03", spend: 3 }), row(), row({ date: "2026-09-02", spend: 2 })];

    expect(buildDailySpendSeries(rows).map((point) => point.date)).toEqual(["2026-09-01", "2026-09-02", "2026-09-03"]);
  });
});

describe("buildProjectSpendBreakdown", () => {
  it("sums a project's spend and tokens across every day into a single row", () => {
    const secondDayRow: Partial<ProjectDailySpendRow> = {
      date: "2026-09-02",
      spend: 2,
      total_tokens: 20,
      successful_requests: 1,
      failed_requests: 2,
    };
    const rows = [row({ total_tokens: 10, api_requests: 2, failed_requests: 0 }), row(secondDayRow)];

    expect(buildProjectSpendBreakdown(rows)).toEqual([
      {
        project_id: "project-alpha",
        project_alias: "Project Alpha",
        spend: 3,
        requests: 5,
        successful_requests: 3,
        failed_requests: 2,
        tokens: 30,
      },
    ]);
  });

  it("sorts projects by spend descending", () => {
    const rows = [
      row({ project_id: "project-cheap", project_alias: "Cheap" }),
      row({ project_id: "project-costly", project_alias: "Costly", spend: 9 }),
    ];

    expect(buildProjectSpendBreakdown(rows).map((r) => r.project_id)).toEqual(["project-costly", "project-cheap"]);
  });

  it("falls back to the project id when the project has no alias", () => {
    const rows = [row({ project_id: "project-untitled", project_alias: null })];

    expect(buildProjectSpendBreakdown(rows)[0].project_alias).toBe("project-untitled");
  });

  it("disambiguates two projects that share the same human-set alias", () => {
    const rows = [
      row({ project_id: "project-one", project_alias: "Production" }),
      row({ project_id: "project-two", project_alias: "Production" }),
    ];

    const aliases = buildProjectSpendBreakdown(rows).map((r) => r.project_alias);
    expect(new Set(aliases).size).toBe(2);
    expect(aliases.every((alias) => alias.includes("Production"))).toBe(true);
  });

  it("leaves a unique alias untouched", () => {
    const rows = [row({ project_id: "project-alpha", project_alias: "Project Alpha" })];

    expect(buildProjectSpendBreakdown(rows)[0].project_alias).toBe("Project Alpha");
  });
});
