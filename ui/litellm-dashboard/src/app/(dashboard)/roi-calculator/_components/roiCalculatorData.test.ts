import { describe, expect, it } from "vitest";

import {
  coverageLabel,
  effortNote,
  estimateLabel,
  filterPulls,
  formatMoney,
  formatNumber,
  formatSyncedAt,
  peopleCsv,
} from "./roiCalculatorData";
import type { ROIPull } from "./roiCalculatorData";

const pull = (overrides: Partial<ROIPull>): ROIPull => ({
  repo: "org/repo",
  number: 42,
  title: "Improve request routing",
  url: "https://github.com/org/repo/pull/42",
  login: "alice",
  emails: ["alice@example.com"],
  profile_email: "alice@example.com",
  merged_at: "2026-09-12T00:00:00Z",
  head_sha: "abc",
  additions: 10,
  deletions: 2,
  changed_files: 1,
  commit_count: 1,
  incomplete_metadata: false,
  estimate: { status: "estimated", hours: 4.5, reasoning: "Metadata-based estimate.", cached: false },
  email: "alice@example.com",
  match_method: "profile email",
  matched: true,
  ...overrides,
});

const summary = {
  metrics: { matched_prs: 1, merged_prs: 2 },
};

describe("ROI calculator display helpers", () => {
  it("formats spend and estimated hours without losing null values", () => {
    expect(formatMoney(1234.5)).toBe("$1,234.50");
    expect(formatMoney(0.0001)).toBe("<$0.01");
    expect(formatMoney(0)).toBe("$0.00");
    expect(formatMoney(null)).toBe("—");
    expect(formatNumber(4.25)).toBe("4.3");
    expect(formatNumber(null)).toBe("—");
  });

  it("formats report sync timestamps in UTC", () => {
    expect(formatSyncedAt("2026-09-30T12:00:00Z")).toBe("Sep 30, 2026, 12:00 PM UTC");
    expect(formatSyncedAt("invalid")).toBe("invalid");
  });

  it("keeps caveat copy tied to the estimate basis and reports match coverage", () => {
    expect(effortNote("without_ai")).toContain("not actual hours worked or hours saved");
    expect(effortNote(null)).toContain("Earlier estimates");
    expect(coverageLabel(summary)).toBe("1 of 2 PRs have email matches");
  });

  it("labels estimates and filters PRs by title, repository, number, or login", () => {
    const matchingPull = pull({});
    expect(estimateLabel(matchingPull.estimate)).toBe("4.5 hrs");
    expect(estimateLabel({ status: "needs_review", hours: null, reasoning: "", cached: false })).toBe("Needs review");
    expect(filterPulls([matchingPull], "ROUTING")).toEqual([matchingPull]);
    expect(filterPulls([matchingPull], "nobody")).toEqual([]);
  });
});

it("exports precise spend, cohort eligibility and safely quoted CSV values", () => {
  const exportSummary = {
    start: "2026-09-01",
    end: "2026-09-30",
    effort_basis: "without_ai",
    people: [
      {
        email: '=HYPERLINK("bad")',
        logins: ["alice", "bob"],
        spend: 0.0001,
        hours: 4,
        prs: 1,
        pending_prs: 0,
        eligible: true,
        cost_per_hour: 0.000025,
      },
    ],
  };
  const csv = peopleCsv(exportSummary);
  expect(csv.split("\r\n")).toHaveLength(2);
  expect(csv).toContain('"\'=HYPERLINK(""bad"")","alice;bob","0.0001","4","1","0","true","0.000025"');
  expect(csv).toContain('"2026-09-01","2026-09-30","without_ai"');
});
