import { describe, expect, it } from "vitest";

import {
  coverageLabel,
  effortNote,
  estimateLabel,
  estimatorModelOptions,
  filterPulls,
  formatMoney,
  formatNumber,
  formatSyncedAt,
  highestCostPulls,
  peopleCsv,
} from "./roiCalculatorData";
import type { ROIPull } from "./roiCalculatorData";

const pull = (overrides: Partial<ROIPull>): ROIPull => ({
  source_repo: "github.com/org/repo",
  source_branch: "feature",
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
  it("ranks only attributed PR costs, limits the overview to five and preserves report order", () => {
    const pulls = [2, 6, 1, 4, 3, 5].map((spend) =>
      pull({
        number: spend,
        branch_cost: { status: "matched", spend, requests: 1, repo: "github.com/org/repo", branch: "feature" },
      }),
    );
    pulls.push(
      pull({
        number: 99,
        branch_cost: { status: "ambiguous", spend: 99, requests: 1, repo: "github.com/org/repo", branch: "feature" },
      }),
    );
    pulls.push(pull({ number: 100 }));
    expect(highestCostPulls(pulls).map((item) => item.number)).toEqual([6, 5, 4, 3, 2]);
    expect(pulls.map((item) => item.number)).toEqual([2, 6, 1, 4, 3, 5, 99, 100]);
  });
  it("formats spend and estimated hours without losing null values", () => {
    expect(formatMoney(1234.5)).toBe("$1,234.50");
    expect(formatMoney(0.0001)).toBe("$0.0001");
    expect(formatMoney(0.000186)).toBe("$0.000186");
    expect(formatMoney(0.0000001)).toBe("<$0.000001");
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
    expect(coverageLabel(summary)).toBe("1 of 2 matched");
  });

  it("labels estimates and filters PRs by title, repository, number, or login", () => {
    const matchingPull = pull({});
    expect(estimateLabel(matchingPull.estimate)).toBe("4.5 hrs");
    const incompleteEstimate = { status: "needs_review" as const, hours: null, reasoning: "", cached: false };
    expect(estimateLabel(incompleteEstimate)).toBe("Needs review");
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
        id: "export-person",
        estimated_prs: 1,
        match_methods: ["manual"],
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

it("recommends the real Luna model and preserves its gateway name for requests", () => {
  expect(
    estimatorModelOptions({
      available_models: ["general", "fast-estimator"],
      estimator_models: [
        { model_name: "general", provider_models: ["anthropic/claude-haiku"] },
        { model_name: "fast-estimator", provider_models: ["openai/gpt-6-luna"] },
      ],
    }),
  ).toEqual([
    {
      value: "fast-estimator",
      label: "GPT-6 Luna",
      sublabel: "Recommended · Gateway name: fast-estimator",
      recommended: true,
    },
    { value: "general", label: "anthropic/claude-haiku", sublabel: "Gateway name: general", recommended: false },
  ]);
});

it("does not invent available models or recommend an alias pointing to a different model", () => {
  expect(
    estimatorModelOptions({
      available_models: ["gpt-6-luna"],
      estimator_models: [{ model_name: "gpt-6-luna", provider_models: ["custom-model"] }],
    }),
  ).toEqual([{ value: "gpt-6-luna", label: "custom-model", sublabel: "Gateway name: gpt-6-luna", recommended: false }]);
  expect(estimatorModelOptions({ available_models: [], estimator_models: [] })).toEqual([]);
});
