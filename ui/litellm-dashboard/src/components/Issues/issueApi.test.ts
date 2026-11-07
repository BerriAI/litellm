import { describe, expect, it } from "vitest";
import { issueComparisonText, issueResource } from "./issueApi";
import { SPEND_ISSUE } from "./issueFixtures";

describe("issue comparisons", () => {
  it("labels gateway issues without repeating the scope", () => {
    expect(
      issueResource({
        ...SPEND_ISSUE,
        scope_label: "Gateway",
        cohort: { scope_type: "deployment", scope_id: null, model_pairs: [] },
      }),
    ).toBe("Gateway");
  });

  it.each([
    ["spend", "spend", 30, "$30.00 vs $0.00"],
    ["error_rate", "reliability", 8, "8% vs 0%"],
  ] as const)("shows %s from a zero baseline without an infinite ratio", (metric, family, value, expected) => {
    const comparison = {
      metric,
      direction: "increase" as const,
      observed_value: value,
      baseline_value: 0,
      threshold: 2,
      baseline_start: "2026-10-28",
      baseline_end: "2026-11-03",
    };
    const issue = { ...SPEND_ISSUE, family, comparisons: [comparison] };
    expect(issueComparisonText(issue)).toBe(expected);
  });
});
