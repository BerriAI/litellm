import { describe, expect, it } from "vitest";
import { issueComparisonText, issueHeadline, issueResource } from "./issueApi";
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
    ["spend", "spend", 30, "$30.00 on Nov 4, 2026 UTC, compared with the usual $0.00."],
    ["error_rate", "reliability", 8, "8% on Nov 4, 2026 UTC, compared with the usual 0%."],
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
    expect(issueHeadline(issue)).not.toMatch(/Infinity|NaN/);
  });
});

it("explains a complete traffic drop without dividing by zero", () => {
  const comparison = {
    ...SPEND_ISSUE.comparisons![0],
    metric: "requests" as const,
    direction: "decrease" as const,
    baseline_value: 100,
    observed_value: 0,
  };
  expect(issueHeadline({ ...SPEND_ISSUE, family: "traffic", comparisons: [comparison] })).toBe(
    "Requests to Checkout key fell 100%",
  );
});
