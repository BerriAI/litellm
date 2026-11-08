import { describe, expect, it } from "vitest";
import { issueHistory } from "./IssueHistoryChart";
import { SPEND_ISSUE } from "./issueFixtures";

describe("saved alert history", () => {
  it("keeps missing days as gaps and real zero observations as zero", () => {
    const issue = {
      ...SPEND_ISSUE,
      comparisons: SPEND_ISSUE.comparisons?.map((comparison) => ({
        ...comparison,
        history: [
          { day: comparison.baseline_start, value: 30 },
          { day: SPEND_ISSUE.observed_on, value: 0 },
        ],
      })),
    };
    const { data, metric } = issueHistory(issue);
    expect(metric).toBe("spend");
    expect(data).toHaveLength(8);
    expect(data[0]).toEqual({ day: "2026-10-28", value: 30 });
    expect(data[1]).toEqual({ day: "2026-10-29", value: null });
    expect(data[7]).toEqual({ day: "2026-11-04", value: 0 });
  });
});
