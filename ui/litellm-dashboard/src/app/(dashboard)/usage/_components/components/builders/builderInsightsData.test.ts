import { describe, expect, it } from "vitest";
import {
  dailySeries,
  sortBuilders,
  teamAgents,
  teamTotals,
  type BuilderInsightBuilder,
  type BuilderTotalsInput,
} from "./builderInsightsData";

const totalsBuilder = (
  overrides: Partial<BuilderTotalsInput & Pick<BuilderInsightBuilder, "prs">>,
): BuilderTotalsInput & Pick<BuilderInsightBuilder, "prs"> => ({
  spend: 0,
  prs: 0,
  prsDevin: 0,
  spendPerPr: null,
  tokens: 0,
  cacheHitRate: 0,
  ...overrides,
});

describe("teamTotals", () => {
  it("combines spend and PR totals, median builder cost, top-three share, and token-weighted cache hit rate", () => {
    const totals = teamTotals([
      totalsBuilder({ spend: 200, prs: 10, prsDevin: 2, spendPerPr: 20, tokens: 1000, cacheHitRate: 0.9 }),
      totalsBuilder({ spend: 100, prs: 5, prsDevin: 3, spendPerPr: 20, tokens: 3000, cacheHitRate: 0.8 }),
      totalsBuilder({ spend: 50, prs: 1, spendPerPr: 50 }),
      totalsBuilder({ spend: 50 }),
    ]);

    expect(totals).toEqual({
      spend: 400,
      prs: 16,
      prsDevin: 5,
      builderCount: 4,
      medianSpendPerPr: 20,
      top3Share: 0.875,
      cacheHitRate: 0.825,
    });
  });

  it("returns null efficiency and zero shares for an empty team", () => {
    expect(teamTotals([])).toEqual({
      spend: 0,
      prs: 0,
      prsDevin: 0,
      builderCount: 0,
      medianSpendPerPr: null,
      top3Share: 0,
      cacheHitRate: 0,
    });
  });
});

describe("teamAgents", () => {
  it("sums spend and requests, counts builders at the five-percent threshold, and ranks by spend", () => {
    expect(
      teamAgents([
        {
          spend: 100,
          agents: [
            { id: "codex", spend: 60, requests: 6, share: 0.6 },
            { id: "python", spend: 4, requests: 2, share: 0.04 },
            { id: "claude-code", spend: 36, requests: 3, share: 0.36 },
          ],
        },
        {
          spend: 200,
          agents: [
            { id: "codex", spend: 10, requests: 1, share: 0.05 },
            { id: "python", spend: 190, requests: 20, share: 0.95 },
          ],
        },
      ]),
    ).toEqual([
      { id: "python", spend: 194, requests: 22, builders: 1, share: 194 / 300 },
      { id: "codex", spend: 70, requests: 7, builders: 2, share: 70 / 300 },
      { id: "claude-code", spend: 36, requests: 3, builders: 1, share: 0.12 },
    ]);
  });
});

describe("sortBuilders", () => {
  const builders = [
    { id: "high", spend: 300, prs: 3, spendPerPr: 100 },
    { id: "low", spend: 100, prs: 8, spendPerPr: 12.5 },
    { id: "no-prs", spend: 50, prs: 0, spendPerPr: null },
  ] as const;

  it("sorts by spend, PR count, and efficiency with null efficiency last", () => {
    expect(sortBuilders(builders, "spend").map((builder) => builder.id)).toEqual(["high", "low", "no-prs"]);
    expect(sortBuilders(builders, "prs").map((builder) => builder.id)).toEqual(["low", "high", "no-prs"]);
    expect(sortBuilders(builders, "efficiency").map((builder) => builder.id)).toEqual(["low", "high", "no-prs"]);
  });
});

describe("dailySeries", () => {
  it("sorts daily spend in date order and returns chart labels and values", () => {
    expect(
      dailySeries({
        daily: [
          { date: "2026-10-02", spend: 20, requests: 2 },
          { date: "2026-10-01", spend: 10, requests: 1 },
        ],
      }),
    ).toEqual([
      { date: "2026-10-01", label: "Oct 1", spend: 10, requests: 1 },
      { date: "2026-10-02", label: "Oct 2", spend: 20, requests: 2 },
    ]);
  });
});
