import { describe, expect, it } from "vitest";
import {
  aggregateModels,
  builderInitials,
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
  it("shares 7-day agent spend across agents, counts builders at five percent, and ranks by spend", () => {
    expect(
      teamAgents([
        {
          spend: 100,
          agents: [
            { id: "codex", spend: 6, requests: 6, share: 0.06 },
            { id: "python", spend: 2, requests: 2, share: 0.02 },
            { id: "claude-code", spend: 4, requests: 3, share: 0.04 },
          ],
        },
        {
          spend: 200,
          agents: [
            { id: "codex", spend: 2, requests: 1, share: 0.05 },
          ],
        },
      ]),
    ).toEqual([
      { id: "codex", spend: 8, requests: 7, builders: 2, share: 8 / 14 },
      { id: "claude-code", spend: 4, requests: 3, builders: 0, share: 4 / 14 },
      { id: "python", spend: 2, requests: 2, builders: 0, share: 2 / 14 },
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

describe("aggregateModels", () => {
  it("merges leading provider prefixes while preserving spend and request totals", () => {
    expect(
      aggregateModels([
        { model: "gpt-6-astra", spend: 100, requests: 10 },
        { model: "openai/gpt-6-astra", spend: 40, requests: 4 },
        { model: "anthropic/claude-5", spend: 30, requests: 3 },
        { model: "bedrock/sonnet-5", spend: 20, requests: 2 },
        { model: "vertex_ai/gemini-4", spend: 15, requests: 1 },
        { model: "azure/gpt-6", spend: 10, requests: 1 },
        { model: "custom/openai/gpt-6-astra", spend: 5, requests: 1 },
      ]),
    ).toEqual([
      { model: "gpt-6-astra", spend: 140, requests: 14 },
      { model: "claude-5", spend: 30, requests: 3 },
      { model: "sonnet-5", spend: 20, requests: 2 },
      { model: "gemini-4", spend: 15, requests: 1 },
      { model: "gpt-6", spend: 10, requests: 1 },
      { model: "custom/openai/gpt-6-astra", spend: 5, requests: 1 },
    ]);
  });
});

describe("builderInitials", () => {
  it("uses initials for multiple name parts and a capitalized two-letter monogram otherwise", () => {
    expect(builderInitials("Tin")).toBe("Ti");
    expect(builderInitials("Moe Li")).toBe("ML");
  });
});
