import { describe, expect, it } from "vitest";
import {
  aggregateModels,
  builderCostPerPr,
  builderInitials,
  builderVerdictDotClass,
  sortBuilders,
  teamTotals,
  type BuilderTotalsInput,
} from "./builderInsightsData";

const totalsBuilder = (overrides: Partial<BuilderTotalsInput>): BuilderTotalsInput => ({
  spend: 0,
  prs: 0,
  prsDevin: 0,
  ...overrides,
});

describe("teamTotals", () => {
  it("combines spend and PR totals and counts visible builders", () => {
    const totals = teamTotals([
      totalsBuilder({ spend: 200, prs: 10, prsDevin: 2 }),
      totalsBuilder({ spend: 100, prs: 5, prsDevin: 3 }),
      totalsBuilder({ spend: 50, prs: 1 }),
      totalsBuilder({ spend: 50 }),
    ]);
    const expectedTotals = { spend: 400, prs: 16, prsDevin: 5, builderCount: 4 };

    expect(totals).toEqual(expectedTotals);
  });

  it("returns zero totals for an empty team", () => {
    const emptyTotals = { spend: 0, prs: 0, prsDevin: 0, builderCount: 0 };
    expect(teamTotals([])).toEqual(emptyTotals);
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

describe("builderCostPerPr", () => {
  it("does not calculate a cost per PR when there are no merged PRs", () => {
    expect(builderCostPerPr({ spend: 100, prs: 0, spendPerPr: 25 })).toBeNull();
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
    expect(builderInitials("Alex")).toBe("Al");
    expect(builderInitials("Jordan Lee")).toBe("JL");
  });
});

describe("builderVerdictDotClass", () => {
  it("maps each verdict to its indicator color", () => {
    expect(builderVerdictDotClass("productive")).toBe("bg-emerald-500");
    expect(builderVerdictDotClass("mixed")).toBe("bg-amber-500");
    expect(builderVerdictDotClass("expensive")).toBe("bg-red-500");
    expect(builderVerdictDotClass("none")).toBe("bg-muted-foreground/40");
  });
});
