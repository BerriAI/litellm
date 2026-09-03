import { describe, expect, it } from "vitest";
import { buildCostBreakdownTiles, buildSummaryTiles, hasFlatCost } from "./entityUsageSummary";

const metadata = {
  total_spend: 100,
  total_flat_cost: 40,
  total_api_requests: 12,
  total_successful_requests: 10,
  total_failed_requests: 2,
  total_tokens: 3456,
};

describe("hasFlatCost", () => {
  it("is false when there is no flat cost to report", () => {
    expect(hasFlatCost({ ...metadata, total_flat_cost: 0 })).toBe(false);
    const { total_flat_cost, ...noFlat } = metadata;
    expect(hasFlatCost(noFlat)).toBe(false);
  });

  it("is true once a flat cost has accrued", () => {
    expect(hasFlatCost(metadata)).toBe(true);
  });
});

describe("buildSummaryTiles", () => {
  it("keeps the row at five tiles either way so adding flat cost never narrows the cards", () => {
    expect(buildSummaryTiles(metadata, false)).toHaveLength(5);
    expect(buildSummaryTiles(metadata, true)).toHaveLength(5);
  });

  it("shows request-only spend under the original title when there is no flat cost", () => {
    const [first] = buildSummaryTiles(metadata, false);
    expect(first.title).toBe("Total Spend");
    expect(first.value).toBe("$100.00");
    expect(first.expandable).toBeUndefined();
  });

  it("rolls flat cost into a single expandable Total Cost tile", () => {
    const [first] = buildSummaryTiles(metadata, true);
    expect(first.title).toBe("Total Cost");
    expect(first.value).toBe("$140.00");
    expect(first.expandable).toBe(true);
    expect(first.tooltip).toBeTruthy();
  });

  it("never renders the breakdown titles in the top row", () => {
    const titles = buildSummaryTiles(metadata, true).map((t) => t.title);
    expect(titles).not.toContain("Flat Cost");
    expect(titles).not.toContain("Request Cost");
  });

  it("treats a missing flat cost as zero", () => {
    const { total_flat_cost, ...noFlat } = metadata;
    expect(buildSummaryTiles(noFlat, true)[0].value).toBe("$100.00");
  });
});

describe("buildCostBreakdownTiles", () => {
  it("splits the total into request cost and flat cost", () => {
    const byTitle = Object.fromEntries(buildCostBreakdownTiles(metadata).map((t) => [t.title, t.value]));
    expect(byTitle["Request Cost"]).toBe("$100.00");
    expect(byTitle["Flat Cost"]).toBe("$40.00");
  });

  it("adds up to the Total Cost tile so the expanded view reconciles", () => {
    const parse = (v: string) => Number(v.replace(/[$,]/g, ""));
    const parts = buildCostBreakdownTiles(metadata).map((t) => parse(t.value));
    expect(parts[0] + parts[1]).toBe(parse(buildSummaryTiles(metadata, true)[0].value));
  });

  it("explains each part, including that flat cost is outside budgets", () => {
    const byTitle = Object.fromEntries(buildCostBreakdownTiles(metadata).map((t) => [t.title, t.tooltip]));
    expect(byTitle["Request Cost"]).toBeTruthy();
    expect(byTitle["Flat Cost"]).toContain("budget");
  });

  it("treats a missing flat cost as zero", () => {
    const { total_flat_cost, ...noFlat } = metadata;
    const byTitle = Object.fromEntries(buildCostBreakdownTiles(noFlat).map((t) => [t.title, t.value]));
    expect(byTitle["Flat Cost"]).toBe("$0.00");
  });
});
