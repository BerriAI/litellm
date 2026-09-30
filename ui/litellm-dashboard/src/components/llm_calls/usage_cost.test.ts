import { describe, expect, it } from "vitest";
import { parseUsageCost } from "./usage_cost";

describe("parseUsageCost", () => {
  it("keeps finite numbers, including zero", () => {
    expect(parseUsageCost(0)).toBe(0);
    expect(parseUsageCost(0.000063)).toBe(0.000063);
  });

  it("keeps numeric strings", () => {
    expect(parseUsageCost("0.00019")).toBe(0.00019);
    expect(parseUsageCost(" 0.00019 ")).toBe(0.00019);
  });

  it("drops blank strings instead of fabricating a zero cost", () => {
    expect(parseUsageCost("")).toBeUndefined();
    expect(parseUsageCost("   ")).toBeUndefined();
    expect(parseUsageCost("\t\n")).toBeUndefined();
  });

  it("drops strings with a numeric prefix instead of truncating them", () => {
    expect(parseUsageCost("1oops")).toBeUndefined();
    expect(parseUsageCost("0.5 USD")).toBeUndefined();
  });

  it("drops non-finite numbers", () => {
    expect(parseUsageCost(Number.NaN)).toBeUndefined();
    expect(parseUsageCost(Number.POSITIVE_INFINITY)).toBeUndefined();
  });

  it("drops values that are not numbers or strings", () => {
    expect(parseUsageCost(null)).toBeUndefined();
    expect(parseUsageCost(undefined)).toBeUndefined();
    expect(parseUsageCost(true)).toBeUndefined();
    expect(parseUsageCost([])).toBeUndefined();
    expect(parseUsageCost(["0.5"])).toBeUndefined();
    expect(parseUsageCost({ total_cost: 0.5 })).toBeUndefined();
  });
});
