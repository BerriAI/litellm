import { describe, expect, it } from "vitest";
import { changeBetween, statComparison } from "./usageComparison";

const previousRange = { start: "2026-10-01", end: "2026-10-12" };
const metadata = (spend: number, tokens: number, requests: number) => ({
  total_spend: spend,
  total_tokens: tokens,
  total_api_requests: requests,
});

describe("changeBetween", () => {
  it("reads a zero baseline as New or unchanged, never an infinite percentage", () => {
    expect(changeBetween(5, 0)).toEqual({ kind: "new" });
    expect(changeBetween(0, 0)).toEqual({ kind: "same" });
    expect(changeBetween(0, 4)).toEqual({ kind: "change", percent: -100 });
    expect(changeBetween(15, 10)).toEqual({ kind: "change", percent: 50 });
  });

  it("refuses to compare a value that is not a finite number", () => {
    expect(changeBetween(Number.NaN, 1)).toEqual({ kind: "unavailable" });
    expect(changeBetween(1, Number.POSITIVE_INFINITY)).toEqual({ kind: "unavailable" });
  });
});

describe("statComparison", () => {
  const current = metadata(30, 3000, 300);

  it("compares spend, tokens, requests and cost per request from the same aggregate on both sides", () => {
    const comparison = statComparison({
      current,
      previous: metadata(20, 0, 100),
      previousRange,
      status: "ready",
      requestsComparable: true,
    });
    expect(comparison.spend).toEqual({ kind: "change", percent: 50 });
    expect(comparison.tokens).toEqual({ kind: "new" });
    expect(comparison.requests).toEqual({ kind: "change", percent: 200 });
    expect(comparison.avgCostPerRequest).toEqual({ kind: "change", percent: -50 });
    expect(comparison.previous).toBe(previousRange);
  });

  it("leaves requests uncompared when the tile shows gateway counts the aggregate cannot match", () => {
    const comparison = statComparison({
      current,
      previous: metadata(20, 10, 100),
      previousRange,
      status: "ready",
      requestsComparable: false,
    });
    expect(comparison.requests).toBeNull();
  });

  it("has no cost-per-request change when the previous period had no requests", () => {
    const comparison = statComparison({
      current,
      previous: metadata(0, 0, 0),
      previousRange,
      status: "ready",
      requestsComparable: true,
    });
    expect(comparison.avgCostPerRequest).toEqual({ kind: "unavailable" });
    expect(comparison.spend).toEqual({ kind: "new" });
  });

  it.each([
    ["loading", "pending"],
    ["failed", "unavailable"],
  ] as const)("marks every change %s as %s instead of comparing against zeros", (status, kind) => {
    const comparison = statComparison({ current, previous: null, previousRange, status, requestsComparable: true });
    expect([comparison.spend, comparison.tokens, comparison.requests, comparison.avgCostPerRequest]).toEqual(
      Array(4).fill({ kind }),
    );
  });
});
