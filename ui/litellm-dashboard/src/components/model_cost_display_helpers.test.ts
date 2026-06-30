import { describe, expect, it } from "vitest";
import { formatModelCostEstimate } from "./model_cost_display_helpers";

describe("formatModelCostEstimate", () => {
  it("formats a known model cost per million tokens", () => {
    expect(formatModelCostEstimate(0.00003)).toBe("30.0000");
  });

  it("returns Estimate unavailable when the cost is null", () => {
    expect(formatModelCostEstimate(null)).toBe("Estimate unavailable");
  });

  it("returns Estimate unavailable when the cost is missing", () => {
    expect(formatModelCostEstimate(undefined)).toBe("Estimate unavailable");
  });

  it.each(["disabled", "gated"] as const)("returns Cost pending for %s costs", (status) => {
    expect(formatModelCostEstimate(0.00003, { status })).toBe("Cost pending");
  });
});
