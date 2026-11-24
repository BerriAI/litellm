import { describe, expect, it } from "vitest";
import { customerSurchargeBreakdown } from "./CustomerSurchargeBreakdown";

describe("customerSurchargeBreakdown", () => {
  it("splits spend into base cost and surcharge", () => {
    const breakdown = customerSurchargeBreakdown(10.2, 0.2);
    expect(breakdown?.surcharge).toBe(0.2);
    expect(breakdown?.baseCost).toBeCloseTo(10);
  });

  it.each([0, null, undefined, -1])("shows nothing when the surcharge is %s", (surcharge) => {
    expect(customerSurchargeBreakdown(10, surcharge)).toBeNull();
  });

  it("never reports a surcharge larger than the spend", () => {
    expect(customerSurchargeBreakdown(1, 5)).toEqual({ baseCost: 0, surcharge: 1 });
  });
});
