import { describe, expect, it } from "vitest";
import { counterLabel, formatCost, totalUnits, unpricedSummary } from "./usageUnits";

describe("formatCost", () => {
  it("renders a dash when nothing was priced", () => {
    expect(formatCost(null)).toBe("—");
    expect(formatCost(undefined)).toBe("—");
  });

  it("keeps an explicit zero as a real price rather than a dash", () => {
    expect(formatCost(0)).toBe("$0.0000");
  });

  it("shows four decimals for the sub-cent amounts guardrail units cost", () => {
    expect(formatCost(0.0003)).toBe("$0.0003");
    expect(formatCost(12.5)).toBe("$12.5000");
  });

  it("flags amounts below the displayed precision instead of rounding them to zero", () => {
    expect(formatCost(0.00001)).toBe("< $0.0001");
  });
});

describe("totalUnits", () => {
  it("sums every counter", () => {
    expect(totalUnits({ contentPolicyUnits: 3, sensitiveInformationPolicyUnits: 4 })).toBe(7);
  });

  it("is zero for no counters", () => {
    expect(totalUnits({})).toBe(0);
  });
});

describe("counterLabel", () => {
  it("turns a Bedrock counter name into words without the Units suffix", () => {
    expect(counterLabel("sensitiveInformationPolicyUnits")).toBe("Sensitive Information Policy");
    expect(counterLabel("contentPolicyUnits")).toBe("Content Policy");
  });

  it("leaves a name it cannot split alone apart from capitalising it", () => {
    expect(counterLabel("units")).toBe("Units");
  });
});

describe("unpricedSummary", () => {
  it("is null when every unit was priced", () => {
    expect(unpricedSummary({})).toBeNull();
    expect(unpricedSummary({ contentPolicyUnits: 0 })).toBeNull();
  });

  it("counts unpriced units across counters with a pluralised label", () => {
    expect(unpricedSummary({ contentPolicyUnits: 1200, someFutureCounter: 34 })).toBe("1,234 units unpriced");
    expect(unpricedSummary({ someFutureCounter: 1 })).toBe("1 unit unpriced");
  });
});
