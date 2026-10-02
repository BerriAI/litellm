import { describe, expect, it } from "vitest";
import {
  counterLabel,
  counterMathRow,
  formatCost,
  formatUnitPrice,
  pricingIssueUrl,
  totalUnits,
  unitPrice,
  unitsMathRows,
  unpricedSummary,
  type CounterMath,
} from "./usageUnits";

const counterOf = (counter: string, units: number, unpriced: number, cost: number | null): CounterMath => ({
  counter,
  units,
  unpriced,
  cost,
});

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

describe("unitPrice", () => {
  it("backs the per-unit price out of the priced share only", () => {
    expect(unitPrice(counterOf("contentPolicyUnits", 1200, 200, 0.15))).toBeCloseTo(0.00015, 10);
  });

  it("is null when nothing was priced", () => {
    expect(unitPrice(counterOf("someFutureCounter", 7, 7, null))).toBeNull();
    expect(unitPrice(counterOf("someFutureCounter", 7, 7, 0))).toBeNull();
  });
});

describe("formatUnitPrice", () => {
  it("keeps the significant decimals and drops trailing zeros", () => {
    expect(formatUnitPrice(0.0001)).toBe("$0.0001");
    expect(formatUnitPrice(0.00015)).toBe("$0.00015");
    expect(formatUnitPrice(0)).toBe("$0");
    expect(formatUnitPrice(1)).toBe("$1");
  });

  it("never shows a positive price as free", () => {
    expect(formatUnitPrice(0.0000002)).toBe("< $0.000001");
  });
});

describe("counterMathRow", () => {
  it("shows units × price = cost for a fully priced counter", () => {
    expect(counterMathRow(counterOf("contentPolicyUnits", 1000, 0, 0.15))).toEqual({
      label: "Content Policy",
      parts: ["1,000", "× $0.00015", "= $0.1500"],
      note: null,
    });
  });

  it("prices only the priced share and calls out the rest", () => {
    expect(counterMathRow(counterOf("sensitiveInformationPolicyUnits", 8, 2, 0.0006))).toEqual({
      label: "Sensitive Information Policy",
      parts: ["6", "× $0.0001", "= $0.0006"],
      note: "2 unpriced units left out",
    });
    expect(counterMathRow(counterOf("sensitiveInformationPolicyUnits", 8, 1, 0.0007)).note).toBe(
      "1 unpriced unit left out",
    );
  });

  it("says so when a counter has no known price at all", () => {
    expect(counterMathRow(counterOf("someFutureCounter", 7, 7, null))).toEqual({
      label: "Some Future Counter",
      parts: ["7", "× —", "= —"],
      note: "no known price, left out",
    });
  });

  it("shows a free counter as × $0", () => {
    expect(counterMathRow(counterOf("wordPolicyUnits", 2, 0, 0)).parts).toEqual(["2", "× $0", "= $0.0000"]);
  });
});

describe("unitsMathRows", () => {
  it("lists the counters in order with their counts", () => {
    expect(unitsMathRows({ contentPolicyUnits: 2, wordPolicyUnits: 1200 })).toEqual([
      { label: "Content Policy", parts: ["2"], note: null },
      { label: "Word Policy", parts: ["1,200"], note: null },
    ]);
  });
});

describe("pricingIssueUrl", () => {
  it("prefills the feature request with the provider and the unpriced counters", () => {
    const url = new URL(pricingIssueUrl({ text_records: 5, someFutureCounter: 7 }, "azure/prompt_shield"));

    expect(url.origin + url.pathname).toBe("https://github.com/BerriAI/litellm/issues/new");
    expect(url.searchParams.get("template")).toBe("feature_request.yml");
    expect(url.searchParams.get("title")).toBe("[Feature]: add azure/prompt_shield guardrail pricing to the cost map");
    expect(url.searchParams.get("the-feature")).toContain("text_records, someFutureCounter");
  });

  it("stays generic when no provider is known", () => {
    const url = new URL(pricingIssueUrl({ text_records: 5 }));

    expect(url.searchParams.get("title")).toBe("[Feature]: add guardrail pricing to the cost map");
  });
});
