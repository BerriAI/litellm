import { describe, expect, it } from "vitest";
import {
  counterLabel,
  counterMathLine,
  formatCost,
  formatUnitPrice,
  pricingIssueUrl,
  totalUnits,
  unitPrice,
  unitsSumLine,
  unpricedSummary,
} from "./usageUnits";

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
    expect(unitPrice({ counter: "contentPolicyUnits", units: 1200, unpriced: 200, cost: 0.15 })).toBeCloseTo(
      0.00015,
      10,
    );
  });

  it("is null when nothing was priced", () => {
    expect(unitPrice({ counter: "someFutureCounter", units: 7, unpriced: 7, cost: null })).toBeNull();
    expect(unitPrice({ counter: "someFutureCounter", units: 7, unpriced: 7, cost: 0 })).toBeNull();
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

describe("counterMathLine", () => {
  it("shows units × price = cost for a fully priced counter", () => {
    expect(counterMathLine({ counter: "contentPolicyUnits", units: 1000, unpriced: 0, cost: 0.15 })).toBe(
      "Content Policy: 1,000 × $0.00015 = $0.1500",
    );
  });

  it("prices only the priced share and calls out the rest", () => {
    expect(counterMathLine({ counter: "sensitiveInformationPolicyUnits", units: 8, unpriced: 2, cost: 0.0006 })).toBe(
      "Sensitive Information Policy: 6 × $0.0001 = $0.0006 (2 unpriced left out)",
    );
  });

  it("says so when a counter has no known price at all", () => {
    expect(counterMathLine({ counter: "someFutureCounter", units: 7, unpriced: 7, cost: null })).toBe(
      "Some Future Counter: 7 units with no known price, left out",
    );
    expect(counterMathLine({ counter: "someFutureCounter", units: 1, unpriced: 1, cost: null })).toBe(
      "Some Future Counter: 1 unit with no known price, left out",
    );
  });

  it("shows a free counter as × $0", () => {
    expect(counterMathLine({ counter: "wordPolicyUnits", units: 2, unpriced: 0, cost: 0 })).toBe(
      "Word Policy: 2 × $0 = $0.0000",
    );
  });
});

describe("unitsSumLine", () => {
  it("adds the counters up in order", () => {
    expect(unitsSumLine({ contentPolicyUnits: 2, topicPolicyUnits: 2, wordPolicyUnits: 1200 })).toBe(
      "Content Policy 2 + Topic Policy 2 + Word Policy 1,200 = 1,204",
    );
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
