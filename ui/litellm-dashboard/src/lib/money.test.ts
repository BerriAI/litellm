import { describe, expect, it } from "vitest";
import { formatMoney, formatUsd, fromUsd, toUsd, USD } from "./money";

const eur = { code: "EUR", usd_rate: 0.92 };

describe("money", () => {
  it("converts in both directions while retaining USD values", () => {
    expect(fromUsd(100, eur)).toBe(92);
    expect(toUsd(92, eur)).toBe(100);
    expect(fromUsd(toUsd(100, eur), eur)).toBeCloseTo(100);
    expect(toUsd(10, USD)).toBe(10);
  });

  it("formats currencies, precision, abbreviation and small values", () => {
    expect(formatMoney(100, eur)).toBe("€92.00");
    expect(formatUsd(1234.5, { decimals: 1 })).toBe("$1,234.5");
    expect(formatMoney(2000, eur, { abbreviate: true, decimals: 1 })).toBe("€1.8K");
    expect(formatUsd(1_200_000, { abbreviate: true, decimals: 0 })).toBe("$1M");
    expect(formatUsd(0.0000004, { decimals: 6, showSmallAmount: true })).toBe("< $0.000001");
    expect(formatUsd(-3)).toBe("-$3.00");
  });

  it("handles null, non-finite and optionally hidden zero", () => {
    expect(formatMoney(null, eur)).toBe("-");
    expect(formatMoney(undefined, eur)).toBe("-");
    expect(formatMoney(NaN, eur)).toBe("-");
    expect(formatMoney(0, eur, { showZero: false })).toBe("-");
    expect(formatMoney(0, eur)).toBe("€0.00");
  });
});
