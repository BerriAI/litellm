import { describe, expect, it } from "vitest";
import { estimatedSpend } from "./spendDisplay";

const USD_PER_UNIT = { USD: 1, EUR: 1.25, JPY: 0.01 };
const NOW = new Date("2026-11-14T12:00:00Z");
const ratesOn = (rate_date: string) => ({ usd_per_unit: USD_PER_UNIT, rate_date });

describe("estimatedSpend", () => {
  it.each([
    [42.5, "EUR", 34],
    [0, "EUR", 0],
    [-2.5, "EUR", -2],
    [42.5, "JPY", 4250],
  ])("converts %d USD to %s at the reference rate", (usd, currency, amount) => {
    expect(estimatedSpend(usd, ratesOn("2026-11-13"), currency, NOW)).toEqual({ amount, rateDate: "2026-11-13" });
  });

  it("offers no estimate for a currency the rates do not list", () => {
    expect(estimatedSpend(42.5, ratesOn("2026-11-13"), "XYZ", NOW)).toBeNull();
  });

  it.each([
    ["2026-11-07", true],
    ["2026-11-06", false],
    ["2026-11-15", false],
  ])("uses a rate dated %s only while it is at most seven days old: %s", (rateDate, usable) => {
    expect(estimatedSpend(42.5, ratesOn(rateDate), "EUR", NOW) !== null).toBe(usable);
  });
});
