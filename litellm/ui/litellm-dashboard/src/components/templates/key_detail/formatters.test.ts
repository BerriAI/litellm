import { describe, expect, it } from "vitest";

import { formatMoney, formatRelative } from "./formatters";

describe("formatMoney", () => {
  it("shows dollars with two decimals and thousands separators", () => {
    expect(formatMoney(31480.5)).toBe("$31,480.50");
    expect(formatMoney(0)).toBe("$0.00");
  });

  it("keeps four decimals below a dollar so small spend does not read as zero", () => {
    expect(formatMoney(0.0042)).toBe("$0.0042");
  });

  it("does not pad a sub-dollar amount past the digits it has", () => {
    expect(formatMoney(0.1)).toBe("$0.10");
    expect(formatMoney(0.12345)).toBe("$0.1235");
  });

  it("keeps a sub-cent budget and its spend apart instead of rounding both to $0.00", () => {
    expect(formatMoney(0.000001)).toBe("$0.000001");
    expect(formatMoney(0.0000084)).toBe("$0.0000084");
  });
});

describe("formatRelative", () => {
  const now = Date.parse("2026-11-21T12:00:00Z");

  it("describes recent and older times in the largest whole unit", () => {
    expect(formatRelative("2026-11-21T11:58:00Z", now, "en")).toBe("2 minutes ago");
    expect(formatRelative("2026-11-21T09:00:00Z", now, "en")).toBe("3 hours ago");
    expect(formatRelative("2026-11-19T12:00:00Z", now, "en")).toBe("2 days ago");
  });

  it("calls anything under a minute now", () => {
    expect(formatRelative("2026-11-21T11:59:40Z", now, "en")).toBe("this minute");
  });
});
