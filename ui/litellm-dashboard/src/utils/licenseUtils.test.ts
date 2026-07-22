import { describe, it, expect } from "vitest";
import {
  type LicenseExpiryTier,
  formatExpirationStatus,
  formatExpiryDate,
  getDaysUntilExpiration,
  getLicenseExpiryTier,
} from "./licenseUtils";

const NOW = new Date("2026-07-08T00:00:00Z");

describe("getDaysUntilExpiration", () => {
  it("returns null for a null expiration", () => {
    expect(getDaysUntilExpiration(null, NOW)).toBeNull();
  });

  it("returns null for an unparseable date", () => {
    expect(getDaysUntilExpiration("not-a-date", NOW)).toBeNull();
  });

  it("returns 0 for an expiration on the current UTC day", () => {
    expect(getDaysUntilExpiration("2026-07-08", NOW)).toBe(0);
  });

  it("returns a positive count for future dates", () => {
    expect(getDaysUntilExpiration("2026-07-15", NOW)).toBe(7);
    expect(getDaysUntilExpiration("2026-08-07", NOW)).toBe(30);
  });

  it("returns a negative count for a past date", () => {
    expect(getDaysUntilExpiration("2026-07-07", NOW)).toBe(-1);
  });

  it("is timezone-independent within a UTC day", () => {
    expect(getDaysUntilExpiration("2026-08-07", new Date("2026-07-08T00:00:01Z"))).toBe(30);
    expect(getDaysUntilExpiration("2026-08-07", new Date("2026-07-08T23:59:59Z"))).toBe(30);
  });
});

describe("getLicenseExpiryTier", () => {
  const cases: Array<[string | null, LicenseExpiryTier]> = [
    [null, "none"],
    ["not-a-date", "none"],
    ["2026-08-08", "none"],
    ["2026-08-07", "warning"],
    ["2026-07-16", "warning"],
    ["2026-07-15", "critical"],
    ["2026-07-09", "critical"],
    ["2026-07-08", "critical"],
    ["2026-07-07", "expired"],
    ["2026-01-01", "expired"],
  ];

  it.each(cases)("classifies %s as %s", (date, expected) => {
    expect(getLicenseExpiryTier(date, NOW)).toBe(expected);
  });
});

describe("formatExpiryDate", () => {
  it("formats an ISO date as a human-readable UTC date", () => {
    expect(formatExpiryDate("2026-07-31")).toBe("Jul 31, 2026");
  });

  it("returns the input unchanged when unparseable", () => {
    expect(formatExpiryDate("bogus")).toBe("bogus");
  });
});

describe("formatExpirationStatus", () => {
  it("shows the exact date for a future expiration", () => {
    expect(formatExpirationStatus("2026-08-07", NOW)).toBe("Expires Aug 7, 2026");
  });

  it("still reads as upcoming on the expiration day itself", () => {
    expect(formatExpirationStatus("2026-07-08", NOW)).toBe("Expires Jul 8, 2026");
  });

  it("shows the exact date for a past expiration", () => {
    expect(formatExpirationStatus("2026-07-07", NOW)).toBe("Expired Jul 7, 2026");
  });

  it("returns No expiration for a null date", () => {
    expect(formatExpirationStatus(null, NOW)).toBe("No expiration");
  });

  it("returns No expiration for an unparseable date", () => {
    expect(formatExpirationStatus("not-a-date", NOW)).toBe("No expiration");
  });
});
