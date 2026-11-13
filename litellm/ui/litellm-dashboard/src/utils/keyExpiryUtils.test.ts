import { afterEach, describe, expect, it, vi } from "vitest";
import { calculateExpiryPreviewFromDuration, formatExpiresUtc, isKeyExpired, parseExpiresUtc } from "./keyExpiryUtils";

describe("keyExpiryUtils", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("treats naive ISO strings as UTC", () => {
    const ms = parseExpiresUtc("2020-01-01T12:00:00");
    expect(ms).toBe(Date.parse("2020-01-01T12:00:00Z"));
  });

  it("parses Z-suffixed ISO strings unchanged", () => {
    const iso = "2020-01-01T12:00:00.000Z";
    expect(parseExpiresUtc(iso)).toBe(Date.parse(iso));
  });

  it("formats naive ISO strings as UTC before localizing", () => {
    const naive = "2026-06-05T10:00:00";
    expect(formatExpiresUtc(naive)).toBe(new Date(Date.parse(`${naive}Z`)).toLocaleString());
  });

  it("returns false when expires is missing", () => {
    expect(isKeyExpired(undefined)).toBe(false);
    expect(isKeyExpired(null)).toBe(false);
  });

  it("detects expired keys using UTC comparison", () => {
    vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-06-06T12:00:00Z"));
    expect(isKeyExpired("2026-06-05T12:00:00Z")).toBe(true);
    expect(isKeyExpired("2026-06-06T12:00:00")).toBe(false);
    expect(isKeyExpired("2026-06-07T12:00:00Z")).toBe(false);
  });

  it("returns a preview for supported duration strings", () => {
    vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-06-06T12:00:00Z"));
    expect(calculateExpiryPreviewFromDuration("30d")).toBeTruthy();
  });

  it("returns null for unparseable duration strings", () => {
    expect(calculateExpiryPreviewFromDuration("bogus")).toBeNull();
    expect(calculateExpiryPreviewFromDuration(undefined)).toBeNull();
  });
});
