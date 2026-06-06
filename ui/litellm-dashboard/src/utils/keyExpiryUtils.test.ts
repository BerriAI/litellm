import { afterEach, describe, expect, it, vi } from "vitest";
import { isKeyExpired, parseExpiresUtc } from "./keyExpiryUtils";

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
});
