import { describe, expect, it } from "vitest";
import { isValidRateLimitInput, rateLimitUpdate } from "./userRateLimitPayload";

describe("rateLimitUpdate", () => {
  it("omits an untouched input when the stored value is null", () => {
    expect(rateLimitUpdate(undefined, null)).toBeUndefined();
  });

  it.each([500, "500"])("omits an unchanged stored limit from input %s", (input) => {
    expect(rateLimitUpdate(input, 500)).toBeUndefined();
  });

  it.each(["", null])("sends null when a stored limit is deliberately cleared with %s", (input) => {
    expect(rateLimitUpdate(input, 500)).toBeNull();
  });

  it("sends a new limit entered as a string", () => {
    expect(rateLimitUpdate("100", null)).toBe(100);
  });

  it("preserves zero as a changed limit", () => {
    expect(rateLimitUpdate("0", 500)).toBe(0);
  });

  it("omits a whitespace-only input when no limit was stored", () => {
    expect(rateLimitUpdate("  ", null)).toBeUndefined();
  });
});

describe("isValidRateLimitInput", () => {
  it.each([
    ["empty string", ""],
    ["null", null],
    ["undefined", undefined],
    ["whitespace", "  "],
    ["string zero", "0"],
    ["number zero", 0],
    ["integer string", "12"],
    ["integer", 12],
  ])("accepts %s", (_label, value) => {
    expect(isValidRateLimitInput(value)).toBe(true);
  });

  it.each(["1.5", "-1", "abc", "1e400"])("rejects %s", (value) => {
    expect(isValidRateLimitInput(value)).toBe(false);
  });
});
