import { describe, it, expect } from "vitest";
import { toFiniteNumber } from "./numberUtils";

describe("toFiniteNumber", () => {
  it("returns finite numbers unchanged", () => {
    expect(toFiniteNumber(0)).toBe(0);
    expect(toFiniteNumber(0.005)).toBe(0.005);
    expect(toFiniteNumber(-1.25)).toBe(-1.25);
    expect(toFiniteNumber(1e-9)).toBe(1e-9);
  });

  it("rejects non-finite numbers", () => {
    expect(toFiniteNumber(NaN)).toBeNull();
    expect(toFiniteNumber(Infinity)).toBeNull();
    expect(toFiniteNumber(-Infinity)).toBeNull();
  });

  it("parses stringified numbers", () => {
    expect(toFiniteNumber("0.005")).toBe(0.005);
    expect(toFiniteNumber("42")).toBe(42);
    expect(toFiniteNumber("-1.25")).toBe(-1.25);
    expect(toFiniteNumber("1e-9")).toBe(1e-9);
    expect(toFiniteNumber("  3.14 ")).toBe(3.14);
  });

  it("rejects empty / whitespace-only strings", () => {
    expect(toFiniteNumber("")).toBeNull();
    expect(toFiniteNumber("   ")).toBeNull();
    expect(toFiniteNumber("\t\n")).toBeNull();
  });

  it("rejects non-numeric strings", () => {
    expect(toFiniteNumber("abc")).toBeNull();
    expect(toFiniteNumber("0.005 USD")).toBeNull();
    expect(toFiniteNumber("NaN")).toBeNull();
    expect(toFiniteNumber("Infinity")).toBeNull();
  });

  it("returns null for null / undefined", () => {
    expect(toFiniteNumber(null)).toBeNull();
    expect(toFiniteNumber(undefined)).toBeNull();
  });

  it("returns null for non-string non-number values", () => {
    expect(toFiniteNumber({})).toBeNull();
    expect(toFiniteNumber([])).toBeNull();
    expect(toFiniteNumber(true)).toBeNull();
    expect(toFiniteNumber(false)).toBeNull();
  });

  it("regression for litellm#27095: stringified cost survives without crashing", () => {
    // This is the exact shape seen in mcp_info JSONB column when the value is
    // serialized as a string instead of a number.
    const stringifiedCost = "0.005";
    const coerced = toFiniteNumber(stringifiedCost);
    expect(coerced).not.toBeNull();
    expect(coerced!.toFixed(4)).toBe("0.0050");
  });
});
