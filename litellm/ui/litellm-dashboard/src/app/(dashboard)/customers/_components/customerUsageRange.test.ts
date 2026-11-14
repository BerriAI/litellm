import { describe, expect, it } from "vitest";
import { customerPresetRange, customerRangeParser } from "./customerUsageRange";

describe("customer usage calendar ranges", () => {
  it.each([
    ["month", "2024-02-15T12:00:00Z", "2024-02-01..2024-02-29"],
    ["previous", "2025-01-01T00:00:00Z", "2024-12-01..2024-12-31"],
    ["YTD", "2025-01-01T00:00:00Z", "2025-01-01..2025-01-01"],
    ["YTD", "2025-11-12T00:01:00Z", "2025-01-01..2025-11-12"],
  ] as const)("resolves %s at %s as UTC calendar days", (preset, now, expected) => {
    expect(customerRangeParser.serialize(customerPresetRange(preset, new Date(now)))).toBe(expected);
  });

  it("round trips a single calendar day without shifting its timezone", () => {
    const range = customerRangeParser.parse("2024-02-29..2024-02-29");
    expect(range).not.toBeNull();
    expect(customerRangeParser.serialize(range!)).toBe("2024-02-29..2024-02-29");
  });

  it("accepts the API's 400-day boundary across leap years and daylight saving", () => {
    const range = customerRangeParser.parse("2024-01-01..2025-02-04");
    expect(range).not.toBeNull();
    expect(customerRangeParser.serialize(range!)).toBe("2024-01-01..2025-02-04");
  });

  it.each([
    "2023-02-29..2023-03-01",
    "2024-02-30..2024-03-01",
    "2024-03-01..2024-02-01",
    "2024-02-01",
    "2024-02-01..2024-02-29..2024-03-01",
    "2024-2-01..2024-02-29",
    "2024-01-01..2025-02-05",
  ])("rejects invalid URL range %s", (range) => {
    expect(customerRangeParser.parse(range)).toBeNull();
  });
});
