import { describe, expect, it } from "vitest";
import { createLoader } from "nuqs";
import {
  customerDateSelection,
  customerDatePatch,
  customerPresetRange,
  customerRangeParser,
  customerUsageDateParsers,
} from "./customerUsageRange";

const loadDates = createLoader(customerUsageDateParsers);

describe("customer usage calendar ranges", () => {
  it.each([
    ["month", "2024-02-15T12:00:00Z", "2024-02-01..2024-02-29"],
    ["previous", "2025-01-01T00:00:00Z", "2024-12-01..2024-12-31"],
    ["YTD", "2025-01-01T00:00:00Z", "2025-01-01..2025-01-01"],
    ["YTD", "2025-11-12T00:01:00Z", "2025-01-01..2025-11-12"],
    ["MTD", "2025-01-01T00:01:00Z", "2025-01-01..2025-01-01"],
    ["today", "2025-01-01T00:01:00Z", "2025-01-01..2025-01-01"],
    ["7d", "2025-01-01T00:01:00Z", "2024-12-25..2025-01-01"],
    ["30d", "2024-03-01T23:59:00Z", "2024-01-31..2024-03-01"],
  ] as const)("resolves %s at %s as UTC calendar days", (preset, now, expected) => {
    expect(customerRangeParser.serialize(customerPresetRange(preset, new Date(now)))).toBe(expected);
  });

  it("round trips a single calendar day without shifting its timezone", () => {
    const range = customerRangeParser.parse("2024-02-29..2024-02-29");
    expect(range).not.toBeNull();
    expect(customerRangeParser.serialize(range!)).toBe("2024-02-29..2024-02-29");
  });

  it.each([
    ["?range=2024-02-01..2024-02-29&period=previous", "2024-02-01..2024-02-29", "previous"],
    ["?from=2024-02-01&to=2024-02-29", "2024-02-01..2024-02-29", null],
    ["?range=MTD&from=2024-01-01&to=2024-01-03", "2024-03-01..2024-03-01", "MTD"],
    ["?range=garbage", "2024-03-01..2024-03-31", "month"],
    ["?from=2024-02-30&to=2024-03-02", "2024-03-01..2024-03-31", "month"],
    ["?from=2024-03-02&to=2024-03-01", "2024-03-01..2024-03-31", "month"],
    ["?from=2024-01-01&to=2025-02-05", "2024-03-01..2024-03-31", "month"],
  ])("resolves saved links and falls back for invalid dates in %s", (query, expected, preset) => {
    const selection = customerDateSelection(loadDates(query), new Date("2024-03-01T00:30:00Z"));
    expect(customerRangeParser.serialize(selection.dateValue)).toBe(expected);
    expect(selection.preset).toBe(preset);
  });

  it("replaces a previous preset with the exact calendar selection", () => {
    const value = { from: new Date(2024, 2, 9), to: new Date(2024, 2, 11) };
    const custom = { range: null, period: null, from: "2024-03-09", to: "2024-03-11" };
    const relative = { range: "7d", period: null, from: null, to: null };
    expect(customerDatePatch(value, null)).toEqual(custom);
    expect(customerDatePatch(value, "7d")).toEqual(relative);
    expect(customerDatePatch({ from: value.to, to: value.from }, null)).toBeNull();
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
