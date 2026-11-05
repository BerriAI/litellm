import { describe, expect, it } from "vitest";
import { parseActivityDay } from "./entityActivityState";

describe("activity date links", () => {
  it.each(["2026-11-05", "2024-02-29"])("accepts a calendar day: %s", (value) => {
    expect(parseActivityDay.parse(value)).toBe(value);
  });
  it.each(["", "today", "2026-2-1", "2026-02-29", "2026-11-05T12:00:00Z", "2026-13-01"])("rejects %s", (value) => {
    expect(parseActivityDay.parse(value)).toBeNull();
  });
});
