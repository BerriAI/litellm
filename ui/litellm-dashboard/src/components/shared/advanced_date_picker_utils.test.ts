import { describe, expect, it } from "vitest";
import { getInclusiveDayRange } from "./advanced_date_picker_utils";

describe("getInclusiveDayRange", () => {
  it.each([
    [7, "2026-10-13", "2026-10-07"],
    [30, "2026-10-13", "2026-09-14"],
    [7, "2026-01-03", "2025-12-28"],
    [30, "2026-03-10", "2026-02-09"],
    [7, "2026-11-03", "2026-10-28"],
  ] as const)("includes exactly %i calendar days ending on %s", (days, today, expectedStart) => {
    expect(getInclusiveDayRange(days, new Date(`${today}T12:00:00`))).toEqual({
      from: new Date(`${expectedStart}T00:00:00.000`),
      to: new Date(`${today}T23:59:59.999`),
    });
  });
});
