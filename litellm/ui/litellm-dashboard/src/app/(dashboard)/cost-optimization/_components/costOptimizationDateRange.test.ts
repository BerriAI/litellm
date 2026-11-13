import { describe, expect, it } from "vitest";

import { costOptimizationDateRange } from "./costOptimizationDateRange";

describe("costOptimizationDateRange", () => {
  it.each([
    ["2026-10-14", "2026-10-08", "12:34:56"],
    ["2026-03-02", "2026-02-24", "12:34:56"],
    ["2028-03-02", "2028-02-25", "12:34:56"],
    ["2026-01-02", "2025-12-27", "12:34:56"],
    ["2026-03-09", "2026-03-03", "00:30:00"],
    ["2026-11-02", "2026-10-27", "23:30:00"],
  ])("includes exactly seven local calendar days ending on %s", (today, firstDay, time) => {
    const now = new Date(`${today}T${time}`);

    expect(costOptimizationDateRange(now)).toEqual({
      from: new Date(`${firstDay}T00:00:00`),
      to: new Date(`${today}T23:59:59.999`),
    });
    expect(now).toEqual(new Date(`${today}T${time}`));
  });
});
