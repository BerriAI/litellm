import { describe, expect, it } from "vitest";

import { BUDGET_WINDOW_OPTIONS } from "./BudgetWindowsEditor";

const hintFor = (value: string): string => {
  const option = BUDGET_WINDOW_OPTIONS.find((o) => o.value === value);
  if (!option) throw new Error(`no budget window option for ${value}`);
  return option.resetHint;
};

describe("BUDGET_WINDOW_OPTIONS reset hints", () => {
  it("describes the weekly window as resetting on Monday, matching the backend", () => {
    expect(hintFor("7d")).toContain("Monday");
    expect(hintFor("7d")).not.toContain("Sunday");
  });

  it("does not claim a fixed UTC reset, since the backend honors the configured timezone", () => {
    for (const option of BUDGET_WINDOW_OPTIONS) {
      expect(option.resetHint).not.toContain("midnight UTC");
    }
  });

  it("points at litellm_settings.timezone and its UTC default for every window that resets at a time of day", () => {
    for (const value of ["24h", "7d", "30d"]) {
      expect(hintFor(value)).toContain("litellm_settings.timezone");
      expect(hintFor(value)).toContain("defaults to UTC");
    }
  });
});
