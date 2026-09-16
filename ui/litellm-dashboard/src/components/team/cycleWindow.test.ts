import { describe, expect, it, vi } from "vitest";
import { describeCycleWindow } from "./cycleWindow";

const formatDate = (iso: string): string => `formatted:${iso}`;

describe("describeCycleWindow", () => {
  it("describes a missing budget as never resetting", () => {
    expect(describeCycleWindow(null, formatDate)).toBe("Never resets");
  });

  it("describes a never-resetting budget duration", () => {
    expect(
      describeCycleWindow({ budget_duration: "none", budget_reset_at: "2026-07-15T12:00:00.000Z" }, formatDate),
    ).toBe("Never resets");
  });

  it("describes a budget with a pending reset date", () => {
    const formatDateSpy = vi.fn(formatDate);

    expect(describeCycleWindow({ budget_duration: "30d", budget_reset_at: null }, formatDateSpy)).toBe(
      "Resets every 30d",
    );
    expect(formatDateSpy).not.toHaveBeenCalled();
  });

  it("describes the reset date and duration", () => {
    expect(
      describeCycleWindow({ budget_duration: "30d", budget_reset_at: "2026-07-15T12:00:00.000Z" }, formatDate),
    ).toBe("Resets formatted:2026-07-15T12:00:00.000Z (every 30d)");
  });
});
