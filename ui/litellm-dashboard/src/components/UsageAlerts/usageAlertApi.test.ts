import { describe, expect, it } from "vitest";
import { formatAlertChange } from "./usageAlertApi";

describe("saved comparison summaries", () => {
  it("uses each saved comparison's ratio for increases and decreases", () => {
    expect(formatAlertChange({ direction: "increase", observed_value: 420, baseline_value: 120 })).toBe("↑ 3.5×");
    expect(formatAlertChange({ direction: "decrease", observed_value: 120, baseline_value: 420 })).toBe("↓ 3.5×");
  });

  it("describes zero baselines and complete drops without an infinite ratio", () => {
    expect(formatAlertChange({ direction: "increase", observed_value: 10, baseline_value: 0 })).toBe("New activity");
    expect(formatAlertChange({ direction: "decrease", observed_value: 0, baseline_value: 100 })).toBe("↓ 100%");
  });
});
