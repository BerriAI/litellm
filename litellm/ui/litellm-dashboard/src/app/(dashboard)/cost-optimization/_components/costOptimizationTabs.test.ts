import { describe, expect, it } from "vitest";

import { COST_OPTIMIZATION_TABS, visibleCostOptimizationTab } from "./costOptimizationTabs";

describe("visibleCostOptimizationTab", () => {
  it.each(COST_OPTIMIZATION_TABS)("keeps %s for a role that can view proxy-wide cost data", (tab) => {
    expect(visibleCostOptimizationTab(tab, true)).toBe(tab);
  });

  it.each(COST_OPTIMIZATION_TABS)("shows the overall tab instead of %s to a role that cannot", (tab) => {
    expect(visibleCostOptimizationTab(tab, false)).toBe("usage");
  });
});
