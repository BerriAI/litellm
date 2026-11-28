export const COST_OPTIMIZATION_TABS = ["usage", "compression", "caching", "autorouter-usage"] as const;
export type CostOptimizationTab = (typeof COST_OPTIMIZATION_TABS)[number];
export const DEFAULT_COST_OPTIMIZATION_TAB: CostOptimizationTab = "usage";

export const visibleCostOptimizationTab = (
  requested: CostOptimizationTab,
  canViewProxyWideCostData: boolean,
): CostOptimizationTab => (canViewProxyWideCostData ? requested : DEFAULT_COST_OPTIMIZATION_TAB);
