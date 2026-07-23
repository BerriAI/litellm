import { createTabRoutes } from "@/utils/tabRoutes";

export const costOptimizationRoutes = createTabRoutes("cost-optimization", [
  "compression",
  "autorouter",
  "caching",
] as const);

export type CostOptimizationTabSlug = (typeof costOptimizationRoutes.slugs)[number];
