import { migratedHref } from "@/utils/migratedPages";

export const COST_OPTIMIZATION_BASE_SEGMENT = "cost-optimization";

export const COST_OPTIMIZATION_TAB_SLUGS = ["compression", "autorouter", "caching"] as const;

export type CostOptimizationTabSlug = (typeof COST_OPTIMIZATION_TAB_SLUGS)[number];

export function costOptimizationTabHref(slug: string): string {
  const base = migratedHref(COST_OPTIMIZATION_BASE_SEGMENT);
  return slug ? `${base}/${slug}/` : `${base}/`;
}

export function slugFromPathname(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  const idx = parts.indexOf(COST_OPTIMIZATION_BASE_SEGMENT);
  if (idx === -1) {
    return "";
  }
  return parts[idx + 1] ?? "";
}
