import { migratedHref } from "@/utils/migratedPages";

export const CACHING_BASE_SEGMENT = "caching";

export const CACHE_TAB_SLUGS = ["health", "settings", "coordination-redis"] as const;

export type CacheTabSlug = (typeof CACHE_TAB_SLUGS)[number];

export function cacheTabHref(slug: string): string {
  const base = migratedHref(CACHING_BASE_SEGMENT);
  return slug ? `${base}/${slug}/` : `${base}/`;
}

export function slugFromPathname(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  const idx = parts.indexOf(CACHING_BASE_SEGMENT);
  if (idx === -1) {
    return "";
  }
  return parts[idx + 1] ?? "";
}
