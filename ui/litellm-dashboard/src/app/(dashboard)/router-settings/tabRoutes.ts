import { migratedHref } from "@/utils/migratedPages";

export const ROUTER_SETTINGS_BASE_SEGMENT = "router-settings";

export const ROUTER_SETTINGS_TAB_SLUGS = ["routing-groups", "fallbacks", "prompt-caching", "general"] as const;

export type RouterSettingsTabSlug = (typeof ROUTER_SETTINGS_TAB_SLUGS)[number];

export function routerSettingsTabHref(slug: string): string {
  const base = migratedHref(ROUTER_SETTINGS_BASE_SEGMENT);
  return slug ? `${base}/${slug}/` : `${base}/`;
}

export function slugFromPathname(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  const idx = parts.indexOf(ROUTER_SETTINGS_BASE_SEGMENT);
  if (idx === -1) {
    return "";
  }
  return parts[idx + 1] ?? "";
}
