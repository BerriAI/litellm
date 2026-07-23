import { migratedHref } from "@/utils/migratedPages";

export const LOGS_BASE_SEGMENT = "logs";

export const LOGS_TAB_SLUGS = ["audit", "deleted-keys", "deleted-teams"] as const;

export type LogsTabSlug = (typeof LOGS_TAB_SLUGS)[number];

export function logsTabHref(slug: string): string {
  const base = migratedHref(LOGS_BASE_SEGMENT);
  return slug ? `${base}/${slug}/` : `${base}/`;
}

export function slugFromPathname(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  const idx = parts.indexOf(LOGS_BASE_SEGMENT);
  if (idx === -1) {
    return "";
  }
  return parts[idx + 1] ?? "";
}
