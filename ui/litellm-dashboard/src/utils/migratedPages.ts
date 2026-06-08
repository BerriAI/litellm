import { serverRootPath } from "@/components/networking";

/**
 * Single source of truth for pages cut over from the legacy `?page=` switch in
 * app/page.tsx to path-based routes under app/(dashboard)/.
 *
 * Key = legacy page id emitted by the sidebar. Value = route segment under (dashboard)/.
 * Add an entry to route the sidebar and deep links to the new path and redirect the
 * legacy `?page=` URL; remove it to roll back. Empty until a page is migrated.
 */
export const MIGRATED_PAGES: Record<string, string> = {};

export function migratedHref(routeSegment: string): string {
  const raw = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = raw.replace(/^\/+|\/+$/g, "");
  let base = trimmed ? `/${trimmed}/` : "/";

  if (serverRootPath && serverRootPath !== "/") {
    const cleanRoot = serverRootPath.replace(/\/+$/, "");
    const cleanBase = base.replace(/^\/+/, "");
    base = `${cleanRoot}/${cleanBase}`;
  }

  return `${base}${routeSegment}`;
}
