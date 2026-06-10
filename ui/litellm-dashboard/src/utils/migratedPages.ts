import { serverRootPath } from "@/components/networking";

/**
 * Single source of truth for pages cut over from the legacy `?page=` switch in
 * app/page.tsx to path-based routes under app/(dashboard)/.
 *
 * Key = legacy page id emitted by the sidebar. Value = route segment under (dashboard)/.
 * Add an entry to route the sidebar and deep links to the new path and redirect the
 * legacy `?page=` URL; remove it to roll back.
 */
export const MIGRATED_PAGES: Record<string, string> = {
  api_ref: "api-reference",
  // Legacy alias: older bookmarks used the hyphenated ?page=api-reference form.
  "api-reference": "api-reference",
  budgets: "budgets",
  caching: "caching",
  "cost-tracking": "cost-tracking",
  guardrails: "guardrails",
  "guardrails-monitor": "guardrails-monitor",
  logs: "logs",
  "mcp-servers": "mcp-servers",
  memory: "memory",
  policies: "policies",
  projects: "projects",
  prompts: "prompts",
  "search-tools": "search-tools",
  // Canonical sidebar key first, alias after: legacyKeyForPathname returns the first match.
  skills: "skills",
  "claude-code-plugins": "skills",
  "tag-management": "tag-management",
  "tool-policies": "tool-policies",
  "transform-request": "transform-request",
  "ui-theme": "ui-theme",
  "vector-stores": "vector-stores",
  workflows: "workflows",
  "access-groups": "access-groups",
};

function uiBase(): string {
  const root = serverRootPath && serverRootPath !== "/" ? `/${serverRootPath.replace(/^\/+|\/+$/g, "")}` : "";
  return `${root}/ui`;
}

/** Absolute (same-origin) href for a migrated route segment, e.g. "api-reference" -> "/ui/api-reference". */
export function migratedHref(routeSegment: string): string {
  return `${uiBase()}/${routeSegment.replace(/^\/+/, "")}`;
}

/** Href for a not-yet-migrated page, served by the legacy `?page=` switch at the UI root. */
export function legacyPageHref(pageKey: string): string {
  return `${uiBase()}/?page=${pageKey}`;
}

/**
 * Reverse-maps a path-routed location back to its legacy page id, e.g. "/ui/api-reference" -> "api_ref".
 * Returns the first key whose segment matches, so when several keys share a segment (aliases) the
 * canonical sidebar key must be listed before its aliases in MIGRATED_PAGES.
 */
export function legacyKeyForPathname(pathname: string): string | null {
  const base = uiBase();
  const rel = (pathname.startsWith(base) ? pathname.slice(base.length) : pathname).replace(/^\/+|\/+$/g, "");
  for (const [key, segment] of Object.entries(MIGRATED_PAGES)) {
    if (rel === segment) return key;
  }
  return null;
}
