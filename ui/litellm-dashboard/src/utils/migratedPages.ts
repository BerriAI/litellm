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
  "api-keys": "api-keys",
  models: "models-and-endpoints",
  api_ref: "api-reference",
  // Legacy alias: older bookmarks used the hyphenated ?page=api-reference form.
  "api-reference": "api-reference",
  "llm-playground": "playground",
  projects: "projects",
  chat: "chat",
  "access-groups": "access-groups",
  budgets: "budgets",
  workflows: "workflows",
  "guardrails-monitor": "guardrails-monitor",
  "mcp-servers": "mcp-servers",
  "search-tools": "search-tools",
  "tag-management": "tag-management",
  "vector-stores": "vector-stores",
  memory: "memory",
  policies: "policies",
  guardrails: "guardrails",
  prompts: "prompts",
  "tool-policies": "tool-policies",
  skills: "skills",
  // Legacy alias: the old switch matched ?page=claude-code-plugins for the same panel.
  "claude-code-plugins": "skills",
  caching: "caching",
  "cost-tracking": "cost-tracking",
  "transform-request": "transform-request",
  "ui-theme": "ui-theme",
  logs: "logs",
  "admin-panel": "admin-panel",
  "logging-and-alerts": "logging-and-alerts",
  "model-hub-table": "model-hub-table",
  // The modern usage dashboard; the legacy ?page=usage report routes to /old-usage.
  new_usage: "usage",
  usage: "old-usage",
  "cost-optimization": "cost-optimization",
  agents: "agents",
  "router-settings": "router-settings",
  users: "users",
  teams: "teams",
  organizations: "organizations",
};

function uiBase(): string {
  // next dev serves the app at the root; only the proxy mounts the static export under /ui
  // (and optionally under server_root_path). Inlined at build time, so production is unaffected.
  if (process.env.NODE_ENV === "development") {
    return "";
  }
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

/** Reverse-maps a path-routed location back to its legacy page id, e.g. "/ui/api-reference" -> "api_ref". */
export function legacyKeyForPathname(pathname: string): string | null {
  const base = uiBase();
  const rel = (pathname.startsWith(base) ? pathname.slice(base.length) : pathname).replace(/^\/+|\/+$/g, "");
  for (const [key, segment] of Object.entries(MIGRATED_PAGES)) {
    if (rel === segment) return key;
  }
  return null;
}
