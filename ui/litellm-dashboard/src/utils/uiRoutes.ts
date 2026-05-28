/**
 * Shared helpers for resolving sidebar / dashboard navigation hrefs.
 *
 * The LiteLLM proxy always mounts the static UI at \`/ui\` (see
 * \`app.mount("/ui", StaticFiles(...))\` in
 * \`litellm/proxy/proxy_server.py\`). All sidebar links must therefore
 * be anchored to that mount regardless of \`NEXT_PUBLIC_BASE_URL\`.
 *
 * Keep this module dependency-light — both \`leftnav.tsx\` and
 * \`(dashboard)/layout.tsx\` import from it.
 */
import { serverRootPath } from "@/components/networking";

/** Resolve the UI root URL: \`<serverRootPath>/ui/\`. */
export function uiRootBase(): string {
  const root = serverRootPath && serverRootPath !== "/" ? serverRootPath.replace(/\/+$/, "") : "";
  return `${root}/ui/`;
}

/** Build an absolute href for a migrated (path-based) page. */
export function migratedHref(routeSegment: string): string {
  return `${uiRootBase()}${routeSegment.replace(/^\/+/, "")}`;
}

/**
 * Build an absolute href for a legacy (query-param) page.
 *
 * Only emits \`?page=<page>\`; we intentionally do not carry over any
 * other query parameters from \`window.location.search\` because each
 * legacy page renders its own panel and reusing a stale filter (e.g.
 * \`someFilter=value\` from a previous panel) silently leaks state.
 */
export function legacyHref(page: string): string {
  return `${uiRootBase()}?page=${encodeURIComponent(page)}`;
}
