import { serverRootPath } from "@/lib/serverRootPath";

function uiBase(): string {
  const root = serverRootPath && serverRootPath !== "/" ? `/${serverRootPath.replace(/^\/+|\/+$/g, "")}` : "";
  return `${root}/ui`;
}

/**
 * Browser-absolute (same-origin) href for a dashboard route segment, e.g. "api-reference" -> "/ui/api-reference".
 * Use for raw <a> tags and window.location navigations. For <Link> and router.push, pass the app-relative
 * path ("/api-reference") instead: the router prepends the /ui basePath itself.
 */
export function uiHref(routeSegment: string): string {
  return `${uiBase()}/${routeSegment.replace(/^\/+/, "")}`;
}

/** App-relative path for router.push, from a uiHref-built href: "/ui/teams?team=x" -> "/teams?team=x". */
export function appHrefFromUiHref(href: string): string {
  const base = uiBase();
  if (href === base) return "/";
  return href.startsWith(`${base}/`) || href.startsWith(`${base}?`) ? href.slice(base.length) : href;
}

/** First route segment of an app-relative pathname (usePathname already strips the /ui basePath). */
export function routeSegmentForPathname(pathname: string): string {
  return pathname.replace(/^\/+/, "").split("/")[0];
}
