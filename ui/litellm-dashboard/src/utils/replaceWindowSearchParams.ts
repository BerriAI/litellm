/**
 * Update the current URL's search params without a Next soft-navigation.
 * Required when the static dashboard is served under /ui (basePath is empty, so
 * router.replace("/ui/...") does not reliably update useSearchParams).
 */
export function replaceWindowSearchParams(mutate: (params: URLSearchParams) => void): void {
  if (typeof window === "undefined") {
    return;
  }
  const params = new URLSearchParams(window.location.search);
  mutate(params);
  const qs = params.toString();
  const next = `${window.location.pathname}${qs ? `?${qs}` : ""}${window.location.hash}`;
  window.history.replaceState(null, "", next);
}
