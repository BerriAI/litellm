export function navigateWithParams(mutate: (params: URLSearchParams) => void, mode: "push" | "replace" = "push"): void {
  const params = new URLSearchParams(window.location.search);
  mutate(params);
  const qs = params.toString();
  const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  if (mode === "replace") {
    window.history.replaceState(null, "", url);
  } else {
    window.history.pushState(null, "", url);
  }
}
