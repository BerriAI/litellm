import { backgroundFetchClient } from "./api";

export const LENS_UI_AUTH = "bearer-v1";
export type LensCredentials = { accessToken: string; headerName: string };
export type LensWorkspaceAvailability =
  | { state: "served" | "unconfigured" | "unavailable" }
  | { state: "unsupported"; browserUrl: string };

export const lensWorkspaceAvailability = async (
  url: string,
  { accessToken, headerName }: LensCredentials,
  signal: AbortSignal,
): Promise<LensWorkspaceAvailability> => {
  if (signal.aborted) return { state: "unavailable" };
  const probe = new AbortController();
  const abortProbe = () => probe.abort();
  signal.addEventListener("abort", abortProbe, { once: true });
  const timeout = setTimeout(abortProbe, 5000);
  try {
    const lensUrl = new URL(url, window.location.href);
    const response = await fetch(lensUrl, {
      signal: probe.signal,
      headers: { Accept: "application/json" },
    });
    if (signal.aborted || (!response.ok && response.status !== 404)) return { state: "unavailable" };
    const sameOrigin =
      lensUrl.origin === window.location.origin &&
      new URL(response.url || lensUrl.href).origin === window.location.origin;
    const supported = response.ok && sameOrigin && response.headers.get("X-Lens-UI-Auth") === LENS_UI_AUTH;
    const { data: connection } = await backgroundFetchClient.GET("/lens/service", {
      signal,
      credentials: "include",
      headers: { [headerName]: `Bearer ${accessToken}` },
    });
    if (connection?.configured === false) return { state: "unconfigured" };
    if (connection?.connected !== true) return { state: "unavailable" };
    return supported ? { state: "served" } : { state: "unsupported", browserUrl: connection.browser_url ?? "" };
  } catch {
    return { state: "unavailable" };
  } finally {
    clearTimeout(timeout);
    signal.removeEventListener("abort", abortProbe);
  }
};
