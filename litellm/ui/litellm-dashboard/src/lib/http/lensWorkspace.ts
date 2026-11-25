import { backgroundFetchClient } from "./api";
import type { components } from "./schema";
import { getCookie, storeLoginToken } from "@/utils/cookieUtils";
import { z } from "zod";

export type LensWorkspaceAvailability =
  | { state: "served" | "unconfigured" | "unavailable" | "session-changed" }
  | { state: "unsupported"; browserUrl: string };

const sessionResponse = z.object({ token: z.string().min(1) });

export const lensWorkspaceAvailability = async (
  url: string,
  dashboardToken: string | null,
  signal: AbortSignal,
): Promise<LensWorkspaceAvailability> => {
  try {
    const lensUrl = new URL(url, window.location.href);
    const response = await fetch(lensUrl, { signal, headers: { Accept: "application/json" } });
    if (!response.ok && response.status !== 404) return { state: "unavailable" };
    const sameOrigin = lensUrl.origin === window.location.origin;
    if (response.ok && sameOrigin) {
      const options: RequestInit = {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({
          token: dashboardToken,
        } satisfies components["schemas"]["Body_restore_dashboard_session_lens_session_post"]),
        signal,
        credentials: "include",
        redirect: "error",
      };
      const session = await fetch(new URL("../session", lensUrl), options);
      if (!session.ok) return { state: "unavailable" };
      const body: unknown = await session.json();
      const selected = sessionResponse.safeParse(body);
      if (!selected.success || signal.aborted) return { state: "unavailable" };
      if (selected.data.token !== dashboardToken || selected.data.token !== getCookie("token")) {
        storeLoginToken(selected.data.token);
      }
      if (getCookie("token") !== selected.data.token) return { state: "unavailable" };
      if (selected.data.token !== dashboardToken) return { state: "session-changed" };
    }
    const { data: connection } = await backgroundFetchClient.GET("/lens/service", {
      signal,
      credentials: "include",
    });
    if (connection?.configured === false) return { state: "unconfigured" };
    if (connection?.connected !== true) return { state: "unavailable" };
    return response.status === 404 || !sameOrigin
      ? { state: "unsupported", browserUrl: connection.browser_url ?? "" }
      : { state: "served" };
  } catch {
    return { state: "unavailable" };
  }
};
