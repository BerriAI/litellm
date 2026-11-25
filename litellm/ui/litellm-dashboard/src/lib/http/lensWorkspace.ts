import { backgroundFetchClient } from "./api";
import { getCookie } from "@/utils/cookieUtils";

export type LensWorkspaceAvailability =
  | { state: "served" | "unconfigured" | "unavailable" }
  | { state: "unsupported"; browserUrl: string };

export const lensWorkspaceAvailability = async (
  url: string,
  signal: AbortSignal,
): Promise<LensWorkspaceAvailability> => {
  try {
    const { data: connection } = await backgroundFetchClient.GET("/lens/service", {
      signal,
      credentials: "include",
    });
    if (connection?.configured === false) return { state: "unconfigured" };
    if (connection?.connected !== true) return { state: "unavailable" };
    const token = getCookie("token");
    if (token) {
      const options: RequestInit = {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
        signal,
        credentials: "include",
        redirect: "error",
      };
      const session = await fetch(new URL("../session", url), options);
      if (!session.ok) return { state: "unavailable" };
    }
    const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
    if (response.ok) return { state: "served" };
    return response.status === 404
      ? { state: "unsupported", browserUrl: connection.browser_url ?? "" }
      : { state: "unavailable" };
  } catch {
    return { state: "unavailable" };
  }
};
