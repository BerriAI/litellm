import { sessionLogoutCall } from "@/components/networking";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl } from "@/utils/returnUrlUtils";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

/**
 * Revokes the session key server-side, then clears client state. Exported for
 * flows that navigate somewhere other than PROXY_LOGOUT_URL (worker switch,
 * forced password reset). The server call must happen BEFORE the cookies are
 * cleared (the token authenticates it) and is best-effort: local logout must
 * still complete when the server is unreachable.
 */
export async function revokeSessionAndClearClientState(accessToken: string | null): Promise<void> {
  if (accessToken) {
    try {
      await sessionLogoutCall(accessToken);
    } catch {
      // Best-effort: the key still expires server-side at its session TTL.
    }
  }
  clearTokenCookies();
  clearStoredReturnUrl();
  localStorage.removeItem("litellm_selected_worker_id");
  localStorage.removeItem("litellm_worker_url");
}

/**
 * Shared sign-out handler. Used by both the top navbar and the sidebar footer so
 * the two entry points can never drift on which client state gets cleared.
 */
export function useLogout(accessToken: string | null): () => void {
  const proxySettings = useProxySettings(accessToken);

  return () => {
    void revokeSessionAndClearClientState(accessToken).finally(() => {
      window.location.href = proxySettings.PROXY_LOGOUT_URL || "";
    });
  };
}
