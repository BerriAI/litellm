import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl } from "@/utils/returnUrlUtils";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

/**
 * Shared sign-out handler. Used by both the top navbar and the sidebar footer so
 * the two entry points can never drift on which client state gets cleared.
 */
export function useLogout(accessToken: string | null): () => void {
  const proxySettings = useProxySettings(accessToken);

  return () => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    window.location.href = proxySettings.PROXY_LOGOUT_URL || "";
  };
}
