import { getProxyBaseUrl } from "@/components/networking";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { clearStoredReturnUrl, getLoginUrl } from "@/utils/returnUrlUtils";
import { useQueryClient } from "@tanstack/react-query";
import useProxySettings, { ensureProxySettings } from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

/**
 * Shared sign-out handler. Used by both the top navbar and the sidebar footer so
 * the two entry points can never drift on which client state gets cleared.
 */
export function useLogout(accessToken: string | null): () => Promise<void> {
  const queryClient = useQueryClient();
  useProxySettings(accessToken);

  return async () => {
    clearTokenCookies();
    clearStoredReturnUrl();
    localStorage.removeItem("litellm_selected_worker_id");
    localStorage.removeItem("litellm_worker_url");
    const settings = await ensureProxySettings(queryClient, accessToken).catch(() => null);
    window.location.replace(settings?.PROXY_LOGOUT_URL || getLoginUrl(getProxyBaseUrl()));
  };
}
