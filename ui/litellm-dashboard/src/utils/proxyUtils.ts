import { getProxyBaseUrl, getProxyUISettings } from "@/components/networking";

export const fetchProxySettings = async (accessToken: string | null) => {
  if (!accessToken) return null;

  try {
    const proxySettings = await getProxyUISettings(accessToken);
    return proxySettings;
  } catch (error) {
    console.error("Error fetching proxy settings:", error);
    return null;
  }
};

/**
 * Resolve the base URL shown in UI docs/snippets.
 * Prefer LITELLM_UI_API_DOC_BASE_URL (public docs host) over PROXY_BASE_URL / runtime proxy.
 */
export function resolveUiApiDocBaseUrl(
  proxySettings?: {
    PROXY_BASE_URL?: string | null;
    LITELLM_UI_API_DOC_BASE_URL?: string | null;
  } | null,
): string {
  const customDocBaseUrl = proxySettings?.LITELLM_UI_API_DOC_BASE_URL?.trim();
  if (customDocBaseUrl) {
    return customDocBaseUrl;
  }
  const proxyBaseFromSettings = proxySettings?.PROXY_BASE_URL?.trim();
  if (proxyBaseFromSettings) {
    return proxyBaseFromSettings;
  }
  return getProxyBaseUrl();
}
