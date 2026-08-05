export interface MCPConnectProxySettings {
  PROXY_BASE_URL?: string | null;
  LITELLM_UI_API_DOC_BASE_URL?: string | null;
}

export function resolveMCPConnectBaseUrl(
  proxySettings: MCPConnectProxySettings | undefined,
  fallbackBaseUrl: string,
): string {
  const customDocBaseUrl = proxySettings?.LITELLM_UI_API_DOC_BASE_URL;
  if (customDocBaseUrl && customDocBaseUrl.trim()) {
    return customDocBaseUrl.trim();
  }

  const proxyBaseUrl = proxySettings?.PROXY_BASE_URL;
  if (proxyBaseUrl && proxyBaseUrl.trim()) {
    return proxyBaseUrl.trim();
  }

  return fallbackBaseUrl;
}
