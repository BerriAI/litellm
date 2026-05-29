/**
 * Shared utilities for MCP OAuth2 PKCE flow hooks.
 *
 * These helpers are used by both useToolsOAuthFlow and useUserMcpOAuthFlow
 * to avoid divergence in URL construction and storage cleanup logic.
 */

import { getProxyBaseUrl, serverRootPath } from "@/components/networking";

/**
 * Build the OAuth callback URL for the current UI deployment.
 *
 * In the browser, derive the `/ui` prefix from the current pathname so the
 * callback works regardless of how the proxy is mounted.  Outside the browser
 * (SSR), fall back to the configured proxy base URL and server root path.
 */
export const buildCallbackUrl = (): string => {
  if (typeof window !== "undefined") {
    const path = window.location.pathname || "";
    const idx = path.indexOf("/ui");
    const prefix = idx >= 0 ? path.slice(0, idx + 3).replace(/\/+$/, "") : "";
    return `${window.location.origin}${prefix}/mcp/oauth/callback`;
  }
  const base = (getProxyBaseUrl() || "").replace(/\/+$/, "");
  const root = serverRootPath && serverRootPath !== "/" ? serverRootPath : "";
  return `${base}${root}/ui/mcp/oauth/callback`;
};

/**
 * Remove the given keys from sessionStorage, ignoring errors (e.g. storage
 * disabled by browser privacy settings).
 */
export const clearStorage = (...keys: string[]): void => {
  keys.forEach((k) => {
    try {
      window.sessionStorage.removeItem(k);
    } catch (_) {}
  });
};
