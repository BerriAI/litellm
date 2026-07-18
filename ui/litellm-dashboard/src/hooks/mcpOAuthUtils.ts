/**
 * Shared utilities for MCP OAuth2 PKCE flow hooks.
 *
 * These helpers are used by both useToolsOAuthFlow and useUserMcpOAuthFlow
 * to avoid divergence in URL construction and storage cleanup logic.
 */

import { getProxyBaseUrl, serverRootPath } from "@/components/networking";

/**
 * sessionStorage key used to restore the MCP server detail view on the Tools
 * tab after a full-page OAuth redirect. The OBO authorize flow redirects to the
 * IdP and back to the MCP Servers page; without this the user lands on the
 * server list and useUserMcpOAuthFlow never re-mounts to persist the credential.
 * Mirrors the admin edit flow's EDIT_OAUTH_UI_STATE_KEY.
 */
export const TOOLS_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-tools-state";

/**
 * sessionStorage flag set by useUserMcpOAuthFlow right before it navigates the whole page
 * to the upstream IdP to authorize one server. ConnectFlowBanner's auto-finish-on-close
 * handler skips while this is set, so authorizing a server is not mistaken for the user
 * leaving the gateway DCR connect flow.
 */
export const PERSERVER_CONNECTING_KEY = "litellm-mcp-perserver-connecting";

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
