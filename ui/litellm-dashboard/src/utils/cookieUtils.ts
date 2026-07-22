/**
 * Utility functions for managing cookies
 */

import { clearAllMcpTokens } from "./mcpTokenStore";

/**
 * Returns the cookie path for the UI.
 * Derives the path from window.location.pathname so it works when
 * LiteLLM is deployed behind a subpath (e.g. /myapp/ui instead of /ui).
 * No imports from networking.tsx to avoid circular dependencies.
 */
function getUiCookiePath(): string {
  if (typeof window === "undefined") return "/ui";
  // Match "/ui" only as a full path segment (followed by "/" or end of string)
  // to avoid false matches like "/my-ui-tool/login" → "/my-ui".
  // The UI mounts at the last "/ui" segment (SERVER_ROOT_PATH + "/ui"), so use the
  // last match to stay correct when the root path itself contains a "/ui" segment.
  const matches = [...window.location.pathname.matchAll(/\/ui(?=\/|$)/g)];
  const lastMatch = matches[matches.length - 1];
  if (lastMatch && lastMatch.index !== undefined) {
    return window.location.pathname.substring(0, lastMatch.index + 3);
  }
  return "/ui";
}

/**
 * Clears the token cookie from both root and /ui paths
 */
export function clearTokenCookies() {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return;
  }

  // Get the current domain
  const domain = window.location.hostname;

  // Clear with various combinations of path and SameSite
  // Include current path in case of custom server root path
  const currentPath = window.location.pathname;
  const uiCookiePath = getUiCookiePath();
  const paths = ["/", uiCookiePath];

  // Clear at the server root path (e.g. "/litellm") too, since the server-set
  // auth cookie is scoped there when SERVER_ROOT_PATH is configured.
  const serverRootPath = uiCookiePath.replace(/\/ui$/, "");
  if (serverRootPath && !paths.includes(serverRootPath)) {
    paths.push(serverRootPath);
  }

  // Add the current path directory if it's different from root and /ui
  if (currentPath && currentPath !== "/" && !currentPath.startsWith("/ui")) {
    const dirPath = currentPath.substring(0, currentPath.lastIndexOf("/") + 1);
    if (dirPath && !paths.includes(dirPath)) {
      paths.push(dirPath);
    }
  }

  const sameSiteValues = ["Lax", "Strict", "None"];

  paths.forEach((path) => {
    // Basic clearing
    document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path};`;

    // With domain
    document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain};`;

    // Try different SameSite values
    sameSiteValues.forEach((sameSite) => {
      const secureFlag = sameSite === "None" ? " Secure;" : "";
      document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; SameSite=${sameSite};${secureFlag}`;
      document.cookie = `token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=${path}; domain=${domain}; SameSite=${sameSite};${secureFlag}`;
    });
  });

  try {
    sessionStorage.removeItem("token");
  } catch {
    // sessionStorage may be unavailable
  }

  clearAllMcpTokens();
}

/**
 * Stores the login token so the UI can read it even when a reverse proxy
 * (e.g. nginx-ingress) adds HttpOnly to the server-set cookie.
 *
 * Strategy:
 *  1. Set a JS-accessible cookie at path "/ui". Because nginx only modifies
 *     server-set Set-Cookie headers, a cookie created via document.cookie will
 *     never carry HttpOnly. Using path "/ui" avoids colliding with the
 *     server-set HttpOnly cookie at path "/".
 *  2. Also store in sessionStorage as a secondary fallback.
 */
export function storeLoginToken(token: string) {
  if (typeof window === "undefined") return;
  if (!token || !token.trim()) return;

  // 1. JS-accessible cookie at /ui — survives same-tab navigations and
  //    is readable by getCookie() via document.cookie.
  try {
    const secure = window.location.protocol === "https:" ? "; Secure" : "";
    const cookiePath = getUiCookiePath();
    document.cookie = `token=${encodeURIComponent(token)}; path=${cookiePath}; SameSite=Lax${secure}`;
  } catch {
    // cookie setting may fail in restrictive environments
  }

  // 2. sessionStorage backup
  try {
    sessionStorage.setItem("token", token);
  } catch {
    // sessionStorage may be unavailable (e.g. private browsing quota exceeded)
  }
}

/**
 * Reads a cookie value directly from document.cookie with no fallback.
 *
 * Use this in flows that decide whether to redirect on the basis of "is the user
 * still authenticated?". sessionStorage is per-origin and survives a logout
 * triggered from a different origin (e.g. dev UI on :3000 cannot reach
 * sessionStorage on the proxy origin :4000), which produces an infinite
 * logout/login redirect.
 */
export function getCookieFromDocument(name: string) {
  if (typeof document === "undefined") return null;
  const row = document.cookie.split("; ").find((r) => r.startsWith(name + "="));
  if (!row) return null;
  const raw = row.split("=").slice(1).join("=");
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

/**
 * Gets a cookie value by name
 * @param name The name of the cookie to retrieve
 * @returns The cookie value or null if not found
 */
export function getCookie(name: string) {
  const fromCookie = getCookieFromDocument(name);
  if (fromCookie !== null) return fromCookie;
  // Fallback to sessionStorage — covers the case where a reverse proxy
  // added HttpOnly to the server-set cookie, making it invisible to JS.
  if (name === "token" && typeof window !== "undefined") {
    try {
      return sessionStorage.getItem(name);
    } catch {
      return null;
    }
  }
  return null;
}
