/**
 * Utility functions for managing cookies
 */

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
  const match = window.location.pathname.match(/\/ui(?=\/|$)/);
  if (match && match.index !== undefined) {
    return window.location.pathname.substring(0, match.index + 3);
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
 * Gets a cookie value by name
 * @param name The name of the cookie to retrieve
 * @returns The cookie value or null if not found
 */
export function getCookie(name: string) {
  if (typeof document === "undefined") return null;
  const row = document.cookie.split("; ").find((r) => r.startsWith(name + "="));
  if (row) {
    const raw = row.split("=").slice(1).join("=");
    try {
      return decodeURIComponent(raw);
    } catch {
      return raw;
    }
  }
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
