/**
 * Utility functions for managing return URLs during authentication flows.
 *
 * When a user is redirected to login, we store the original URL so they can be
 * redirected back after successful authentication.
 *
 * NOTE: We use cookies instead of sessionStorage because the SSO flow may cross
 * different ports (e.g., localhost:3000 -> localhost:4000), and sessionStorage
 * is not shared across different origins. Cookies on the same hostname are shared
 * across different ports.
 */

const RETURN_URL_COOKIE_NAME = "litellm_return_url";
const RETURN_URL_PARAM = "redirect_to";

/**
 * Gets the current URL with all query parameters.
 * Returns null if running on server-side.
 */
export function getCurrentUrl(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.location.href;
}

/**
 * Sets a cookie with the given name and value.
 * Automatically adds Secure flag when running over HTTPS.
 */
function setCookie(name: string, value: string, maxAgeSeconds: number = 300): void {
  if (typeof document === "undefined") {
    return;
  }
  // Set cookie with path=/ so it's available across all paths
  // Use SameSite=Lax to allow the cookie to be sent on navigation from external sites (SSO redirect)
  // Add Secure flag when running over HTTPS to prevent cookie from being sent over unencrypted connections
  const isSecure = typeof window !== "undefined" && window.location.protocol === "https:";
  const secureFlag = isSecure ? "; Secure" : "";
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAgeSeconds}; SameSite=Lax${secureFlag}`;
}

/**
 * Gets a cookie value by name.
 */
function getCookie(name: string): string | null {
  if (typeof document === "undefined") {
    return null;
  }
  const match = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
  if (match) {
    try {
      return decodeURIComponent(match[2]);
    } catch {
      return match[2];
    }
  }
  return null;
}

/**
 * Deletes a cookie by name.
 */
function deleteCookie(name: string): void {
  if (typeof document === "undefined") {
    return;
  }
  document.cookie = `${name}=; path=/; max-age=0`;
}

/**
 * Stores the current URL in a cookie before redirecting to login.
 * This allows us to redirect the user back to their original destination after login.
 * Cookie expires in 5 minutes (300 seconds).
 */
export function storeReturnUrl(): void {
  if (typeof window === "undefined") {
    return;
  }

  const currentUrl = getCurrentUrl();
  if (currentUrl) {
    setCookie(RETURN_URL_COOKIE_NAME, currentUrl, 300);
  }
}

/**
 * Retrieves the stored return URL from the cookie.
 * Returns null if no return URL is stored or if running on server-side.
 */
export function getStoredReturnUrl(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return getCookie(RETURN_URL_COOKIE_NAME);
}

/**
 * Clears the stored return URL from the cookie.
 * Should be called after redirecting to the return URL.
 */
export function clearStoredReturnUrl(): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    deleteCookie(RETURN_URL_COOKIE_NAME);
  } catch (error) {
    console.error("Failed to clear return URL cookie:", error);
  }
}

/**
 * Gets the return URL from URL query parameters.
 * Used when the return URL is passed via query string to the login page.
 */
export function getReturnUrlFromParams(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  const searchParams = new URLSearchParams(window.location.search);
  return searchParams.get(RETURN_URL_PARAM);
}

/**
 * Builds a login URL with the return URL as a query parameter.
 *
 * @param baseLoginUrl - The base login URL (e.g., "/ui/login")
 * @param returnUrl - The URL to redirect to after login (defaults to current URL)
 */
export function buildLoginUrlWithReturn(baseLoginUrl: string, returnUrl?: string): string {
  const url = returnUrl || getCurrentUrl();

  if (!url) {
    return baseLoginUrl;
  }

  // Don't add return URL if we're already on the login page
  if (url.includes("/login")) {
    return baseLoginUrl;
  }

  const separator = baseLoginUrl.includes("?") ? "&" : "?";
  return `${baseLoginUrl}${separator}${RETURN_URL_PARAM}=${encodeURIComponent(url)}`;
}

/**
 * Gets the best return URL to use after login.
 * Priority:
 * 1. URL query parameter (redirect_to)
 * 2. Cookie
 * 3. null (caller should use default)
 */
export function getReturnUrl(): string | null {
  // First check URL params
  const paramUrl = getReturnUrlFromParams();
  if (paramUrl) {
    return paramUrl;
  }

  // Then check cookie
  const storedUrl = getStoredReturnUrl();
  if (storedUrl) {
    return storedUrl;
  }

  return null;
}

/**
 * Checks if we're running in a development environment.
 * Returns true for localhost, 127.0.0.1, IPv6 localhost, or .local domains.
 * This determines whether cross-port redirects are allowed (dev only).
 */
function isDevEnvironment(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  const hostname = window.location.hostname;
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "::1" ||
    hostname.startsWith("127.") || // Full IPv4 loopback range (127.0.0.0/8)
    hostname.endsWith(".local") // Common dev domain suffix
  );
}

/**
 * Validates a return URL to prevent open redirect attacks.
 * - Always allows relative URLs (starting with / but not //)
 * - In dev (localhost): allows same hostname with any port
 * - In production: requires exact origin match (protocol + hostname + port)
 *
 * @param url - The URL to validate
 * @returns true if the URL is safe to redirect to
 */
export function isValidReturnUrl(url: string): boolean {
  if (!url) {
    return false;
  }

  // Allow relative URLs
  if (url.startsWith("/") && !url.startsWith("//")) {
    return true;
  }

  // For absolute URLs, validate against current origin
  if (typeof window === "undefined") {
    return false;
  }

  try {
    const returnUrlObj = new URL(url);
    const currentHostname = window.location.hostname;

    // Hostname must always match
    if (returnUrlObj.hostname !== currentHostname) {
      return false;
    }

    // In dev environments (localhost), allow any port on the same hostname
    // This supports SSO flows that cross ports (e.g., localhost:3000 -> localhost:4000)
    if (isDevEnvironment()) {
      return true;
    }

    // In production, require exact origin match (protocol + hostname + port)
    return returnUrlObj.origin === window.location.origin;
  } catch {
    // Invalid URL
    return false;
  }
}

export function normalizeUrlForCompare(url: string): string {
  if (typeof window === "undefined") {
    return url;
  }

  try {
    const parsed = new URL(url, window.location.origin);
    let pathname = parsed.pathname;
    if (pathname.length > 1 && pathname.endsWith("/")) {
      pathname = pathname.slice(0, -1);
    }

    const params = new URLSearchParams(parsed.search);
    const sortedParams = new URLSearchParams();
    Array.from(params.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .forEach(([key, value]) => {
        sortedParams.append(key, value);
      });

    const search = sortedParams.toString();
    const hash = parsed.hash || "";
    return `${parsed.origin}${pathname}${search ? `?${search}` : ""}${hash}`;
  } catch {
    return url;
  }
}

/**
 * Gets and clears the return URL in one operation.
 * Returns the validated return URL or null if invalid/not found.
 *
 * Priority:
 * 1. If redirect_to param is valid, use it and clear cookie
 * 2. If redirect_to param is invalid/missing, check cookie
 * 3. Only clear cookie when we have a valid URL to return
 */
export function consumeReturnUrl(): string | null {
  // Check URL param first
  const paramUrl = getReturnUrlFromParams();
  if (paramUrl) {
    if (isValidReturnUrl(paramUrl)) {
      clearStoredReturnUrl();
      return paramUrl;
    }
    // Log rejected URLs in development for debugging
    if (isDevEnvironment()) {
      console.warn("[returnUrlUtils] Invalid return URL in params rejected:", paramUrl);
    }
  }

  // Fall back to cookie
  const storedUrl = getStoredReturnUrl();
  if (storedUrl) {
    if (isValidReturnUrl(storedUrl)) {
      clearStoredReturnUrl();
      return storedUrl;
    }
    // Log rejected URLs in development for debugging
    if (isDevEnvironment()) {
      console.warn("[returnUrlUtils] Invalid return URL in cookie rejected:", storedUrl);
    }
  }

  // No valid URL found - don't clear cookie (nothing to clear or already invalid)
  return null;
}
