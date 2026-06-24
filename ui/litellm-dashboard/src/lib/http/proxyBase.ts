import { resolveApiBase } from "./resolveApiBase";

const isLocal = process.env.NODE_ENV === "development";
// In dev, if NEXT_PUBLIC_USE_REWRITES=true the Next.js dev server proxies API calls
// to the backend — use relative URLs (null) so rewrites can intercept them.
const resolveDefaultBase = (fallback: string | null): string | null =>
  process.env.NEXT_PUBLIC_BASE_URL
    ? process.env.NEXT_PUBLIC_BASE_URL
    : isLocal && process.env.NEXT_PUBLIC_USE_REWRITES !== "true"
      ? "http://localhost:4000"
      : fallback;

export const defaultProxyBaseUrl = resolveDefaultBase(null);
const defaultServerRootPath = "/";
export let serverRootPath = defaultServerRootPath;

const WORKER_URL_KEY = "litellm_worker_url";
// If a worker URL is in localStorage, use it as the initial proxyBaseUrl.
// This survives page navigation and the sessionStorage.clear() in user_dashboard.
const _rawWorkerUrl = typeof window !== "undefined" ? window.localStorage.getItem(WORKER_URL_KEY) : null;
// Validate stored worker URL — reject non-HTTP schemes to prevent exfiltration
const _initialWorkerUrl = (() => {
  if (!_rawWorkerUrl) return null;
  try {
    const parsed = new URL(_rawWorkerUrl);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") return _rawWorkerUrl;
  } catch {
    /* invalid URL */
  }
  // Invalid URL in storage — clear it
  if (typeof window !== "undefined") window.localStorage.removeItem(WORKER_URL_KEY);
  return null;
})();

export let proxyBaseUrl: string | null = _initialWorkerUrl ?? defaultProxyBaseUrl;

export const getWindowLocation = () => {
  if (typeof window === "undefined") {
    return null;
  }
  return window.location;
};

export const updateProxyBaseUrl = (serverRootPath: string, receivedProxyBaseUrl: string | null = null) => {
  /**
   * Special function for updating the proxy base url. Should only be called by getUiConfig.
   */
  // If a worker URL is in localStorage, don't let getUiConfig overwrite it
  if (typeof window !== "undefined" && window.localStorage.getItem(WORKER_URL_KEY)) {
    return;
  }
  proxyBaseUrl = resolveApiBase({
    explicitBase: receivedProxyBaseUrl || resolveDefaultBase(getWindowLocation()?.origin ?? null),
    serverRootPath,
  });
};

export const getProxyBaseUrl = (): string => {
  if (proxyBaseUrl) {
    return proxyBaseUrl;
  }
  const browserLocation = getWindowLocation();
  return browserLocation?.origin ?? "";
};

/**
 * Switch API calls to point at a worker (or back to the control plane).
 * Persists to localStorage so it survives page navigation and the
 * sessionStorage.clear() in user_dashboard. Also updates the module-level
 * proxyBaseUrl so in-flight code in this JS execution sees the new value
 * immediately.
 */
function isValidHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export function switchToWorkerUrl(workerUrl: string | null): void {
  if (workerUrl && !isValidHttpUrl(workerUrl)) {
    return;
  }
  if (typeof window !== "undefined") {
    if (workerUrl) {
      window.localStorage.setItem(WORKER_URL_KEY, workerUrl);
    } else {
      window.localStorage.removeItem(WORKER_URL_KEY);
    }
  }
  proxyBaseUrl = workerUrl ?? defaultProxyBaseUrl;
}

// Global variable for the header name
export let globalLitellmHeaderName: string = "Authorization";

// Function to set the global header name
export function setGlobalLitellmHeaderName(headerName: string = "Authorization") {
  console.log(`setGlobalLitellmHeaderName: ${headerName}`);
  globalLitellmHeaderName = headerName;
}

// Function to get the global header name
export function getGlobalLitellmHeaderName(): string {
  return globalLitellmHeaderName;
}
