/**
 * Session-storage-backed OAuth token store for MCP servers.
 * Tokens are keyed by server_id and cleared automatically when the browser
 * session ends (tab/window close).  Never written to localStorage.
 */

const KEY_PREFIX = "mcp-session-token:";

interface StoredToken {
  access_token: string;
  expires_at: number;
  refresh_token?: string;
  token_type: string;
}

interface TokenInput {
  access_token: string;
  expires_in?: number;
  refresh_token?: string;
  token_type?: string;
}

const DEFAULT_TTL_MS = 3600 * 1000; // 1 hour

export function setToken(serverId: string, data: TokenInput): void {
  if (typeof window === "undefined") return;
  const stored: StoredToken = {
    access_token: data.access_token,
    expires_at: Date.now() + (data.expires_in != null ? data.expires_in * 1000 : DEFAULT_TTL_MS),
    token_type: data.token_type ?? "bearer",
    ...(data.refresh_token ? { refresh_token: data.refresh_token } : {}),
  };
  try {
    window.sessionStorage.setItem(KEY_PREFIX + serverId, JSON.stringify(stored));
  } catch {
    // Silently ignore storage errors (private browsing, quota exceeded, etc.)
  }
}

export function getToken(serverId: string): StoredToken | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(KEY_PREFIX + serverId);
    if (!raw) return null;
    return JSON.parse(raw) as StoredToken;
  } catch {
    return null;
  }
}

export function removeToken(serverId: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(KEY_PREFIX + serverId);
  } catch {
    // Silently ignore
  }
}

export function isTokenValid(serverId: string): boolean {
  const token = getToken(serverId);
  if (!token) return false;
  return token.expires_at > Date.now();
}
