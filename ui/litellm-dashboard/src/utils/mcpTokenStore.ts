/**
 * Session-storage-backed OAuth token store for MCP servers.
 * Tokens are keyed by LiteLLM user id + server_id and cleared when the browser
 * session ends (tab/window close). Never written to localStorage.
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

function storageKey(serverId: string, userId?: string | null): string {
  const userPart = userId?.trim() || "_anonymous";
  return `${KEY_PREFIX}${userPart}:${serverId}`;
}

export function setToken(
  serverId: string,
  data: TokenInput,
  userId?: string | null,
): void {
  if (typeof window === "undefined") return;
  const stored: StoredToken = {
    access_token: data.access_token,
    expires_at: Date.now() + (data.expires_in != null ? data.expires_in * 1000 : DEFAULT_TTL_MS),
    token_type: data.token_type ?? "bearer",
    ...(data.refresh_token ? { refresh_token: data.refresh_token } : {}),
  };
  try {
    window.sessionStorage.setItem(storageKey(serverId, userId), JSON.stringify(stored));
  } catch {
    // Silently ignore storage errors (private browsing, quota exceeded, etc.)
  }
}

export function getToken(
  serverId: string,
  userId?: string | null,
): StoredToken | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(storageKey(serverId, userId));
    if (!raw) return null;
    return JSON.parse(raw) as StoredToken;
  } catch {
    return null;
  }
}

export function removeToken(serverId: string, userId?: string | null): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(storageKey(serverId, userId));
  } catch {
    // Silently ignore
  }
}

export function isTokenValid(serverId: string, userId?: string | null): boolean {
  const token = getToken(serverId, userId);
  if (!token) return false;
  return token.expires_at > Date.now();
}

/** Remove all MCP session tokens (e.g. on logout or user switch). */
export function clearAllMcpTokens(): void {
  if (typeof window === "undefined") return;
  try {
    const keysToRemove: string[] = [];
    for (let i = 0; i < window.sessionStorage.length; i++) {
      const key = window.sessionStorage.key(i);
      if (key?.startsWith(KEY_PREFIX)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((key) => window.sessionStorage.removeItem(key));
  } catch {
    // Silently ignore
  }
}
