/**
 * Browser-side PKCE helpers (S5-05).
 *
 * Use:
 *   const session = await beginPkce({ clientId, redirectUri, baseUrl });
 *   window.location.href = session.authorizeUrl;
 *   // ... user comes back to redirectUri with ?code=&state=
 *   const tokens = await completePkce(baseUrl, { code });
 *
 * State persists in sessionStorage (NOT localStorage — per the project's
 * key-storage convention) so it survives the redirect round-trip without
 * leaking through tabs that close in the meantime.
 */

const STORAGE_KEY = "xct_pkce_session";

export interface PkceConfig {
  clientId: string;
  redirectUri: string;
  baseUrl: string;
  scope?: string;
}

export interface PkceSession {
  clientId: string;
  redirectUri: string;
  codeVerifier: string;
  state: string;
  authorizeUrl: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  token_type: "Bearer" | string;
  expires_in: number;
  scope?: string;
}

export async function beginPkce(config: PkceConfig): Promise<PkceSession> {
  const codeVerifier = randomUrlSafe(43);
  const state = randomUrlSafe(22);
  const codeChallenge = await s256Challenge(codeVerifier);

  const params = new URLSearchParams({
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    response_type: "code",
    state,
    code_challenge: codeChallenge,
    code_challenge_method: "S256",
  });
  if (config.scope) params.set("scope", config.scope);

  const authorizeUrl = `${trimRight(config.baseUrl)}/oauth/authorize?${params.toString()}`;

  const session: PkceSession = {
    clientId: config.clientId,
    redirectUri: config.redirectUri,
    codeVerifier,
    state,
    authorizeUrl,
  };

  if (typeof window !== "undefined" && window.sessionStorage) {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }

  return session;
}

export async function completePkce(
  baseUrl: string,
  args: { code: string; state?: string; clientSecret?: string },
): Promise<TokenResponse> {
  const session = loadSession();
  if (!session) {
    throw new Error("No PKCE session in sessionStorage. Did beginPkce() run on this origin?");
  }
  if (args.state !== undefined && args.state !== session.state) {
    throw new Error(`OAuth state mismatch — expected ${session.state} got ${args.state}`);
  }

  const form = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: session.clientId,
    code: args.code,
    code_verifier: session.codeVerifier,
    redirect_uri: session.redirectUri,
  });
  if (args.clientSecret) form.set("client_secret", args.clientSecret);

  const resp = await fetch(`${trimRight(baseUrl)}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`oauth/token failed (${resp.status}): ${body}`);
  }
  // Clear the session — it's single-use.
  if (typeof window !== "undefined" && window.sessionStorage) {
    window.sessionStorage.removeItem(STORAGE_KEY);
  }
  return (await resp.json()) as TokenResponse;
}

export async function refreshAccessToken(
  baseUrl: string,
  args: { clientId: string; refreshToken: string; clientSecret?: string },
): Promise<TokenResponse> {
  const form = new URLSearchParams({
    grant_type: "refresh_token",
    client_id: args.clientId,
    refresh_token: args.refreshToken,
  });
  if (args.clientSecret) form.set("client_secret", args.clientSecret);
  const resp = await fetch(`${trimRight(baseUrl)}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`refresh failed (${resp.status}): ${body}`);
  }
  return (await resp.json()) as TokenResponse;
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

function loadSession(): PkceSession | null {
  if (typeof window === "undefined" || !window.sessionStorage) return null;
  const raw = window.sessionStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PkceSession;
  } catch {
    return null;
  }
}

function trimRight(url: string): string {
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

function randomUrlSafe(length: number): string {
  // Use crypto.getRandomValues so we cover both browser and Node 19+ which
  // ships a global `crypto`.
  const bytes = new Uint8Array(Math.ceil((length * 3) / 4));
  globalThis.crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes).slice(0, length);
}

async function s256Challenge(verifier: string): Promise<string> {
  const data = new TextEncoder().encode(verifier);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

function base64UrlEncode(buf: Uint8Array): string {
  let str = "";
  for (let i = 0; i < buf.length; i++) str += String.fromCharCode(buf[i]);
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
