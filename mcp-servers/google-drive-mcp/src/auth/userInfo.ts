// ---------------------------------------------------------------------------
// Userinfo lookup — resolves the stable Google subject id + email for a
// freshly authenticated OAuth2Client. Called by `manage_accounts add` only;
// failures are tolerated (the caller falls back to a `pendingIdentity` record).
// ---------------------------------------------------------------------------

import { OAuth2Client } from 'google-auth-library';

const USERINFO_URL = 'https://www.googleapis.com/oauth2/v3/userinfo';
const DEFAULT_TIMEOUT_MS = 5_000;

export interface UserInfo {
  email: string;
  sub: string;
  /** Google Workspace hosted domain. Absent for consumer accounts — team
   * mode's domain allowlist fails closed on absence. */
  hd?: string;
}

/**
 * Fetch `{ sub, email }` from the OpenID Connect userinfo endpoint.
 *
 * Requires the `openid` and `https://www.googleapis.com/auth/userinfo.email`
 * scopes to be granted on the access token. Times out after `timeoutMs` and
 * throws on any HTTP or parse error — callers should catch and degrade
 * gracefully.
 */
export async function fetchUserInfo(
  client: OAuth2Client,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<UserInfo> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await client.request<{ sub?: string; email?: string; hd?: string }>({
      url: USERINFO_URL,
      method: 'GET',
      signal: controller.signal,
    });
    const { sub, email, hd } = res.data;
    if (!sub || !email) {
      throw new Error(
        `Userinfo response missing sub or email (got keys: ${Object.keys(res.data ?? {}).join(', ')}).`,
      );
    }
    return { sub, email, ...(hd ? { hd } : {}) };
  } finally {
    clearTimeout(timer);
  }
}
