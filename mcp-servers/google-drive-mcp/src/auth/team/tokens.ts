// ---------------------------------------------------------------------------
// Token minting for team mode. All secrets are 256-bit crypto-random values;
// prefixes make leaked tokens greppable/attributable without weakening them.
// Only SHA-256 hashes of these values are ever persisted (no salt needed —
// these are high-entropy random values, not passwords).
// ---------------------------------------------------------------------------

import * as crypto from 'crypto';

export const ACCESS_TOKEN_PREFIX = 'mcp_at_';
export const REFRESH_TOKEN_PREFIX = 'mcp_rt_';
export const AUTH_CODE_PREFIX = 'mcp_ac_';

function randomToken(prefix: string): string {
  return prefix + crypto.randomBytes(32).toString('base64url');
}

export function mintAccessToken(): string {
  return randomToken(ACCESS_TOKEN_PREFIX);
}

export function mintRefreshToken(): string {
  return randomToken(REFRESH_TOKEN_PREFIX);
}

export function mintAuthorizationCode(): string {
  return randomToken(AUTH_CODE_PREFIX);
}

/** State parameter for the Google hop. */
export function mintState(): string {
  return crypto.randomBytes(32).toString('base64url');
}

export function newFamilyId(): string {
  return crypto.randomUUID();
}

export function sha256Hex(value: string): string {
  return crypto.createHash('sha256').update(value).digest('hex');
}
