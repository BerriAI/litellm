// ---------------------------------------------------------------------------
// Team-mode configuration: env vars / CLI flags → validated TeamConfig.
//
// Everything here is validated at startup so a misconfigured deployment fails
// at boot with an actionable message, never on a user's first sign-in.
// ---------------------------------------------------------------------------

import * as path from 'path';
import { isExternalTokenMode, isServiceAccountMode } from '../externalAuth.js';
import { resolveOAuthScopes, USERINFO_SCOPES } from '../scopes.js';
import { getSecureTokenPath } from '../utils.js';

export const GOOGLE_CALLBACK_PATH = '/oauth/google/callback';

export interface TeamConfig {
  issuerUrl: URL;
  /** Derived: <issuer>/oauth/google/callback — registered at Google. */
  googleRedirectUri: string;
  /** Lowercased Workspace domains; empty = any Google account may join. */
  allowedDomains: string[];
  /** DCR redirect-URI allowlist (exact match); empty = open registration. */
  allowedRedirectUris: string[];
  /** MCP access-token TTL in ms. */
  tokenTtlMs: number;
  store: 'memory' | 'file';
  storePath: string;
  /** Express `trust proxy` hop count; undefined = not behind a proxy. */
  trustProxy?: number;
  /** Host-header allowlist for createMcpExpressApp (issuer hostname + extras). */
  allowedHosts: string[];
  /** Scopes requested from Google per user (configured set + userinfo). */
  googleScopes: string[];
  /** Scopes advertised to MCP clients (configured set, without userinfo). */
  advertisedScopes: string[];
}

export function isLoopbackHost(hostname: string): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]' || hostname === '::1';
}

function splitCsv(value: string | undefined): string[] {
  return (value ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Build and validate the team-mode configuration. Throws with an actionable
 * message on any misconfiguration; the CLI layer turns that into exit(1).
 */
export function loadTeamConfig(opts: {
  transport: string;
  issuerUrlArg?: string;
  env?: NodeJS.ProcessEnv;
}): TeamConfig {
  const env = opts.env ?? process.env;

  if (opts.transport !== 'http') {
    throw new Error('Team mode requires the HTTP transport. Start with --transport http.');
  }
  if (isServiceAccountMode()) {
    throw new Error(
      'Team mode is incompatible with service-account mode. Unset GOOGLE_APPLICATION_CREDENTIALS.',
    );
  }
  if (isExternalTokenMode()) {
    throw new Error(
      'Team mode is incompatible with external-token mode. Unset GOOGLE_DRIVE_MCP_ACCESS_TOKEN.',
    );
  }

  const rawIssuer = opts.issuerUrlArg ?? env.MCP_TEAM_ISSUER_URL;
  if (!rawIssuer) {
    throw new Error(
      'Team mode requires the public issuer URL. Set --issuer-url <url> or MCP_TEAM_ISSUER_URL ' +
        '(e.g. https://drive-mcp.example.com).',
    );
  }
  let issuerUrl: URL;
  try {
    issuerUrl = new URL(rawIssuer);
  } catch {
    throw new Error(`Invalid issuer URL: "${rawIssuer}".`);
  }
  if (issuerUrl.search || issuerUrl.hash) {
    throw new Error('The issuer URL must not contain a query string or fragment.');
  }
  if (issuerUrl.pathname !== '/') {
    // Team mode mounts its OAuth endpoints (and the Google callback) at the host
    // root; `new URL('/oauth/google/callback', issuer)` would silently drop a
    // base path, so the derived redirect URI would not match what a reverse
    // proxy forwards. Fail fast rather than mint an unusable callback URL.
    throw new Error(
      `The issuer URL must not contain a path (got "${issuerUrl.pathname}"). Team mode serves ` +
        'its OAuth endpoints at the host root; use a dedicated host/subdomain instead.',
    );
  }
  // Mirror the SDK's checkIssuerUrl so this fails at boot, not first request.
  if (issuerUrl.protocol !== 'https:' && !isLoopbackHost(issuerUrl.hostname)) {
    throw new Error(
      `The issuer URL must use https (got "${rawIssuer}"). Plain http is allowed only for ` +
        'localhost development.',
    );
  }

  const tokenTtlSeconds = parseInt(env.MCP_TEAM_TOKEN_TTL ?? '3600', 10);
  if (!Number.isFinite(tokenTtlSeconds) || tokenTtlSeconds < 60 || tokenTtlSeconds > 86_400) {
    throw new Error(
      `Invalid MCP_TEAM_TOKEN_TTL: "${env.MCP_TEAM_TOKEN_TTL}". Expected seconds in [60, 86400].`,
    );
  }

  const store = env.MCP_TEAM_STORE ?? 'file';
  if (store !== 'file' && store !== 'memory') {
    throw new Error(`Invalid MCP_TEAM_STORE: "${store}". Expected "file" or "memory".`);
  }
  const storePath = env.MCP_TEAM_STORE_PATH
    ? path.resolve(env.MCP_TEAM_STORE_PATH)
    : path.join(path.dirname(getSecureTokenPath()), 'team-store.json');

  const allowedDomains = splitCsv(env.MCP_TEAM_ALLOWED_DOMAINS).map((d) => d.toLowerCase());

  const allowedRedirectUris = splitCsv(env.MCP_TEAM_ALLOWED_REDIRECT_URIS);
  for (const uri of allowedRedirectUris) {
    try {
      new URL(uri);
    } catch {
      throw new Error(`Invalid entry in MCP_TEAM_ALLOWED_REDIRECT_URIS: "${uri}".`);
    }
  }

  let trustProxy: number | undefined;
  if (env.MCP_TRUST_PROXY !== undefined) {
    trustProxy = parseInt(env.MCP_TRUST_PROXY, 10);
    if (!Number.isFinite(trustProxy) || trustProxy < 0) {
      throw new Error(
        `Invalid MCP_TRUST_PROXY: "${env.MCP_TRUST_PROXY}". Expected the number of trusted ` +
          'proxy hops (e.g. 1 for Cloud Run).',
      );
    }
  }

  // Without a Host allowlist the SDK applies no host validation at all for
  // non-localhost binds — derive one from the issuer, extendable via env.
  // Passing allowedHosts replaces the SDK's loopback defaults, so a loopback
  // issuer re-adds all loopback spellings (localhost vs 127.0.0.1).
  // Lowercase every entry: the SDK's Host-header check is case-sensitive and
  // compares against `new URL('http://' + host).hostname`, which is always
  // lowercase — so a mixed-case env entry would never match and would 403 every
  // request through that hostname.
  const allowedHosts = [
    ...new Set(
      [
        issuerUrl.hostname,
        ...(isLoopbackHost(issuerUrl.hostname) ? ['localhost', '127.0.0.1', '[::1]'] : []),
        ...splitCsv(env.MCP_HTTP_ALLOWED_HOSTS),
      ].map((h) => h.toLowerCase()),
    ),
  ];

  const advertisedScopes = resolveOAuthScopes();
  return {
    issuerUrl,
    googleRedirectUri: new URL(GOOGLE_CALLBACK_PATH, issuerUrl).href,
    allowedDomains,
    allowedRedirectUris,
    tokenTtlMs: tokenTtlSeconds * 1000,
    store,
    storePath,
    trustProxy,
    allowedHosts,
    // Userinfo scopes are mandatory in team mode: sub/email ARE the identity.
    googleScopes: [...advertisedScopes, ...USERINFO_SCOPES],
    advertisedScopes,
  };
}
