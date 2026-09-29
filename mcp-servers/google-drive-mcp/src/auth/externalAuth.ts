// ---------------------------------------------------------------------------
// External authentication modes: Service Account & pre-obtained OAuth tokens
// ---------------------------------------------------------------------------

import { readFileSync } from 'node:fs';
import { OAuth2Client } from 'google-auth-library';
import { GoogleAuth, GoogleAuthOptions } from 'google-auth-library';
import { resolveOAuthScopes } from './scopes.js';

// ---------------------------------------------------------------------------
// Service Account mode
// ---------------------------------------------------------------------------

/** True when `GOOGLE_APPLICATION_CREDENTIALS` is set (standard Google convention). */
export function isServiceAccountMode(): boolean {
  return !!process.env.GOOGLE_APPLICATION_CREDENTIALS;
}

export type ActiveAuthMode = 'service_account' | 'external_token' | 'oauth';

/**
 * Env vars whose mere presence overrides the local `tokens.json` OAuth flow.
 * Keyed by the mode they force. Used to explain to users *why* their
 * authenticated `tokens.json` is being bypassed (see issue #137).
 */
export const AUTH_MODE_OVERRIDE_ENV_VARS: Record<Exclude<ActiveAuthMode, 'oauth'>, string> = {
  service_account: 'GOOGLE_APPLICATION_CREDENTIALS',
  external_token: 'GOOGLE_DRIVE_MCP_ACCESS_TOKEN',
};

/**
 * The single source of truth for which auth mode `authenticate()` (src/auth.ts)
 * selects, based purely on env-var presence. Service-account and external-token
 * modes take strict priority over the local `tokens.json` OAuth flow;
 * `authenticate()` switches on this value.
 */
export function getActiveAuthMode(): ActiveAuthMode {
  if (isServiceAccountMode()) return 'service_account';
  if (isExternalTokenMode()) return 'external_token';
  return 'oauth';
}

/**
 * The user-facing warning for the issue #137 trap: an override env var forces
 * service-account/external-token mode while an authenticated `tokens.json`
 * exists on disk and is therefore silently ignored. Returns `null` when no such
 * token file exists (nothing is being bypassed). The caller supplies the token
 * path and its existence so both the startup warning and `authGetStatus` emit
 * identical wording; the set of currently-active override env vars is read from
 * `process.env` here (consistent with the mode predicates in this module).
 */
export function describeBypassedTokens(
  mode: Exclude<ActiveAuthMode, 'oauth'>,
  tokenPath: string,
  tokenExists: boolean,
): string | null {
  if (!tokenExists) return null;
  const envVar = AUTH_MODE_OVERRIDE_ENV_VARS[mode];
  // Every override var that is currently set — unsetting only the winning one
  // just hands control to the next override, so tokens.json stays bypassed.
  const setOverrideVars = Object.values(AUTH_MODE_OVERRIDE_ENV_VARS).filter(
    (v) => !!process.env[v],
  );
  const remedy =
    setOverrideVars.length > 1
      ? `Unset ${setOverrideVars.join(' and ')} to use your authenticated Google account`
      : `Unset ${envVar} to use your authenticated Google account`;
  return `The local OAuth token at ${tokenPath} exists but is IGNORED because ` +
    `${envVar} is set (active auth mode: ${mode}). ${remedy} (see issue #137).`;
}

/**
 * Build `GoogleAuth` options from the current environment.
 *
 * When `GOOGLE_DRIVE_MCP_SUBJECT` is set, the returned options include
 * `clientOptions.subject`, which instructs `GoogleAuth` to mint a JWT that
 * impersonates the given user via domain-wide delegation. The Workspace admin
 * must have authorized the service account's client ID for the requested
 * scopes under Security → API controls → Manage Domain-wide Delegation.
 *
 * Scopes are resolved via {@link resolveOAuthScopes}, so `GOOGLE_DRIVE_MCP_SCOPES`
 * narrows the SA's authority the same way it does for interactive OAuth.
 *
 * Exported so tests can assert the shape without hitting the filesystem.
 */
export function buildServiceAccountAuthOptions(): GoogleAuthOptions {
  const keyFile = process.env.GOOGLE_APPLICATION_CREDENTIALS!;
  const subject = process.env.GOOGLE_DRIVE_MCP_SUBJECT?.trim();

  const options: GoogleAuthOptions = {
    keyFile,
    scopes: resolveOAuthScopes(),
  };

  if (subject) {
    options.clientOptions = { subject };
  }

  return options;
}

/**
 * Credential `type` values `GoogleAuth` accepts from a file pointed at by
 * `GOOGLE_APPLICATION_CREDENTIALS`. It is the standard ADC variable, so it may
 * legitimately hold more than a service-account key — e.g. the `authorized_user`
 * file written by `gcloud auth application-default login`, or a workload-identity
 * config. Validation below stays permissive about which of these it is.
 */
const ADC_CREDENTIAL_TYPES = new Set([
  'service_account',
  'authorized_user',
  'external_account',
  'external_account_authorized_user',
  'impersonated_service_account',
  'gdch_service_account',
]);

/**
 * Fail fast on a malformed credentials file.
 *
 * google-auth-library v10 no longer rejects one: `getClient()` *resolves* with a
 * credential-less JWT client when the file is unparseable or missing its fields
 * (only an absent file throws). Without this guard the server would report
 * "Service account authentication successful" while holding no credentials, and
 * every later call would fail with a misleading `Method doesn't allow
 * unregistered callers`. v9 surfaced this immediately, so this restores that.
 *
 * Read errors (ENOENT and friends) propagate untouched — they already name the
 * path and are clear. Messages never include file contents: the file holds a
 * private key.
 */
export function validateCredentialsFile(keyFile: string): void {
  const raw = readFileSync(keyFile, 'utf-8');

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(
      `Credentials file at ${keyFile} is not valid JSON. ` +
        'It may be truncated or corrupted; re-download the key file.',
    );
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`Credentials file at ${keyFile} must contain a JSON object.`);
  }

  const cred = parsed as { type?: unknown; client_email?: unknown; private_key?: unknown };
  const type = typeof cred.type === 'string' ? cred.type : undefined;

  if (type !== undefined && !ADC_CREDENTIAL_TYPES.has(type)) {
    throw new Error(
      `Credentials file at ${keyFile} has unrecognized type "${type}". ` +
        `Expected one of: ${[...ADC_CREDENTIAL_TYPES].join(', ')}.`,
    );
  }

  // Only service-account keys are checked field-by-field; the other ADC shapes
  // carry different fields and are left to GoogleAuth.
  if (type === 'service_account' || type === undefined) {
    const missing = (['client_email', 'private_key'] as const).filter(
      (f) => typeof cred[f] !== 'string' || !(cred[f] as string).trim(),
    );
    if (missing.length) {
      throw new Error(
        `Credentials file at ${keyFile} is missing required field(s): ${missing.join(', ')}. ` +
          'A service account key needs both; re-download it from the Google Cloud console.',
      );
    }
  }
}

/**
 * Create an authorized client from a service account JSON key file.
 * `GoogleAuth` handles JWT signing and token refresh automatically.
 */
export async function createServiceAccountAuth(): Promise<any> {
  const options = buildServiceAccountAuthOptions();
  const subject = process.env.GOOGLE_DRIVE_MCP_SUBJECT?.trim();
  console.error(
    `Using service account credentials from ${options.keyFile}` +
      (subject ? ` (impersonating ${subject} via domain-wide delegation)` : ''),
  );

  validateCredentialsFile(options.keyFile as string);

  const auth = new GoogleAuth(options);
  const client = await auth.getClient();
  console.error('Service account authentication successful');
  return client;
}

// ---------------------------------------------------------------------------
// External OAuth Token mode
// ---------------------------------------------------------------------------

/** True when `GOOGLE_DRIVE_MCP_ACCESS_TOKEN` is set. */
export function isExternalTokenMode(): boolean {
  return !!process.env.GOOGLE_DRIVE_MCP_ACCESS_TOKEN;
}

/**
 * Validate that the env-var combination makes sense.
 * Throws with an actionable message on mis-configuration.
 */
export function validateExternalTokenConfig(): void {
  const accessToken = process.env.GOOGLE_DRIVE_MCP_ACCESS_TOKEN?.trim();
  if (!accessToken) {
    throw new Error(
      'GOOGLE_DRIVE_MCP_ACCESS_TOKEN is set but empty. Provide a valid OAuth access token.'
    );
  }

  const refreshToken = process.env.GOOGLE_DRIVE_MCP_REFRESH_TOKEN?.trim();
  const clientId = process.env.GOOGLE_DRIVE_MCP_CLIENT_ID?.trim();
  const clientSecret = process.env.GOOGLE_DRIVE_MCP_CLIENT_SECRET?.trim();

  if (refreshToken) {
    if (!clientId || !clientSecret) {
      throw new Error(
        'GOOGLE_DRIVE_MCP_REFRESH_TOKEN is set but GOOGLE_DRIVE_MCP_CLIENT_ID and/or ' +
          'GOOGLE_DRIVE_MCP_CLIENT_SECRET are missing. All three are required for automatic token refresh.'
      );
    }
  }

  // Warn about partial client credential sets (one without the other)
  if ((clientId && !clientSecret) || (!clientId && clientSecret)) {
    throw new Error(
      'Both GOOGLE_DRIVE_MCP_CLIENT_ID and GOOGLE_DRIVE_MCP_CLIENT_SECRET must be provided together.'
    );
  }
}

/**
 * Create an OAuth2Client pre-loaded with externally-obtained credentials.
 * When a refresh token + client credentials are provided, the client will
 * auto-refresh transparently.
 */
export function createExternalOAuth2Client(): OAuth2Client {
  const accessToken = process.env.GOOGLE_DRIVE_MCP_ACCESS_TOKEN!.trim();
  const refreshToken = process.env.GOOGLE_DRIVE_MCP_REFRESH_TOKEN?.trim();
  const clientId = process.env.GOOGLE_DRIVE_MCP_CLIENT_ID?.trim();
  const clientSecret = process.env.GOOGLE_DRIVE_MCP_CLIENT_SECRET?.trim();

  const oauth2Client = new OAuth2Client(clientId, clientSecret);

  oauth2Client.setCredentials({
    access_token: accessToken,
    refresh_token: refreshToken || undefined,
  });

  if (!refreshToken) {
    console.error(
      'Warning: No refresh token provided. The access token will not auto-refresh when it expires.'
    );
  } else {
    console.error('External OAuth tokens configured with auto-refresh support.');
  }

  return oauth2Client;
}
