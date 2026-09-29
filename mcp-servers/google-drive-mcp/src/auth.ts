// Main authentication module that re-exports and orchestrates the modular components
import { OAuth2Client } from 'google-auth-library';
import { AccountClientFactory } from './auth/accountClientFactory.js';
import { AccountResolver } from './auth/accountResolver.js';
import { AccountStore, buildRecordFromV1 } from './auth/accountStore.js';
import { initializeOAuth2Client } from './auth/client.js';
import {
  createExternalOAuth2Client,
  createServiceAccountAuth,
  describeBypassedTokens,
  getActiveAuthMode,
  validateExternalTokenConfig,
} from './auth/externalAuth.js';
import { getSecureTokenPath } from './auth/utils.js';
import { resolveOAuthScopes } from './auth/scopes.js';
import { AuthServer } from './auth/server.js';
import { SessionStore } from './auth/sessionStore.js';
import {
  DEFAULT_TOKEN_REFRESH_CONFIG,
  installTokenEndpointTimeout,
  type TokenRefreshConfig,
} from './auth/tokenRefresh.js';
import { AccountRecord, AuthMode } from './auth/types.js';
import { existsSync } from 'fs';

export { initializeOAuth2Client } from './auth/client.js';
export { AuthServer } from './auth/server.js';
export { SCOPE_ALIASES, SCOPE_PRESETS, DEFAULT_SCOPES, resolveOAuthScopes } from './auth/scopes.js';
export {
  isServiceAccountMode, createServiceAccountAuth,
  isExternalTokenMode, validateExternalTokenConfig,
  createExternalOAuth2Client,
} from './auth/externalAuth.js';
export { AccountStore } from './auth/accountStore.js';
export { AccountResolver } from './auth/accountResolver.js';
export { AccountClientFactory } from './auth/accountClientFactory.js';
export { SessionStore, STDIO_SESSION_ID } from './auth/sessionStore.js';
export type {
  AccountRecord,
  AccountTargeting,
  AuthMode,
  RedactedAccountView,
  SessionState,
  ToolOpKind,
} from './auth/types.js';

export interface AuthSystem {
  mode: AuthMode;
  store: AccountStore;
  factory: AccountClientFactory;
  resolver: AccountResolver;
  sessions: SessionStore;
}

/**
 * When an env var forces service-account/external-token mode, the user's
 * authenticated `tokens.json` is silently ignored. If such a file exists, say
 * so loudly on stderr — this is the trap behind issue #137, where a broad
 * `auth/drive` token appears valid yet every Drive call comes back empty
 * because the process is actually acting as an empty service account.
 */
function warnIfLocalTokensBypassed(): void {
  const mode = getActiveAuthMode();
  if (mode === 'oauth') return;
  try {
    const tokenPath = getSecureTokenPath();
    const msg = describeBypassedTokens(mode, tokenPath, existsSync(tokenPath));
    if (msg) console.error(`⚠  ${msg}`);
  } catch {
    // A diagnostic warning must never break authentication.
  }
}

/**
 * Build the multi-account auth system.
 *
 * - Detects mode from env (service account > external token > local OAuth).
 * - Constructs AccountStore/Factory/Resolver/Sessions.
 * - Seeds synthetic accounts for non-local-OAuth modes.
 * - For local OAuth: loads v2 tokens.json (or migrates v1).
 *   If no accounts are registered, runs the interactive auth flow via AuthServer.
 */
export async function buildAuthSystem(
  opts: {
    interactiveIfEmpty?: boolean;
    openBrowser?: boolean;
    /** Timeout/retry policy for OAuth token refreshes; defaults to the
     * runtime defaults (15s per attempt, one retry). */
    tokenRefresh?: TokenRefreshConfig;
  } = {},
): Promise<AuthSystem> {
  console.error('Initializing authentication...');

  const mode = getActiveAuthMode();
  const tokenRefresh = opts.tokenRefresh ?? DEFAULT_TOKEN_REFRESH_CONFIG;

  if (mode === 'service_account') {
    warnIfLocalTokensBypassed();
    const client = await createServiceAccountAuth();
    const store = new AccountStore({ mode: 'service-account' });
    await store.reload();
    store.setSyntheticAccount(buildSyntheticRecord('service-account'), client);
    return assembleSystem('service-account', store, tokenRefresh);
  }

  if (mode === 'external_token') {
    warnIfLocalTokensBypassed();
    validateExternalTokenConfig();
    const client = createExternalOAuth2Client();
    // This client bypasses the factory, so its refreshes (with a configured
    // refresh token) are the library's own implicit ones; bound those too.
    installTokenEndpointTimeout(client, tokenRefresh);
    const store = new AccountStore({ mode: 'external-token' });
    await store.reload();
    store.setSyntheticAccount(buildSyntheticRecord('external-token'), client);
    return assembleSystem('external-token', store, tokenRefresh);
  }

  // Local OAuth mode
  const store = new AccountStore({ mode: 'local-oauth' });
  try {
    await store.reload();
  } catch (err) {
    // A corrupt/unreadable tokens.json must not brick every request. Move the bad
    // file aside (non-destructive) and start empty so the first-time auth flow
    // below can re-establish credentials — restoring the pre-PR self-heal.
    const backup = await store.quarantineCorruptFile();
    console.error(
      `Authentication: tokens.json was unreadable (${(err as Error).message}). ` +
        (backup ? `Moved it to ${backup}. ` : '') +
        'Starting fresh — you will be prompted to re-authenticate.',
    );
  }

  if (store.list().length === 0) {
    if (opts.interactiveIfEmpty === false) {
      // Caller (e.g. the `auth` CLI) drives its own additive consent flow, so
      // don't launch the legacy first-time browser auth here.
      return assembleSystem('local-oauth', store, tokenRefresh);
    }
    // First-time auth: run the interactive browser flow. Tokens are persisted
    // additively into the v2 store under the reserved alias 'default' — the
    // flat v1 file is never written.
    const oauth2Client = await initializeOAuth2Client();
    const authServer = new AuthServer(oauth2Client, {
      onTokens: async (tokens) => {
        // Same record a v1→v2 migration would have produced: alias 'default',
        // pendingIdentity, sub derived from the token material, scope from
        // tokens.scope (DEFAULT_SCOPES fallback). A throw here renders the
        // failure page and keeps authCompletedSuccessfully false.
        const record = buildRecordFromV1(tokens);
        await store.upsert(record);
        if (!store.getDefault()) {
          await store.setDefault(record.alias);
        }
      },
    });
    const started = await authServer.start(opts.openBrowser ?? true);
    if (!started) {
      throw new Error('Authentication failed. Please check your credentials and try again.');
    }
    // Wait for the OAuth callback. authCompletedSuccessfully flips only after
    // onTokens has resolved, so the store already holds the persisted account
    // — no re-read or token validation needed here (the factory refreshes
    // lazily on first client use).
    await new Promise<void>((resolve) => {
      const poll = setInterval(async () => {
        if (authServer.authCompletedSuccessfully) {
          clearInterval(poll);
          await authServer.stop();
          resolve();
        }
      }, 1000);
    });
  } else {
    console.error(`Authentication: loaded ${store.list().length} account(s) from ${store.getFilePath()}`);
  }

  return assembleSystem('local-oauth', store, tokenRefresh);
}

function assembleSystem(
  mode: AuthMode,
  store: AccountStore,
  tokenRefresh: TokenRefreshConfig,
): AuthSystem {
  const sessions = new SessionStore();
  const factory = new AccountClientFactory(store, tokenRefresh);
  const resolver = new AccountResolver(store, sessions);
  return { mode, store, factory, resolver, sessions };
}

function buildSyntheticRecord(alias: 'service-account' | 'external-token'): AccountRecord {
  const scope = resolveOAuthScopes().join(' ');
  const now = new Date().toISOString();
  return {
    alias,
    email: 'unknown',
    sub: `synthetic:${alias}`,
    accessToken: '',
    refreshToken: '',
    scope,
    tokenType: 'Bearer',
    expiryDate: 0,
    addedAt: now,
    lastRefreshedAt: now,
    pendingIdentity: true,
  };
}

/**
 * Authenticate and return the active OAuth2Client.
 *
 * Back-compat shim: callers that only want the raw client (e.g. tests, the
 * service-account priority test) keep working. New callers should use
 * `buildAuthSystem()` and pull clients from the factory per-alias.
 */
export async function authenticate(): Promise<OAuth2Client> {
  // Preserve legacy fast paths used by tests: synthetic-mode callers expect
  // the mode-specific client directly without touching AccountStore.
  const mode = getActiveAuthMode();
  if (mode === 'service_account') {
    return (await createServiceAccountAuth()) as OAuth2Client;
  }
  if (mode === 'external_token') {
    validateExternalTokenConfig();
    return createExternalOAuth2Client();
  }

  const system = await buildAuthSystem();
  const defaultAlias = system.store.getDefault() ?? system.store.list()[0]?.alias;
  if (!defaultAlias) {
    throw new Error('Authentication completed but no active account is available.');
  }
  return system.factory.getClient(defaultAlias);
}

