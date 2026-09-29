// ---------------------------------------------------------------------------
// AccountClientFactory — per-alias OAuth2Client cache with refresh dedupe.
//
// - Builds one `OAuth2Client` per alias (from the shared client_id/client_secret
//   in gcp-oauth.keys.json) and caches it.
// - Attaches a `'tokens'` listener that merges refresh results into the
//   underlying AccountStore, so rotating access tokens persist to disk.
// - Dedupes concurrent refreshes of the same alias so N in-flight tool calls
//   trigger at most one refresh request to Google.
// - Bounds every refresh (per-attempt timeout plus one retry, see
//   tokenRefresh.ts) so a stalled token endpoint fails the call fast instead
//   of parking every deduped caller on it (#169).
// - In synthetic modes (service-account / external-token / test), returns
//   the pre-seeded client from the store unchanged.
// ---------------------------------------------------------------------------

import { Credentials, OAuth2Client } from 'google-auth-library';
import { TimeoutError } from '../utils/retry.js';
import { AccountStore } from './accountStore.js';
import { loadCredentials } from './client.js';
import {
  DEFAULT_TOKEN_REFRESH_CONFIG,
  isInvalidGrant,
  REFRESH_BUFFER_MS,
  refreshAccessTokenBounded,
  type TokenRefreshConfig,
} from './tokenRefresh.js';
import { AccountRecord } from './types.js';
import { describeErrorForLog } from './utils.js';

export class AccountClientFactory {
  private clients = new Map<string, OAuth2Client>();
  private inflightRefresh = new Map<string, Promise<void>>();
  private baseCredentialsPromise: Promise<{ client_id: string; client_secret?: string }> | null = null;

  constructor(
    private store: AccountStore,
    private readonly refreshConfig: TokenRefreshConfig = DEFAULT_TOKEN_REFRESH_CONFIG,
  ) {}

  /**
   * Return an OAuth2Client (or GoogleAuth-derived client) for the given alias,
   * validated/refreshed if necessary.
   */
  async getClient(alias: string): Promise<OAuth2Client> {
    const record = this.store.get(alias);
    if (!record) throw new Error(`No account registered with alias "${alias}".`);

    const synthetic = this.store.getSyntheticClient(alias);
    if (synthetic) return synthetic as OAuth2Client;

    const client = await this.ensureClient(alias, record);
    await this.refreshIfNeeded(alias, client, record);
    return client;
  }

  /** Drop a cached client (e.g. after `remove`, or a re-consent that
   *  supersedes the grant it was built from). */
  evict(alias: string): void {
    this.clients.delete(alias);
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private async ensureClient(alias: string, record: AccountRecord): Promise<OAuth2Client> {
    const existing = this.clients.get(alias);
    if (existing) return existing;

    const base = await this.loadBaseCredentials();
    const client = new OAuth2Client({
      clientId: base.client_id,
      clientSecret: base.client_secret || undefined,
    });
    client.setCredentials({
      access_token: record.accessToken || undefined,
      refresh_token: record.refreshToken || undefined,
      expiry_date: record.expiryDate || undefined,
      scope: record.scope || undefined,
      token_type: record.tokenType,
    });

    client.on('tokens', (newCreds) => {
      // Intentionally fire-and-forget: the library emits synchronously in the
      // middle of a request. We persist asynchronously via the write queue.
      this.persistRefreshedTokens(alias, newCreds).catch((err) => {
        // Never log the raw error: a gaxios error would print the request
        // config, including token material.
        console.error(
          `Failed to persist refreshed tokens for "${alias}": ${describeErrorForLog(err)}`,
        );
      });
    });

    this.clients.set(alias, client);
    return client;
  }

  private async loadBaseCredentials(): Promise<{ client_id: string; client_secret?: string }> {
    if (!this.baseCredentialsPromise) {
      this.baseCredentialsPromise = loadCredentials();
    }
    return this.baseCredentialsPromise;
  }

  private async refreshIfNeeded(
    alias: string,
    client: OAuth2Client,
    record: AccountRecord,
  ): Promise<void> {
    const expiry = client.credentials.expiry_date ?? record.expiryDate;
    const needsRefresh = expiry
      ? Date.now() >= expiry - REFRESH_BUFFER_MS
      : !client.credentials.access_token;

    if (!needsRefresh) return;
    if (!client.credentials.refresh_token) {
      // No refresh token — nothing we can do. Let the next API call surface a 401.
      return;
    }

    const inflight = this.inflightRefresh.get(alias);
    if (inflight) return inflight;

    const p = (async () => {
      try {
        const credentials = await refreshAccessTokenBounded(
          client,
          this.refreshConfig,
          alias,
          (message, data) => console.error(`${message}${data ? ` ${JSON.stringify(data)}` : ''}`),
        );
        // The `'tokens'` listener handles persistence. `refreshAccessToken`
        // already calls `setCredentials` internally.
        if (!credentials.access_token) {
          throw new Error('Token refresh returned no access_token.');
        }
      } catch (err) {
        if (isInvalidGrant(err)) {
          // Refresh token revoked or expired. Surface an actionable error instead
          // of an opaque downstream 401 so the user knows how to recover.
          throw new Error(
            `Account '${alias}' authorization was revoked or has expired. ` +
              `Run:  manage_accounts add ${alias}  to reconnect it.`,
          );
        }
        if (err instanceof TimeoutError) {
          // The token endpoint stalled through every attempt. Fail the call now
          // with a clear message: swallowing (as for the transient case below)
          // would hand the caller's API request to the library's own implicit
          // refresh, which would stall on the same endpoint again (#169).
          throw Object.assign(
            new Error(
              `Account '${alias}' could not refresh its access token: Google's token ` +
                `endpoint did not respond within ${this.refreshConfig.tokenRefreshTimeout}ms. ` +
                `Check network/proxy access to oauth2.googleapis.com and retry; ` +
                `--token-refresh-timeout raises the limit.`,
            ),
            { cause: err },
          );
        }
        // Never log the raw error: gaxios embeds the refresh POST body (refresh
        // token + client secret) in err.config.
        console.error(`Token refresh failed for "${alias}": ${describeErrorForLog(err)}`);
        // Transient failure — don't throw; let the caller's API call surface it.
      }
    })().finally(() => {
      this.inflightRefresh.delete(alias);
    });
    this.inflightRefresh.set(alias, p);
    return p;
  }

  private async persistRefreshedTokens(alias: string, newCreds: Credentials): Promise<void> {
    const current = this.store.get(alias);
    if (!current) return; // account was removed while a refresh was in flight
    const updated: AccountRecord = {
      ...current,
      accessToken: newCreds.access_token ?? current.accessToken,
      refreshToken: newCreds.refresh_token ?? current.refreshToken,
      expiryDate: newCreds.expiry_date ?? current.expiryDate,
      scope: newCreds.scope ?? current.scope,
      lastRefreshedAt: new Date().toISOString(),
    };
    await this.store.upsert(updated);
  }
}
