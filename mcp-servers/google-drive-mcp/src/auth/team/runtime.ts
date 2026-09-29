// ---------------------------------------------------------------------------
// Team-mode runtime assembly: config → store + Google IdP client + provider +
// callback handler, plus the periodic sweeper that bounds store growth.
// Tests inject an in-memory store and a fake IdP through `overrides`.
// ---------------------------------------------------------------------------

import type { RequestHandler } from 'express';
import { loadWebCredentials } from '../client.js';
import { describeErrorForLog } from '../utils.js';
import { makeGoogleCallbackHandler } from './callback.js';
import { TeamClientFactory } from './clientFactory.js';
import type { TeamConfig } from './config.js';
import { FileTeamStore } from './fileStore.js';
import { GoogleIdp, GoogleIdpClient } from './googleIdp.js';
import { InMemoryTeamStore } from './memoryStore.js';
import { TeamOAuthProvider } from './provider.js';
import type { TokenRefreshConfig } from '../tokenRefresh.js';
import type { TeamStore } from './types.js';

const SWEEP_INTERVAL_MS = 60 * 1000;

export interface TeamRuntime {
  config: TeamConfig;
  store: TeamStore;
  idp: GoogleIdp;
  provider: TeamOAuthProvider;
  clientFactory: TeamClientFactory;
  callbackHandler: RequestHandler;
  /** Stops the background sweep timer (tests, shutdown). */
  stop(): void;
  /** Drain queued durable store writes so nothing is lost on shutdown. */
  flush(): Promise<void>;
}

export interface TeamRuntimeOverrides {
  store?: TeamStore;
  idp?: GoogleIdp;
  /** Google OAuth client pair for the per-user client factory. Defaults to the
   * loaded web credentials; tests with a fake IdP get inert placeholders. */
  clientCredentials?: { client_id: string; client_secret: string };
  /** Timeout/retry policy for per-user Google token refreshes (the runtime
   * config's `tokenRefreshTimeout` / `retryMax` / `retryBaseDelay`). */
  tokenRefresh?: TokenRefreshConfig;
}

export async function createTeamRuntime(
  config: TeamConfig,
  overrides: TeamRuntimeOverrides = {},
): Promise<TeamRuntime> {
  const store =
    overrides.store ??
    (config.store === 'file' ? new FileTeamStore(config.storePath) : new InMemoryTeamStore());
  await store.init();

  let idp = overrides.idp;
  let credentials = overrides.clientCredentials;
  if (!idp) {
    credentials ??= await loadWebCredentials(config.googleRedirectUri);
    idp = new GoogleIdpClient({
      clientId: credentials.client_id,
      clientSecret: credentials.client_secret,
      redirectUri: config.googleRedirectUri,
      scopes: config.googleScopes,
      hdHint: config.allowedDomains.length === 1 ? config.allowedDomains[0] : undefined,
    });
  }
  // A faked IdP (tests) never reaches Google, so placeholder credentials are safe.
  credentials ??= { client_id: 'unused-test-client', client_secret: 'unused-test-secret' };

  const provider = new TeamOAuthProvider({ store, idp, config });
  const clientFactory = new TeamClientFactory(store, credentials, overrides.tokenRefresh);
  const callbackHandler = makeGoogleCallbackHandler({
    store,
    idp,
    config,
    // A re-authorization replaces the Google grant; cached clients built from
    // the old grant must not survive it.
    onUserAuthorized: (sub) => clientFactory.evict(sub),
  });

  const sweeper = setInterval(() => {
    store.sweepExpired().catch((err) => {
      console.error(`[team-auth] Store sweep failed: ${describeErrorForLog(err)}`);
    });
  }, SWEEP_INTERVAL_MS);
  sweeper.unref();

  return {
    config,
    store,
    idp,
    provider,
    clientFactory,
    callbackHandler,
    stop: () => clearInterval(sweeper),
    flush: () => store.flush(),
  };
}
