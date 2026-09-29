// ---------------------------------------------------------------------------
// Bounded OAuth token refresh (issue #169).
//
// `OAuth2Client.refreshAccessToken()` takes no timeout or AbortSignal, and
// google-auth-library dedupes refreshes internally: while a token POST is in
// flight, every further refresh call — a retry, or the library's own implicit
// refresh inside the next API call — re-awaits that same promise. A stalled
// token endpoint can therefore only be handled by actually cancelling the POST;
// a Promise.race on its own would release the caller while leaving every retry
// pinned to the stuck request.
//
// This module does two things:
//  - installs, once per client, a gaxios request interceptor that attaches an
//    AbortSignal to requests aimed at the token endpoint — and nothing else:
//    the same transporter carries every Drive/Docs call, and a large upload
//    must not inherit a 15-second budget;
//  - runs `refreshAccessToken()` under `withRetry`, handing each attempt's
//    signal to that interceptor so a timed-out attempt aborts its POST and the
//    retry sends a fresh one.
//
// Requests the library issues on its own (an API call that crosses the expiry
// buffer, a 401-triggered refresh) get a "floor" signal that fires after the
// same timeout, so no token-endpoint request is ever unbounded.
//
// gaxios is duck-typed so this compiles and behaves the same against gaxios 6
// (google-auth-library 9) and gaxios 7 (google-auth-library 10): both expose
// `interceptors.request`, both honour `config.signal`. Aborting through our own
// AbortController rather than `AbortSignal.timeout` matters: gaxios 7 retries a
// TimeoutError-reasoned abort up to `noResponseRetries` times but never an
// AbortError, and gaxios 6 reports both as AbortError.
// ---------------------------------------------------------------------------

import type { Credentials, OAuth2Client } from 'google-auth-library';
import { RUNTIME_DEFAULTS, type RuntimeConfig } from '../utils/cliArgs.js';
import { withRetry } from '../utils/retry.js';

/** Buffer before access-token expiry that triggers a proactive refresh (ms). */
export const REFRESH_BUFFER_MS = 5 * 60 * 1000;

/** The slice of the runtime config a token refresh needs. */
export type TokenRefreshConfig = Pick<
  RuntimeConfig,
  'tokenRefreshTimeout' | 'retryMax' | 'retryBaseDelay'
>;

export const DEFAULT_TOKEN_REFRESH_CONFIG: TokenRefreshConfig = RUNTIME_DEFAULTS;

/**
 * A refresh is retried at most once regardless of `--retry-max` (0 still
 * disables retries). Two stalled attempts already prove the endpoint is
 * unreachable, and the default budget (2 x 15s plus backoff, ~31s) has to stay
 * under the ~60s tool timeout common in MCP clients — four attempts (~67s)
 * would look exactly like the hang this is meant to bound.
 */
export const MAX_TOKEN_REFRESH_RETRIES = 1;

/**
 * Detect an OAuth `invalid_grant` (refresh token revoked or expired) across the
 * shapes google-auth-library / gaxios surface it in.
 */
export function isInvalidGrant(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const e = err as { response?: { data?: { error?: unknown } }; message?: unknown };
  if (e.response?.data?.error === 'invalid_grant') return true;
  return typeof e.message === 'string' && e.message.includes('invalid_grant');
}

// ---------------------------------------------------------------------------
// gaxios plumbing (duck-typed; see the header for why)
// ---------------------------------------------------------------------------

interface TokenRequestConfig {
  url?: string | URL;
  signal?: AbortSignal;
}

interface RequestInterceptor {
  resolved: (config: TokenRequestConfig) => Promise<TokenRequestConfig>;
}

interface GaxiosLike {
  interceptors: { request: { add(interceptor: RequestInterceptor): unknown } };
}

function isGaxiosLike(candidate: unknown): candidate is GaxiosLike {
  if (!candidate || typeof candidate !== 'object') return false;
  const c = candidate as { interceptors?: { request?: { add?: unknown } } };
  return typeof c.interceptors?.request?.add === 'function';
}

/**
 * google-auth-library 9 exposes the Gaxios instance as `client.gaxios` (its
 * `transporter` is a DefaultTransporter wrapper around `.instance`); in v10
 * `client.transporter` is the Gaxios itself.
 */
function findGaxios(client: OAuth2Client): GaxiosLike | undefined {
  const c = client as unknown as { gaxios?: unknown; transporter?: { instance?: unknown } };
  return [c.gaxios, c.transporter, c.transporter?.instance].find(isGaxiosLike);
}

/** A signal that aborts when either input aborts. (Node 18 has no AbortSignal.any.) */
function linkSignals(existing: AbortSignal | undefined, added: AbortSignal): AbortSignal {
  if (!existing) return added;
  if (existing.aborted) return existing;
  if (added.aborted) return added;
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  existing.addEventListener('abort', onAbort, { once: true });
  added.addEventListener('abort', onAbort, { once: true });
  return controller.signal;
}

/** A signal that aborts after `ms`. The timer never keeps the process alive. */
function floorSignal(ms: number): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms).unref();
  return controller.signal;
}

interface RefreshState {
  /** Signal of the proactive refresh attempt currently in flight, if any. */
  attemptSignal?: AbortSignal;
}

const states = new WeakMap<OAuth2Client, RefreshState>();

/**
 * Bound every request `client` makes to Google's token endpoint. Installs the
 * interceptor once per client (idempotent) and returns the per-client state
 * `refreshAccessTokenBounded` uses to hand its attempt signal over. A
 * `tokenRefreshTimeout` of 0 installs nothing.
 */
export function installTokenEndpointTimeout(
  client: OAuth2Client,
  cfg: TokenRefreshConfig,
): RefreshState {
  let state = states.get(client);
  if (state) return state;
  state = {};
  states.set(client, state);

  const gaxios = findGaxios(client);
  // An unrecognised transporter (none exists in this codebase) still gets the
  // caller-side deadline from withRetry; only the in-flight POST goes uncancelled.
  if (!gaxios || cfg.tokenRefreshTimeout <= 0) return state;

  const tokenUrl = String(client.endpoints.oauth2TokenUrl);
  const shared = state;
  gaxios.interceptors.request.add({
    resolved: async (config) => {
      if (String(config.url) !== tokenUrl) return config;
      // One-shot hand-over: gaxios' own retries re-run the request with this
      // config rather than the interceptor, and a later attempt must never
      // inherit an earlier attempt's signal.
      const signal = shared.attemptSignal ?? floorSignal(cfg.tokenRefreshTimeout);
      shared.attemptSignal = undefined;
      config.signal = linkSignals(config.signal, signal);
      return config;
    },
  });
  return state;
}

/**
 * `client.refreshAccessToken()` with a per-attempt deadline and (at most one)
 * retry. Rejects with `TimeoutError` when the token endpoint stalls through
 * every attempt; any other failure is rethrown untouched, so `invalid_grant`
 * surfaces on the first attempt exactly as before.
 *
 * If the library dedupes this call onto an implicit refresh it already has in
 * flight, the interceptor never sees our attempt; that request is bounded by
 * the floor signal instead, which adds at most one extra timeout.
 */
export async function refreshAccessTokenBounded(
  client: OAuth2Client,
  cfg: TokenRefreshConfig,
  label: string,
  log: (message: string, data?: unknown) => void = () => {},
): Promise<Credentials> {
  const state = installTokenEndpointTimeout(client, cfg);
  return withRetry(
    async (signal) => {
      state.attemptSignal = signal;
      try {
        const { credentials } = await client.refreshAccessToken();
        return credentials;
      } finally {
        // Clear only our own signal: an aborted attempt can settle after the
        // next attempt has already registered its signal.
        if (state.attemptSignal === signal) state.attemptSignal = undefined;
      }
    },
    {
      apiTimeout: cfg.tokenRefreshTimeout,
      retryMax: Math.min(cfg.retryMax, MAX_TOKEN_REFRESH_RETRIES),
      retryBaseDelay: cfg.retryBaseDelay,
    },
    `token refresh (${label})`,
    log,
  );
}
