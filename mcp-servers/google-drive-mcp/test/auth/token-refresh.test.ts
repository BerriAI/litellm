import assert from 'node:assert/strict';
import test from 'node:test';
import { OAuth2Client } from 'google-auth-library';

import {
  installTokenEndpointTimeout,
  isInvalidGrant,
  refreshAccessTokenBounded,
  type TokenRefreshConfig,
} from '../../src/auth/tokenRefresh.js';
import { TimeoutError } from '../../src/utils/retry.js';
import {
  installFakeTokenEndpoint,
  invalidGrantResponse,
  stallThen,
  stallUntilAborted,
  tokenResponse,
} from '../helpers/fake-token-endpoint.js';

// ---------------------------------------------------------------------------
// Bounded token refresh (#169).
//
// `refreshAccessToken()` has no timeout and google-auth-library dedupes
// refreshes internally, so a stalled token POST pins every later refresh —
// retries included — to itself. These tests drive the REAL refresh path of a
// real OAuth2Client against a fake transport and pin that: the deadline aborts
// the in-flight POST (each retry is a new request, not a re-await of the stuck
// one), a stall through every attempt surfaces as TimeoutError, a transient
// stall recovers on the retry, invalid_grant is not retried, the library's own
// implicit refreshes are bounded too, and ordinary API requests are untouched.
// Real timers with small budgets; a retry adds up to 199 ms of jitter.
// ---------------------------------------------------------------------------

const TOKEN_URL = 'https://oauth2.googleapis.com/token';

function makeClient(overrides: Partial<Record<string, unknown>> = {}): OAuth2Client {
  const client = new OAuth2Client({ clientId: 'test-client-id', clientSecret: 'test-secret' });
  client.setCredentials({
    access_token: 'old-access',
    refresh_token: 'persistent-refresh',
    // Expired: the next refresh/getAccessToken must hit the token endpoint.
    expiry_date: Date.now() - 60_000,
    ...overrides,
  });
  return client;
}

const cfg = (o: Partial<TokenRefreshConfig> = {}): TokenRefreshConfig => ({
  tokenRefreshTimeout: 30,
  retryMax: 1,
  retryBaseDelay: 0,
  ...o,
});

test('a stall through every attempt rejects with TimeoutError and aborts each POST', async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);

  await assert.rejects(refreshAccessTokenBounded(client, cfg(), 'work'), (err: unknown) => {
    assert.ok(err instanceof TimeoutError);
    assert.match((err as Error).message, /token refresh \(work\) timed out after 30ms/);
    return true;
  });

  // Two distinct requests: the retry was a fresh POST, not a re-await of the
  // first (which google-auth-library would otherwise have handed back).
  assert.equal(endpoint.calls.length, 2);
  assert.ok(endpoint.calls.every((c) => c.url === TOKEN_URL && c.method === 'POST'));
  assert.ok(endpoint.calls[0].signal?.aborted, 'first attempt was not aborted');
  assert.ok(endpoint.calls[1].signal?.aborted, 'second attempt was not aborted');
  assert.notEqual(endpoint.calls[0].signal, endpoint.calls[1].signal);
  // Credentials are untouched by a failed refresh.
  assert.equal(client.credentials.access_token, 'old-access');
});

test('a transient stall recovers on the retry and updates credentials once', async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, stallThen(tokenResponse('new-access')));
  const tokenEvents: unknown[] = [];
  client.on('tokens', (t) => tokenEvents.push(t));

  const credentials = await refreshAccessTokenBounded(client, cfg(), 'work');

  assert.equal(credentials.access_token, 'new-access');
  assert.equal(client.credentials.access_token, 'new-access');
  assert.equal(client.credentials.refresh_token, 'persistent-refresh');
  assert.ok((client.credentials.expiry_date ?? 0) > Date.now());
  assert.equal(endpoint.calls.length, 2);
  assert.ok(endpoint.calls[0].signal?.aborted);
  assert.equal(endpoint.calls[1].signal?.aborted, false);
  assert.equal(tokenEvents.length, 1, 'the abandoned attempt must not emit tokens');
});

test('refresh retries are capped at one regardless of retryMax', async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);

  await assert.rejects(
    refreshAccessTokenBounded(client, cfg({ retryMax: 5 }), 'work'),
    (err: unknown) => err instanceof TimeoutError,
  );
  assert.equal(endpoint.calls.length, 2);
});

test('retryMax 0 disables the refresh retry', async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);

  await assert.rejects(
    refreshAccessTokenBounded(client, cfg({ retryMax: 0 }), 'work'),
    (err: unknown) => err instanceof TimeoutError,
  );
  assert.equal(endpoint.calls.length, 1);
});

test('invalid_grant surfaces on the first attempt and is not retried', async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, invalidGrantResponse);

  await assert.rejects(refreshAccessTokenBounded(client, cfg(), 'work'), (err: unknown) => {
    assert.ok(isInvalidGrant(err), `expected invalid_grant, got: ${String(err)}`);
    assert.ok(!(err instanceof TimeoutError));
    return true;
  });
  assert.equal(endpoint.calls.length, 1);
});

test('tokenRefreshTimeout 0 disables the deadline and attaches no signal', async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, async (call) => {
    await new Promise((r) => setTimeout(r, 40));
    return tokenResponse('new-access')(call);
  });

  const credentials = await refreshAccessTokenBounded(client, cfg({ tokenRefreshTimeout: 0 }), 'work');

  assert.equal(credentials.access_token, 'new-access');
  assert.equal(endpoint.calls.length, 1);
  assert.equal(endpoint.calls[0].signal, undefined);
});

test("the library's own implicit refresh is bounded by the floor signal", async () => {
  const client = makeClient();
  const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);
  installTokenEndpointTimeout(client, cfg({ tokenRefreshTimeout: 30 }));
  // The floor timer is unref'd (it must never hold the server open), so with a
  // fake transport nothing else keeps the loop alive until it fires.
  const keepAlive = setTimeout(() => {}, 5_000);

  try {
    // No proactive refresh in flight: getAccessToken() refreshes on its own.
    const start = Date.now();
    await assert.rejects(client.getAccessToken());
    assert.ok(Date.now() - start < 2_000, 'implicit refresh was not bounded');
    assert.equal(endpoint.calls.length, 1);
    assert.equal(endpoint.calls[0].url, TOKEN_URL);
    assert.ok(endpoint.calls[0].signal?.aborted);
  } finally {
    clearTimeout(keepAlive);
  }
});

test('requests to other URLs are left alone', async () => {
  // Fresh credentials so client.request() does not refresh first.
  const client = makeClient({ expiry_date: Date.now() + 3600_000 });
  const endpoint = installFakeTokenEndpoint(client, async () => ({ data: { user: 'x' } }));
  installTokenEndpointTimeout(client, cfg());

  const res = await client.request<{ user: string }>({
    url: 'https://www.googleapis.com/drive/v3/about',
  });

  assert.equal(res.data.user, 'x');
  assert.equal(endpoint.calls.length, 1);
  assert.equal(endpoint.calls[0].signal, undefined);
});

test('the interceptor is installed once per client', async () => {
  const client = makeClient();
  installFakeTokenEndpoint(client, tokenResponse('a'));
  // v9 exposes the Gaxios as `client.gaxios`; v10's transporter is the Gaxios.
  const gaxios = (client as unknown as { gaxios?: unknown }).gaxios ?? client.transporter;
  const interceptors = (gaxios as { interceptors: { request: Set<unknown> } }).interceptors.request;
  const before = interceptors.size;

  await refreshAccessTokenBounded(client, cfg(), 'work');
  client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
  await refreshAccessTokenBounded(client, cfg(), 'work');

  assert.equal(interceptors.size, before + 1);
});

test('an unrecognised transporter still gets the caller-side deadline', async () => {
  const client = makeClient();
  // A bare Transporter with no interceptor hook: the POST cannot be cancelled,
  // but the caller must still be released with a TimeoutError.
  (client as unknown as { transporter: unknown }).transporter = {
    request: () => new Promise(() => {}),
  };

  const start = Date.now();
  await assert.rejects(
    refreshAccessTokenBounded(client, cfg(), 'work'),
    (err: unknown) => err instanceof TimeoutError,
  );
  assert.ok(Date.now() - start < 2_000);
});
