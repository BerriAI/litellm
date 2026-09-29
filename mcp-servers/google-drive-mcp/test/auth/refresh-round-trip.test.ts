import assert from 'node:assert/strict';
import test, { mock } from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

import { AccountStore } from '../../src/auth/accountStore.js';
import { AccountClientFactory } from '../../src/auth/accountClientFactory.js';
import type { AccountRecord, TokenFileV2 } from '../../src/auth/types.js';
import {
  installFakeTokenEndpoint,
  stallThen,
  stallUntilAborted,
  tokenResponse,
} from '../helpers/fake-token-endpoint.js';

// ---------------------------------------------------------------------------
// Token-refresh round-trip.
//
// AccountClientFactory wires a `'tokens'` listener on every per-alias
// OAuth2Client. When Google's library emits that event after a refresh, the
// listener must merge the new credentials with the existing record (especially
// preserving `refreshToken`, which Google rarely rotates) and persist back to
// AccountStore. This test drives that path with a real OAuth2Client (no
// network) by emitting the event manually.
// ---------------------------------------------------------------------------

async function setupTmpCredentials(): Promise<{
  tokenPath: string;
  credsPath: string;
  cleanup: () => Promise<void>;
  saved: Record<string, string | undefined>;
}> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-refresh-'));
  const tokenPath = path.join(dir, 'tokens.json');
  const credsPath = path.join(dir, 'gcp-oauth.keys.json');

  // Minimal valid credentials file — only client_id is needed for OAuth2Client
  // construction; no network is hit during this test.
  await fs.writeFile(
    credsPath,
    JSON.stringify({
      installed: {
        client_id: 'test-client-id.apps.googleusercontent.com',
        client_secret: 'test-client-secret',
        redirect_uris: ['http://localhost:3000/oauth2callback'],
      },
    }),
  );

  const saved = {
    GOOGLE_DRIVE_MCP_TOKEN_PATH: process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH,
    GOOGLE_DRIVE_OAUTH_CREDENTIALS: process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS,
  };
  process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH = tokenPath;
  process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS = credsPath;

  const cleanup = async () => {
    if (saved.GOOGLE_DRIVE_MCP_TOKEN_PATH === undefined)
      delete process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH;
    else process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH = saved.GOOGLE_DRIVE_MCP_TOKEN_PATH;
    if (saved.GOOGLE_DRIVE_OAUTH_CREDENTIALS === undefined)
      delete process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS;
    else process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS = saved.GOOGLE_DRIVE_OAUTH_CREDENTIALS;
  };

  return { tokenPath, credsPath, cleanup, saved };
}

function makeRecord(overrides: Partial<AccountRecord> = {}): AccountRecord {
  const now = new Date().toISOString();
  return {
    alias: 'work',
    email: 'work@example.com',
    sub: 'sub-work',
    accessToken: 'old-access',
    refreshToken: 'persistent-refresh',
    scope: 'https://www.googleapis.com/auth/drive',
    tokenType: 'Bearer',
    // Future expiry so getClient does NOT kick off a real (failing) network
    // refresh that races the manually-emitted 'tokens' event below. Tests that
    // need the refresh path stub it explicitly (see the invalid_grant test).
    expiryDate: Date.now() + 3600_000,
    addedAt: now,
    lastRefreshedAt: now,
    ...overrides,
  };
}

/** Wait until `predicate()` returns truthy or `timeoutMs` elapses. Several
 * callers poll the on-disk token file, which AccountClientFactory's `'tokens'`
 * listener writes asynchronously and fire-and-forget (see persistRefreshedTokens
 * in accountClientFactory.ts); the generous default accounts for real disk I/O
 * latency under a loaded machine, not for any in-memory visibility delay. */
async function waitFor(
  predicate: () => boolean | Promise<boolean>,
  timeoutMs = 10_000,
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`waitFor timed out after ${timeoutMs}ms`);
}

/** Read and parse the token file, tolerating "not written yet". */
async function readTokenFile(tokenPath: string): Promise<TokenFileV2 | undefined> {
  try {
    return JSON.parse(await fs.readFile(tokenPath, 'utf-8')) as TokenFileV2;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw err;
  }
}

test('factory persists new access_token from a "tokens" event into AccountStore', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store);
    const client = await factory.getClient('work');

    // Synthesize a refresh-event with a new access_token but no refresh_token
    // (Google's typical case — the existing refresh token must be preserved).
    const newAccess = `new-access-${Date.now()}`;
    const newExpiry = Date.now() + 3600_000;
    client.emit('tokens', {
      access_token: newAccess,
      expiry_date: newExpiry,
      // No refresh_token — must NOT clobber the existing one.
    });

    await waitFor(async () => {
      const onDisk = await readTokenFile(tokenPath);
      return onDisk?.accounts?.work?.accessToken === newAccess;
    });

    const updated = store.get('work')!;
    assert.equal(updated.accessToken, newAccess);
    assert.equal(updated.expiryDate, newExpiry);
    assert.equal(updated.refreshToken, 'persistent-refresh');
    assert.notEqual(updated.lastRefreshedAt, makeRecord().lastRefreshedAt);

    // File on disk reflects the change too.
    const onDisk = await readTokenFile(tokenPath);
    assert.equal(onDisk?.accounts.work?.accessToken, newAccess);
    assert.equal(onDisk?.accounts.work?.refreshToken, 'persistent-refresh');
  } finally {
    await cleanup();
  }
});

test('factory accepts a rotated refresh_token when Google does send one', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store);
    const client = await factory.getClient('work');

    client.emit('tokens', {
      access_token: 'rotated-access',
      refresh_token: 'rotated-refresh',
      expiry_date: Date.now() + 3600_000,
    });

    await waitFor(async () => {
      const onDisk = await readTokenFile(tokenPath);
      return onDisk?.accounts?.work?.refreshToken === 'rotated-refresh';
    });
    assert.equal(store.get('work')!.accessToken, 'rotated-access');

    // File on disk reflects the rotated token too.
    const onDisk = await readTokenFile(tokenPath);
    assert.equal(onDisk?.accounts.work?.accessToken, 'rotated-access');
    assert.equal(onDisk?.accounts.work?.refreshToken, 'rotated-refresh');
  } finally {
    await cleanup();
  }
});

test('factory caches the same OAuth2Client for the same alias across calls', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store);
    const a = await factory.getClient('work');
    const b = await factory.getClient('work');
    assert.equal(a, b, 'same alias should yield the same cached client instance');
  } finally {
    await cleanup();
  }
});

test('factory yields distinct OAuth2Client instances for distinct aliases', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord({ alias: 'work' }));
    await store.upsert(makeRecord({ alias: 'personal', sub: 'sub-personal', refreshToken: 'rt-personal' }));

    const factory = new AccountClientFactory(store);
    const work = await factory.getClient('work');
    const personal = await factory.getClient('personal');
    assert.notEqual(work, personal, 'distinct aliases must yield distinct clients');
  } finally {
    await cleanup();
  }
});

test('factory.evict drops the cached client so the next getClient rebuilds', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store);
    const before = await factory.getClient('work');
    factory.evict('work');
    const after = await factory.getClient('work');
    assert.notEqual(before, after, 'evict should force a fresh client on next getClient');
  } finally {
    await cleanup();
  }
});

test('factory surfaces invalid_grant as an actionable reconnect error (finding 5)', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    // Future expiry so the first getClient does NOT trigger a refresh.
    await store.upsert(makeRecord({ expiryDate: Date.now() + 3600_000 }));

    const factory = new AccountClientFactory(store);
    const client = await factory.getClient('work');

    // Make the cached client look expired and its refresh fail with invalid_grant.
    client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
    (client as unknown as { refreshAccessToken: () => Promise<never> }).refreshAccessToken =
      async () => {
        const err = new Error('invalid_grant') as Error & {
          response?: { data?: { error?: string } };
        };
        err.response = { data: { error: 'invalid_grant' } };
        throw err;
      };

    await assert.rejects(
      () => factory.getClient('work'),
      (err: Error) => {
        assert.match(err.message, /revoked or has expired/);
        assert.match(err.message, /manage_accounts add work/);
        return true;
      },
    );
  } finally {
    await cleanup();
  }
});

test('factory detects invalid_grant in the gaxios-7 error shape', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord({ expiryDate: Date.now() + 3600_000 }));

    const factory = new AccountClientFactory(store);
    const client = await factory.getClient('work');

    client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
    (client as unknown as { refreshAccessToken: () => Promise<never> }).refreshAccessToken =
      async () => {
        // gaxios 7 sets a top-level numeric `status` and still parses the
        // OAuth error body into response.data.
        const err = new Error('Request failed with status code 400') as Error & {
          status?: number;
          response?: { status?: number; data?: { error?: string; error_description?: string } };
        };
        err.status = 400;
        err.response = {
          status: 400,
          data: { error: 'invalid_grant', error_description: 'Token has been expired or revoked.' },
        };
        throw err;
      };

    await assert.rejects(
      () => factory.getClient('work'),
      (err: Error) => {
        assert.match(err.message, /revoked or has expired/);
        assert.match(err.message, /manage_accounts add work/);
        return true;
      },
    );
  } finally {
    await cleanup();
  }
});

// ---------------------------------------------------------------------------
// Bounded refresh through the factory (#169). The fake endpoint replaces the
// network under the REAL refreshAccessToken() path, so the factory's dedupe
// map, the retry, the abort of the stalled POST and the 'tokens' persistence
// are all exercised together.
// ---------------------------------------------------------------------------

const BOUNDED = { tokenRefreshTimeout: 30, retryMax: 1, retryBaseDelay: 0 };

test('a stalled refresh fails every deduped caller fast with an actionable error', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  const logged: string[] = [];
  const errMock = mock.method(console, 'error', (...args: unknown[]) => {
    logged.push(args.map(String).join(' '));
  });
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store, BOUNDED);
    const client = await factory.getClient('work');
    client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
    const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);

    const start = Date.now();
    const results = await Promise.allSettled([factory.getClient('work'), factory.getClient('work')]);
    assert.ok(Date.now() - start < 2_000, 'callers were not released promptly');
    for (const r of results) {
      assert.equal(r.status, 'rejected');
      const message = (r as PromiseRejectedResult).reason.message as string;
      assert.match(message, /Account 'work' could not refresh its access token/);
      assert.match(message, /did not respond within 30ms/);
      assert.match(message, /--token-refresh-timeout/);
    }
    // One deduped refresh: two attempts, each a distinct aborted POST.
    assert.equal(endpoint.calls.length, 2);
    assert.ok(endpoint.calls.every((c) => c.signal?.aborted));
    assert.equal(
      (factory as unknown as { inflightRefresh: Map<string, unknown> }).inflightRefresh.size,
      0,
      'inflight map must clear after a timeout',
    );
    // The retry log names the operation but never the token material.
    const retryLine = logged.find((l) => l.includes('retry 1/1'));
    assert.ok(retryLine, `expected a retry log line, got: ${JSON.stringify(logged)}`);
    assert.ok(!logged.join('\n').includes('persistent-refresh'), 'refresh token leaked into logs');

    // Endpoint recovers: the next call retries from scratch and persists.
    endpoint.handler = tokenResponse('new-access');
    const again = await factory.getClient('work');
    assert.equal(again, client, 'a timeout must not evict the cached client');
    assert.equal(again.credentials.access_token, 'new-access');
    await waitFor(async () => {
      const onDisk = await readTokenFile(tokenPath);
      return onDisk?.accounts?.work?.accessToken === 'new-access';
    });
    assert.equal(store.get('work')!.refreshToken, 'persistent-refresh');
    assert.equal((await readTokenFile(tokenPath))?.accounts.work?.refreshToken, 'persistent-refresh');
  } finally {
    errMock.mock.restore();
    await cleanup();
  }
});

test('a transient stall recovers on the retry and the caller never sees an error', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  const errMock = mock.method(console, 'error', () => {});
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store, BOUNDED);
    const client = await factory.getClient('work');
    client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
    const endpoint = installFakeTokenEndpoint(client, stallThen(tokenResponse('new-access')));

    await factory.getClient('work');
    assert.equal(client.credentials.access_token, 'new-access');
    assert.equal(endpoint.calls.length, 2);
    await waitFor(async () => {
      const onDisk = await readTokenFile(tokenPath);
      return onDisk?.accounts?.work?.accessToken === 'new-access';
    });
  } finally {
    errMock.mock.restore();
    await cleanup();
  }
});
