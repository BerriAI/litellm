import assert from 'node:assert/strict';
import test, { mock } from 'node:test';

import { TeamClientFactory } from '../../src/auth/team/clientFactory.js';
import { InMemoryTeamStore } from '../../src/auth/team/memoryStore.js';
import type { TeamUserRecord } from '../../src/auth/team/types.js';
import {
  installFakeTokenEndpoint,
  invalidGrantResponse,
  stallThen,
  stallUntilAborted,
  tokenResponse,
} from '../helpers/fake-token-endpoint.js';

// ---------------------------------------------------------------------------
// TeamClientFactory refresh bounds (#169).
//
// The per-user factory mirrors AccountClientFactory's refresh mechanics, so it
// had the same unbounded `refreshAccessToken()` call. Pins that a stalled
// token endpoint fails the call fast with a message naming the member (and
// does NOT flag the grant as revoked), that the dedupe map clears so the next
// call retries fresh, that a recovered endpoint persists the new token, and
// that invalid_grant through the real request path still self-heals.
// ---------------------------------------------------------------------------

const SUB = 'sub-alice';
const EMAIL = 'alice@corp.example';

function makeUser(overrides: Partial<TeamUserRecord> = {}): TeamUserRecord {
  const now = new Date().toISOString();
  return {
    sub: SUB,
    email: EMAIL,
    googleRefreshToken: 'team-refresh-token',
    googleAccessToken: 'old-access',
    // Future expiry so the first getClient does NOT trigger a refresh.
    googleTokenExpiry: Date.now() + 3600_000,
    grantedScopes: ['https://www.googleapis.com/auth/drive'],
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
}

async function setup() {
  const store = new InMemoryTeamStore();
  await store.init();
  await store.upsertUser(makeUser());
  const factory = new TeamClientFactory(
    store,
    { client_id: 'test-client-id', client_secret: 'test-secret' },
    { tokenRefreshTimeout: 30, retryMax: 1, retryBaseDelay: 0 },
  );
  const client = await factory.getClient(SUB);
  // Now make the cached client look expired so the next getClient refreshes.
  client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
  return { store, factory, client };
}

async function waitFor(predicate: () => Promise<boolean>, timeoutMs = 10_000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`waitFor timed out after ${timeoutMs}ms`);
}

test('a stalled refresh rejects fast, names the member, and leaves the grant intact', async () => {
  const { store, factory, client } = await setup();
  const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);
  const errMock = mock.method(console, 'error', () => {});
  try {
    const start = Date.now();
    await assert.rejects(
      () => factory.getClient(SUB),
      (err: Error) => {
        assert.match(err.message, /alice@corp\.example/);
        assert.match(err.message, /did not respond within 30ms/);
        assert.match(err.message, /not a revoked grant/);
        return true;
      },
    );
    assert.ok(Date.now() - start < 2_000);
    assert.equal(endpoint.calls.length, 2);

    const user = await store.getUser(SUB);
    assert.equal(user?.needsReauth, undefined, 'a timeout must not flag re-auth');
    assert.equal(user?.googleRefreshToken, 'team-refresh-token');
  } finally {
    errMock.mock.restore();
  }
});

test('after a timeout the next call retries fresh and persists the new token', async () => {
  const { store, factory, client } = await setup();
  const endpoint = installFakeTokenEndpoint(client, stallUntilAborted);
  const errMock = mock.method(console, 'error', () => {});
  try {
    await assert.rejects(() => factory.getClient(SUB), /did not respond/);
    assert.equal((factory as unknown as { inflightRefresh: Map<string, unknown> }).inflightRefresh.size, 0);

    endpoint.handler = tokenResponse('new-access');
    const again = await factory.getClient(SUB);
    assert.equal(again, client, 'the cached client survives a timeout');
    assert.equal(again.credentials.access_token, 'new-access');
    await waitFor(async () => (await store.getUser(SUB))?.googleAccessToken === 'new-access');
    assert.equal((await store.getUser(SUB))?.googleRefreshToken, 'team-refresh-token');
  } finally {
    errMock.mock.restore();
  }
});

test('a transient stall recovers on the retry without surfacing an error', async () => {
  const { store, factory, client } = await setup();
  const endpoint = installFakeTokenEndpoint(client, stallThen(tokenResponse('new-access')));
  const logged: string[] = [];
  const errMock = mock.method(console, 'error', (...args: unknown[]) => {
    logged.push(args.map(String).join(' '));
  });
  try {
    await factory.getClient(SUB);
    assert.equal(client.credentials.access_token, 'new-access');
    assert.equal(endpoint.calls.length, 2);
    const retryLine = logged.find((l) => l.includes('retry 1/1'));
    assert.ok(retryLine, `expected a retry log line, got: ${JSON.stringify(logged)}`);
    assert.match(retryLine!, /^\[team-auth\] /);
    assert.ok(!logged.join('\n').includes('team-refresh-token'), 'refresh token leaked into logs');
    await waitFor(async () => (await store.getUser(SUB))?.googleAccessToken === 'new-access');
  } finally {
    errMock.mock.restore();
  }
});

test('invalid_grant through the real request path still flags re-auth on the first attempt', async () => {
  const { store, factory, client } = await setup();
  const endpoint = installFakeTokenEndpoint(client, invalidGrantResponse);
  const errMock = mock.method(console, 'error', () => {});
  try {
    await assert.rejects(
      () => factory.getClient(SUB),
      (err: Error) => {
        assert.match(err.message, /expired or been revoked/);
        assert.match(err.message, /alice@corp\.example/);
        return true;
      },
    );
    assert.equal(endpoint.calls.length, 1, 'invalid_grant must not be retried');
    assert.equal((await store.getUser(SUB))?.needsReauth, true);
  } finally {
    errMock.mock.restore();
  }
});
