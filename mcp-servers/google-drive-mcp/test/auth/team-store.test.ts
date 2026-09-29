import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

import type { OAuthClientInformationFull } from '@modelcontextprotocol/sdk/shared/auth.js';
import { InMemoryTeamStore } from '../../src/auth/team/memoryStore.js';
import { FileTeamStore } from '../../src/auth/team/fileStore.js';
import {
  AccessTokenRecord,
  AuthorizationCodeRecord,
  PendingAuthorization,
  RefreshTokenRecord,
  TeamStore,
  TeamStoreCapacityError,
  TeamStoreCaps,
  TeamUserRecord,
} from '../../src/auth/team/types.js';

// ---------------------------------------------------------------------------
// TeamStore contract suite, run against both implementations. The single-use
// consume* semantics are the core security property: exactly one of N
// concurrent callers may receive a pending authorization or code.
// ---------------------------------------------------------------------------

interface StoreHandle {
  store: TeamStore;
  filePath?: string;
  cleanup: () => Promise<void>;
}

interface StoreFactory {
  name: string;
  create: (caps?: Partial<TeamStoreCaps>) => Promise<StoreHandle>;
}

const factories: StoreFactory[] = [
  {
    name: 'InMemoryTeamStore',
    create: async (caps) => {
      const store = new InMemoryTeamStore({ caps });
      await store.init();
      return { store, cleanup: async () => {} };
    },
  },
  {
    name: 'FileTeamStore',
    create: async (caps) => {
      const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-teamstore-'));
      const filePath = path.join(dir, 'team-store.json');
      const store = new FileTeamStore(filePath, { caps });
      await store.init();
      return {
        store,
        filePath,
        cleanup: async () => fs.rm(dir, { recursive: true, force: true }),
      };
    },
  },
];

function makeClient(id: string): OAuthClientInformationFull {
  return { client_id: id, redirect_uris: [`https://example.com/cb/${id}`] };
}

function makePending(overrides: Partial<PendingAuthorization> = {}): PendingAuthorization {
  const now = Date.now();
  return {
    googleState: `state-${Math.random().toString(36).slice(2)}`,
    clientId: 'client-1',
    clientState: 'mcp-client-state',
    codeChallenge: 'challenge',
    redirectUri: 'https://example.com/cb/client-1',
    scopes: ['https://www.googleapis.com/auth/drive'],
    createdAt: now,
    expiresAt: now + 600_000,
    ...overrides,
  };
}

function makeCode(overrides: Partial<AuthorizationCodeRecord> = {}): AuthorizationCodeRecord {
  const now = Date.now();
  return {
    codeHash: `hash-${Math.random().toString(36).slice(2)}`,
    clientId: 'client-1',
    sub: 'sub-1',
    codeChallenge: 'challenge',
    redirectUri: 'https://example.com/cb/client-1',
    scopes: ['https://www.googleapis.com/auth/drive'],
    createdAt: now,
    expiresAt: now + 60_000,
    challengeLookups: 0,
    ...overrides,
  };
}

function makeUser(sub: string, overrides: Partial<TeamUserRecord> = {}): TeamUserRecord {
  const now = new Date().toISOString();
  return {
    sub,
    email: `${sub}@example.com`,
    googleRefreshToken: `google-rt-${sub}`,
    grantedScopes: ['https://www.googleapis.com/auth/drive'],
    createdAt: now,
    updatedAt: now,
    ...overrides,
  };
}

function makeAccessToken(overrides: Partial<AccessTokenRecord> = {}): AccessTokenRecord {
  return {
    tokenHash: `at-${Math.random().toString(36).slice(2)}`,
    clientId: 'client-1',
    sub: 'sub-1',
    scopes: ['https://www.googleapis.com/auth/drive'],
    expiresAt: Date.now() + 3600_000,
    familyId: 'family-1',
    ...overrides,
  };
}

function makeRefreshToken(overrides: Partial<RefreshTokenRecord> = {}): RefreshTokenRecord {
  const now = Date.now();
  return {
    tokenHash: `rt-${Math.random().toString(36).slice(2)}`,
    clientId: 'client-1',
    sub: 'sub-1',
    scopes: ['https://www.googleapis.com/auth/drive'],
    familyId: 'family-1',
    createdAt: now,
    expiresAt: now + 30 * 24 * 3600_000,
    ...overrides,
  };
}

for (const factory of factories) {
  test(`${factory.name}: clients round-trip and LRU-evict the idle one at the cap`, async () => {
    const { store, cleanup } = await factory.create({ maxClients: 2 });
    try {
      await store.saveClient(makeClient('a'));
      await store.saveClient(makeClient('b'));
      assert.equal((await store.getClient('a'))?.client_id, 'a');
      assert.equal(await store.getClient('missing'), undefined);

      // At the cap, a new registration evicts the least-recently-used client
      // (here 'a', registered first and never used) instead of throwing — DCR
      // must not be permanently lockable by unauthenticated registrations.
      await store.saveClient(makeClient('c'));
      assert.equal(await store.getClient('a'), undefined, 'idle LRU client evicted');
      assert.ok(await store.getClient('b'));
      assert.ok(await store.getClient('c'));

      // Updating an existing client at the cap is an update, not an insert.
      await store.saveClient({ ...makeClient('b'), client_name: 'renamed' });
      assert.equal((await store.getClient('b'))?.client_name, 'renamed');
      assert.ok(await store.getClient('c'));
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: an active client survives eviction; the idle one is reclaimed`, async () => {
    const { store, cleanup } = await factory.create({ maxClients: 2 });
    try {
      await store.saveClient(makeClient('active'));
      await store.saveClient(makeClient('idle'));
      // Issuing a token marks 'active' as recently used, so 'idle' becomes the
      // LRU victim when a third client registers.
      await store.saveAccessToken(makeAccessToken({ clientId: 'active', sub: 'sub-x' }));

      await store.saveClient(makeClient('newcomer'));
      assert.ok(await store.getClient('active'), 'token-issuing client protected');
      assert.equal(await store.getClient('idle'), undefined, 'idle client reclaimed');
      assert.ok(await store.getClient('newcomer'));
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: updateUser atomically mutates the freshest record and no-ops on a missing sub`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      await store.upsertUser(makeUser('sub-a', { needsReauth: true }));
      await store.updateUser('sub-a', (u) => ({ ...u, needsReauth: false, email: 'renamed@example.com' }));
      const updated = await store.getUser('sub-a');
      assert.equal(updated?.needsReauth, false);
      assert.equal(updated?.email, 'renamed@example.com');
      // Preserves untouched fields.
      assert.equal(updated?.googleRefreshToken, 'google-rt-sub-a');

      // A missing sub is a no-op: the mutator never runs and nothing is created.
      let mutatorRan = false;
      await store.updateUser('ghost', (u) => {
        mutatorRan = true;
        return u;
      });
      assert.equal(mutatorRan, false);
      assert.equal(await store.getUser('ghost'), undefined);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: flush resolves and leaves data intact`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      await store.upsertUser(makeUser('sub-a'));
      await store.flush();
      assert.equal((await store.getUser('sub-a'))?.sub, 'sub-a');
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: superseded tombstones are hard-capped per sub`, async () => {
    const { store, cleanup } = await factory.create({ maxTokensPerSub: 50, maxTombstonesPerSub: 2 });
    try {
      const now = Date.now();
      // Build four tombstones for sub-1 (save fresh, then supersede). Each new
      // mint trims the excess, so the retained set never exceeds the cap and the
      // oldest tombstones are the ones reclaimed.
      const tombstones = [now - 4000, now - 3000, now - 2000, now - 1000].map((createdAt) =>
        makeRefreshToken({ createdAt }),
      );
      for (const t of tombstones) {
        await store.saveRefreshToken(t);
        await store.updateRefreshToken({ ...t, supersededByHash: 'succ', graceUntil: now + 60_000 });
      }

      const survivors = await Promise.all(tombstones.map((t) => store.getRefreshToken(t.tokenHash)));
      assert.ok(
        survivors.filter(Boolean).length <= 2,
        `at most maxTombstonesPerSub survive, got ${survivors.filter(Boolean).length}`,
      );
      assert.equal(survivors[0], undefined, 'the oldest tombstone was reclaimed');
      assert.equal(survivors[1], undefined, 'the next-oldest tombstone was reclaimed');
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: pending authorization is single-use under contention`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const pending = makePending();
      await store.savePendingAuthorization(pending);
      const results = await Promise.all([
        store.consumePendingAuthorization(pending.googleState),
        store.consumePendingAuthorization(pending.googleState),
        store.consumePendingAuthorization(pending.googleState),
      ]);
      const winners = results.filter(Boolean);
      assert.equal(winners.length, 1, 'exactly one consumer must win');
      assert.equal(winners[0]!.clientState, 'mcp-client-state');
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: expired pending authorization is not consumable`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const pending = makePending({ expiresAt: Date.now() - 1 });
      await store.savePendingAuthorization(pending);
      assert.equal(await store.consumePendingAuthorization(pending.googleState), undefined);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: pending cap rejects new but sweeps expired first`, async () => {
    const { store, cleanup } = await factory.create({ maxPendingAuthorizations: 2 });
    try {
      await store.savePendingAuthorization(makePending({ expiresAt: Date.now() - 1 }));
      await store.savePendingAuthorization(makePending());
      // The expired entry is swept to make room for this one.
      await store.savePendingAuthorization(makePending());
      await assert.rejects(
        () => store.savePendingAuthorization(makePending()),
        TeamStoreCapacityError,
      );
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: authorization code is single-use under contention`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const code = makeCode();
      await store.saveAuthorizationCode(code);
      const results = await Promise.all([
        store.consumeAuthorizationCode(code.codeHash),
        store.consumeAuthorizationCode(code.codeHash),
        store.consumeAuthorizationCode(code.codeHash),
      ]);
      assert.equal(results.filter(Boolean).length, 1, 'exactly one consumer must win');
      assert.equal(await store.consumeAuthorizationCode(code.codeHash), undefined);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: expired authorization code is not consumable or peekable`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const code = makeCode({ expiresAt: Date.now() - 1 });
      await store.saveAuthorizationCode(code);
      assert.equal(await store.peekAuthorizationCode(code.codeHash), undefined);

      const code2 = makeCode({ expiresAt: Date.now() - 1 });
      await store.saveAuthorizationCode(code2);
      assert.equal(await store.consumeAuthorizationCode(code2.codeHash), undefined);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: code is burned after too many challenge lookups`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const code = makeCode();
      await store.saveAuthorizationCode(code);
      // MAX_CHALLENGE_LOOKUPS = 3 lookups are allowed…
      assert.ok(await store.peekAuthorizationCode(code.codeHash));
      assert.ok(await store.peekAuthorizationCode(code.codeHash));
      assert.ok(await store.peekAuthorizationCode(code.codeHash));
      // …the 4th burns the code entirely.
      assert.equal(await store.peekAuthorizationCode(code.codeHash), undefined);
      assert.equal(await store.consumeAuthorizationCode(code.codeHash), undefined);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: users round-trip and enforce the cap for new subs only`, async () => {
    const { store, cleanup } = await factory.create({ maxUsers: 2 });
    try {
      await store.upsertUser(makeUser('sub-a'));
      await store.upsertUser(makeUser('sub-b'));
      assert.equal((await store.getUser('sub-a'))?.email, 'sub-a@example.com');

      await assert.rejects(() => store.upsertUser(makeUser('sub-c')), TeamStoreCapacityError);
      await store.upsertUser(makeUser('sub-a', { needsReauth: true }));
      assert.equal((await store.getUser('sub-a'))?.needsReauth, true);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: access tokens round-trip, delete, and evict oldest at per-sub cap`, async () => {
    const { store, cleanup } = await factory.create({ maxTokensPerSub: 2 });
    try {
      const oldest = makeAccessToken({ expiresAt: Date.now() + 1000 });
      const middle = makeAccessToken({ expiresAt: Date.now() + 2000 });
      await store.saveAccessToken(oldest);
      await store.saveAccessToken(middle);
      // Another user's tokens don't count toward sub-1's cap.
      await store.saveAccessToken(makeAccessToken({ sub: 'sub-other' }));

      const newest = makeAccessToken({ expiresAt: Date.now() + 3000 });
      await store.saveAccessToken(newest);
      assert.equal(await store.getAccessToken(oldest.tokenHash), undefined, 'oldest evicted');
      assert.ok(await store.getAccessToken(middle.tokenHash));
      assert.ok(await store.getAccessToken(newest.tokenHash));

      await store.deleteAccessToken(newest.tokenHash);
      assert.equal(await store.getAccessToken(newest.tokenHash), undefined);
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: refresh tokens rotate-update and superseded ones don't count toward the cap`, async () => {
    const { store, cleanup } = await factory.create({ maxTokensPerSub: 2 });
    try {
      const live1 = makeRefreshToken({ createdAt: Date.now() - 3000 });
      const live2 = makeRefreshToken({ createdAt: Date.now() - 2000 });
      await store.saveRefreshToken(live1);
      await store.saveRefreshToken(live2);

      // Rotate live1: superseded records are tombstones, not active tokens.
      await store.updateRefreshToken({
        ...live1,
        supersededByHash: 'successor',
        graceUntil: Date.now() + 60_000,
      });

      // Only one active token remains (live2), so saving another must NOT evict.
      const live3 = makeRefreshToken({ createdAt: Date.now() - 1000 });
      await store.saveRefreshToken(live3);
      assert.ok(await store.getRefreshToken(live1.tokenHash), 'tombstone kept');
      assert.ok(await store.getRefreshToken(live2.tokenHash), 'live token under cap kept');
      assert.ok(await store.getRefreshToken(live3.tokenHash));

      // Now at the active cap (live2 + live3): the next save evicts the oldest
      // ACTIVE token (live2), never the tombstone.
      const live4 = makeRefreshToken({ createdAt: Date.now() });
      await store.saveRefreshToken(live4);
      assert.ok(await store.getRefreshToken(live1.tokenHash), 'tombstone still kept');
      assert.equal(await store.getRefreshToken(live2.tokenHash), undefined, 'oldest live evicted');
      assert.ok(await store.getRefreshToken(live3.tokenHash));
      assert.ok(await store.getRefreshToken(live4.tokenHash));
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: revokeTokensForSub removes all of a user's tokens and nobody else's`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const mineA = makeAccessToken({ sub: 'victim' });
      const mineR = makeRefreshToken({ sub: 'victim' });
      const otherA = makeAccessToken({ sub: 'bystander' });
      await store.saveAccessToken(mineA);
      await store.saveRefreshToken(mineR);
      await store.saveAccessToken(otherA);

      await store.revokeTokensForSub('victim');
      assert.equal(await store.getAccessToken(mineA.tokenHash), undefined);
      assert.equal(await store.getRefreshToken(mineR.tokenHash), undefined);
      assert.ok(await store.getAccessToken(otherA.tokenHash));
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: revokeTokenFamily removes access and refresh tokens of that family only`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const famA = makeAccessToken({ familyId: 'stolen' });
      const famR = makeRefreshToken({ familyId: 'stolen' });
      const other = makeRefreshToken({ familyId: 'innocent' });
      await store.saveAccessToken(famA);
      await store.saveRefreshToken(famR);
      await store.saveRefreshToken(other);

      await store.revokeTokenFamily('stolen');
      assert.equal(await store.getAccessToken(famA.tokenHash), undefined);
      assert.equal(await store.getRefreshToken(famR.tokenHash), undefined);
      assert.ok(await store.getRefreshToken(other.tokenHash));
    } finally {
      await cleanup();
    }
  });

  test(`${factory.name}: sweepExpired drops expired tokens but keeps unexpired tombstones`, async () => {
    const { store, cleanup } = await factory.create();
    try {
      const now = Date.now();
      const expiredA = makeAccessToken({ expiresAt: now - 1 });
      const liveA = makeAccessToken({ expiresAt: now + 3600_000 });
      const expiredR = makeRefreshToken({ expiresAt: now - 1 });
      const tombstone = makeRefreshToken({
        expiresAt: now + 3600_000,
        supersededByHash: 'x',
        graceUntil: now - 1000, // grace long past — must still survive the sweep
      });
      await store.saveAccessToken(expiredA);
      await store.saveAccessToken(liveA);
      await store.saveRefreshToken(expiredR);
      await store.saveRefreshToken(tombstone);

      await store.sweepExpired(now);
      assert.equal(await store.getAccessToken(expiredA.tokenHash), undefined);
      assert.ok(await store.getAccessToken(liveA.tokenHash));
      assert.equal(await store.getRefreshToken(expiredR.tokenHash), undefined);
      assert.ok(
        await store.getRefreshToken(tombstone.tokenHash),
        'reuse-detection tombstone must survive until expiresAt',
      );
    } finally {
      await cleanup();
    }
  });
}

// ---------------------------------------------------------------------------
// FileTeamStore-only behavior.
// ---------------------------------------------------------------------------

async function makeFileStore(): Promise<{ dir: string; filePath: string; store: FileTeamStore }> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-teamstore-'));
  const filePath = path.join(dir, 'team-store.json');
  const store = new FileTeamStore(filePath);
  await store.init();
  return { dir, filePath, store };
}

test('FileTeamStore: durable sections survive a restart; pending/codes do not', async () => {
  const { dir, filePath, store } = await makeFileStore();
  try {
    await store.saveClient(makeClient('a'));
    await store.upsertUser(makeUser('sub-a'));
    const at = makeAccessToken();
    const rt = makeRefreshToken();
    await store.saveAccessToken(at);
    await store.saveRefreshToken(rt);
    const pending = makePending();
    await store.savePendingAuthorization(pending);
    const code = makeCode();
    await store.saveAuthorizationCode(code);

    const reopened = new FileTeamStore(filePath);
    await reopened.init();
    assert.equal((await reopened.getClient('a'))?.client_id, 'a');
    assert.equal((await reopened.getUser('sub-a'))?.googleRefreshToken, 'google-rt-sub-a');
    assert.ok(await reopened.getAccessToken(at.tokenHash));
    assert.ok(await reopened.getRefreshToken(rt.tokenHash));
    assert.equal(await reopened.consumePendingAuthorization(pending.googleState), undefined);
    assert.equal(await reopened.consumeAuthorizationCode(code.codeHash), undefined);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: store file is created with mode 0600', async () => {
  const { dir, filePath } = await makeFileStore();
  try {
    const stat = await fs.stat(filePath);
    assert.equal(stat.mode & 0o777, 0o600);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: raw MCP token strings never appear in the file', async () => {
  const { dir, filePath, store } = await makeFileStore();
  try {
    // Simulates the provider contract: only hashes are handed to the store.
    const rawToken = 'mcp_at_super-secret-raw-token-value';
    await store.saveAccessToken(makeAccessToken({ tokenHash: 'a'.repeat(64) }));
    const content = await fs.readFile(filePath, 'utf-8');
    assert.ok(!content.includes(rawToken));
    assert.ok(content.includes('a'.repeat(64)));
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: corrupt file is quarantined and the store starts empty', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-teamstore-'));
  const filePath = path.join(dir, 'team-store.json');
  try {
    await fs.writeFile(filePath, '{"version":1,"clients":{TRUNCATED');
    const store = new FileTeamStore(filePath);
    await store.init();
    assert.equal(await store.getClient('anything'), undefined);

    const entries = await fs.readdir(dir);
    assert.ok(
      entries.some((e) => e.startsWith('team-store.json.corrupt-')),
      `expected a quarantine backup, got: ${entries.join(', ')}`,
    );
    // A fresh, valid file was re-established.
    const parsed = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    assert.equal(parsed.version, 1);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: a non-ENOENT read error is NOT quarantined — init throws and leaves the file untouched', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-teamstore-'));
  // A directory at the store path makes readFile fail with EISDIR — an IO error
  // that is NOT content corruption. The store must refuse to start rather than
  // rename a possibly-intact file aside and wipe every member's grant.
  const storePath = path.join(dir, 'team-store.json');
  await fs.mkdir(storePath);
  try {
    const store = new FileTeamStore(storePath);
    await assert.rejects(() => store.init(), /could not be read/);
    // No quarantine backup was created.
    const entries = await fs.readdir(dir);
    assert.ok(
      !entries.some((e) => e.includes('.corrupt-')),
      `expected no quarantine backup, got: ${entries.join(', ')}`,
    );
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: flush drains queued writes to disk', async () => {
  const { dir, filePath, store } = await makeFileStore();
  try {
    // Enqueue several durable writes without awaiting each, then flush.
    const pending = [
      store.upsertUser(makeUser('sub-a')),
      store.saveClient(makeClient('client-a')),
      store.saveAccessToken(makeAccessToken({ clientId: 'client-a' })),
    ];
    await store.flush();
    // flush resolves only once the queue tail has run, so all writes are on disk.
    await Promise.all(pending);
    const parsed = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    assert.ok(parsed.users['sub-a']);
    assert.ok(parsed.clients['client-a']);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: client LRU metadata persists across a restart', async () => {
  const { dir, filePath, store } = await makeFileStore();
  try {
    await store.saveClient(makeClient('a'));
    await store.saveClient(makeClient('b'));
    // Mark 'b' recently used so 'a' is the unambiguous LRU victim after reopen.
    await store.saveAccessToken(makeAccessToken({ clientId: 'b', sub: 'sub-b' }));

    const reopened = new FileTeamStore(filePath, { caps: { maxClients: 2 } });
    await reopened.init();
    await reopened.saveClient(makeClient('c'));
    assert.equal(await reopened.getClient('a'), undefined, 'persisted-LRU client evicted');
    assert.ok(await reopened.getClient('b'));
    assert.ok(await reopened.getClient('c'));
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: init fails on an unwritable store path', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-teamstore-'));
  try {
    // Parent "directory" is actually a file → mkdir/write must fail.
    const blocker = path.join(dir, 'blocker');
    await fs.writeFile(blocker, 'not a directory');
    const store = new FileTeamStore(path.join(blocker, 'team-store.json'));
    await assert.rejects(() => store.init());
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: concurrent writers on the same path do not clobber each other', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-teamstore-'));
  const filePath = path.join(dir, 'team-store.json');
  try {
    const a = new FileTeamStore(filePath);
    const b = new FileTeamStore(filePath);
    await a.init();
    await b.init();

    await a.upsertUser(makeUser('sub-a'));
    await b.upsertUser(makeUser('sub-b'));

    const parsed = JSON.parse(await fs.readFile(filePath, 'utf-8'));
    assert.ok(parsed.users['sub-a'], 'writer A record survived');
    assert.ok(parsed.users['sub-b'], 'writer B record survived');
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('FileTeamStore: a stale temp file from a crashed write does not affect loading', async () => {
  const { dir, filePath, store } = await makeFileStore();
  try {
    await store.upsertUser(makeUser('sub-a'));
    // Simulate a crash between temp-write and rename.
    await fs.writeFile(`${filePath}.tmp.99999.0`, '{"version":1,"clients":{GARBAGE');

    const reopened = new FileTeamStore(filePath);
    await reopened.init();
    assert.equal((await reopened.getUser('sub-a'))?.sub, 'sub-a');
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});
