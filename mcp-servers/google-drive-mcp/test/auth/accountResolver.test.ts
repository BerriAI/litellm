import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { AccountResolver, coversScopes } from '../../src/auth/accountResolver.js';
import { AccountStore } from '../../src/auth/accountStore.js';
import { SessionStore, STDIO_SESSION_ID } from '../../src/auth/sessionStore.js';
import { AccountRecord } from '../../src/auth/types.js';

const SCOPE_DRIVE = 'https://www.googleapis.com/auth/drive';
const SCOPE_READONLY = 'https://www.googleapis.com/auth/drive.readonly';
const SCOPE_DOCS = 'https://www.googleapis.com/auth/documents';

async function mktmp(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-resolver-'));
  return path.join(dir, 'tokens.json');
}

function makeRecord(alias: string, scope: string): AccountRecord {
  const now = new Date().toISOString();
  return {
    alias,
    email: `${alias}@example.com`,
    sub: `sub-${alias}`,
    accessToken: '',
    refreshToken: '',
    scope,
    tokenType: 'Bearer',
    expiryDate: Date.now() + 3600_000,
    addedAt: now,
    lastRefreshedAt: now,
  };
}

async function buildResolver(accounts: AccountRecord[], defaultAlias?: string) {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  for (const a of accounts) await store.upsert(a);
  if (defaultAlias) await store.setDefault(defaultAlias);
  const sessions = new SessionStore();
  const resolver = new AccountResolver(store, sessions);
  return { store, sessions, resolver };
}

// ---------------------------------------------------------------------------
// coversScopes — any-of semantics
// ---------------------------------------------------------------------------

test('coversScopes returns true when acceptable list is empty', () => {
  assert.equal(coversScopes('', []), true);
  assert.equal(coversScopes(SCOPE_DRIVE, []), true);
});

test('coversScopes requires exact match for at least one acceptable scope', () => {
  assert.equal(coversScopes(SCOPE_READONLY, [SCOPE_DRIVE]), false);
  assert.equal(coversScopes(SCOPE_DRIVE, [SCOPE_DRIVE]), true);
});

test('coversScopes is any-of: granted matches any one acceptable scope', () => {
  // Granted only drive; acceptable list includes drive + docs → granted has
  // drive → true.
  assert.equal(coversScopes(SCOPE_DRIVE, [SCOPE_DRIVE, SCOPE_DOCS]), true);
  // Granted only readonly; acceptable list drive + docs → no match → false.
  assert.equal(coversScopes(SCOPE_READONLY, [SCOPE_DRIVE, SCOPE_DOCS]), false);
});

// ---------------------------------------------------------------------------
// Explicit param
// ---------------------------------------------------------------------------

test('resolver: explicit string param returns single', async () => {
  const { resolver } = await buildResolver([
    makeRecord('work', SCOPE_DRIVE),
    makeRecord('personal', SCOPE_READONLY),
  ]);
  const t = await resolver.resolve('work', 'write', {
    sessionId: STDIO_SESSION_ID,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.kind, 'single');
  assert.equal(t.accounts[0].alias, 'work');
  assert.equal(t.resolutionReason, 'explicit-param');
});

test('resolver: explicit string unknown alias throws', async () => {
  const { resolver } = await buildResolver([makeRecord('work', SCOPE_DRIVE)]);
  await assert.rejects(
    () => resolver.resolve('nope', 'read', { sessionId: STDIO_SESSION_ID, acceptableScopes: [] }),
    /Unknown account/,
  );
});

test('resolver: explicit string without acceptable scopes throws the scope-shortage message', async () => {
  const { resolver } = await buildResolver([makeRecord('personal', SCOPE_READONLY)]);
  await assert.rejects(
    () =>
      resolver.resolve('personal', 'write', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    (err: Error) => {
      assert.match(err.message, /Account 'personal' is connected but lacks the required scope/);
      // Recovery must be non-destructive: a single in-place re-consent, no remove.
      assert.match(err.message, /manage_accounts add personal/);
      assert.doesNotMatch(err.message, /manage_accounts remove/);
      assert.match(err.message, new RegExp(SCOPE_DRIVE.replace(/\//g, '\\/')));
      return true;
    },
  );
});

test('resolver: explicit array on read returns fanout', async () => {
  const { resolver } = await buildResolver([
    makeRecord('a', SCOPE_DRIVE),
    makeRecord('b', SCOPE_DRIVE),
  ]);
  const t = await resolver.resolve(['a', 'b'], 'read', {
    sessionId: STDIO_SESSION_ID,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.kind, 'fanout');
  assert.deepEqual(
    t.accounts.map((a) => a.alias),
    ['a', 'b'],
  );
});

test('resolver: explicit array on write throws', async () => {
  const { resolver } = await buildResolver([
    makeRecord('a', SCOPE_DRIVE),
    makeRecord('b', SCOPE_DRIVE),
  ]);
  await assert.rejects(
    () =>
      resolver.resolve(['a', 'b'], 'write', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    /only be an array on read tools/,
  );
});

// ---------------------------------------------------------------------------
// Defaults (session / global)
// ---------------------------------------------------------------------------

test('resolver: session default beats global default', async () => {
  const { resolver, sessions } = await buildResolver(
    [makeRecord('work', SCOPE_DRIVE), makeRecord('personal', SCOPE_DRIVE)],
    'work',
  );
  const sid = 'session-1';
  sessions.getOrCreate(sid).defaultAccountAlias = 'personal';

  const t = await resolver.resolve(undefined, 'write', {
    sessionId: sid,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.accounts[0].alias, 'personal');
  assert.equal(t.resolutionReason, 'session-default');
});

test('resolver: global default used when no session default', async () => {
  const { resolver } = await buildResolver(
    [makeRecord('work', SCOPE_DRIVE), makeRecord('personal', SCOPE_DRIVE)],
    'work',
  );
  const t = await resolver.resolve(undefined, 'write', {
    sessionId: STDIO_SESSION_ID,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.accounts[0].alias, 'work');
  assert.equal(t.resolutionReason, 'global-default');
});

test('resolver: session default skipped if scopes insufficient', async () => {
  const { resolver, sessions } = await buildResolver(
    [makeRecord('work', SCOPE_DRIVE), makeRecord('personal', SCOPE_READONLY)],
    'work',
  );
  const sid = 'session-2';
  sessions.getOrCreate(sid).defaultAccountAlias = 'personal';

  // Write tool wants drive; session default (personal/readonly) is skipped;
  // falls to global default (work).
  const t = await resolver.resolve(undefined, 'write', {
    sessionId: sid,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.accounts[0].alias, 'work');
  assert.equal(t.resolutionReason, 'global-default');
});

test('resolver: scope-short global default errors instead of silently routing elsewhere (write)', async () => {
  // default 'personal' lacks drive; 'work' has it. A bare write must NOT be
  // silently rerouted to 'work' — surface the user's configured default instead.
  const { resolver } = await buildResolver(
    [makeRecord('personal', SCOPE_READONLY), makeRecord('work', SCOPE_DRIVE)],
    'personal',
  );
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'write', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    (err: Error) => {
      assert.match(err.message, /Account 'personal' is connected but lacks the required scope/);
      assert.match(err.message, /manage_accounts add personal/);
      // Must name the configured default, not the substitute account.
      assert.doesNotMatch(err.message, /\bwork\b/);
      return true;
    },
  );
});

test('resolver: scope-short default also errors for reads (uniform behavior)', async () => {
  const { resolver } = await buildResolver(
    [makeRecord('personal', SCOPE_READONLY), makeRecord('work', SCOPE_DRIVE)],
    'personal',
  );
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'read', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    /Account 'personal' is connected but lacks the required scope/,
  );
});

test('resolver: scope-short session default errors, naming the session alias', async () => {
  // No global default; the session default 'personal' lacks drive while 'work'
  // has it. Must error naming 'personal' (step-2 recording), not route to 'work'.
  const { resolver, sessions } = await buildResolver([
    makeRecord('personal', SCOPE_READONLY),
    makeRecord('work', SCOPE_DRIVE),
  ]);
  const sid = 'session-scope-short';
  sessions.getOrCreate(sid).defaultAccountAlias = 'personal';
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'write', {
        sessionId: sid,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    (err: Error) => {
      assert.match(err.message, /Account 'personal' is connected but lacks the required scope/);
      assert.doesNotMatch(err.message, /\bwork\b/);
      return true;
    },
  );
});

// ---------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------

test('resolver: sole-authenticated when exactly one account eligible', async () => {
  const { resolver } = await buildResolver([makeRecord('only', SCOPE_DRIVE)]);
  const t = await resolver.resolve(undefined, 'write', {
    sessionId: STDIO_SESSION_ID,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.accounts[0].alias, 'only');
  assert.equal(t.resolutionReason, 'sole-authenticated');
});

test('resolver: merged-eligible fanout for reads when multiple eligible', async () => {
  const { resolver } = await buildResolver([
    makeRecord('a', SCOPE_DRIVE),
    makeRecord('b', SCOPE_DRIVE),
  ]);
  const t = await resolver.resolve(undefined, 'read', {
    sessionId: STDIO_SESSION_ID,
    acceptableScopes: [SCOPE_DRIVE],
  });
  assert.equal(t.kind, 'fanout');
  assert.equal(t.resolutionReason, 'merged-eligible');
  assert.equal(t.accounts.length, 2);
});

test('resolver: ambiguous write points at manage_accounts set_default', async () => {
  const { resolver } = await buildResolver([
    makeRecord('a', SCOPE_DRIVE),
    makeRecord('b', SCOPE_DRIVE),
  ]);
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'write', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    (err: Error) => {
      assert.match(err.message, /Multiple accounts have required scopes/);
      assert.match(err.message, /manage_accounts set_default/);
      assert.match(err.message, /\ba\b.*\bb\b/);
      return true;
    },
  );
});

test('resolver: no accounts at all throws', async () => {
  const { resolver } = await buildResolver([]);
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'read', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    /No accounts are authenticated/,
  );
});

test('resolver: sole-account scope mismatch throws the scope-shortage message', async () => {
  const { resolver } = await buildResolver([makeRecord('only', SCOPE_READONLY)]);
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'write', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    (err: Error) => {
      assert.match(err.message, /Account 'only' is connected but lacks the required scope/);
      // Recovery must be non-destructive: a single in-place re-consent, no remove.
      assert.match(err.message, /manage_accounts add only/);
      assert.doesNotMatch(err.message, /manage_accounts remove/);
      return true;
    },
  );
});

test('resolver: multiple accounts, none eligible, throws the generic shortage message', async () => {
  const { resolver } = await buildResolver([
    makeRecord('a', SCOPE_READONLY),
    makeRecord('b', SCOPE_READONLY),
  ]);
  await assert.rejects(
    () =>
      resolver.resolve(undefined, 'write', {
        sessionId: STDIO_SESSION_ID,
        acceptableScopes: [SCOPE_DRIVE],
      }),
    (err: Error) => {
      assert.match(err.message, /No authenticated account has any of the required scopes/);
      assert.match(err.message, /manage_accounts add/);
      // Recovery advice must be non-destructive: in-place re-consent, no remove.
      assert.doesNotMatch(err.message, /manage_accounts remove/);
      return true;
    },
  );
});
