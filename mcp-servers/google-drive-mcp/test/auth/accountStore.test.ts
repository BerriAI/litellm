import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { AccountStore } from '../../src/auth/accountStore.js';
import { AccountRecord } from '../../src/auth/types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function mktmp(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-accountstore-'));
  return path.join(dir, 'tokens.json');
}

function makeRecord(alias: string, overrides: Partial<AccountRecord> = {}): AccountRecord {
  const now = new Date().toISOString();
  return {
    alias,
    email: `${alias}@example.com`,
    sub: `sub-${alias}`,
    accessToken: 'at-' + alias,
    refreshToken: 'rt-' + alias,
    scope: 'https://www.googleapis.com/auth/drive',
    tokenType: 'Bearer',
    expiryDate: Date.now() + 60 * 60 * 1000,
    addedAt: now,
    lastRefreshedAt: now,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Round-trip + atomic write
// ---------------------------------------------------------------------------

test('AccountStore upsert → reload round-trip', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  await store.upsert(makeRecord('work'));
  await store.setDefault('work');

  const reloaded = new AccountStore({ filePath, mode: 'local-oauth' });
  await reloaded.reload();
  const got = reloaded.get('work');
  assert.ok(got, 'work account should persist');
  assert.equal(got!.email, 'work@example.com');
  assert.equal(reloaded.getDefault(), 'work');
});

test('AccountStore writes v2 shape with version and accounts keys', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  await store.upsert(makeRecord('a'));
  const raw = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  assert.equal(raw.version, 2);
  assert.ok(raw.accounts.a);
  assert.equal(raw.accounts.a.alias, 'a');
});

test('AccountStore remove drops account and clears default if needed', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  await store.upsert(makeRecord('a'));
  await store.upsert(makeRecord('b'));
  await store.setDefault('a');
  await store.remove('a');
  assert.equal(store.get('a'), undefined);
  assert.equal(store.getDefault(), undefined);
  assert.ok(store.get('b'));
});

test('AccountStore setDefault rejects unknown alias', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  await assert.rejects(
    () => store.setDefault('nope'),
    /does not exist/,
  );
});

test('AccountStore concurrent upserts all persist', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  const count = 20;
  const ops: Promise<void>[] = [];
  for (let i = 0; i < count; i++) {
    ops.push(store.upsert(makeRecord(`acct${i}`)));
  }
  await Promise.all(ops);
  const reloaded = new AccountStore({ filePath, mode: 'local-oauth' });
  await reloaded.reload();
  assert.equal(reloaded.list().length, count);
  for (let i = 0; i < count; i++) {
    assert.ok(reloaded.get(`acct${i}`), `acct${i} should be present`);
  }
});

test('AccountStore write failure does not poison subsequent writes', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  // Trigger a write that rejects (see below) and assert it doesn't poison the queue.
  const firstBad: Promise<void> = (async () => {
    // setDefault of a missing alias throws inside the mutate callback → chained rejection.
    await assert.rejects(() => store.setDefault('does-not-exist'), /does not exist/);
  })();
  await firstBad;
  // Subsequent write should still work.
  await store.upsert(makeRecord('recovery'));
  assert.ok(store.get('recovery'));
});

// ---------------------------------------------------------------------------
// v1 → v2 migration
// ---------------------------------------------------------------------------

test('AccountStore migrates v1 single-account file to v2 with backup', async () => {
  const filePath = await mktmp();
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const v1 = {
    access_token: 'ya29.legacy',
    refresh_token: '1//legacy-refresh',
    scope: 'https://www.googleapis.com/auth/drive',
    token_type: 'Bearer',
    expiry_date: Date.now() + 3600 * 1000,
  };
  await fs.writeFile(filePath, JSON.stringify(v1), { mode: 0o600 });

  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();

  const rec = store.get('default');
  assert.ok(rec, 'default account should be created');
  assert.equal(rec!.refreshToken, '1//legacy-refresh');
  assert.equal(rec!.accessToken, 'ya29.legacy');
  assert.equal(rec!.pendingIdentity, true);
  assert.equal(rec!.email, 'unknown');
  assert.ok(rec!.sub.length > 0);
  assert.equal(store.getDefault(), 'default');

  // v2 file now present
  const current = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  assert.equal(current.version, 2);

  // Backup exists
  const dir = path.dirname(filePath);
  const entries = await fs.readdir(dir);
  assert.ok(
    entries.some((e) => e.includes('.v1-backup-')),
    `expected a v1 backup file in ${dir}, saw: ${entries.join(', ')}`,
  );
});

test('AccountStore reload on empty directory yields empty state', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  assert.equal(store.list().length, 0);
  assert.equal(store.getDefault(), undefined);
});

test('AccountStore throws on unknown version', async () => {
  const filePath = await mktmp();
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify({ version: 99, accounts: {} }));
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await assert.rejects(() => store.reload(), /Unrecognized tokens\.json format/);
});

// ---------------------------------------------------------------------------
// listRedacted / synthetic mode
// ---------------------------------------------------------------------------

test('AccountStore listRedacted omits tokens and marks default', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  await store.upsert(makeRecord('a', { scope: 'https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents' }));
  await store.upsert(makeRecord('b'));
  await store.setDefault('a');

  const redacted = store.listRedacted();
  assert.equal(redacted.length, 2);
  const a = redacted.find((r) => r.alias === 'a')!;
  assert.equal(a.isDefault, true);
  assert.deepEqual(a.scopesGranted.sort(), [
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/drive',
  ]);
  // No token fields leaked
  assert.equal((a as unknown as { accessToken?: string }).accessToken, undefined);
  assert.equal((a as unknown as { refreshToken?: string }).refreshToken, undefined);
});

test('AccountStore synthetic mode does not touch disk', async () => {
  const filePath = await mktmp();
  const store = new AccountStore({ filePath, mode: 'service-account' });
  await store.reload();
  store.setSyntheticAccount(makeRecord('service-account'), { request: () => {} });
  // File should not exist — synthetic mode is memory-only.
  let exists = true;
  try {
    await fs.access(filePath);
  } catch {
    exists = false;
  }
  assert.equal(exists, false);
  assert.ok(store.get('service-account'));
  assert.ok(store.getSyntheticClient('service-account'));
});

// ---------------------------------------------------------------------------
// Cross-process safety + corrupt-file self-heal (regression: PR #106 review)
// ---------------------------------------------------------------------------

test('AccountStore merges a concurrent writer instead of clobbering it (finding 2)', async () => {
  const filePath = await mktmp();
  // Two independent stores on the same file, both loaded while empty — simulating
  // two server processes (e.g. Claude Desktop + Cursor) sharing tokens.json.
  const a = new AccountStore({ filePath, mode: 'local-oauth' });
  const b = new AccountStore({ filePath, mode: 'local-oauth' });
  await a.reload();
  await b.reload();

  await a.upsert(makeRecord('work'));
  // b's in-memory snapshot predates 'work'; its write must not drop it.
  await b.upsert(makeRecord('personal'));

  const c = new AccountStore({ filePath, mode: 'local-oauth' });
  await c.reload();
  assert.ok(c.get('work'), "work (written by A) must survive B's write");
  assert.ok(c.get('personal'), 'personal (written by B) must be present');
});

test('AccountStore reload rejects a v2 file with null accounts (finding 14)', async () => {
  const filePath = await mktmp();
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify({ version: 2, accounts: null }));
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await assert.rejects(() => store.reload(), /Unrecognized tokens\.json format/);
});

test('AccountStore reload rejects a v2 file whose accounts is an array (finding 14)', async () => {
  const filePath = await mktmp();
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify({ version: 2, accounts: [] }));
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await assert.rejects(() => store.reload(), /Unrecognized tokens\.json format/);
});

test('AccountStore quarantines a corrupt file and resets to empty (finding 6)', async () => {
  const filePath = await mktmp();
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, '{ this is not valid json');
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  // reload() still throws (so programmatic callers/tests see the error)...
  await assert.rejects(() => store.reload(), /JSON|Unexpected|token/i);

  // ...but the boot layer can self-heal by quarantining the bad file.
  const backup = await store.quarantineCorruptFile();
  assert.ok(backup && backup.includes('.corrupt-'), `expected a .corrupt-* backup, got ${backup}`);
  await store.reload(); // original moved aside → ENOENT → empty
  assert.equal(store.list().length, 0);
  const entries = await fs.readdir(path.dirname(filePath));
  assert.ok(
    entries.some((e) => e.includes('.corrupt-')),
    `expected a corrupt backup file, saw: ${entries.join(', ')}`,
  );
});
