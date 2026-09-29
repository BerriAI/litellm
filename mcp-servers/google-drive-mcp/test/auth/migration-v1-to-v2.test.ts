import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { AccountStore } from '../../src/auth/accountStore.js';
import { coversScopes } from '../../src/auth/accountResolver.js';
import { DEFAULT_SCOPES } from '../../src/auth/scopes.js';

const DRIVE = 'https://www.googleapis.com/auth/drive';
const DRIVE_FILE = 'https://www.googleapis.com/auth/drive.file';

async function mktmpDir(): Promise<string> {
  return fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-migration-'));
}

// ---------------------------------------------------------------------------
// Happy-path v1 → v2 migration
// ---------------------------------------------------------------------------

test('v1 file with full credentials migrates to v2, preserves tokens, backs up', async () => {
  const dir = await mktmpDir();
  const filePath = path.join(dir, 'tokens.json');
  const v1 = {
    access_token: 'ya29.FULL',
    refresh_token: '1//REFRESH',
    scope: 'https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents',
    token_type: 'Bearer',
    expiry_date: Date.now() + 3600 * 1000,
  };
  await fs.writeFile(filePath, JSON.stringify(v1, null, 2), { mode: 0o600 });

  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();

  // Account preserved
  const rec = store.get('default');
  assert.ok(rec);
  assert.equal(rec!.accessToken, 'ya29.FULL');
  assert.equal(rec!.refreshToken, '1//REFRESH');
  assert.equal(rec!.scope, v1.scope);
  assert.equal(rec!.tokenType, 'Bearer');
  assert.equal(rec!.expiryDate, v1.expiry_date);
  assert.equal(rec!.email, 'unknown');
  assert.equal(rec!.pendingIdentity, true);

  // Default set
  assert.equal(store.getDefault(), 'default');

  // Live file is v2
  const live = JSON.parse(await fs.readFile(filePath, 'utf-8'));
  assert.equal(live.version, 2);
  assert.ok(live.accounts.default);
  assert.equal(live.defaultAccount, 'default');

  // Backup preserved
  const entries = await fs.readdir(dir);
  const backup = entries.find((e) => /\.v1-backup-\d+$/.test(e));
  assert.ok(backup, `expected a v1 backup file in ${dir}, saw: ${entries.join(', ')}`);
  const backupContent = JSON.parse(await fs.readFile(path.join(dir, backup!), 'utf-8'));
  assert.deepEqual(backupContent, v1);
});

test('v1 file with only refresh_token still migrates (no access_token)', async () => {
  const dir = await mktmpDir();
  const filePath = path.join(dir, 'tokens.json');
  await fs.writeFile(
    filePath,
    JSON.stringify({ refresh_token: '1//r' }, null, 2),
    { mode: 0o600 },
  );
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await store.reload();
  const rec = store.get('default');
  assert.ok(rec);
  assert.equal(rec!.refreshToken, '1//r');
  assert.equal(rec!.accessToken, '');
  assert.equal(rec!.pendingIdentity, true);
  // A scope-less v1 file must migrate to the full default scope set, not '',
  // so the resolver's scope gate doesn't reject every tool (finding 8).
  assert.equal(rec!.scope, DEFAULT_SCOPES.join(' '));
  assert.ok(
    coversScopes(rec!.scope, [DRIVE, DRIVE_FILE]),
    'migrated scope-less account must remain eligible for drive tools',
  );
});

test('second reload of an already-migrated v2 file is a no-op', async () => {
  const dir = await mktmpDir();
  const filePath = path.join(dir, 'tokens.json');
  await fs.writeFile(
    filePath,
    JSON.stringify({ access_token: 'a', refresh_token: 'r' }),
    { mode: 0o600 },
  );

  const first = new AccountStore({ filePath, mode: 'local-oauth' });
  await first.reload();
  const firstSub = first.get('default')!.sub;

  const second = new AccountStore({ filePath, mode: 'local-oauth' });
  await second.reload();
  assert.equal(second.get('default')!.sub, firstSub);
  // Only one backup file exists — the second reload did not re-migrate.
  const entries = await fs.readdir(dir);
  const backups = entries.filter((e) => /\.v1-backup-\d+$/.test(e));
  assert.equal(backups.length, 1);
});

// ---------------------------------------------------------------------------
// sub is deterministic from refresh token (for identity continuity)
// ---------------------------------------------------------------------------

test('sub is stable across two v1 migrations with the same refresh token', async () => {
  const dir1 = await mktmpDir();
  const dir2 = await mktmpDir();
  const v1 = { access_token: 'a', refresh_token: '1//same-token' };
  await fs.writeFile(path.join(dir1, 'tokens.json'), JSON.stringify(v1));
  await fs.writeFile(path.join(dir2, 'tokens.json'), JSON.stringify(v1));

  const s1 = new AccountStore({ filePath: path.join(dir1, 'tokens.json'), mode: 'local-oauth' });
  const s2 = new AccountStore({ filePath: path.join(dir2, 'tokens.json'), mode: 'local-oauth' });
  await s1.reload();
  await s2.reload();
  assert.equal(s1.get('default')!.sub, s2.get('default')!.sub);
});

// ---------------------------------------------------------------------------
// Corrupted-file detection
// ---------------------------------------------------------------------------

test('malformed JSON causes reload to throw (not silently clobber)', async () => {
  const dir = await mktmpDir();
  const filePath = path.join(dir, 'tokens.json');
  await fs.writeFile(filePath, '{this-is-not-json', { mode: 0o600 });
  const store = new AccountStore({ filePath, mode: 'local-oauth' });
  await assert.rejects(() => store.reload(), /JSON|SyntaxError/);
});
