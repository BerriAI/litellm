import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

// ---------------------------------------------------------------------------
// Boot-time migration via buildAuthSystem.
//
// The unit migration tests exercise AccountStore.reload in isolation; this
// test closes the gap by driving the full system bootstrapper used by the
// server at startup. It seeds a v1 tokens.json under a tmp path, points
// GOOGLE_DRIVE_MCP_TOKEN_PATH at it, and asserts buildAuthSystem returns a
// system with the migrated 'default' account already loaded.
// ---------------------------------------------------------------------------

async function withTokenPath<T>(fn: (tokenPath: string) => Promise<T>): Promise<T> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-boot-'));
  const tokenPath = path.join(dir, 'tokens.json');
  const saved = process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH;
  process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH = tokenPath;
  // Make sure no synthetic-mode env vars sneak in from the parent shell.
  const savedSA = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  const savedET = process.env.GOOGLE_DRIVE_MCP_ACCESS_TOKEN;
  delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
  delete process.env.GOOGLE_DRIVE_MCP_ACCESS_TOKEN;
  try {
    return await fn(tokenPath);
  } finally {
    if (saved === undefined) delete process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH;
    else process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH = saved;
    if (savedSA !== undefined) process.env.GOOGLE_APPLICATION_CREDENTIALS = savedSA;
    if (savedET !== undefined) process.env.GOOGLE_DRIVE_MCP_ACCESS_TOKEN = savedET;
  }
}

test('buildAuthSystem migrates a pre-existing v1 tokens.json on first call', async () => {
  await withTokenPath(async (tokenPath) => {
    const v1 = {
      access_token: 'ya29.legacy',
      refresh_token: '1//legacy-refresh',
      scope: 'https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents',
      token_type: 'Bearer',
      expiry_date: Date.now() + 3600 * 1000,
    };
    await fs.writeFile(tokenPath, JSON.stringify(v1, null, 2), { mode: 0o600 });

    const { buildAuthSystem } = await import('../../src/auth.js');
    const system = await buildAuthSystem();

    assert.equal(system.mode, 'local-oauth');
    const accounts = system.store.list();
    assert.equal(accounts.length, 1, `expected 1 migrated account, saw ${accounts.length}`);
    assert.equal(accounts[0].alias, 'default');
    assert.equal(accounts[0].refreshToken, '1//legacy-refresh');
    assert.equal(accounts[0].accessToken, 'ya29.legacy');
    assert.equal(accounts[0].pendingIdentity, true);
    assert.equal(system.store.getDefault(), 'default');

    // Live file is now v2.
    const live = JSON.parse(await fs.readFile(tokenPath, 'utf-8'));
    assert.equal(live.version, 2);
    assert.ok(live.accounts.default);

    // Backup exists alongside.
    const dir = path.dirname(tokenPath);
    const entries = await fs.readdir(dir);
    const backups = entries.filter((e) => /\.v1-backup-\d+$/.test(e));
    assert.equal(backups.length, 1, `expected one v1 backup, saw: ${entries.join(', ')}`);
  });
});

test('buildAuthSystem on second invocation reads the already-migrated v2 file (no re-migration)', async () => {
  await withTokenPath(async (tokenPath) => {
    const v1 = {
      access_token: 'a',
      refresh_token: 'r',
      scope: 'https://www.googleapis.com/auth/drive',
    };
    await fs.writeFile(tokenPath, JSON.stringify(v1));

    const { buildAuthSystem } = await import('../../src/auth.js');
    const first = await buildAuthSystem();
    const subBefore = first.store.get('default')!.sub;

    // Second call: file is now v2.
    const second = await buildAuthSystem();
    assert.equal(second.store.get('default')!.sub, subBefore);

    // Only one backup exists (no second migration).
    const entries = await fs.readdir(path.dirname(tokenPath));
    const backups = entries.filter((e) => /\.v1-backup-\d+$/.test(e));
    assert.equal(backups.length, 1, `expected 1 backup, saw: ${entries.join(', ')}`);
  });
});

test('buildAuthSystem assembles resolver/factory/sessions over the migrated state', async () => {
  await withTokenPath(async (tokenPath) => {
    await fs.writeFile(
      tokenPath,
      JSON.stringify({
        access_token: 'a',
        refresh_token: 'r',
        scope: 'https://www.googleapis.com/auth/drive',
      }),
    );
    const { buildAuthSystem } = await import('../../src/auth.js');
    const system = await buildAuthSystem();

    // Migrated account is auto-set as the global default → resolver should
    // hit the global-default branch.
    const targeting = await system.resolver.resolve(undefined, 'read', {
      sessionId: 'stdio',
      acceptableScopes: [],
    });
    assert.equal(targeting.kind, 'single');
    assert.equal(targeting.accounts[0].alias, 'default');
    assert.equal(targeting.resolutionReason, 'global-default');

    // SessionStore is fresh.
    assert.equal(system.sessions.size(), 0);
  });
});
