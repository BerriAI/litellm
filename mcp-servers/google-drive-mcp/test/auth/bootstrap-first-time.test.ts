import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { OAuth2Client } from 'google-auth-library';

import { AccountStore, buildRecordFromV1 } from '../../src/auth/accountStore.js';
import { DEFAULT_SCOPES } from '../../src/auth/scopes.js';
import { AuthServer } from '../../src/auth/server.js';
import { buildAuthSystem } from '../../src/auth.js';

// ---------------------------------------------------------------------------
// First-time bootstrap.
//
// buildAuthSystem's empty-store branch persists the fresh OAuth grant straight
// into the v2 store under the reserved alias 'default' via AuthServer's
// onTokens callback — the flat v1 file must never be written and no
// .v1-backup-* artifact may appear. These tests drive that path with a real
// AuthServer over HTTP (no network to Google: getToken is stubbed).
// ---------------------------------------------------------------------------

async function setupTmpEnv(authPort: number): Promise<{
  dir: string;
  tokenPath: string;
  restore: () => void;
}> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-bootstrap-'));
  const tokenPath = path.join(dir, 'tokens.json');
  const credsPath = path.join(dir, 'gcp-oauth.keys.json');

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
    GOOGLE_DRIVE_MCP_AUTH_PORT: process.env.GOOGLE_DRIVE_MCP_AUTH_PORT,
  };
  process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH = tokenPath;
  process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS = credsPath;
  process.env.GOOGLE_DRIVE_MCP_AUTH_PORT = String(authPort);

  const restore = () => {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };

  return { dir, tokenPath, restore };
}

const FAKE_TOKENS = {
  access_token: 'bootstrap-access',
  refresh_token: 'bootstrap-refresh',
  scope: 'https://www.googleapis.com/auth/drive',
  token_type: 'Bearer',
  expiry_date: Date.now() + 3600_000,
};

/** Stub the code-for-tokens exchange; returns a restore function. */
function stubGetToken(tokens: Record<string, unknown> = FAKE_TOKENS): () => void {
  const original = OAuth2Client.prototype.getToken;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (OAuth2Client.prototype as any).getToken = async () => ({ tokens, res: null });
  return () => {
    OAuth2Client.prototype.getToken = original;
  };
}

async function listBackupFiles(dir: string): Promise<string[]> {
  return (await fs.readdir(dir)).filter((f) => f.includes('.v1-backup-'));
}

test('buildRecordFromV1 converts a Credentials-shaped grant to the migration-parity record', () => {
  const record = buildRecordFromV1({
    access_token: 'acc',
    refresh_token: 'ref',
    scope: 'https://www.googleapis.com/auth/drive',
    expiry_date: 123,
    token_type: 'Bearer',
  });
  assert.equal(record.alias, 'default');
  assert.equal(record.email, 'unknown');
  assert.equal(record.pendingIdentity, true);
  assert.equal(record.accessToken, 'acc');
  assert.equal(record.refreshToken, 'ref');
  assert.equal(record.scope, 'https://www.googleapis.com/auth/drive');
  assert.equal(record.expiryDate, 123);

  // Null fields (google-auth Credentials allows null) must not leak through.
  const nullish = buildRecordFromV1({
    access_token: 'only-access',
    refresh_token: null,
    scope: null,
    expiry_date: null,
    token_type: null,
  });
  assert.equal(nullish.refreshToken, '');
  assert.equal(nullish.expiryDate, 0);
  // Missing scope falls back to the full default set (same as v1 migration).
  assert.equal(nullish.scope, DEFAULT_SCOPES.join(' '));

  // sub is deterministic for the same token material, and derived from
  // access_token when there is no refresh_token.
  assert.equal(
    buildRecordFromV1({ refresh_token: 'ref' }).sub,
    buildRecordFromV1({ refresh_token: 'ref' }).sub,
  );
  assert.notEqual(
    buildRecordFromV1({ refresh_token: 'ref' }).sub,
    buildRecordFromV1({ access_token: 'only-access' }).sub,
  );
});

test('AuthServer callback persists v2-only via onTokens (no flat v1 file, no backup)', async () => {
  const { dir, tokenPath, restore } = await setupTmpEnv(18620);
  const restoreGetToken = stubGetToken();
  const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
  await store.reload();

  const authServer = new AuthServer(new OAuth2Client('id', 'secret'), {
    onTokens: async (tokens) => {
      const record = buildRecordFromV1(tokens);
      await store.upsert(record);
      if (!store.getDefault()) await store.setDefault(record.alias);
    },
  });
  try {
    assert.equal(await authServer.start(false), true);
    const port = authServer.getRunningPort();
    assert.ok(port, 'auth server should be listening');

    const res = await fetch(`http://localhost:${port}/oauth2callback?code=x`);
    assert.equal(res.status, 200);
    const body = await res.text();
    // Success page must not disclose the token file location.
    assert.ok(!body.includes(dir), 'success page must not contain the token path');
    assert.match(body, /close this browser window/i);
    assert.equal(authServer.authCompletedSuccessfully, true);

    const onDisk = JSON.parse(await fs.readFile(tokenPath, 'utf8'));
    assert.equal(onDisk.version, 2);
    assert.equal(onDisk.defaultAccount, 'default');
    assert.equal(onDisk.accounts.default.refreshToken, 'bootstrap-refresh');
    assert.equal((await listBackupFiles(dir)).length, 0);
  } finally {
    await authServer.stop();
    restoreGetToken();
    restore();
  }
});

test('buildAuthSystem first-time bootstrap persists alias "default" end-to-end', async () => {
  const { dir, tokenPath, restore } = await setupTmpEnv(18640);
  const restoreGetToken = stubGetToken();
  try {
    const systemPromise = buildAuthSystem({ openBrowser: false });

    // Poll until the callback server is up, then complete the "consent".
    const deadline = Date.now() + 10_000;
    let completed = false;
    while (!completed && Date.now() < deadline) {
      try {
        const res = await fetch('http://localhost:18640/oauth2callback?code=x');
        completed = res.status === 200;
      } catch {
        await new Promise((r) => setTimeout(r, 50));
      }
    }
    assert.equal(completed, true, 'OAuth callback should complete');

    const system = await systemPromise;
    assert.equal(system.mode, 'local-oauth');
    assert.deepEqual(system.store.list().map((r) => r.alias), ['default']);
    assert.equal(system.store.getDefault(), 'default');

    const onDisk = JSON.parse(await fs.readFile(tokenPath, 'utf8'));
    assert.equal(onDisk.version, 2);
    assert.equal(onDisk.accounts.default.accessToken, 'bootstrap-access');
    assert.equal((await listBackupFiles(dir)).length, 0);
  } finally {
    restoreGetToken();
    restore();
  }
});

test('AuthServer failure path: onTokens throw renders failure page, persists nothing', async () => {
  const { tokenPath, restore } = await setupTmpEnv(18660);
  const restoreGetToken = stubGetToken();
  const authServer = new AuthServer(new OAuth2Client('id', 'secret'), {
    onTokens: async () => {
      throw new Error('store exploded');
    },
  });
  try {
    assert.equal(await authServer.start(false), true);
    const port = authServer.getRunningPort();
    assert.ok(port, 'auth server should be listening');

    const res = await fetch(`http://localhost:${port}/oauth2callback?code=x`);
    assert.equal(res.status, 500);
    assert.match(await res.text(), /Authentication Failed/);
    assert.equal(authServer.authCompletedSuccessfully, false);
    await assert.rejects(fs.access(tokenPath), 'tokens.json must not exist');
  } finally {
    await authServer.stop();
    restoreGetToken();
    restore();
  }
});
