import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';
import { google } from 'googleapis';
import { OAuth2Client } from 'google-auth-library';

import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';
import {
  AccountClientFactory,
  AccountResolver,
  AccountStore,
  DEFAULT_SCOPES,
  SessionStore,
  type AuthSystem,
} from '../../src/auth.js';
import type { AccountRecord } from '../../src/auth/types.js';

// ---------------------------------------------------------------------------
// Re-consent must invalidate every cached client (issue #168).
//
// `manage_accounts add <existing-alias>` re-consents in place. Before the fix
// it only wrote the new record: the factory kept handing out the OAuth2Client
// built from the superseded (often revoked) grant, and the per-alias Drive and
// Calendar services stayed bound to it, so every call failed until the server
// was restarted.
//
// This drives the real flow — a local-OAuth AuthSystem over a temp tokens.json,
// the real AuthServer answering a real HTTP callback — with the code-for-tokens
// exchange stubbed, so no network reaches Google.
// ---------------------------------------------------------------------------

const AUTH_PORT = 18680;
const OLD_ACCESS = 'old-access';
const NEW_ACCESS = 'reauth-access';
const HOUR_MS = 60 * 60 * 1000;

function makeStoredRecord(): AccountRecord {
  const now = new Date().toISOString();
  return {
    alias: 'work',
    email: 'work@example.com',
    sub: 'sub-work',
    accessToken: OLD_ACCESS,
    refreshToken: 'old-refresh',
    scope: DEFAULT_SCOPES.join(' '),
    tokenType: 'Bearer',
    // Far-future expiry so nothing in this test triggers a real token refresh.
    expiryDate: Date.now() + HOUR_MS,
    addedAt: now,
    lastRefreshedAt: now,
  };
}

/** Stub the code-for-tokens exchange; returns a restore function. */
function stubGetToken(): () => void {
  const original = OAuth2Client.prototype.getToken;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (OAuth2Client.prototype as any).getToken = async () => ({
    tokens: {
      access_token: NEW_ACCESS,
      refresh_token: 'reauth-refresh',
      scope: DEFAULT_SCOPES.join(' '),
      token_type: 'Bearer',
      expiry_date: Date.now() + HOUR_MS,
    },
    res: null,
  });
  return () => {
    OAuth2Client.prototype.getToken = original;
  };
}

/**
 * Fail the userinfo lookup `addAccountFlow` performs after consent. Keeps the
 * test offline and exercises the documented fallback: a re-consent must not
 * regress a known email/sub when identity resolution fails.
 */
function stubRequestFailure(): () => void {
  const original = OAuth2Client.prototype.request;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (OAuth2Client.prototype as any).request = async () => {
    throw new Error('userinfo unavailable in tests');
  };
  return () => {
    OAuth2Client.prototype.request = original;
  };
}

describe('re-consent evicts cached clients (issue #168)', () => {
  let ctx: TestContext;
  let serverModule: any;
  let sys: AuthSystem;
  let tmpDir: string;
  let tokenPath: string;
  let originalDriveFactory: typeof google.drive;
  let driveAuths: unknown[] = [];
  const savedEnv: Record<string, string | undefined> = {};
  const restores: Array<() => void> = [];

  before(async () => {
    ctx = await setupTestServer();
    serverModule = await import('../../src/index.js');

    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-reauth-'));
    tokenPath = path.join(tmpDir, 'tokens.json');
    const credsPath = path.join(tmpDir, 'gcp-oauth.keys.json');
    await fs.writeFile(
      credsPath,
      JSON.stringify({
        installed: {
          client_id: 'test-client-id.apps.googleusercontent.com',
          client_secret: 'test-client-secret',
          redirect_uris: [`http://127.0.0.1:${AUTH_PORT}/oauth2callback`],
        },
      }),
    );
    await fs.writeFile(
      tokenPath,
      JSON.stringify({
        version: 2,
        defaultAccount: 'work',
        accounts: { work: makeStoredRecord() },
      }),
    );

    for (const key of [
      'GOOGLE_DRIVE_MCP_TOKEN_PATH',
      'GOOGLE_DRIVE_OAUTH_CREDENTIALS',
      'GOOGLE_DRIVE_MCP_AUTH_PORT',
    ]) {
      savedEnv[key] = process.env[key];
    }
    process.env.GOOGLE_DRIVE_MCP_TOKEN_PATH = tokenPath;
    process.env.GOOGLE_DRIVE_OAUTH_CREDENTIALS = credsPath;
    process.env.GOOGLE_DRIVE_MCP_AUTH_PORT = String(AUTH_PORT);

    // A real local-OAuth auth system in place of the harness's 'test' mode one,
    // so the account-lifecycle flows don't short-circuit in requireLocalOAuthMode.
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    const sessions = new SessionStore();
    const factory = new AccountClientFactory(store);
    sys = {
      mode: 'local-oauth',
      store,
      factory,
      resolver: new AccountResolver(store, sessions),
      sessions,
    };
    serverModule._setAuthSystemForTesting(sys);

    // Record the auth client every Drive service is constructed with, so a
    // cached-vs-rebuilt service is directly observable.
    originalDriveFactory = google.drive as any;
    (google as any).drive = (opts: any) => {
      driveAuths.push(opts?.auth);
      return originalDriveFactory(opts);
    };

    restores.push(stubGetToken(), stubRequestFailure());
  });

  after(async () => {
    (google as any).drive = originalDriveFactory;
    for (const restore of restores) restore();
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    // Restore the shared 'test'-mode auth system for any later test in this file
    // and leave the module in the state other suites expect.
    serverModule._setAuthClientForTesting({});
    await ctx.cleanup();
    // The temp dir holds a tokens.json and a credentials file; don't leave them
    // behind in os.tmpdir() on every run.
    if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it('a completed re-consent takes effect without restarting the server', async () => {
    // 1. Prime the caches: this builds the factory client plus the per-alias
    //    Drive and Calendar services from the stored (soon-to-be-stale) grant.
    const primed = await callTool(ctx.client, 'search', { query: 'q', account: 'work' });
    assert.notEqual(primed.isError, true, `priming call errored: ${JSON.stringify(primed)}`);

    const clientBefore = await sys.factory.getClient('work');
    assert.equal(clientBefore.credentials.access_token, OLD_ACCESS);
    assert.ok(driveAuths.length > 0, 'priming call should have built a Drive service');
    driveAuths = [];

    // 2. Re-consent the same alias through the real flow.
    const { completion } = await serverModule._addAccountFlowForTesting('work');
    // 127.0.0.1, not `localhost`: AuthServer binds the loopback IP explicitly so
    // the bind and the redirect URI agree on dual-stack hosts (see auth/server.ts).
    const callback = await fetch(`http://127.0.0.1:${AUTH_PORT}/oauth2callback?code=x`);
    assert.equal(callback.status, 200);
    const record = await completion;

    // The failed userinfo lookup must not regress the known identity.
    assert.equal(record.email, 'work@example.com');
    assert.equal(record.sub, 'sub-work');

    // 3. The factory must hand out a new client carrying the new grant.
    const clientAfter = await sys.factory.getClient('work');
    assert.notEqual(clientAfter, clientBefore, 'factory returned the superseded OAuth2Client');
    assert.equal(clientAfter.credentials.access_token, NEW_ACCESS);

    // 4. And the next tool call must rebuild its Drive service on that client
    //    instead of reusing the one cached under this alias.
    const after = await callTool(ctx.client, 'search', { query: 'q', account: 'work' });
    assert.notEqual(after.isError, true, `post-reauth call errored: ${JSON.stringify(after)}`);
    assert.ok(
      driveAuths.length > 0,
      'post-reauth call reused the cached Drive service instead of rebuilding it',
    );
    for (const auth of driveAuths) {
      assert.notEqual(auth, clientBefore, 'Drive service was rebuilt on the superseded client');
    }

    // 5. The new grant is persisted, so a restart would agree with memory.
    const onDisk = JSON.parse(await fs.readFile(tokenPath, 'utf8'));
    assert.equal(onDisk.accounts.work.accessToken, NEW_ACCESS);
    assert.equal(onDisk.accounts.work.email, 'work@example.com');
  });
});
