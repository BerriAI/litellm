import assert from 'node:assert/strict';
import test, { mock } from 'node:test';
import * as fs from 'node:fs/promises';
import * as os from 'node:os';
import * as path from 'node:path';

import { AccountStore } from '../../src/auth/accountStore.js';
import { AccountClientFactory } from '../../src/auth/accountClientFactory.js';
import { describeErrorForLog } from '../../src/auth/utils.js';
import type { AccountRecord } from '../../src/auth/types.js';

// ---------------------------------------------------------------------------
// Log redaction.
//
// Gaxios errors embed the full request config: for a token refresh that is the
// POST body containing the refresh token and client secret. Passing the raw
// error object to console.error prints all of it. These tests pin that the
// auth layer's error logs never contain credential material.
// ---------------------------------------------------------------------------

const SECRET_REFRESH_TOKEN = '1//super-secret-refresh-token';
const SECRET_CLIENT_SECRET = 'GOCSPX-super-secret-client-secret';

/** A gaxios-shaped error whose config carries the token-refresh POST body. */
function makeGaxiosError(): Error {
  const err = new Error('Request failed with status code 500') as Error & {
    code?: string;
    config?: { url?: string; data?: string };
    response?: { status?: number; data?: { error?: string; error_description?: string } };
  };
  err.code = '500';
  err.config = {
    url: 'https://oauth2.googleapis.com/token',
    data: `refresh_token=${SECRET_REFRESH_TOKEN}&client_secret=${SECRET_CLIENT_SECRET}&grant_type=refresh_token`,
  };
  err.response = { status: 500, data: { error: 'internal_failure' } };
  return err;
}

/** A gaxios-7-shaped error: top-level numeric `status`, deep-copied config,
 * and a native-fetch cause chain that also carries the request payload. */
function makeGaxiosV7Error(): Error {
  const undiciErr = Object.assign(new Error('socket hang up'), {
    code: 'UND_ERR_SOCKET',
    payload: `refresh_token=${SECRET_REFRESH_TOKEN}&client_secret=${SECRET_CLIENT_SECRET}`,
  });
  const fetchErr = new TypeError('fetch failed');
  (fetchErr as Error & { cause?: unknown }).cause = undiciErr;
  const err = new Error('Request failed with status code 500') as Error & {
    status?: number;
    config?: { url?: string; body?: string };
    response?: { status?: number; data?: { error?: string; error_description?: string } };
    cause?: unknown;
  };
  err.status = 500;
  err.config = {
    url: 'https://oauth2.googleapis.com/token',
    body: `refresh_token=${SECRET_REFRESH_TOKEN}&client_secret=${SECRET_CLIENT_SECRET}&grant_type=refresh_token`,
  };
  err.response = { status: 500, data: { error: 'internal_failure' } };
  err.cause = fetchErr;
  return err;
}

test('describeErrorForLog drops the gaxios-7 config and cause chain', () => {
  const rendered = describeErrorForLog(makeGaxiosV7Error());
  assert.ok(!rendered.includes(SECRET_REFRESH_TOKEN));
  assert.ok(!rendered.includes(SECRET_CLIENT_SECRET));
  assert.match(rendered, /Request failed with status code 500/);
  assert.match(rendered, /status=500/);
  assert.match(rendered, /error=internal_failure/);
});

test('describeErrorForLog keeps safe fields and drops the gaxios request config', () => {
  const rendered = describeErrorForLog(makeGaxiosError());
  assert.ok(!rendered.includes(SECRET_REFRESH_TOKEN));
  assert.ok(!rendered.includes(SECRET_CLIENT_SECRET));
  assert.match(rendered, /Request failed with status code 500/);
  assert.match(rendered, /status=500/);
  assert.match(rendered, /error=internal_failure/);
});

test('describeErrorForLog never echoes JSON.parse source fragments', () => {
  let syntaxErr: unknown;
  try {
    JSON.parse(`{"refreshToken": "${SECRET_REFRESH_TOKEN}"`);
  } catch (err) {
    syntaxErr = err;
  }
  const rendered = describeErrorForLog(syntaxErr);
  assert.equal(rendered, 'SyntaxError: invalid JSON');
});

test('describeErrorForLog handles non-object throwables', () => {
  assert.equal(describeErrorForLog('boom'), 'boom');
  assert.equal(describeErrorForLog(undefined), 'undefined');
  assert.equal(describeErrorForLog({}), 'unknown error');
});

// ---------------------------------------------------------------------------
// End-to-end through AccountClientFactory's two logging paths.
// ---------------------------------------------------------------------------

async function setupTmpCredentials(): Promise<{ tokenPath: string; cleanup: () => Promise<void> }> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'gdrive-mcp-redact-'));
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
  return { tokenPath, cleanup };
}

function makeRecord(overrides: Partial<AccountRecord> = {}): AccountRecord {
  const now = new Date().toISOString();
  return {
    alias: 'work',
    email: 'work@example.com',
    sub: 'sub-work',
    accessToken: 'old-access',
    refreshToken: SECRET_REFRESH_TOKEN,
    scope: 'https://www.googleapis.com/auth/drive',
    tokenType: 'Bearer',
    expiryDate: Date.now() + 3600_000,
    addedAt: now,
    lastRefreshedAt: now,
    ...overrides,
  };
}

/** Wait until `predicate()` returns truthy or `timeoutMs` elapses. Generous
 * default: under full-suite parallel load, 2s was routinely exceeded. */
async function waitFor(predicate: () => boolean, timeoutMs = 10_000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (predicate()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`waitFor timed out after ${timeoutMs}ms`);
}

test('transient refresh failure logs no credential material', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  const logged: string[] = [];
  const errMock = mock.method(console, 'error', (...args: unknown[]) => {
    logged.push(args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
  });
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store);
    const client = await factory.getClient('work');

    // Expire the cached credentials and make the refresh fail transiently
    // (non-invalid_grant), which hits the console.error path.
    client.setCredentials({ ...client.credentials, expiry_date: Date.now() - 60_000 });
    (client as unknown as { refreshAccessToken: () => Promise<never> }).refreshAccessToken =
      async () => {
        throw makeGaxiosError();
      };

    // Transient failures are swallowed; the call itself must not reject.
    await factory.getClient('work');

    const refreshLog = logged.find((l) => l.includes('Token refresh failed for "work"'));
    assert.ok(refreshLog, `expected a refresh-failure log, got: ${JSON.stringify(logged)}`);
    const allOutput = logged.join('\n');
    assert.ok(!allOutput.includes(SECRET_REFRESH_TOKEN), 'refresh token leaked into logs');
    assert.ok(!allOutput.includes(SECRET_CLIENT_SECRET), 'client secret leaked into logs');
    assert.match(refreshLog!, /error=internal_failure/);
  } finally {
    errMock.mock.restore();
    await cleanup();
  }
});

test('persist failure after a tokens event logs no credential material', async () => {
  const { tokenPath, cleanup } = await setupTmpCredentials();
  const logged: string[] = [];
  const errMock = mock.method(console, 'error', (...args: unknown[]) => {
    logged.push(args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
  });
  try {
    const store = new AccountStore({ filePath: tokenPath, mode: 'local-oauth' });
    await store.reload();
    await store.upsert(makeRecord());

    const factory = new AccountClientFactory(store);
    const client = await factory.getClient('work');

    // Make persistence fail with a gaxios-shaped error carrying secrets.
    (store as unknown as { upsert: () => Promise<never> }).upsert = async () => {
      throw makeGaxiosError();
    };
    client.emit('tokens', {
      access_token: 'new-access',
      expiry_date: Date.now() + 3600_000,
    });

    await waitFor(() => logged.some((l) => l.includes('Failed to persist refreshed tokens')));
    const allOutput = logged.join('\n');
    assert.ok(!allOutput.includes(SECRET_REFRESH_TOKEN), 'refresh token leaked into logs');
    assert.ok(!allOutput.includes(SECRET_CLIENT_SECRET), 'client secret leaked into logs');
  } finally {
    errMock.mock.restore();
    await cleanup();
  }
});
