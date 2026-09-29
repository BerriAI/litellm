import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  isExternalTokenMode,
  isServiceAccountMode,
  validateExternalTokenConfig,
  createExternalOAuth2Client,
  buildServiceAccountAuthOptions,
  validateCredentialsFile,
  describeBypassedTokens,
} from '../src/auth/externalAuth.js';
import { SCOPE_ALIASES } from '../src/auth/scopes.js';
import { setEnv } from './helpers/env.js';

// ---------------------------------------------------------------------------
// Helpers — save & restore env vars around each test
// ---------------------------------------------------------------------------
const EXTERNAL_VARS = [
  'GOOGLE_DRIVE_MCP_ACCESS_TOKEN',
  'GOOGLE_DRIVE_MCP_REFRESH_TOKEN',
  'GOOGLE_DRIVE_MCP_CLIENT_ID',
  'GOOGLE_DRIVE_MCP_CLIENT_SECRET',
  'GOOGLE_APPLICATION_CREDENTIALS',
  'GOOGLE_DRIVE_MCP_SUBJECT',
  'GOOGLE_DRIVE_MCP_SCOPES',
] as const;

// Wraps a test body: force-clears every EXTERNAL_VAR (so ambient config can't
// leak in), applies `vars`, and restores everything afterward — built on the
// shared withEnv primitive so the save/restore logic lives in one place.
function withEnv(vars: Record<string, string>, fn: () => void | Promise<void>) {
  return async () => {
    const cleared: Record<string, string | undefined> = {};
    for (const v of EXTERNAL_VARS) cleared[v] = undefined;
    const env = setEnv({ ...cleared, ...vars });
    try {
      await fn();
    } finally {
      env.restore();
    }
  };
}

// ---------------------------------------------------------------------------
// isExternalTokenMode
// ---------------------------------------------------------------------------
test('isExternalTokenMode returns true when access token is set', withEnv(
  { GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test' },
  () => { assert.equal(isExternalTokenMode(), true); },
));

test('isExternalTokenMode returns false when access token is not set', withEnv(
  {},
  () => { assert.equal(isExternalTokenMode(), false); },
));

// ---------------------------------------------------------------------------
// isServiceAccountMode
// ---------------------------------------------------------------------------
test('isServiceAccountMode returns true when GOOGLE_APPLICATION_CREDENTIALS is set', withEnv(
  { GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa-key.json' },
  () => { assert.equal(isServiceAccountMode(), true); },
));

test('isServiceAccountMode returns false when not set', withEnv(
  {},
  () => { assert.equal(isServiceAccountMode(), false); },
));

// ---------------------------------------------------------------------------
// buildServiceAccountAuthOptions
// ---------------------------------------------------------------------------
test('buildServiceAccountAuthOptions returns keyFile + default scopes when only GOOGLE_APPLICATION_CREDENTIALS is set', withEnv(
  { GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa-key.json' },
  () => {
    const opts = buildServiceAccountAuthOptions();
    assert.equal(opts.keyFile, '/tmp/sa-key.json');
    assert.ok(Array.isArray(opts.scopes), 'scopes should be an array');
    assert.ok((opts.scopes as string[]).length > 0, 'default scopes should be non-empty');
    assert.equal(opts.clientOptions, undefined, 'clientOptions must be omitted when no subject is set');
  },
));

test('buildServiceAccountAuthOptions sets clientOptions.subject for domain-wide delegation', withEnv(
  {
    GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa-key.json',
    GOOGLE_DRIVE_MCP_SUBJECT: 'bot@example.com',
  },
  () => {
    const opts = buildServiceAccountAuthOptions();
    assert.deepEqual(opts.clientOptions, { subject: 'bot@example.com' });
  },
));

test('buildServiceAccountAuthOptions trims whitespace around the subject', withEnv(
  {
    GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa-key.json',
    GOOGLE_DRIVE_MCP_SUBJECT: '  bot@example.com  ',
  },
  () => {
    const opts = buildServiceAccountAuthOptions();
    assert.deepEqual(opts.clientOptions, { subject: 'bot@example.com' });
  },
));

test('buildServiceAccountAuthOptions omits clientOptions when GOOGLE_DRIVE_MCP_SUBJECT is blank', withEnv(
  {
    GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa-key.json',
    GOOGLE_DRIVE_MCP_SUBJECT: '   ',
  },
  () => {
    const opts = buildServiceAccountAuthOptions();
    assert.equal(opts.clientOptions, undefined);
  },
));

test('buildServiceAccountAuthOptions honors GOOGLE_DRIVE_MCP_SCOPES', withEnv(
  {
    GOOGLE_APPLICATION_CREDENTIALS: '/tmp/sa-key.json',
    GOOGLE_DRIVE_MCP_SCOPES: 'drive.file,documents',
  },
  () => {
    const opts = buildServiceAccountAuthOptions();
    assert.deepEqual(opts.scopes, [SCOPE_ALIASES['drive.file'], SCOPE_ALIASES['documents']]);
  },
));

// ---------------------------------------------------------------------------
// validateExternalTokenConfig
// ---------------------------------------------------------------------------
test('validates successfully with access token only', withEnv(
  { GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test' },
  () => { assert.doesNotThrow(() => validateExternalTokenConfig()); },
));

test('throws on empty access token', withEnv(
  { GOOGLE_DRIVE_MCP_ACCESS_TOKEN: '  ' },
  () => {
    assert.throws(
      () => validateExternalTokenConfig(),
      /GOOGLE_DRIVE_MCP_ACCESS_TOKEN is set but empty/,
    );
  },
));

test('throws when refresh token set without client credentials', withEnv(
  {
    GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test',
    GOOGLE_DRIVE_MCP_REFRESH_TOKEN: '1//refresh',
  },
  () => {
    assert.throws(
      () => validateExternalTokenConfig(),
      /GOOGLE_DRIVE_MCP_CLIENT_ID and\/or.*GOOGLE_DRIVE_MCP_CLIENT_SECRET are missing/,
    );
  },
));

test('throws when only client ID provided without client secret', withEnv(
  {
    GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test',
    GOOGLE_DRIVE_MCP_CLIENT_ID: 'id.apps.googleusercontent.com',
  },
  () => {
    assert.throws(
      () => validateExternalTokenConfig(),
      /Both GOOGLE_DRIVE_MCP_CLIENT_ID and GOOGLE_DRIVE_MCP_CLIENT_SECRET must be provided together/,
    );
  },
));

test('validates successfully with full credential set', withEnv(
  {
    GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test',
    GOOGLE_DRIVE_MCP_REFRESH_TOKEN: '1//refresh',
    GOOGLE_DRIVE_MCP_CLIENT_ID: 'id.apps.googleusercontent.com',
    GOOGLE_DRIVE_MCP_CLIENT_SECRET: 'GOCSPX-secret',
  },
  () => { assert.doesNotThrow(() => validateExternalTokenConfig()); },
));

// ---------------------------------------------------------------------------
// createExternalOAuth2Client
// ---------------------------------------------------------------------------
test('creates OAuth2Client with access token', withEnv(
  { GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test-token' },
  () => {
    const client = createExternalOAuth2Client();
    assert.equal(client.credentials.access_token, 'ya29.test-token');
    assert.equal(client.credentials.refresh_token, undefined);
  },
));

test('creates OAuth2Client with full credentials', withEnv(
  {
    GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test-token',
    GOOGLE_DRIVE_MCP_REFRESH_TOKEN: '1//refresh-token',
    GOOGLE_DRIVE_MCP_CLIENT_ID: 'test-client-id',
    GOOGLE_DRIVE_MCP_CLIENT_SECRET: 'test-client-secret',
  },
  () => {
    const client = createExternalOAuth2Client();
    assert.equal(client.credentials.access_token, 'ya29.test-token');
    assert.equal(client.credentials.refresh_token, '1//refresh-token');
  },
));

// ---------------------------------------------------------------------------
// authenticate() integration — priority order
// ---------------------------------------------------------------------------
test('authenticate prefers service account over external token', withEnv(
  {
    GOOGLE_APPLICATION_CREDENTIALS: '/tmp/nonexistent-sa-key.json',
    GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test',
  },
  async () => {
    // Service account mode should be chosen, but it will fail because the key
    // file doesn't exist. The important thing is that it doesn't fall through
    // to external token mode.
    const { authenticate } = await import('../src/auth.js');
    await assert.rejects(
      () => authenticate(),
      // GoogleAuth will throw about the missing key file
      (err: any) => err !== undefined,
    );
    // Confirm service account mode was selected (external token mode would succeed)
    assert.equal(isServiceAccountMode(), true);
    assert.equal(isExternalTokenMode(), true); // both set, but SA takes priority
  },
));

test('authenticate uses external token when no service account', withEnv(
  {
    GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test',
  },
  async () => {
    const { authenticate } = await import('../src/auth.js');
    const client = await authenticate();
    assert.equal(client.credentials.access_token, 'ya29.test');
  },
));

// ---------------------------------------------------------------------------
// describeBypassedTokens — issue #137 remediation advice
// ---------------------------------------------------------------------------

test('describeBypassedTokens returns null when no local token file exists', withEnv(
  { GOOGLE_APPLICATION_CREDENTIALS: '/x/sa.json' },
  () => {
    assert.equal(describeBypassedTokens('service_account', '/home/u/tokens.json', false), null);
  },
));

test('describeBypassedTokens names only the one set override var', withEnv(
  { GOOGLE_APPLICATION_CREDENTIALS: '/x/sa.json' },
  () => {
    const msg = describeBypassedTokens('service_account', '/home/u/tokens.json', true);
    assert.ok(msg);
    assert.ok(msg!.includes('Unset GOOGLE_APPLICATION_CREDENTIALS to use your authenticated Google account'));
    assert.ok(!msg!.includes('GOOGLE_DRIVE_MCP_ACCESS_TOKEN'), 'does not mention an unset override var');
  },
));

test('describeBypassedTokens tells the user to unset BOTH override vars when both are set (finding #6)', withEnv(
  { GOOGLE_APPLICATION_CREDENTIALS: '/x/sa.json', GOOGLE_DRIVE_MCP_ACCESS_TOKEN: 'ya29.test' },
  () => {
    // Unsetting only the winning var just hands control to the other override,
    // so tokens.json stays bypassed — the remedy must name both.
    const msg = describeBypassedTokens('service_account', '/home/u/tokens.json', true);
    assert.ok(msg);
    assert.ok(
      msg!.includes('Unset GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_DRIVE_MCP_ACCESS_TOKEN'),
      'names both override vars in the remedy',
    );
  },
));

// ---------------------------------------------------------------------------
// validateCredentialsFile
//
// google-auth-library v10 stopped rejecting malformed credentials files:
// getClient() resolves with a credential-less JWT client, so the server would
// claim "authentication successful" and then fail every call with a misleading
// "unregistered callers" error. These pin the fail-fast guard that restores v9
// behavior — and pin that it stays permissive toward non-service-account ADC
// files, which GOOGLE_APPLICATION_CREDENTIALS may legitimately point at.
// ---------------------------------------------------------------------------

const PRIVATE_KEY = '-----BEGIN PRIVATE KEY-----\nsuper-secret-key-material\n-----END PRIVATE KEY-----\n';

function writeCredFile(contents: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'gdrive-mcp-cred-'));
  const file = join(dir, 'creds.json');
  writeFileSync(file, contents);
  return file;
}

test('validateCredentialsFile accepts a well-formed service account key', () => {
  const file = writeCredFile(JSON.stringify({
    type: 'service_account',
    client_email: 'sa@example.iam.gserviceaccount.com',
    private_key: PRIVATE_KEY,
  }));
  assert.doesNotThrow(() => validateCredentialsFile(file));
});

test('validateCredentialsFile rejects invalid JSON naming the file', () => {
  const file = writeCredFile('{ this is not valid json ');
  assert.throws(() => validateCredentialsFile(file), (err: Error) => {
    assert.match(err.message, /not valid JSON/);
    assert.ok(err.message.includes(file), 'error should name the offending file');
    return true;
  });
});

test('validateCredentialsFile rejects a service account key missing its fields', () => {
  const file = writeCredFile(JSON.stringify({ type: 'service_account' }));
  assert.throws(() => validateCredentialsFile(file), (err: Error) => {
    assert.match(err.message, /client_email/);
    assert.match(err.message, /private_key/);
    return true;
  });
});

test('validateCredentialsFile rejects a bare JSON object with no credential fields', () => {
  const file = writeCredFile(JSON.stringify({ not: 'a valid sa key' }));
  assert.throws(() => validateCredentialsFile(file), /missing required field/);
});

test('validateCredentialsFile rejects a non-object and an unknown type', () => {
  assert.throws(() => validateCredentialsFile(writeCredFile('["array"]')), /must contain a JSON object/);
  assert.throws(
    () => validateCredentialsFile(writeCredFile(JSON.stringify({ type: 'not_a_real_type' }))),
    /unrecognized type/,
  );
});

test('validateCredentialsFile accepts non-service-account ADC credential types', () => {
  // `gcloud auth application-default login` writes an authorized_user file;
  // workload identity writes external_account. Neither carries client_email,
  // and both are valid targets for GOOGLE_APPLICATION_CREDENTIALS.
  const authorizedUser = writeCredFile(JSON.stringify({
    type: 'authorized_user',
    client_id: 'x.apps.googleusercontent.com',
    client_secret: 'shh',
    refresh_token: '1//token',
  }));
  assert.doesNotThrow(() => validateCredentialsFile(authorizedUser));

  const externalAccount = writeCredFile(JSON.stringify({
    type: 'external_account',
    audience: '//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/x',
    subject_token_type: 'urn:ietf:params:oauth:token-type:jwt',
    token_url: 'https://sts.googleapis.com/v1/token',
  }));
  assert.doesNotThrow(() => validateCredentialsFile(externalAccount));
});

test('validateCredentialsFile never echoes key material into the error', () => {
  // The file holds a private key; a validation error must not leak it.
  const file = writeCredFile(JSON.stringify({ type: 'service_account', private_key: PRIVATE_KEY }));
  assert.throws(() => validateCredentialsFile(file), (err: Error) => {
    assert.ok(!err.message.includes('super-secret-key-material'), 'key material leaked into error');
    return true;
  });
});

test('validateCredentialsFile lets a missing file surface as ENOENT', () => {
  assert.throws(
    () => validateCredentialsFile(join(tmpdir(), 'gdrive-mcp-definitely-missing.json')),
    /ENOENT/,
  );
});
