import assert from 'node:assert/strict';
import test from 'node:test';

import { computeAuthStatus } from '../../src/tools/drive.js';

const base = {
  authMode: 'oauth' as const,
  tokenFileExists: true,
  hasRefreshToken: true,
  grantedScopes: [] as string[],
  missingScopes: [] as string[],
  identityError: false,
  warningCount: 0,
};

const DRIVE = 'https://www.googleapis.com/auth/drive';
const DRIVE_FILE = 'https://www.googleapis.com/auth/drive.file';

test('oauth mode with no token file is needs_reauth and outranks a failing identity', () => {
  assert.equal(
    computeAuthStatus({ ...base, tokenFileExists: false, identityError: true }),
    'needs_reauth',
  );
});

test('oauth mode with no refresh token is needs_reauth', () => {
  assert.equal(computeAuthStatus({ ...base, hasRefreshToken: false }), 'needs_reauth');
});

test('a reliable scope shortfall reports scope_mismatch even when identity also fails (finding #4)', () => {
  // Under-scoped token: about.get 403s for lack of drive/drive.readonly. The
  // actionable fix is re-auth-with-scopes, so scope_mismatch must win — not
  // identity_error, which would send the user chasing a credentials problem.
  assert.equal(
    computeAuthStatus({
      ...base,
      authMode: 'service_account',
      grantedScopes: [DRIVE_FILE],
      missingScopes: [DRIVE],
      identityError: true,
    }),
    'scope_mismatch',
  );
});

test('an identity failure with unknown (empty) granted scopes is NOT masked as scope_mismatch', () => {
  // grantedScopes is often empty for disk-loaded tokens, which makes
  // missingScopes falsely list everything. That false positive must not mask a
  // genuine wrong/empty identity (e.g. an empty service account).
  assert.equal(
    computeAuthStatus({
      ...base,
      authMode: 'service_account',
      grantedScopes: [],
      missingScopes: [DRIVE],
      identityError: true,
    }),
    'identity_error',
  );
});

test('a scope shortfall with empty granted scopes still falls through to scope_mismatch when identity is fine', () => {
  assert.equal(
    computeAuthStatus({
      ...base,
      authMode: 'service_account',
      grantedScopes: [],
      missingScopes: [DRIVE],
    }),
    'scope_mismatch',
  );
});

test('warning is reported only when no specific diagnosis applies', () => {
  assert.equal(computeAuthStatus({ ...base, authMode: 'service_account', warningCount: 1 }), 'warning');
});

test('a fully healthy state is ok', () => {
  assert.equal(computeAuthStatus({ ...base, authMode: 'service_account' }), 'ok');
});
