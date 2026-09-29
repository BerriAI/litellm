import assert from 'node:assert/strict';
import test from 'node:test';
import {
  resolveOAuthScopes,
  resolveAddAccountScopes,
  DEFAULT_SCOPES,
  USERINFO_SCOPES,
  SCOPE_ALIASES,
} from '../../src/auth/scopes.js';

const ENV = 'GOOGLE_DRIVE_MCP_SCOPES';

// node --test runs each file in its own process, so mutating this env var here is
// isolated from other test files; we still save/restore to keep tests independent.
function withScopeEnv(value: string | undefined, fn: () => void): void {
  const saved = process.env[ENV];
  try {
    if (value === undefined) delete process.env[ENV];
    else process.env[ENV] = value;
    fn();
  } finally {
    if (saved === undefined) delete process.env[ENV];
    else process.env[ENV] = saved;
  }
}

// ---------------------------------------------------------------------------
// resolveAddAccountScopes — the scopes `manage_accounts add` requests (finding 15)
// ---------------------------------------------------------------------------

test('resolveAddAccountScopes honors GOOGLE_DRIVE_MCP_SCOPES (least privilege)', () => {
  withScopeEnv('drive.readonly', () => {
    const scopes = resolveAddAccountScopes();
    // Requests exactly the operator's scope, plus userinfo for identity discovery.
    assert.ok(scopes.includes(SCOPE_ALIASES['drive.readonly']), 'includes drive.readonly');
    for (const s of USERINFO_SCOPES) {
      assert.ok(scopes.includes(s), `includes userinfo scope ${s}`);
    }
    // Must NOT over-request scopes the operator did not ask for.
    assert.ok(!scopes.includes(SCOPE_ALIASES['drive']), 'excludes full drive');
    assert.ok(!scopes.includes(SCOPE_ALIASES['calendar']), 'excludes calendar');
  });
});

test('resolveAddAccountScopes honors custom https:// scopes', () => {
  const custom = 'https://www.googleapis.com/auth/drive.metadata.readonly';
  withScopeEnv(custom, () => {
    const scopes = resolveAddAccountScopes();
    assert.ok(scopes.includes(custom), 'includes the custom scope');
    for (const s of USERINFO_SCOPES) {
      assert.ok(scopes.includes(s), `includes userinfo scope ${s}`);
    }
  });
});

test('resolveAddAccountScopes falls back to DEFAULT_SCOPES + userinfo when env unset', () => {
  withScopeEnv(undefined, () => {
    const scopes = resolveAddAccountScopes();
    for (const s of DEFAULT_SCOPES) {
      assert.ok(scopes.includes(s), `includes default scope ${s}`);
    }
    for (const s of USERINFO_SCOPES) {
      assert.ok(scopes.includes(s), `includes userinfo scope ${s}`);
    }
  });
});

// ---------------------------------------------------------------------------
// resolveOAuthScopes — env parsing (first direct coverage)
// ---------------------------------------------------------------------------

test('resolveOAuthScopes maps aliases and dedupes', () => {
  withScopeEnv('drive,documents,drive', () => {
    assert.deepEqual(resolveOAuthScopes(), [
      SCOPE_ALIASES['drive'],
      SCOPE_ALIASES['documents'],
    ]);
  });
});

test('resolveOAuthScopes returns DEFAULT_SCOPES when the env var is unset', () => {
  withScopeEnv(undefined, () => {
    assert.deepEqual(resolveOAuthScopes(), [...DEFAULT_SCOPES]);
  });
});

test('resolveOAuthScopes throws on an unknown alias', () => {
  withScopeEnv('not-a-real-scope', () => {
    assert.throws(() => resolveOAuthScopes(), /Unknown OAuth scope alias/);
  });
});
