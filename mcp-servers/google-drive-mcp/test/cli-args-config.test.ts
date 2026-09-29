import assert from 'node:assert/strict';
import test from 'node:test';

import { loadRuntimeConfig } from '../src/utils/cliArgs.js';

// ---------------------------------------------------------------------------
// Helper — save & restore GOOGLE_DRIVE_MCP_DISABLE_RESOURCES around each test
// ---------------------------------------------------------------------------
const VAR = 'GOOGLE_DRIVE_MCP_DISABLE_RESOURCES';

function withVar(value: string | undefined, fn: () => void) {
  return () => {
    const saved = process.env[VAR];
    if (value === undefined) delete process.env[VAR];
    else process.env[VAR] = value;
    try {
      fn();
    } finally {
      if (saved === undefined) delete process.env[VAR];
      else process.env[VAR] = saved;
    }
  };
}

// ---------------------------------------------------------------------------
// disableResources — default
// ---------------------------------------------------------------------------
test('disableResources defaults to false when unset and no flag', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig([]).disableResources, false);
}));

// ---------------------------------------------------------------------------
// disableResources — env var truthy values
// ---------------------------------------------------------------------------
for (const v of ['1', 'true', 'yes', 'on', 'TRUE', '  Yes  ', ' ON ']) {
  test(`disableResources is true for env value ${JSON.stringify(v)}`, withVar(v, () => {
    assert.equal(loadRuntimeConfig([]).disableResources, true);
  }));
}

// ---------------------------------------------------------------------------
// disableResources — env var falsy values
// ---------------------------------------------------------------------------
for (const v of ['0', 'false', 'no', 'off', 'FALSE', 'Off', ' No ']) {
  test(`disableResources is false for env value ${JSON.stringify(v)}`, withVar(v, () => {
    assert.equal(loadRuntimeConfig([]).disableResources, false);
  }));
}

// ---------------------------------------------------------------------------
// disableResources — unrecognized env value falls back to the default (false)
// ---------------------------------------------------------------------------
test('disableResources falls back to default for unrecognized env value', withVar('maybe', () => {
  assert.equal(loadRuntimeConfig([]).disableResources, false);
}));

// enable/disable are intentionally NOT accepted: on a negated env var they form
// a double negative. They must behave like any other unrecognized value.
for (const v of ['enable', 'enabled', 'disable', 'disabled']) {
  test(`disableResources ignores non-alias env value ${JSON.stringify(v)} (falls back to default)`, withVar(v, () => {
    assert.equal(loadRuntimeConfig([]).disableResources, false);
  }));
}

// ---------------------------------------------------------------------------
// disableResources — --no-resources CLI flag
// ---------------------------------------------------------------------------
test('--no-resources flag sets disableResources true', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--no-resources']).disableResources, true);
}));

test('--no-resources flag overrides a falsy env value', withVar('0', () => {
  assert.equal(loadRuntimeConfig(['--no-resources']).disableResources, true);
}));

// ---------------------------------------------------------------------------
// disableResources — --no-resources=<bool> value form
// ---------------------------------------------------------------------------
test('--no-resources=true sets disableResources true', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--no-resources=true']).disableResources, true);
}));

test('--no-resources=false re-enables, overriding a truthy env value', withVar('1', () => {
  assert.equal(loadRuntimeConfig(['--no-resources=false']).disableResources, false);
}));

test('--no-resources=off re-enables resources', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--no-resources=off']).disableResources, false);
}));

test('--no-resources=<garbage> falls back to the bare-flag intent (disable)', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--no-resources=maybe']).disableResources, true);
}));

test('--no-resources= (empty value) disables (bare-flag intent)', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--no-resources=']).disableResources, true);
}));

// enable/disable are not aliases, so --no-resources=disabled is garbage and
// follows the bare-flag intent (disable) — no double-negative trap.
test('--no-resources=disabled is treated as garbage and disables', withVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--no-resources=disabled']).disableResources, true);
}));

// ---------------------------------------------------------------------------
// Sanity — other runtime config defaults are unaffected
// ---------------------------------------------------------------------------
test('unrelated runtime config defaults are preserved', withVar(undefined, () => {
  const cfg = loadRuntimeConfig([]);
  assert.equal(cfg.apiTimeout, 120_000);
  assert.equal(cfg.tokenRefreshTimeout, 15_000);
  assert.equal(cfg.retryMax, 3);
  assert.equal(cfg.retryBaseDelay, 1_000);
}));

// ---------------------------------------------------------------------------
// tokenRefreshTimeout (#169) — env fallback, flag override, 0 = disabled
// ---------------------------------------------------------------------------
const REFRESH_VAR = 'GOOGLE_DRIVE_MCP_TOKEN_REFRESH_TIMEOUT';

function withRefreshVar(value: string | undefined, fn: () => void) {
  return () => {
    const saved = process.env[REFRESH_VAR];
    if (value === undefined) delete process.env[REFRESH_VAR];
    else process.env[REFRESH_VAR] = value;
    try {
      fn();
    } finally {
      if (saved === undefined) delete process.env[REFRESH_VAR];
      else process.env[REFRESH_VAR] = saved;
    }
  };
}

test('tokenRefreshTimeout is read from the env var', withRefreshVar('4000', () => {
  assert.equal(loadRuntimeConfig([]).tokenRefreshTimeout, 4_000);
}));

test('--token-refresh-timeout sets tokenRefreshTimeout', withRefreshVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--token-refresh-timeout=2500']).tokenRefreshTimeout, 2_500);
}));

test('--token-refresh-timeout overrides the env var', withRefreshVar('4000', () => {
  assert.equal(loadRuntimeConfig(['--token-refresh-timeout=2500']).tokenRefreshTimeout, 2_500);
}));

test('--token-refresh-timeout=0 disables the refresh timeout', withRefreshVar(undefined, () => {
  assert.equal(loadRuntimeConfig(['--token-refresh-timeout=0']).tokenRefreshTimeout, 0);
}));

test('garbage tokenRefreshTimeout values fall back to the default', withRefreshVar('soon', () => {
  assert.equal(loadRuntimeConfig(['--token-refresh-timeout=-5']).tokenRefreshTimeout, 15_000);
}));

test('--token-refresh-timeout leaves --api-timeout alone', withRefreshVar(undefined, () => {
  const cfg = loadRuntimeConfig(['--token-refresh-timeout=2500']);
  assert.equal(cfg.apiTimeout, 120_000);
}));
