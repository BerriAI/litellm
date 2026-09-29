import assert from 'node:assert/strict';
import test from 'node:test';

import { loadTeamConfig } from '../../src/auth/team/config.js';

// ---------------------------------------------------------------------------
// loadTeamConfig validation. env is passed explicitly so these are hermetic
// (no process.env mutation).
// ---------------------------------------------------------------------------

function makeEnv(overrides: Record<string, string> = {}): NodeJS.ProcessEnv {
  return { MCP_TEAM_ISSUER_URL: 'https://drive-mcp.example.com', ...overrides };
}

test('a root issuer yields a host-root Google callback URI', () => {
  const config = loadTeamConfig({ transport: 'http', env: makeEnv() });
  assert.equal(
    config.googleRedirectUri,
    'https://drive-mcp.example.com/oauth/google/callback',
  );
});

test('MCP_HTTP_ALLOWED_HOSTS entries are lowercased for the case-sensitive SDK host check', () => {
  const config = loadTeamConfig({
    transport: 'http',
    env: makeEnv({ MCP_HTTP_ALLOWED_HOSTS: 'Extra.Example.COM, Another.Host' }),
  });
  // The SDK compares the Host header (always lowercased by URL parsing) against
  // this list case-sensitively, so mixed-case entries would never match.
  assert.ok(config.allowedHosts.includes('extra.example.com'));
  assert.ok(config.allowedHosts.includes('another.host'));
  assert.ok(config.allowedHosts.every((h) => h === h.toLowerCase()));
});

test('a path-bearing issuer URL is rejected at boot', () => {
  // new URL(callbackPath, issuer) would silently drop the base path, so the
  // derived redirect URI would not match what a reverse proxy forwards.
  assert.throws(
    () =>
      loadTeamConfig({
        transport: 'http',
        env: makeEnv({ MCP_TEAM_ISSUER_URL: 'https://example.com/mcp' }),
      }),
    /must not contain a path/,
  );
});

test('a query string or fragment on the issuer is still rejected', () => {
  assert.throws(
    () =>
      loadTeamConfig({
        transport: 'http',
        env: makeEnv({ MCP_TEAM_ISSUER_URL: 'https://example.com/?x=1' }),
      }),
    /query string or fragment/,
  );
});
