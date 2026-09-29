import assert from 'node:assert/strict';
import test from 'node:test';
import { OAuth2Client } from 'google-auth-library';

import { AuthServer } from '../src/auth/server.js';

// ---------------------------------------------------------------------------
// Regression test for the OAuth callback-server loopback bind.
//
// The callback server MUST bind to the IPv4 loopback address 127.0.0.1, not
// all interfaces (the historical default when no host is passed to
// `listen()`). A regression here would silently re-expose the short-lived
// auth server to the LAN with the whole suite still green, so this asserts
// the bind address explicitly.
//
// `startServerOnAvailablePort()` is the private port-binding path. It does
// NOT call `loadCredentials()` (only `start()` does), so no OAuth/credential
// mocking is needed — a throwaway OAuth2Client satisfies the constructor and
// is never dereferenced by the binder.
// ---------------------------------------------------------------------------

test('auth callback server binds to 127.0.0.1, not all interfaces', async () => {
  // Use a high, uncommon port range to avoid colliding with the default
  // 3000–3004 (dev servers, etc.). Restore the env var afterwards so it
  // cannot leak into anything else running in this process.
  const savedPort = process.env.GOOGLE_DRIVE_MCP_AUTH_PORT;
  process.env.GOOGLE_DRIVE_MCP_AUTH_PORT = '45123';

  // onTokens is required (AuthServer never persists tokens itself), but this
  // test only exercises the port binder, so a no-op callback suffices.
  const server = new AuthServer(new OAuth2Client(), { onTokens: async () => {} });
  try {
    const boundPort = await (
      server as unknown as {
        startServerOnAvailablePort(): Promise<number | null>;
      }
    ).startServerOnAvailablePort();

    assert.notEqual(boundPort, null, 'server should have bound to a port');

    // The security invariant: bound to the IPv4 loopback address only.
    assert.equal(
      server.getServerAddress(),
      '127.0.0.1',
      'callback server must bind to 127.0.0.1, not 0.0.0.0/::',
    );

    // Port should be within the configured range (the binder auto-advances
    // through 45123–45127 on EADDRINUSE, so assert membership, not equality).
    const runningPort = server.getRunningPort();
    assert.ok(
      runningPort !== null && runningPort >= 45123 && runningPort <= 45127,
      `running port ${runningPort} should be within 45123–45127`,
    );
    assert.equal(runningPort, boundPort);
  } finally {
    await server.stop();
    if (savedPort === undefined) {
      delete process.env.GOOGLE_DRIVE_MCP_AUTH_PORT;
    } else {
      process.env.GOOGLE_DRIVE_MCP_AUTH_PORT = savedPort;
    }
  }
});
