import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { google } from 'googleapis';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

// ---------------------------------------------------------------------------
// Multi-account dispatch routing.
//
// Goal: with two synthetic accounts seeded, prove that the dispatch layer
// hands the correct per-alias OAuth client through to the Drive service
// factory. We do this by wrapping `google.drive` to record the `auth`
// argument, then asserting the marker on the auth client matches the alias
// the call was routed to.
// ---------------------------------------------------------------------------

describe('Multi-account dispatch routing', () => {
  let ctx: TestContext;
  let serverModule: any;
  let originalDriveFactory: typeof google.drive;
  const authCalls: Array<{ marker: string | undefined }> = [];

  before(async () => {
    ctx = await setupTestServer();
    serverModule = await import('../../src/index.js');

    // Seed two extra synthetic accounts on top of the default 'test' account.
    await serverModule._addSyntheticAccountForTesting('alpha', { _marker: 'alpha-client' });
    await serverModule._addSyntheticAccountForTesting('beta', { _marker: 'beta-client' });

    // Wrap google.drive so we record the auth marker for every construction.
    originalDriveFactory = google.drive as any;
    (google as any).drive = (opts: any) => {
      const marker = opts?.auth?._marker;
      authCalls.push({ marker });
      return originalDriveFactory(opts);
    };
  });

  after(async () => {
    (google as any).drive = originalDriveFactory;
    // Reset to a single 'test' account so subsequent test files start clean.
    serverModule._setAuthClientForTesting({});
    await ctx.cleanup();
  });

  beforeEach(() => {
    authCalls.length = 0;
    ctx.mocks.drive.tracker.reset();
  });

  it("routes account='alpha' to the alpha client", async () => {
    const result = await callTool(ctx.client, 'search', {
      query: 'q',
      account: 'alpha',
    });
    assert.notEqual(result.isError, true);
    const markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      markers.includes('alpha-client'),
      `expected alpha-client in auth markers, saw: ${JSON.stringify(markers)}`,
    );
    assert.ok(
      !markers.includes('beta-client'),
      `unexpected beta-client in markers: ${JSON.stringify(markers)}`,
    );
  });

  it("routes account='beta' to the beta client", async () => {
    const result = await callTool(ctx.client, 'search', {
      query: 'q',
      account: 'beta',
    });
    assert.notEqual(result.isError, true);
    const markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      markers.includes('beta-client'),
      `expected beta-client in auth markers, saw: ${JSON.stringify(markers)}`,
    );
    assert.ok(
      !markers.includes('alpha-client'),
      `unexpected alpha-client in markers: ${JSON.stringify(markers)}`,
    );
  });

  it('manage_accounts list reports all three synthetic accounts', async () => {
    const result = await callTool(ctx.client, 'manage_accounts', { action: 'list' });
    assert.notEqual(result.isError, true);
    const payload = JSON.parse(result.content[0].text!);
    const aliases = (payload.accounts as Array<{ alias: string }>).map((a) => a.alias).sort();
    assert.deepEqual(aliases, ['alpha', 'beta', 'test']);
  });

  it('omitting account falls back to the global default', async () => {
    // The harness's 'test' account is the default — _addSyntheticAccountForTesting
    // explicitly preserves it. A bare call should route there.
    const result = await callTool(ctx.client, 'search', { query: 'q' });
    assert.notEqual(result.isError, true);
    // 'test' was injected with a non-marked client `{ request: ... }`, so its
    // auth marker is undefined. Both alpha and beta markers must be absent.
    const markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      !markers.includes('alpha-client') && !markers.includes('beta-client'),
      `expected default-account routing (no alpha/beta markers), saw: ${JSON.stringify(markers)}`,
    );
  });

  it("rejects account='nonexistent' before the handler runs", async () => {
    const result = await callTool(ctx.client, 'search', {
      query: 'q',
      account: 'nonexistent',
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /Unknown account/);
    // No drive service should have been instantiated for this rejected call.
    const markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      !markers.includes('alpha-client') && !markers.includes('beta-client'),
      `dispatch should not have built a drive service for the rejected call, saw: ${JSON.stringify(markers)}`,
    );
  });

  it('rejects an array account argument instead of silently routing to the default', async () => {
    // The resolver supports arrays only for a not-yet-wired fanout; coercing the
    // array to the default would return partial results. It must error instead.
    const result = await callTool(ctx.client, 'search', {
      query: 'q',
      account: ['alpha', 'beta'] as unknown as string,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /single account alias|one call per account/);
    // Threw before dispatch — no drive service built for either account.
    const markers = authCalls.map((c) => c.marker).filter((m) => m !== undefined);
    assert.ok(
      !markers.includes('alpha-client') && !markers.includes('beta-client'),
      `array-account call should not have built any drive service, saw: ${JSON.stringify(markers)}`,
    );
  });

  it('rejects a non-string (number) account argument', async () => {
    const result = await callTool(ctx.client, 'search', {
      query: 'q',
      account: 42 as unknown as string,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text!, /single account alias|one call per account/);
  });
});
