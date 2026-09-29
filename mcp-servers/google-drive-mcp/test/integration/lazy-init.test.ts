import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

describe('Lazy service initialization', () => {
  let ctx: TestContext;
  let serverModule: any;

  before(async () => {
    ctx = await setupTestServer();
    // Get direct access to server module for _setAuthClientForTesting
    serverModule = await import('../../src/index.js');
  });

  after(async () => {
    // Restore a valid auth client so other tests aren't broken
    serverModule._setAuthClientForTesting({});
    await ctx.cleanup();
  });

  beforeEach(() => {
    ctx.mocks.drive.tracker.reset();
    ctx.mocks.calendar.tracker.reset();
  });

  describe('Drive service caching', () => {
    it('reuses cached Drive service across calls', async () => {
      // Reset to fresh state
      const client1 = { id: 'client-1' };
      serverModule._setAuthClientForTesting(client1);

      // Two calls to a Drive tool should only create the service once
      // (google.drive() is called once, then cached)
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));

      await callTool(ctx.client, 'search', { query: 'test1' });
      await callTool(ctx.client, 'search', { query: 'test2' });

      // files.list should be called twice (one per search)
      assert.equal(ctx.mocks.drive.tracker.getCalls('files.list').length, 2);
    });

    it('recreates Drive service when auth client changes', async () => {
      const client1 = { id: 'client-1' };
      serverModule._setAuthClientForTesting(client1);

      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));

      await callTool(ctx.client, 'search', { query: 'test1' });

      // Change auth client — this should null out cached services
      const client2 = { id: 'client-2' };
      serverModule._setAuthClientForTesting(client2);

      await callTool(ctx.client, 'search', { query: 'test2' });

      // Both calls succeeded (service was recreated for client2)
      assert.equal(ctx.mocks.drive.tracker.getCalls('files.list').length, 2);
    });
  });

  describe('Calendar service caching', () => {
    it('reuses cached Calendar service across calls', async () => {
      serverModule._setAuthClientForTesting({ id: 'client-cal' });

      await callTool(ctx.client, 'listCalendars', {});
      await callTool(ctx.client, 'listCalendars', {});

      assert.equal(ctx.mocks.calendar.tracker.getCalls('calendarList.list').length, 2);
    });
  });

  describe('Service independence', () => {
    it('Calendar is not created when only Drive tools are used', async () => {
      serverModule._setAuthClientForTesting({ id: 'client-drive-only' });
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));

      await callTool(ctx.client, 'search', { query: 'test' });

      // Calendar service should not have been called
      assert.equal(ctx.mocks.calendar.tracker.calls.length, 0);
    });

    it('Drive is not created when only Calendar tools are used', async () => {
      serverModule._setAuthClientForTesting({ id: 'client-cal-only' });
      ctx.mocks.drive.tracker.reset();

      await callTool(ctx.client, 'listCalendars', {});

      // Drive service should not have been called
      assert.equal(ctx.mocks.drive.tracker.calls.length, 0);
    });
  });

  describe('Auth client reset', () => {
    it('_setAuthClientForTesting clears cached services', async () => {
      // First call creates Drive service
      serverModule._setAuthClientForTesting({ id: 'client-a' });
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      await callTool(ctx.client, 'search', { query: 'before-reset' });

      // Reset with a new client — cached services should be cleared
      serverModule._setAuthClientForTesting({ id: 'client-b' });
      await callTool(ctx.client, 'search', { query: 'after-reset' });

      // Both calls succeeded, meaning the service was recreated after reset
      assert.equal(ctx.mocks.drive.tracker.getCalls('files.list').length, 2);
    });
  });
});
