import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

describe('Docs listing tools', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.drive.tracker.reset();
  });

  // --- listGoogleDocs ---
  describe('listGoogleDocs', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: {
          files: [{
            id: 'doc-1', name: 'My Document', modifiedTime: '2025-01-01',
            webViewLink: 'https://docs.google.com/doc-1',
            owners: [{ displayName: 'Owner', emailAddress: 'owner@test.com' }],
          }],
        },
      }));
      const res = await callTool(ctx.client, 'listGoogleDocs', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('My Document'));
    });

    it('passes corpora=allDrives so shared-drive docs are listed', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      await callTool(ctx.client, 'listGoogleDocs', {});
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.corpora, 'allDrives');
      assert.equal(args.supportsAllDrives, true);
      assert.equal(args.includeItemsFromAllDrives, true);
      ctx.mocks.drive.service.files.list._resetImpl();
    });

    it('defaults to modifiedTime desc, not oldest-first (issue #167)', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      await callTool(ctx.client, 'listGoogleDocs', {});
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      assert.equal(listCalls[listCalls.length - 1].args[0].orderBy, 'modifiedTime desc');

      await callTool(ctx.client, 'listGoogleDocs', { orderBy: 'name' });
      const afterExplicit = ctx.mocks.drive.tracker.getCalls('files.list');
      assert.equal(afterExplicit[afterExplicit.length - 1].args[0].orderBy, 'name');
      ctx.mocks.drive.service.files.list._resetImpl();
    });

    it('no results', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      const res = await callTool(ctx.client, 'listGoogleDocs', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('No Google Docs'));
    });
  });

  // --- getDocumentInfo ---
  describe('getDocumentInfo', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: {
          id: 'doc-1', name: 'My Document', mimeType: 'application/vnd.google-apps.document',
          createdTime: '2025-01-01', modifiedTime: '2025-01-02',
          webViewLink: 'https://docs.google.com/doc-1', shared: true,
          owners: [{ displayName: 'Owner', emailAddress: 'owner@test.com' }],
        },
      }));
      const res = await callTool(ctx.client, 'getDocumentInfo', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('My Document'));
    });

    it('passes supportsAllDrives so shared-drive documents resolve', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'doc-1', name: 'My Document', mimeType: 'application/vnd.google-apps.document' },
      }));
      await callTool(ctx.client, 'getDocumentInfo', { documentId: 'doc-1' });
      const getCalls = ctx.mocks.drive.tracker.getCalls('files.get');
      assert.equal(getCalls[getCalls.length - 1].args[0].supportsAllDrives, true);
      ctx.mocks.drive.service.files.get._resetImpl();
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'getDocumentInfo', {});
      assert.equal(res.isError, true);
    });
  });
});
