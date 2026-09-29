import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

describe('Tables & Media tools', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.drive.tracker.reset();
    ctx.mocks.docs.tracker.reset();
  });

  // --- insertTable ---
  describe('insertTable', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'insertTable', {
        documentId: 'doc-1', rows: 3, columns: 4, index: 1,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('3x4'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'insertTable', {});
      assert.equal(res.isError, true);
    });
  });

  // --- editTableCell ---
  describe('editTableCell', () => {
    it('happy path', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          body: {
            content: [
              {
                table: {
                  tableRows: [
                    {
                      tableCells: [
                        { startIndex: 5, endIndex: 10, content: [{ paragraph: { elements: [{ textRun: { content: 'old\n' } }] } }] },
                      ],
                    },
                  ],
                },
                startIndex: 2,
              },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'editTableCell', {
        documentId: 'doc-1', tableStartIndex: 2, rowIndex: 0, columnIndex: 0,
        textContent: 'new value',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('edited cell'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'editTableCell', {});
      assert.equal(res.isError, true);
    });
  });

  // --- insertImageFromUrl ---
  describe('insertImageFromUrl', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'insertImageFromUrl', {
        documentId: 'doc-1', imageUrl: 'https://example.com/image.png', index: 1,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('inserted image'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'insertImageFromUrl', {});
      assert.equal(res.isError, true);
    });
  });

  // --- insertLocalImage ---
  describe('insertLocalImage', () => {
    it('validation error on missing args', async () => {
      const res = await callTool(ctx.client, 'insertLocalImage', {});
      assert.equal(res.isError, true);
    });
  });
});
