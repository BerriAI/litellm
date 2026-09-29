import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach, afterEach } from 'node:test';
import { Readable } from 'node:stream';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

// Create reusable mock document structures for testing common document and tab configurations
const mockDocs = {
  // Simple single-tab document
  singleTab: (content = 'Hello World\n') => ({
    documentId: 'doc-1',
    title: 'My Doc',
    body: {
      content: [{ paragraph: { elements: [{ textRun: { content } }] } }],
    },
  }),

  // Multi-tab document
  multiTab: () => ({
    documentId: 'doc-1',
    title: 'Multi-Tab Doc',
    tabs: [
      {
        tabProperties: { tabId: 'tab-1', title: 'Tab1' },
        documentTab: {
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' }, startIndex: 1, endIndex: 11 }] } }] },
        },
      },
      {
        tabProperties: { tabId: 'tab-2', title: 'Tab2' },
        documentTab: {
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Second tab\n' }, startIndex: 1, endIndex: 12 }] } }] },
        },
      },
    ],
  }),

  // Fully nested document (all 3 levels)
  fullyNested: () => ({
    documentId: 'doc-1',
    title: 'Nested Tab Doc',
    tabs: [
      {
        tabProperties: { tabId: 'tab-1', title: 'Tab1' },
        documentTab: {
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' }, startIndex: 1, endIndex: 11 }] } }] },
        },
        childTabs: [
          {
            tabProperties: { tabId: 'tab-1-1', title: 'Tab1.1' },
            documentTab: {
              body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First child\n' }, startIndex: 1, endIndex: 13 }] } }] },
            },
          },
          {
            tabProperties: { tabId: 'tab-1-2', title: 'Tab1.2' },
            documentTab: {
              body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Second child\n' }, startIndex: 1, endIndex: 14 }] } }] },
            },
            childTabs: [
              {
                tabProperties: { tabId: 'tab-1-2-1', title: 'Tab1.2.1' },
                documentTab: {
                  body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First grandchild\n' }, startIndex: 1, endIndex: 18 }] } }] },
                },
              },
            ],
          },
        ],
      },
      {
        tabProperties: { tabId: 'tab-2', title: 'Tab2' },
        documentTab: {
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Second tab\n' }, startIndex: 1, endIndex: 12 }] } }] },
        },
      },
    ],
  }),

  // Single parent with nested children (for edge case testing)
  singleParentNested: () => ({
    documentId: 'doc-1', title: 'Nested Tab Doc',
    tabs: [
      {
        tabProperties: { tabId: 'tab-1', title: 'Tab1' },
        documentTab: {
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' }, startIndex: 1, endIndex: 11 }] } }] },
        },
        childTabs: [
          {
            tabProperties: { tabId: 'tab-1-1', title: 'Tab1.1' },
            documentTab: {
              body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First child\n' }, startIndex: 1, endIndex: 13 }] } }] },
            },
          },
          {
            tabProperties: { tabId: 'tab-1-2', title: 'Tab1.2' },
            documentTab: {
              body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Second child\n' }, startIndex: 1, endIndex: 14 }] } }] },
            },
            childTabs: [
              {
                tabProperties: { tabId: 'tab-1-2-1', title: 'Tab1.2.1' },
                documentTab: {
                  body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First grandchild\n' }, startIndex: 1, endIndex: 18 }] } }] },
                },
              },
            ],
          },
        ],
      },
    ],
  }),
};

describe('Docs tools', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.drive.tracker.reset();
    ctx.mocks.docs.tracker.reset();
    // Reset stub impls that individual blocks override on drive.files.get so a
    // per-block override (e.g. insertText/deleteRange forcing a Google-Docs
    // mimeType) does not leak into later tests and make the suite order-dependent.
    ctx.mocks.drive.service.files.get._resetImpl();
    // Same for docs.documents.get: several blocks install per-test document
    // fixtures, which must not leak into later blocks.
    ctx.mocks.docs.service.documents.get._resetImpl();
  });

  // --- createGoogleDoc ---
  describe('createGoogleDoc', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      ctx.mocks.drive.service.files.create._setImpl(async () => ({
        data: { id: 'doc-1', name: 'My Doc', webViewLink: 'https://docs.google.com/doc-1' },
      }));
      const res = await callTool(ctx.client, 'createGoogleDoc', { name: 'My Doc', content: 'Hello' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('My Doc'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'createGoogleDoc', {});
      assert.equal(res.isError, true);
    });
  });

  // --- updateGoogleDoc ---
  describe('updateGoogleDoc', () => {
    it('happy path', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ endIndex: 10 }] },
        },
      }));
      const res = await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'doc-1', content: 'New content' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Updated Google Doc'));

      // Non-tabId path is now one atomic batchUpdate, like the tabId path: the
      // delete and insert can no longer half-apply and leave the doc wiped.
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 1);
      const kinds = calls[0]!.args[0].requestBody.requests.map((r: any) => Object.keys(r)[0]);
      assert.deepEqual(kinds, ['deleteContentRange', 'insertText', 'updateParagraphStyle']);
    });

    it('with tabId issues a single atomic batchUpdate scoped to the tab', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-Tab Doc',
          tabs: [
            { tabProperties: { tabId: 'tab-1', title: 'Tab1' }, documentTab: { body: { content: [{ endIndex: 5 }] } } },
            { tabProperties: { tabId: 'tab-2', title: 'Tab2' }, documentTab: { body: { content: [{ endIndex: 20 }] } } },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'doc-1', content: 'New tab content', tabId: 'tab-2' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('tab: tab-2'));

      // Verify documents.get was called with includeTabsContent.
      const getCalls = ctx.mocks.docs.tracker.getCalls('documents.get');
      assert.equal(getCalls[getCalls.length - 1]?.args?.[0]?.includeTabsContent, true);

      // Exactly one batchUpdate — atomic.
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 1);

      const requests = calls[0]?.args?.[0]?.requestBody?.requests;
      assert.equal(requests?.length, 3);
      assert.equal(requests[0].deleteContentRange.range.tabId, 'tab-2');
      assert.equal(requests[0].deleteContentRange.range.startIndex, 1);
      assert.equal(requests[0].deleteContentRange.range.endIndex, 19);
      assert.equal(requests[1].insertText.location.tabId, 'tab-2');
      assert.equal(requests[1].insertText.location.index, 1);
      assert.equal(requests[1].insertText.text, 'New tab content');
      assert.equal(requests[2].updateParagraphStyle.range.tabId, 'tab-2');
    });

    it('with tabId finds nested child tab', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Nested',
          tabs: [
            {
              tabProperties: { tabId: 'tab-1', title: 'Tab1' },
              documentTab: { body: { content: [{ endIndex: 5 }] } },
              childTabs: [
                { tabProperties: { tabId: 'tab-1-1', title: 'Child' }, documentTab: { body: { content: [{ endIndex: 8 }] } } },
              ],
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'doc-1', content: 'deep', tabId: 'tab-1-1' });
      assert.equal(res.isError, false);

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 1);
      const requests = calls[0]?.args?.[0]?.requestBody?.requests;
      assert.equal(requests[0].deleteContentRange.range.tabId, 'tab-1-1');
      assert.equal(requests[0].deleteContentRange.range.endIndex, 7);
    });

    it('with tabId on empty tab: skips deleteContentRange', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-Tab Doc',
          tabs: [
            { tabProperties: { tabId: 'tab-1', title: 'Empty' }, documentTab: { body: { content: [{ endIndex: 1 }] } } },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'doc-1', content: 'fresh', tabId: 'tab-1' });
      assert.equal(res.isError, false);

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 1);
      const requests = calls[0]?.args?.[0]?.requestBody?.requests;
      assert.equal(requests?.length, 2);
      assert.ok('insertText' in requests[0]);
      assert.ok('updateParagraphStyle' in requests[1]);
    });

    it('unknown tabId returns error and issues no batchUpdate', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-Tab Doc',
          tabs: [
            { tabProperties: { tabId: 'tab-1', title: 'Tab1' }, documentTab: { body: { content: [{ endIndex: 5 }] } } },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'doc-1', content: 'x', tabId: 'missing' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('Tab with ID "missing" not found'));
      assert.ok(res.content[0].text!.includes('listDocumentTabs'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 0);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'updateGoogleDoc', {});
      assert.equal(res.isError, true);
    });
  });

  // --- insertText ---
  describe('insertText', () => {
    beforeEach(() => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'doc-1', name: 'My Doc', mimeType: 'application/vnd.google-apps.document', parents: ['root'] },
      }));
    });

    it('happy path', async () => {
      const res = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: 'inserted', index: 1 });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('inserted'));
    });

    it('with tabId forwards tabId to Location', async () => {
      const res = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: 'hello', index: 1, tabId: 'tab-7' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('tab-7'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const lastCall = calls[calls.length - 1];
      const requests = lastCall?.args?.[0]?.requestBody?.requests;
      assert.equal(requests?.length, 1);
      assert.equal(requests[0].insertText.location.tabId, 'tab-7');
      assert.equal(requests[0].insertText.location.index, 1);
      assert.equal(requests[0].insertText.text, 'hello');
    });

    it('rejects index 0 on a Google Doc (1-based)', async () => {
      const res = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: 'x', index: 0 });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('1-based'));
    });

    it('textToFind inserts after the match by default and before with position', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Hello target world\n' }, startIndex: 1, endIndex: 20 }] } }] },
        },
      }));

      const after = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: '!', textToFind: 'target' });
      assert.equal(after.isError, false);
      let calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      // "target" spans doc indices 7-13, so after = 13.
      assert.equal(calls[calls.length - 1].args[0].requestBody.requests[0].insertText.location.index, 13);

      const before = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: '>', textToFind: 'target', position: 'before' });
      assert.equal(before.isError, false);
      calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls[calls.length - 1].args[0].requestBody.requests[0].insertText.location.index, 7);
    });

    it('textToFind not found is an error, and index+textToFind together are refused', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: { documentId: 'doc-1', body: { content: [] } },
      }));
      const missing = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: 'x', textToFind: 'nope' });
      assert.equal(missing.isError, true);
      assert.ok(missing.content[0].text!.includes('not found'));

      const both = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: 'x', index: 1, textToFind: 'y' });
      assert.equal(both.isError, true);
      assert.ok(both.content[0].text!.includes('exactly one'));
    });

    it('textToFind after a match that ends at the final paragraph break inserts before that break', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          body: { content: [{ startIndex: 1, endIndex: 11, paragraph: { elements: [{ textRun: { content: 'keep tail\n' }, startIndex: 1, endIndex: 11 }] } }] },
        },
      }));
      // "tail\n" spans 6-11 and 11 is the segment end, so the insert lands at 10.
      const res = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: '!', textToFind: 'tail\n' });
      assert.equal(res.isError, false);
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls[calls.length - 1].args[0].requestBody.requests[0].insertText.location.index, 10);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'insertText', {});
      assert.equal(res.isError, true);
    });
  });

  // --- deleteRange ---
  describe('deleteRange', () => {
    beforeEach(() => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'doc-1', name: 'My Doc', mimeType: 'application/vnd.google-apps.document', parents: ['root'] },
      }));
    });

    it('happy path', async () => {
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', startIndex: 1, endIndex: 5 });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('deleted'));
    });

    it('with tabId forwards tabId to Range', async () => {
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', startIndex: 1, endIndex: 5, tabId: 'tab-7' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('tab-7'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const lastCall = calls[calls.length - 1];
      const requests = lastCall?.args?.[0]?.requestBody?.requests;
      assert.equal(requests?.length, 1);
      assert.equal(requests[0].deleteContentRange.range.tabId, 'tab-7');
      assert.equal(requests[0].deleteContentRange.range.startIndex, 1);
      assert.equal(requests[0].deleteContentRange.range.endIndex, 5);
    });

    it('validation: endIndex must be > startIndex', async () => {
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', startIndex: 5, endIndex: 2 });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.toLowerCase().includes('end index'));
    });

    it('rejects startIndex 0 on a Google Doc (1-based)', async () => {
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', startIndex: 0, endIndex: 3 });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('1-based'));
    });

    it('textToFind deletes the matched range', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'keep DELETE keep\n' }, startIndex: 1, endIndex: 18 }] } }] },
        },
      }));
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', textToFind: 'DELETE ' });
      assert.equal(res.isError, false);
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const range = calls[calls.length - 1].args[0].requestBody.requests[0].deleteContentRange.range;
      // "DELETE " spans doc indices 6-13.
      assert.deepEqual({ startIndex: range.startIndex, endIndex: range.endIndex }, { startIndex: 6, endIndex: 13 });
    });

    it('textToFind ending at the final paragraph break keeps that break', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          body: { content: [{ startIndex: 1, endIndex: 11, paragraph: { elements: [{ textRun: { content: 'keep tail\n' }, startIndex: 1, endIndex: 11 }] } }] },
        },
      }));
      // "tail\n" spans 6-11; 11 is the segment end, so the range is trimmed to 6-10.
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', textToFind: 'tail\n' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('paragraph break kept'));
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const range = calls[calls.length - 1].args[0].requestBody.requests[0].deleteContentRange.range;
      assert.deepEqual({ startIndex: range.startIndex, endIndex: range.endIndex }, { startIndex: 6, endIndex: 10 });

      // A match that is only the final break has nothing left to delete.
      const onlyBreak = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', textToFind: '\n' });
      assert.equal(onlyBreak.isError, true);
      assert.ok(onlyBreak.content[0].text!.includes('final paragraph break'));
      assert.equal(ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length, calls.length);
    });

    it('refuses indices and textToFind together, including a single stray index', async () => {
      for (const extra of [{ startIndex: 1, endIndex: 3 }, { startIndex: 1 }, { endIndex: 3 }]) {
        const res = await callTool(ctx.client, 'deleteRange', { documentId: 'doc-1', textToFind: 'x', ...extra });
        assert.equal(res.isError, true, `expected refusal for ${JSON.stringify(extra)}`);
        assert.ok(res.content[0].text!.includes('not both'), res.content[0].text);
      }
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 0);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'deleteRange', {});
      assert.equal(res.isError, true);
    });
  });

  // --- insertText / deleteRange on text/* files ---
  describe('text-file editing', () => {
    // files.get is a single stub reached both for the metadata read and the
    // alt:'media' content download, so branch on params.alt.
    function stubTextFile(content: string, mimeType = 'text/plain', name = 'notes.txt') {
      ctx.mocks.drive.service.files.get._setImpl(async (p: any) =>
        p?.alt === 'media'
          ? { data: Readable.from(Buffer.from(content, 'utf-8')) }
          : { data: { id: 'file-1', name, mimeType, parents: ['root'] } });
    }

    function lastWrittenBody(): Buffer {
      const updates = ctx.mocks.drive.tracker.getCalls('files.update');
      return updates[updates.length - 1].args[0].media.body as Buffer;
    }

    afterEach(() => {
      ctx.mocks.drive.service.files.get._resetImpl();
    });

    it('insertText inserts at a code-point offset', async () => {
      stubTextFile('Hello World');
      const res = await callTool(ctx.client, 'insertText', { documentId: 'file-1', text: 'X', index: 5 });
      assert.equal(res.isError, false);
      assert.equal(lastWrittenBody().toString('utf-8'), 'HelloX World');
    });

    it('insertText appends at end of file (index === length)', async () => {
      stubTextFile('abc');
      const res = await callTool(ctx.client, 'insertText', { documentId: 'file-1', text: 'Z', index: 3 });
      assert.equal(res.isError, false);
      assert.equal(lastWrittenBody().toString('utf-8'), 'abcZ');
    });

    it('insertText preserves emoji (no surrogate corruption)', async () => {
      // '😀' is one code point (2 UTF-16 units); insert after it at code-point index 1.
      stubTextFile('😀abc');
      const res = await callTool(ctx.client, 'insertText', { documentId: 'file-1', text: 'X', index: 1 });
      assert.equal(res.isError, false);
      const written = lastWrittenBody().toString('utf-8');
      assert.equal(written, '😀Xabc');
      assert.ok(!written.includes('�'));
    });

    it('insertText past end of file errors', async () => {
      stubTextFile('abc');
      const res = await callTool(ctx.client, 'insertText', { documentId: 'file-1', text: 'X', index: 99 });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('beyond end of file'));
    });

    it('insertText rejects tabId on a text file', async () => {
      stubTextFile('abc');
      const res = await callTool(ctx.client, 'insertText', { documentId: 'file-1', text: 'X', index: 0, tabId: 'tab-1' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('tabId is not supported'));
    });

    it('deleteRange removes a code-point range', async () => {
      stubTextFile('Hello World');
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'file-1', startIndex: 5, endIndex: 11 });
      assert.equal(res.isError, false);
      assert.equal(lastWrittenBody().toString('utf-8'), 'Hello');
    });

    it('deleteRange preserves surrounding emoji (no surrogate corruption)', async () => {
      // 'a😀b': code points a(0) 😀(1) b(2); delete the emoji [1,2).
      stubTextFile('a😀b');
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'file-1', startIndex: 1, endIndex: 2 });
      assert.equal(res.isError, false);
      const written = lastWrittenBody().toString('utf-8');
      assert.equal(written, 'ab');
      assert.ok(!written.includes('�'));
    });

    it('deleteRange over the whole file writes empty content (not a silent no-op)', async () => {
      stubTextFile('hello\n'); // 6 code points
      const res = await callTool(ctx.client, 'deleteRange', { documentId: 'file-1', startIndex: 0, endIndex: 6 });
      assert.equal(res.isError, false);
      const updates = ctx.mocks.drive.tracker.getCalls('files.update');
      assert.equal(updates.length, 1); // the write actually happened
      const body = updates[0].args[0].media.body as Buffer;
      assert.ok(Buffer.isBuffer(body));
      assert.equal(body.length, 0); // empty Buffer is truthy → uploaded, so the file is emptied
    });

    it('editing works when metadata read fails (drive.file scope fallback → Docs API)', async () => {
      // Simulate drive.file: metadata files.get throws; media read never happens
      // because the handler falls back to the Google-Docs (batchUpdate) path.
      ctx.mocks.drive.service.files.get._setImpl(async () => { throw new Error('File not found: 404'); });
      const res = await callTool(ctx.client, 'insertText', { documentId: 'doc-1', text: 'x', index: 1 });
      assert.equal(res.isError, false);
      assert.equal(ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length, 1);
    });
  });

  // --- readGoogleDoc ---
  describe('readGoogleDoc', () => {
    it('happy path (text format)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Hello World\n' } }] } }] },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Hello World'));
    });

    it('reads multi-tab document', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-Tab Doc',
          tabs: [
            {
              tabProperties: { tabId: 'tab-1', title: 'Tab1' },
              documentTab: {
                body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' } }] } }] },
              },
            },
            {
              tabProperties: { tabId: 'tab-2', title: 'Tab2' },
              documentTab: {
                body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Second tab\n' } }] } }] },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1 ==='));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab2 ==='));
      assert.ok(res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('Second tab'));
    });

    it('reads specific tab by tabId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-Tab Doc',
          tabs: [
            {
              tabProperties: { tabId: 'tab-1', title: 'Tab1' },
              documentTab: {
                body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' } }] } }] },
              },
            },
            {
              tabProperties: { tabId: 'tab-2', title: 'Tab2' },
              documentTab: {
                body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Second tab\n' } }] } }] },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', tabId: 'tab-2' });
      assert.equal(res.isError, false);
      assert.ok(!res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('Second tab'));
    });

    it('reads specific nested tab by tabId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.fullyNested(),
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', tabId: 'tab-1-2' });
      assert.equal(res.isError, false);
      assert.ok(!res.content[0].text!.includes('First tab'));
      assert.ok(!res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('Second child'));
      assert.ok(!res.content[0].text!.includes('Second tab'));
    });
    
    it('reads specific nested tab by tabId when the document has only one tab with child tabs', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.singleParentNested(),
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', tabId: 'tab-1-2' });
      assert.equal(res.isError, false);
      assert.ok(!res.content[0].text!.includes('First tab'));
      assert.ok(!res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('Second child'));
      assert.ok(!res.content[0].text!.includes('Second tab'));
    });

    it('reads deeply nested grandchild tab by tabId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.fullyNested(),
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', tabId: 'tab-1-2-1' });
      assert.equal(res.isError, false);
      assert.ok(!res.content[0].text!.includes('First tab'));
      assert.ok(!res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('First grandchild'));
    });

    it('reads all tabs including nested when no tabId specified', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.fullyNested(),
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      // Should include all tabs with proper hierarchy
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1 ==='));
      assert.ok(res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.1 ==='));
      assert.ok(res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2 ==='));
      assert.ok(res.content[0].text!.includes('Second child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2.1 ==='));
      assert.ok(res.content[0].text!.includes('First grandchild'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab2 ==='));
      assert.ok(res.content[0].text!.includes('Second tab'));
    });
    
    it('reads all tabs including nested when no tabId specified and the document has only one tab with child tabs ', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.singleParentNested(),
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1' });
      assert.equal(res.isError, false);

      // Should include all tabs with proper hierarchy
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1 ==='));
      assert.ok(res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.1 ==='));
      assert.ok(res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2 ==='));
      assert.ok(res.content[0].text!.includes('Second child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2.1 ==='));
      assert.ok(res.content[0].text!.includes('First grandchild'));
    });

    it('returns error for unknown tabId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-Tab Doc',
          tabs: [
            {
              tabProperties: { tabId: 'tab-1', title: 'Tab1' },
              documentTab: {
                body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' } }] } }] },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', tabId: 'nonexistent' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('not found'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'readGoogleDoc', {});
      assert.equal(res.isError, true);
    });

    it('renders inline images as markdown with objectId in title (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with image',
          body: {
            content: [{
              paragraph: {
                elements: [
                  { inlineObjectElement: { inlineObjectId: 'obj-1' } },
                  { textRun: { content: '\n' } },
                ],
              },
            }],
          },
          inlineObjects: {
            'obj-1': {
              inlineObjectProperties: {
                embeddedObject: {
                  description: 'Architecture diagram',
                  imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' },
                },
              },
            },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('![Architecture diagram](https://lh3.googleusercontent.com/xyz "objectId=obj-1")'), res.content[0].text!);
    });

    it('renders heading paragraphs as ATX hashes (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with headings',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'HEADING_1' }, elements: [{ textRun: { content: 'Top level\n' } }] } },
              { paragraph: { paragraphStyle: { namedStyleType: 'HEADING_3' }, elements: [{ textRun: { content: 'Third level\n' } }] } },
              { paragraph: { paragraphStyle: { namedStyleType: 'NORMAL_TEXT' }, elements: [{ textRun: { content: 'Plain body\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('# Top level\n'), text);
      assert.ok(text.includes('### Third level\n'), text);
      assert.ok(text.includes('Plain body\n'), text);
      assert.ok(!text.includes('# Plain body'), text);
    });

    it('does not emit a bare hash for an empty heading paragraph (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with empty heading',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'HEADING_2' }, elements: [{ textRun: { content: '\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(!text.includes('## '), text);
    });

    it('leaves headings unprefixed in text format', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with headings',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'HEADING_1' }, elements: [{ textRun: { content: 'Top level\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'text' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Top level'), text);
      assert.ok(!text.includes('#'), text);
    });

    it('wraps bold, italic and strikethrough runs in emphasis (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with emphasis',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'plain ' } },
                { textRun: { content: 'bold', textStyle: { bold: true } } },
                { textRun: { content: ' and ' } },
                { textRun: { content: 'italic', textStyle: { italic: true } } },
                { textRun: { content: ' and ' } },
                { textRun: { content: 'struck', textStyle: { strikethrough: true } } },
                { textRun: { content: '\n' } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('plain **bold** and *italic* and ~~struck~~'), text);
    });

    it('keeps emphasis markers tight around a run that ends the paragraph (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with trailing emphasis',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'lead ' } },
                { textRun: { content: 'bold tail\n', textStyle: { bold: true } } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // The newline must fall outside the markers, otherwise the emphasis never closes.
      assert.ok(text.includes('lead **bold tail**\n'), JSON.stringify(text));
      assert.ok(!text.includes('\n**'), JSON.stringify(text));
    });

    it('leaves emphasis unmarked in text format', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with emphasis',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'bold', textStyle: { bold: true } } },
                { textRun: { content: '\n' } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'text' });
      assert.equal(res.isError, false);
      assert.ok(!res.content[0].text!.includes('*'), res.content[0].text!);
    });

    it('renders bulleted and numbered list items with nesting (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with lists',
          body: {
            content: [
              { paragraph: { bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'first\n' } }] } },
              { paragraph: { bullet: { listId: 'l1', nestingLevel: 1 }, elements: [{ textRun: { content: 'nested\n' } }] } },
              { paragraph: { bullet: { listId: 'l2', nestingLevel: 0 }, elements: [{ textRun: { content: 'step one\n' } }] } },
            ],
          },
          lists: {
            l1: { listProperties: { nestingLevels: [{ glyphSymbol: '●' }, { glyphSymbol: '○' }] } },
            l2: { listProperties: { nestingLevels: [{ glyphType: 'DECIMAL' }] } },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('- first\n'), JSON.stringify(text));
      assert.ok(text.includes('  - nested\n'), JSON.stringify(text));
      assert.ok(text.includes('1. step one\n'), JSON.stringify(text));
    });

    it('renders a bulleted heading as a list item, not a heading (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with bulleted heading',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'HEADING_2' }, bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'item\n' } }] } },
            ],
          },
          lists: { l1: { listProperties: { nestingLevels: [{ glyphSymbol: '●' }] } } },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('- item\n'), JSON.stringify(text));
      assert.ok(!text.includes('## item'), JSON.stringify(text));
    });

    it('renders tables as pipe tables (format=markdown)', async () => {
      const cell = (content: string) => ({ content: [{ paragraph: { elements: [{ textRun: { content } }] } }] });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with table',
          body: {
            content: [
              { table: { tableRows: [
                { tableCells: [cell('Owner'), cell('Role')] },
                { tableCells: [cell('Eero'), cell('CEO')] },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('| Owner | Role |'), JSON.stringify(text));
      assert.ok(text.includes('| --- | --- |'), JSON.stringify(text));
      assert.ok(text.includes('| Eero | CEO |'), JSON.stringify(text));
    });

    it('escapes pipe characters inside table cells (format=markdown)', async () => {
      const cell = (content: string) => ({ content: [{ paragraph: { elements: [{ textRun: { content } }] } }] });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with pipe',
          body: {
            content: [
              { table: { tableRows: [{ tableCells: [cell('a | b'), cell('c')] }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('| a \\| b | c |'), JSON.stringify(res.content[0].text!));
    });

    it('separates consecutive paragraphs with a blank line (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with paragraphs',
          body: {
            content: [
              { paragraph: { elements: [{ textRun: { content: 'First para\n' } }] } },
              { paragraph: { elements: [{ textRun: { content: 'Second para\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // A single newline is only a soft break — the two paragraphs would render
      // as one.
      assert.ok(text.includes('First para\n\nSecond para\n'), JSON.stringify(text));
    });

    it('separates a heading from the paragraph that follows it (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with heading',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'HEADING_1' }, elements: [{ textRun: { content: 'Section\n' } }] } },
              { paragraph: { elements: [{ textRun: { content: 'Body\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('# Section\n\nBody\n'), JSON.stringify(res.content[0].text!));
    });

    it('keeps consecutive list items on adjacent lines (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with list',
          body: {
            content: [
              { paragraph: { bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'one\n' } }] } },
              { paragraph: { bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'two\n' } }] } },
            ],
          },
          lists: { l1: { listProperties: { nestingLevels: [{ glyphSymbol: '●' }] } } },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      // Blank lines between items would make the list loose.
      assert.ok(res.content[0].text!.includes('- one\n- two\n'), JSON.stringify(res.content[0].text!));
    });

    it('maps TITLE and SUBTITLE named styles to headings (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with title styles',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'TITLE' }, elements: [{ textRun: { content: 'The Title\n' } }] } },
              { paragraph: { paragraphStyle: { namedStyleType: 'SUBTITLE' }, elements: [{ textRun: { content: 'The Subtitle\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('# The Title\n'), JSON.stringify(text));
      assert.ok(text.includes('## The Subtitle\n'), JSON.stringify(text));
    });

    it('does not repeat the document title when the body carries the same TITLE paragraph', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Quarterly Review', revisionId: 'rev-1',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'TITLE' }, elements: [{ textRun: { content: 'Quarterly Review\n' } }] } },
              { paragraph: { elements: [{ textRun: { content: 'Body\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.equal(text, 'revisionId: rev-1\n# Quarterly Review\n\nBody\n');
    });

    it('still prepends the document title when the body has no matching TITLE paragraph', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Real Title', revisionId: 'rev-2',
          body: {
            content: [
              { paragraph: { paragraphStyle: { namedStyleType: 'TITLE' }, elements: [{ textRun: { content: 'Something else\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      assert.equal(res.content[0].text!, 'revisionId: rev-2\n# Real Title\n\n# Something else\n');
    });

    it('indents a nested item past an ordered parent marker (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with ordered nesting',
          body: {
            content: [
              { paragraph: { bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'step one\n' } }] } },
              { paragraph: { bullet: { listId: 'l1', nestingLevel: 1 }, elements: [{ textRun: { content: 'detail\n' } }] } },
            ],
          },
          lists: {
            l1: { listProperties: { nestingLevels: [{ glyphType: 'DECIMAL' }, { glyphSymbol: '○' }] } },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // `1. ` is three columns wide, so a two-space indent would leave the child
      // outside the parent item and CommonMark would render a sibling list.
      assert.ok(text.includes('1. step one\n   - detail\n'), JSON.stringify(text));
    });

    it('pads rows so a merged header cell does not truncate body columns (format=markdown)', async () => {
      const cell = (content: string, columnSpan?: number) => ({
        content: [{ paragraph: { elements: [{ textRun: { content } }] } }],
        ...(columnSpan ? { tableCellStyle: { columnSpan } } : {}),
      });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with merged header',
          body: {
            content: [
              { table: { tableRows: [
                { tableCells: [cell('Quarterly plan', 3)] },
                { tableCells: [cell('a'), cell('b'), cell('c')] },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // GFM sizes the table from the header: a one-column header would drop
      // columns b and c from the rendering entirely.
      assert.ok(text.includes('| Quarterly plan |  |  |\n| --- | --- | --- |\n| a | b | c |'), JSON.stringify(text));
    });

    it('preserves multi-paragraph table cell structure with <br> (format=markdown)', async () => {
      const multiCell = {
        content: [
          { paragraph: { bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'item one\n' } }] } },
          { paragraph: { bullet: { listId: 'l1', nestingLevel: 0 }, elements: [{ textRun: { content: 'item two\n' } }] } },
        ],
      };
      const cell = (content: string) => ({ content: [{ paragraph: { elements: [{ textRun: { content } }] } }] });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with multiline cell',
          body: {
            content: [
              { table: { tableRows: [{ tableCells: [multiCell, cell('other')] }] } },
            ],
          },
          lists: { l1: { listProperties: { nestingLevels: [{ glyphSymbol: '●' }] } } },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // Space-joining would render "- item one - item two" with no boundary.
      assert.ok(text.includes('| - item one<br>- item two | other |'), JSON.stringify(text));
    });

    it('does not fuse emphasis delimiters between abutting styled runs (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with abutting runs',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 're', textStyle: { bold: true } } },
                { textRun: { content: 'ally', textStyle: { bold: true, italic: true } } },
                { textRun: { content: '\n' } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // Wrapping each run separately yields `**re*****ally***`, which renders as
      // literal asterisks.
      assert.ok(!text.includes('*****'), JSON.stringify(text));
      assert.ok(text.includes('**re*ally***\n'), JSON.stringify(text));
    });

    it('merges runs that Docs split without a style change (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with split run',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'bold ', textStyle: { bold: true } } },
                { textRun: { content: 'across runs', textStyle: { bold: true } } },
                { textRun: { content: '\n' } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('**bold across runs**\n'), JSON.stringify(res.content[0].text!));
    });

    it('escapes markdown metacharacters in document text (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with metacharacters',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'a*b', textStyle: { bold: true } } },
                { textRun: { content: ' and [1]\n' } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // Unescaped, the inner `*` pairs with a surrounding marker and breaks the
      // bold span.
      assert.ok(text.includes('**a\\*b**'), JSON.stringify(text));
      assert.ok(text.includes('\\[1\\]'), JSON.stringify(text));
    });

    it('leaves intraword underscores unescaped (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with snake_case',
          body: {
            content: [
              { paragraph: { elements: [{ textRun: { content: 'call snake_case_name now\n' } }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      // CommonMark does not treat intraword `_` as emphasis, so escaping it
      // would only add noise.
      assert.ok(res.content[0].text!.includes('snake_case_name'), JSON.stringify(res.content[0].text!));
    });

    it('renders person, rich link, footnote and horizontal rule elements (format=markdown)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with inline elements',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'Owner: ' } },
                { person: { personProperties: { name: 'Ada', email: 'ada@example.com' } } },
                { textRun: { content: ' see ' } },
                { richLink: { richLinkProperties: { title: 'Spec', uri: 'https://example.com/spec' } } },
                { footnoteReference: { footnoteNumber: '1' } },
                { textRun: { content: '\n' } },
              ] } },
              { paragraph: { elements: [{ horizontalRule: {} }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('@Ada (ada@example.com)'), JSON.stringify(text));
      assert.ok(text.includes('[Spec](https://example.com/spec)'), JSON.stringify(text));
      assert.ok(text.includes('[^1]'), JSON.stringify(text));
      assert.ok(text.includes('---'), JSON.stringify(text));
    });

    it('leaves person chips out of text format', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with person',
          body: {
            content: [
              { paragraph: { elements: [
                { textRun: { content: 'Owner: ' } },
                { person: { personProperties: { name: 'Ada', email: 'ada@example.com' } } },
                { textRun: { content: '\n' } },
              ] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'text' });
      assert.equal(res.isError, false);
      // Text output is unchanged by the markdown work.
      assert.ok(!res.content[0].text!.includes('@Ada'), JSON.stringify(res.content[0].text!));
    });

    it('keeps tables tab-separated in text format', async () => {
      const cell = (content: string) => ({ content: [{ paragraph: { elements: [{ textRun: { content } }] } }] });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with table',
          body: {
            content: [
              { table: { tableRows: [{ tableCells: [cell('Owner'), cell('Role')] }] } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'text' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Owner\tRole\t'), JSON.stringify(text));
      assert.ok(!text.includes('|'), JSON.stringify(text));
    });

    it('renders inline images as a single-line placeholder (format=text)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with image',
          body: {
            content: [{
              paragraph: {
                elements: [
                  { textRun: { content: 'Before ' } },
                  { inlineObjectElement: { inlineObjectId: 'obj-1' } },
                  { textRun: { content: ' after\n' } },
                ],
              },
            }],
          },
          inlineObjects: {
            'obj-1': {
              inlineObjectProperties: {
                embeddedObject: { imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' } },
              },
            },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('[image: objectId=obj-1 contentUri=https://lh3.googleusercontent.com/xyz]'), text);
      // no markdown syntax in text format
      assert.ok(!text.includes('!['), 'text format should not emit markdown image syntax');
    });

    it('falls back to [image] when the inlineObjects map is missing', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with orphan image',
          body: {
            content: [{
              paragraph: {
                elements: [
                  { inlineObjectElement: { inlineObjectId: 'obj-1' } },
                  { textRun: { content: '\n' } },
                ],
              },
            }],
            // no inlineObjects map
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('[image]'), res.content[0].text!);
    });

    it('resolves a multi-tab image against the correct tab inlineObjects map', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-tab image doc',
          tabs: [
            {
              tabProperties: { tabId: 'tab-1', title: 'Tab1' },
              documentTab: {
                body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First tab\n' } }] } }] },
                inlineObjects: {},
              },
            },
            {
              tabProperties: { tabId: 'tab-2', title: 'Tab2' },
              documentTab: {
                body: {
                  content: [{
                    paragraph: {
                      elements: [
                        { inlineObjectElement: { inlineObjectId: 'obj-2' } },
                        { textRun: { content: '\n' } },
                      ],
                    },
                  }],
                },
                inlineObjects: {
                  'obj-2': {
                    inlineObjectProperties: {
                      embeddedObject: { imageProperties: { contentUri: 'https://lh3.googleusercontent.com/tab2' } },
                    },
                  },
                },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('contentUri=https://lh3.googleusercontent.com/tab2'), res.content[0].text!);
    });

    it('readGoogleDocPaginated carries inline images (proves format threading)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Paginated image doc',
          body: {
            content: [{
              paragraph: {
                elements: [
                  { inlineObjectElement: { inlineObjectId: 'obj-1' } },
                  { textRun: { content: '\n' } },
                ],
              },
            }],
          },
          inlineObjects: {
            'obj-1': {
              inlineObjectProperties: {
                embeddedObject: {
                  description: 'Diagram',
                  imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' },
                },
              },
            },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const envelope = JSON.parse(res.content[0].text!);
      assert.ok(envelope.content.includes('![Diagram](https://lh3.googleusercontent.com/xyz "objectId=obj-1")'), envelope.content);
    });

    it('prefers the durable sourceUri over the ephemeral contentUri in markdown', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with sourced image',
          body: {
            content: [{
              paragraph: {
                elements: [
                  { inlineObjectElement: { inlineObjectId: 'obj-1' } },
                  { textRun: { content: '\n' } },
                ],
              },
            }],
          },
          inlineObjects: {
            'obj-1': {
              inlineObjectProperties: {
                embeddedObject: {
                  description: 'Diagram',
                  imageProperties: {
                    contentUri: 'https://lh3.googleusercontent.com/ephemeral',
                    sourceUri: 'https://example.com/durable.png',
                  },
                },
              },
            },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('![Diagram](https://example.com/durable.png "objectId=obj-1")'), text);
      assert.ok(!text.includes('lh3.googleusercontent.com'), 'ephemeral contentUri must not be used when a sourceUri exists');
    });

    it('percent-encodes spaces and parentheses in the markdown image URL', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with spaced uri',
          body: {
            content: [{
              paragraph: {
                elements: [
                  { inlineObjectElement: { inlineObjectId: 'obj-1' } },
                  { textRun: { content: '\n' } },
                ],
              },
            }],
          },
          inlineObjects: {
            'obj-1': {
              inlineObjectProperties: {
                embeddedObject: {
                  imageProperties: { sourceUri: 'https://example.com/a b(1).png' },
                },
              },
            },
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'doc-1', format: 'markdown' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('![](https://example.com/a%20b%281%29.png "objectId=obj-1")'), text);
    });
  });

  // --- listDocumentTabs ---
  describe('listDocumentTabs', () => {
    it('happy path', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: { documentId: 'doc-1', title: 'My Doc', body: { content: [] } },
      }));
      const res = await callTool(ctx.client, 'listDocumentTabs', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'listDocumentTabs', {});
      assert.equal(res.isError, true);
    });
  });

  // --- applyTextStyle ---
  describe('applyTextStyle', () => {
    it('happy path with index range', async () => {
      const res = await callTool(ctx.client, 'applyTextStyle', {
        documentId: 'doc-1', startIndex: 1, endIndex: 5, bold: true,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('applied text style'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'applyTextStyle', {});
      assert.equal(res.isError, true);
    });

    it('accepts baselineOffset as the only style option', async () => {
      const res = await callTool(ctx.client, 'applyTextStyle', {
        documentId: 'doc-1', startIndex: 1, endIndex: 5, baselineOffset: 'SUPERSCRIPT',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('applied text style'));
    });

    it('rejects an invalid baselineOffset value', async () => {
      const res = await callTool(ctx.client, 'applyTextStyle', {
        documentId: 'doc-1', startIndex: 1, endIndex: 5, baselineOffset: 'MIDDLE',
      });
      assert.equal(res.isError, true);
    });
  });

  // --- applyParagraphStyle ---
  describe('applyParagraphStyle', () => {
    it('happy path with index range', async () => {
      const res = await callTool(ctx.client, 'applyParagraphStyle', {
        documentId: 'doc-1', startIndex: 1, endIndex: 5, alignment: 'CENTER',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('applied paragraph style'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'applyParagraphStyle', {});
      assert.equal(res.isError, true);
    });
  });

  // --- formatGoogleDocText / formatGoogleDocParagraph aliases ---
  describe('format alias tools', () => {
    it('formatGoogleDocText delegates successfully', async () => {
      const res = await callTool(ctx.client, 'formatGoogleDocText', {
        documentId: 'doc-1', startIndex: 1, endIndex: 5, bold: true,
      });
      assert.equal(res.isError, false);
    });

    it('formatGoogleDocParagraph delegates successfully', async () => {
      const res = await callTool(ctx.client, 'formatGoogleDocParagraph', {
        documentId: 'doc-1', startIndex: 1, endIndex: 5, alignment: 'CENTER',
      });
      assert.equal(res.isError, false);
    });
  });

  // --- findAndReplaceInDoc ---
  describe('findAndReplaceInDoc', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'Hello', replaceText: 'Hi',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Replaced'));
    });

    it('dryRun counts matches without replacing', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Hello Hello World\n' } }] } }] },
        },
      }));
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'Hello', replaceText: 'Hi', dryRun: true,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('found 2 occurrence'));
    });

    it('with tabId scopes replacement via tabsCriteria', async () => {
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'Hello', replaceText: 'Hi', tabId: 'tab-2',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('tab-2'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const lastCall = calls[calls.length - 1];
      const requests = lastCall?.args?.[0]?.requestBody?.requests;
      assert.equal(requests?.length, 1);
      assert.deepEqual(requests[0].replaceAllText.tabsCriteria, { tabIds: ['tab-2'] });
      assert.equal(requests[0].replaceAllText.containsText.text, 'Hello');
    });

    it('without tabId omits tabsCriteria', async () => {
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'Hello', replaceText: 'Hi',
      });
      assert.equal(res.isError, false);

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const lastCall = calls[calls.length - 1];
      const requests = lastCall?.args?.[0]?.requestBody?.requests;
      assert.equal(requests[0].replaceAllText.tabsCriteria, undefined);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {});
      assert.equal(res.isError, true);
    });

    it('expectedCount mismatch aborts without writing', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Hello Hello World\n' } }] } }] },
        },
      }));
      const before = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length;

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'Hello', replaceText: 'Hi', expectedCount: 1,
      });

      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /Aborted without writing.*expectedCount=1.*found 2/s);
      assert.equal(
        ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length, before,
        'a failed guard must not reach batchUpdate',
      );
    });

    it('expectedCount overshoot names substring collisions as the likely cause', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'cat catalog cathode\n' } }] } }] },
        },
      }));
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'cat', replaceText: 'dog', expectedCount: 1,
      });

      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /substring collisions/);
      assert.match(res.content[0].text!, /longer, unique findText/);
    });

    it('warns when fewer occurrences change than were verified pre-write', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Hello Hello\n' } }] } }] },
        },
      }));
      // The guard sees 2, but the document is edited between the count and the
      // write, so the API reports only 1 changed.
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => ({
        data: { replies: [{ replaceAllText: { occurrencesChanged: 1 } }] },
      }));

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'Hello', replaceText: 'Hi', expectedCount: 2,
      });

      assert.equal(res.isError, false);
      assert.match(res.content[0].text!, /WARNING: 2 matches were verified/);
      assert.match(res.content[0].text!, /edited concurrently/);
    });

    it('a newline-bearing replaceText compiles to delete+insert, not replaceAllText', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc', revisionId: 'rev-9',
          body: {
            content: [{
              paragraph: { elements: [{ startIndex: 1, textRun: { content: 'Hello World\n' } }] },
            }],
          },
        },
      }));

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'World', replaceText: 'line one\nline two',
      });

      assert.equal(res.isError, false);
      assert.match(res.content[0].text!, /Replaced 1 occurrence/);

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const requests = calls[calls.length - 1]?.args?.[0]?.requestBody?.requests;
      assert.equal(requests.length, 2, 'one deleteContentRange plus one insertText');
      assert.equal(requests[0].replaceAllText, undefined, 'replaceAllText mangles newlines');
      assert.deepEqual(requests[0].deleteContentRange.range, { startIndex: 7, endIndex: 12 });
      assert.equal(requests[1].insertText.location.index, 7);
      assert.equal(requests[1].insertText.text, 'line one\nline two');
    });

    // The mock honours a `fields` mask with respect to revisionId, because the
    // API does. The lock below is taken from this read, so a projection that
    // leaves revisionId out silently unlocks the batch — and since #219 gates
    // the stale-revision message on whether a lock was sent, it silently
    // disables that message too. Both would regress with no test failing if
    // the mock handed back revisionId whatever was asked for. Only revisionId
    // is modelled; the rest of the projection is not what these tests check.
    function suppliesRevisionId(fields: unknown): boolean {
      if (typeof fields !== 'string' || fields.trim() === '') return true;
      return /(^|[(,\s])revisionId([),\s]|$)/.test(fields);
    }

    function docWithRevisionAndMatch(documentId: string, revisionId?: string) {
      ctx.mocks.docs.service.documents.get._setImpl(async (params: any) => ({
        data: {
          documentId, title: 'My Doc',
          ...(revisionId && suppliesRevisionId(params?.fields) ? { revisionId } : {}),
          body: {
            content: [{
              paragraph: { elements: [{ startIndex: 1, textRun: { content: 'Hello World\n' } }] },
            }],
          },
        },
      }));
    }

    it('pins the multi-line batch to the revision it located against', async () => {
      docWithRevisionAndMatch('doc-pin', 'rev-9');

      await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-pin', findText: 'World', replaceText: 'a\nb',
      });
      let calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.deepEqual(calls[calls.length - 1]?.args?.[0]?.requestBody.writeControl, { requiredRevisionId: 'rev-9' });

      await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-pin', findText: 'World', replaceText: 'a\nb', ifRevisionId: 'rev-caller',
      });
      calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.deepEqual(calls[calls.length - 1]?.args?.[0]?.requestBody.writeControl, { requiredRevisionId: 'rev-caller' });
    });

    it('sends no writeControl when the locating read has no revisionId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-no-rev', title: 'My Doc',
          body: {
            content: [{
              paragraph: { elements: [{ startIndex: 1, textRun: { content: 'Hello World\n' } }] },
            }],
          },
        },
      }));

      await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-no-rev', findText: 'World', replaceText: 'a\nb',
      });
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const writeControl = calls[calls.length - 1]?.args?.[0]?.requestBody.writeControl;
      assert.equal(writeControl, undefined, 'no revisionId to lock against means no writeControl at all, not { requiredRevisionId: undefined }');
    });

    it('maps a stale-revision batchUpdate failure to a message telling the caller to re-read and retry', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-stale', title: 'My Doc', revisionId: 'rev-9',
          body: {
            content: [{
              paragraph: { elements: [{ startIndex: 1, textRun: { content: 'Hello World\n' } }] },
            }],
          },
        },
      }));
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => {
        throw Object.assign(
          new Error("The provided revision doesn't match the document's current revision."),
          { status: 400 },
        );
      });

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-stale', findText: 'World', replaceText: 'a\nb',
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /document changed between the read and the write/);
      assert.match(res.content[0].text!, /[Rr]e-read the document and retry/);

      ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    });

    // The mapping above rewrites the API's own explanation, so it has to fire
    // on exactly the failure it explains. Google documents the 400 for a
    // requiredRevisionId mismatch but not its message text, so the condition
    // is "this write carried a lock", not "the message mentions a revision".
    // These two cases pin both sides of it; without them the handler could map
    // every 400 to a revision mismatch and nothing would fail.
    it('keeps the API message when it maps a stale-revision failure', async () => {
      docWithRevisionAndMatch('doc-keep', 'rev-9');
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => {
        throw Object.assign(new Error('Some specific reason from Google.'), { status: 400 });
      });

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-keep', findText: 'World', replaceText: 'a\nb',
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /document changed between the read and the write/);
      // A 400 from some other cause is still mapped, because the lock is the
      // only reliable signal — so the original text must survive for it to be
      // diagnosable at all.
      assert.match(res.content[0].text!, /Some specific reason from Google\./);

      ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    });

    it('passes a 400 through unmapped when the write carried no lock', async () => {
      // No revisionId on the read and no ifRevisionId from the caller means no
      // writeControl was sent, so a 400 cannot be the lock failing.
      docWithRevisionAndMatch('doc-nolock', undefined);
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => {
        throw Object.assign(new Error('Invalid requests[0]: bad index.'), { status: 400 });
      });

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-nolock', findText: 'World', replaceText: 'a\nb',
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /Invalid requests\[0\]: bad index\./);
      assert.doesNotMatch(res.content[0].text!, /revision mismatch/);

      ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    });

    it('passes a non-400 failure through unmapped even when the write carried a lock', async () => {
      docWithRevisionAndMatch('doc-403', 'rev-9');
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => {
        throw Object.assign(new Error('The caller does not have permission.'), { status: 403 });
      });

      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-403', findText: 'World', replaceText: 'a\nb',
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /does not have permission/);
      assert.doesNotMatch(res.content[0].text!, /revision mismatch/);

      ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    });

    it('refuses a multi-line findText rather than silently mismatching', async () => {
      const res = await callTool(ctx.client, 'findAndReplaceInDoc', {
        documentId: 'doc-1', findText: 'a\nb', replaceText: 'x\ny',
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /Multi-line findText is not supported/);
    });
  });

  // --- listComments ---
  describe('listComments', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.comments.list._setImpl(async () => ({
        data: { comments: [{ id: 'c1', content: 'Nice!', author: { displayName: 'User' }, createdTime: '2025-01-01' }] },
      }));
      const res = await callTool(ctx.client, 'listComments', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Nice!'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'listComments', {});
      assert.equal(res.isError, true);
    });

    it('passes pagination params', async () => {
      ctx.mocks.drive.service.comments.list._setImpl(async () => ({
        data: { comments: [{ id: 'c1', content: 'Hi', author: { displayName: 'User' }, createdTime: '2025-01-01' }] },
      }));
      await callTool(ctx.client, 'listComments', { documentId: 'doc-1', pageSize: 10, pageToken: 'tok' });
      const calls = ctx.mocks.drive.tracker.getCalls('comments.list');
      const lastArgs = calls[calls.length - 1].args[0];
      assert.equal(lastArgs.pageSize, 10);
      assert.equal(lastArgs.pageToken, 'tok');
    });

    it('returns nextPageToken', async () => {
      ctx.mocks.drive.service.comments.list._setImpl(async () => ({
        data: {
          comments: [{ id: 'c1', content: 'Hi', author: { displayName: 'User' }, createdTime: '2025-01-01' }],
          nextPageToken: 'next-page',
        },
      }));
      const res = await callTool(ctx.client, 'listComments', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('next-page'));
    });

    it('passes includeDeleted', async () => {
      ctx.mocks.drive.service.comments.list._setImpl(async () => ({
        data: { comments: [] },
      }));
      await callTool(ctx.client, 'listComments', { documentId: 'doc-1', includeDeleted: true });
      const calls = ctx.mocks.drive.tracker.getCalls('comments.list');
      const lastArgs = calls[calls.length - 1].args[0];
      assert.equal(lastArgs.includeDeleted, true);
    });
  });

  // --- getComment ---
  describe('getComment', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'getComment', { documentId: 'doc-1', commentId: 'c1' });
      assert.equal(res.isError, false);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'getComment', {});
      assert.equal(res.isError, true);
    });
  });

  // --- addComment ---
  describe('addComment', () => {
    const flatDoc = () => ({
      documentId: 'doc-1', title: 'My Doc',
      body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Hello World\n' }, startIndex: 1, endIndex: 13 }] } }] },
    });

    it('happy path: quotes the range and sends NO anchor', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: flatDoc() }));
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', startIndex: 1, endIndex: 6, commentText: 'Great!',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Comment added'));

      const body = ctx.mocks.drive.tracker.getCalls('comments.create').at(-1)?.args?.[0]?.requestBody;
      assert.equal(body.quotedFileContent.value, 'Hello');
      assert.equal(body.quotedFileContent.mimeType, 'text/plain');
      // The anchor Docs cannot resolve is what made the editor render these
      // threads as "Original content was deleted". It must not come back.
      assert.ok(!('anchor' in body), 'addComment must not send an anchor');
    });

    it('targets by textToFind and quotes the requested instance', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: { content: [{ paragraph: { elements: [{ textRun: { content: 'target and target again\n' }, startIndex: 1, endIndex: 25 }] } }] },
        },
      }));
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', textToFind: 'target', matchInstance: 2, commentText: 'second one',
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('comments.create').at(-1)?.args?.[0]?.requestBody;
      assert.equal(body.quotedFileContent.value, 'target');
      assert.ok(res.content[0].text!.includes('[12-18]'));
    });

    it('quotes text inside a table', async () => {
      // The previous implementation scanned only top-level body paragraphs, so
      // a range inside a table quoted '' and the comment lost its passage.
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'My Doc',
          body: {
            content: [{
              table: {
                tableRows: [{
                  tableCells: [{
                    content: [{ paragraph: { elements: [{ textRun: { content: 'CellText\n' }, startIndex: 5, endIndex: 14 }] } }],
                  }],
                }],
              },
            }],
          },
        },
      }));
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', startIndex: 5, endIndex: 9, commentText: 'about this cell',
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('comments.create').at(-1)?.args?.[0]?.requestBody;
      assert.equal(body.quotedFileContent.value, 'Cell');
    });

    it('quotes from the requested tab in a multi-tab document', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: mockDocs.multiTab() }));
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', tabId: 'tab-2', startIndex: 1, endIndex: 7, commentText: 'on tab 2',
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('comments.create').at(-1)?.args?.[0]?.requestBody;
      // Index spaces are per-tab: [1-7) is 'Second' in tab-2, 'First ' in tab-1.
      assert.equal(body.quotedFileContent.value, 'Second');
    });

    it('rejects a range that covers no text instead of commenting on nothing', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: flatDoc() }));
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', startIndex: 500, endIndex: 520, commentText: 'nowhere',
      });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('covers no text'));
      assert.equal(ctx.mocks.drive.tracker.getCalls('comments.create').length, 0);
    });

    it('reports when textToFind is absent', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: flatDoc() }));
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', textToFind: 'nonexistent', commentText: 'x',
      });
      assert.equal(res.isError, true);
      assert.equal(ctx.mocks.drive.tracker.getCalls('comments.create').length, 0);
    });

    it('validation: endIndex must be > startIndex', async () => {
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', startIndex: 5, endIndex: 2, commentText: 'test',
      });
      assert.equal(res.isError, true);
    });

    it('validation: requires indices or textToFind', async () => {
      const res = await callTool(ctx.client, 'addComment', {
        documentId: 'doc-1', commentText: 'no target',
      });
      assert.equal(res.isError, true);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'addComment', {});
      assert.equal(res.isError, true);
    });
  });

  // --- replyToComment ---
  describe('replyToComment', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'replyToComment', {
        documentId: 'doc-1', commentId: 'c1', replyText: 'Thanks!',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Reply added'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'replyToComment', {});
      assert.equal(res.isError, true);
    });
  });

  // --- deleteComment ---
  // --- getGoogleDocContent ---
  describe('getGoogleDocContent', () => {
    it('reads multi-tab document', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.multiTab(),
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1 (tabId=tab-1) ==='));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab2 (tabId=tab-2) ==='));
      assert.ok(res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('Second tab'));
    });

    it('reads multi-tab document with nested tabs', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.fullyNested(),
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      // Should include all tabs with proper hierarchy
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1 (tabId=tab-1) ==='));
      assert.ok(res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.1 (tabId=tab-1-1) ==='));
      assert.ok(res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2 (tabId=tab-1-2) ==='));
      assert.ok(res.content[0].text!.includes('Second child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2.1 (tabId=tab-1-2-1) ==='));
      assert.ok(res.content[0].text!.includes('First grandchild'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab2 (tabId=tab-2) ==='));
      assert.ok(res.content[0].text!.includes('Second tab'));
    });

    it('reads multi-tab document with nested tabs when the document has only one parent tab with child tabs', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: mockDocs.singleParentNested(),
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      // Should include all tabs with proper hierarchy
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1 (tabId=tab-1) ==='));
      assert.ok(res.content[0].text!.includes('First tab'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.1 (tabId=tab-1-1) ==='));
      assert.ok(res.content[0].text!.includes('First child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2 (tabId=tab-1-2) ==='));
      assert.ok(res.content[0].text!.includes('Second child'));
      assert.ok(res.content[0].text!.includes('=== Tab: Tab1.2.1 (tabId=tab-1-2-1) ==='));
      assert.ok(res.content[0].text!.includes('First grandchild'));
    });

    it('falls back to body for single-tab doc', async () => {
      // Default mock has no tabs array, just body.content
      ctx.mocks.docs.service.documents.get._resetImpl();
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Hello World'));
      assert.ok(!res.content[0].text!.includes('=== Tab:'));
    });

    it('includes formatting when includeFormatting is true', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Styled Doc',
          tabs: [
            {
              tabProperties: { title: 'Main' },
              documentTab: {
                body: {
                  content: [{
                    paragraph: {
                      elements: [{
                        textRun: {
                          content: 'Bold heading\n',
                          textStyle: {
                            bold: true,
                            weightedFontFamily: { fontFamily: 'Roboto' },
                            fontSize: { magnitude: 18 },
                            foregroundColor: { color: { rgbColor: { red: 1, green: 0, blue: 0 } } },
                          },
                        },
                        startIndex: 1,
                        endIndex: 14,
                      }],
                    },
                  }],
                },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1', includeFormatting: true });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('font="Roboto"'), 'should include font name');
      assert.ok(text.includes('size=18pt'), 'should include font size');
      assert.ok(text.includes('style=bold'), 'should include bold style');
      assert.ok(text.includes('color=#ff0000'), 'should include foreground color');
      assert.ok(text.includes('--- Fonts summary ---'), 'should include fonts summary');
      assert.ok(text.includes('Roboto: sizes [18 pt], styles [bold]'), 'fonts summary should list Roboto with sizes and styles');
    });

    it('excludes formatting by default', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Styled Doc',
          tabs: [
            {
              tabProperties: { title: 'Main' },
              documentTab: {
                body: {
                  content: [{
                    paragraph: {
                      elements: [{
                        textRun: {
                          content: 'Normal text\n',
                          textStyle: {
                            bold: true,
                            weightedFontFamily: { fontFamily: 'Arial' },
                            fontSize: { magnitude: 12 },
                          },
                        },
                        startIndex: 1,
                        endIndex: 13,
                      }],
                    },
                  }],
                },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(!text.includes('font='), 'should not include font metadata');
      assert.ok(!text.includes('--- Fonts summary ---'), 'should not include fonts summary');
      assert.ok(text.includes('Normal text'), 'should still include text content');
    });

    it('surfaces superscript/subscript runs on the formatted read path', async () => {
      // Without this, text written by applyTextStyle({ baselineOffset })
      // reads back indistinguishable from unformatted text.
      const styledRun = (content: string, baselineOffset: string, startIndex: number, endIndex: number) => ({
        textRun: { content, textStyle: { baselineOffset } },
        startIndex,
        endIndex,
      });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Baseline Doc',
          tabs: [
            {
              tabProperties: { title: 'Main' },
              documentTab: {
                body: {
                  content: [{
                    paragraph: {
                      elements: [
                        styledRun('E = mc\n', 'NONE', 1, 8),
                        styledRun('2\n', 'SUPERSCRIPT', 8, 10),
                        styledRun('H2O\n', 'SUBSCRIPT', 10, 14),
                      ],
                    },
                  }],
                },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1', includeFormatting: true });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('baseline=superscript'), 'should mark the superscript run');
      assert.ok(text.includes('baseline=subscript'), 'should mark the subscript run');
      // NONE is what the API reports for ordinary text; emitting it would force
      // a meta line onto every unformatted run.
      assert.ok(!text.includes('baseline=none'), 'should not mark normal-baseline runs');
      const normalLine = text.split('\n').find((l) => l.includes('E = mc'))!;
      assert.ok(!normalLine.includes('baseline='), 'normal run should carry no baseline marker');
    });

    it('includes formatting with multi-tab', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Multi-Tab Styled',
          tabs: [
            {
              tabProperties: { title: 'Tab1' },
              documentTab: {
                body: {
                  content: [{
                    paragraph: {
                      elements: [{
                        textRun: {
                          content: 'First\n',
                          textStyle: { italic: true, weightedFontFamily: { fontFamily: 'Georgia' }, fontSize: { magnitude: 14 } },
                        },
                        startIndex: 1,
                        endIndex: 7,
                      }],
                    },
                  }],
                },
              },
            },
            {
              tabProperties: { title: 'Tab2' },
              documentTab: {
                body: {
                  content: [{
                    paragraph: {
                      elements: [{
                        textRun: {
                          content: 'Second\n',
                          textStyle: { bold: true, weightedFontFamily: { fontFamily: 'Georgia' }, fontSize: { magnitude: 10 } },
                        },
                        startIndex: 1,
                        endIndex: 8,
                      }],
                    },
                  }],
                },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1', includeFormatting: true });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // This fixture's tabProperties carry no tabId, so the header advertises
      // none — the id is only printed when there is one to pass to editTableCell.
      assert.ok(text.includes('=== Tab: Tab1 ==='), 'should have tab headers');
      assert.ok(text.includes('=== Tab: Tab2 ==='), 'should have tab headers');
      assert.ok(text.includes('style=italic'), 'should show italic in Tab1');
      assert.ok(text.includes('style=bold'), 'should show bold in Tab2');
      assert.ok(text.includes('--- Fonts summary ---'), 'should include fonts summary');
      assert.ok(text.includes('Georgia: sizes [10, 14 pt], styles [bold, italic]'), 'fonts summary should aggregate Georgia with sizes and styles');
    });

    it('includes tab headers only when multiple tabs', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Single-Tab Doc',
          tabs: [
            {
              tabProperties: { title: 'Only Tab' },
              documentTab: {
                body: {
                  content: [{ paragraph: { elements: [{ textRun: { content: 'Content here\n' }, startIndex: 1, endIndex: 14 }] } }],
                },
              },
            },
          ],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Content here'));
      assert.ok(!res.content[0].text!.includes('=== Tab:'));
    });

    it('extracts person chips', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with chips',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { textRun: { content: 'Assigned to ' }, startIndex: 0, endIndex: 12 },
                      { person: { personProperties: { name: 'Alice', email: 'alice@example.com' } }, startIndex: 12, endIndex: 13 },
                      { textRun: { content: '\n' }, startIndex: 13, endIndex: 14 },
                    ],
                  },
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('@Alice (alice@example.com)'));
    });

    it('extracts rich links as markdown', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with links',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { textRun: { content: 'See ' }, startIndex: 0, endIndex: 4 },
                      { richLink: { richLinkProperties: { title: 'Design Doc', uri: 'https://docs.google.com/doc/123' } }, startIndex: 4, endIndex: 5 },
                      { textRun: { content: '\n' }, startIndex: 5, endIndex: 6 },
                    ],
                  },
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('[Design Doc](https://docs.google.com/doc/123)'));
    });

    it('extracts inline images with description, uri, and size on one line', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with image',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 0, endIndex: 1 },
                      { textRun: { content: '\n' }, startIndex: 1, endIndex: 2 },
                    ],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      description: 'Architecture diagram',
                      imageProperties: {
                        contentUri: 'https://lh3.googleusercontent.com/xyz',
                        sourceUri: 'https://example.com/a.png',
                      },
                      size: {
                        width: { magnitude: 468, unit: 'PT' },
                        height: { magnitude: 286, unit: 'PT' },
                      },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      // The whole image token must live on a single line.
      const imageLine = text.split('\n').find(l => l.includes('objectId=obj-1'));
      assert.ok(imageLine, 'image token should be present on one line');
      assert.ok(imageLine!.includes('alt="Architecture diagram"'));
      assert.ok(imageLine!.includes('contentUri=https://lh3.googleusercontent.com/xyz'));
      assert.ok(imageLine!.includes('sourceUri=https://example.com/a.png'));
      assert.ok(imageLine!.includes('size=468x286pt'));
    });

    it('surfaces sourceUri when contentUri is absent', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with source-only image',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 0, endIndex: 1 },
                      { textRun: { content: '\n' }, startIndex: 1, endIndex: 2 },
                    ],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      imageProperties: { sourceUri: 'https://example.com/a.png' },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('sourceUri=https://example.com/a.png'));
      assert.ok(!text.includes('contentUri='), 'contentUri should be omitted when absent');
    });

    it('escapes brackets and quotes in image alt text', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with bracketed alt',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 0, endIndex: 1 },
                      { textRun: { content: '\n' }, startIndex: 1, endIndex: 2 },
                    ],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      description: 'Chart [v2] "final"',
                      imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('alt="Chart \\[v2\\] \\"final\\""'), `alt not escaped: ${text}`);
    });

    it('labels an inline image with its true 1-index span, not the placeholder length', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with indexed image',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { textRun: { content: 'See ' }, startIndex: 1, endIndex: 5 },
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 5, endIndex: 6 },
                      { textRun: { content: ' here\n' }, startIndex: 6, endIndex: 12 },
                    ],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      imageProperties: { contentUri: 'https://lh3.googleusercontent.com/' + 'x'.repeat(120) },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      const imageLine = text.split('\n').find(l => l.includes('objectId=obj-1'));
      assert.ok(imageLine, 'image token should be present');
      // The displayed edit range must be the image's real 1-index span [5-6],
      // never [5 .. 5+placeholderLength] (which would over-delete following text).
      assert.ok(imageLine!.startsWith('[5-6] '), `expected [5-6] span, got: ${imageLine}`);
    });

    it('keeps a multi-line alt-text description on a single placeholder line', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with multi-line alt',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 1, endIndex: 2 },
                      { textRun: { content: '\n' }, startIndex: 2, endIndex: 3 },
                    ],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      description: 'Line one\nLine two',
                      imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      const imageLine = text.split('\n').find(l => l.includes('objectId=obj-1'));
      assert.ok(imageLine, 'image token should be present on one line');
      // The newline in the alt text must be collapsed to a space, not leak into output.
      assert.ok(imageLine!.includes('alt="Line one Line two"'), `alt not single-lined: ${imageLine}`);
    });

    it('escapes square brackets in image URIs so the placeholder is not truncated', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with bracketed uri',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 1, endIndex: 2 },
                      { textRun: { content: '\n' }, startIndex: 2, endIndex: 3 },
                    ],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      imageProperties: { sourceUri: 'https://example.com/a]b.png' },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      const imageLine = text.split('\n').find(l => l.includes('objectId=obj-1'));
      assert.ok(imageLine, 'image token should be present');
      // The `]` in the URI must be escaped so it does not close the [image: ...] delimiter early.
      assert.ok(imageLine!.includes('sourceUri=https://example.com/a\\]b.png'), `uri not escaped: ${imageLine}`);
      assert.ok(imageLine!.trimEnd().endsWith(']'), 'placeholder should still be closed by a trailing ]');
    });

    it('renders inline images inside table cells without embedded pipes', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with table image',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  startIndex: 0,
                  endIndex: 2,
                  table: {
                    tableRows: [{
                      tableCells: [{
                        content: [{
                          paragraph: {
                            elements: [
                              { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 1, endIndex: 2 },
                            ],
                          },
                        }],
                      }],
                    }],
                  },
                }],
              },
              inlineObjects: {
                'obj-1': {
                  inlineObjectProperties: {
                    embeddedObject: {
                      imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' },
                    },
                  },
                },
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      const imageLine = text.split('\n').find(l => l.includes('objectId=obj-1'));
      assert.ok(imageLine, 'image token should appear in the table row');
      // The token itself (from `[image:` to its closing `]`) must contain no `|`,
      // so it can't break the surrounding `| cell |` table structure.
      const start = imageLine!.indexOf('[image:');
      const token = imageLine!.slice(start, imageLine!.indexOf(']', start) + 1);
      assert.ok(!token.includes('|'), `image token should not contain a pipe: ${token}`);
    });

    it('extracts footnote references', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with footnote',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { textRun: { content: 'Important claim' }, startIndex: 0, endIndex: 15 },
                      { footnoteReference: { footnoteNumber: '1', footnoteId: 'fn-1' }, startIndex: 15, endIndex: 16 },
                      { textRun: { content: '\n' }, startIndex: 16, endIndex: 17 },
                    ],
                  },
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('[^1]'));
    });

    it('extracts horizontal rules', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with hr',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [
                  { paragraph: { elements: [{ textRun: { content: 'Above\n' }, startIndex: 0, endIndex: 6 }] } },
                  { paragraph: { elements: [{ horizontalRule: {}, startIndex: 6, endIndex: 7 }] } },
                  { paragraph: { elements: [{ textRun: { content: 'Below\n' }, startIndex: 7, endIndex: 13 }] } },
                ],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('---'));
    });

    it('escapes brackets in rich link titles', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with bracketed link',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { richLink: { richLinkProperties: { title: 'Budget [Draft]', uri: 'https://docs.google.com/doc/456' } }, startIndex: 0, endIndex: 1 },
                      { textRun: { content: '\n' }, startIndex: 1, endIndex: 2 },
                    ],
                  },
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Budget \\[Draft\\]'), 'brackets in title should be escaped');
      assert.ok(text.includes('(https://docs.google.com/doc/456)'), 'URL should be preserved');
    });

    it('shows [image] placeholder when inlineObjects map is missing', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with orphan image',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  paragraph: {
                    elements: [
                      { textRun: { content: 'Before ' }, startIndex: 0, endIndex: 7 },
                      { inlineObjectElement: { inlineObjectId: 'obj-1' }, startIndex: 7, endIndex: 8 },
                      { textRun: { content: ' after\n' }, startIndex: 8, endIndex: 15 },
                    ],
                  },
                }],
              },
              // no inlineObjects map
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('[image]'), 'should show placeholder even without inlineObjects map');
    });

    it('extracts tables as markdown', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with table',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [
                  { paragraph: { elements: [{ textRun: { content: 'Before table\n' }, startIndex: 0, endIndex: 13 }] } },
                  {
                    table: {
                      tableRows: [
                        { tableCells: [
                          { content: [{ paragraph: { elements: [{ textRun: { content: 'Owner' }, startIndex: 14, endIndex: 19 }] } }] },
                          { content: [{ paragraph: { elements: [{ textRun: { content: 'Role' }, startIndex: 20, endIndex: 24 }] } }] },
                        ]},
                        { tableCells: [
                          { content: [{ paragraph: { elements: [{ textRun: { content: 'Eero' }, startIndex: 25, endIndex: 29 }] } }] },
                          { content: [{ paragraph: { elements: [{ textRun: { content: 'CEO' }, startIndex: 30, endIndex: 33 }] } }] },
                        ]},
                      ],
                    },
                    startIndex: 13,
                    endIndex: 50,
                  },
                  { paragraph: { elements: [{ textRun: { content: 'After table\n' }, startIndex: 50, endIndex: 62 }] } },
                ],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('| Owner | Role |'));
      assert.ok(res.content[0].text!.includes('| --- | --- |'));
      assert.ok(res.content[0].text!.includes('| Eero | CEO |'));
      assert.ok(res.content[0].text!.includes('Before table'));
      assert.ok(res.content[0].text!.includes('After table'));
    });

    it('extracts table of contents content', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with TOC',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [
                  {
                    tableOfContents: {
                      content: [
                        { paragraph: { elements: [{ textRun: { content: '1. Introduction\n' }, startIndex: 0, endIndex: 16 }] } },
                        { paragraph: { elements: [{ textRun: { content: '2. Overview\n' }, startIndex: 16, endIndex: 28 }] } },
                      ],
                    },
                  },
                  { paragraph: { elements: [{ textRun: { content: 'Body text here\n' }, startIndex: 28, endIndex: 43 }] } },
                ],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('1. Introduction'));
      assert.ok(res.content[0].text!.includes('2. Overview'));
      assert.ok(res.content[0].text!.includes('Body text here'));
    });

    it('extracts multi-row table with empty cells', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with sparse table',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  table: {
                    tableRows: [
                      { tableCells: [
                        { content: [{ paragraph: { elements: [{ textRun: { content: 'Field' }, startIndex: 1, endIndex: 6 }] } }] },
                        { content: [{ paragraph: { elements: [{ textRun: { content: 'Value' }, startIndex: 7, endIndex: 12 }] } }] },
                      ]},
                      { tableCells: [
                        { content: [{ paragraph: { elements: [{ textRun: { content: 'Status' }, startIndex: 13, endIndex: 19 }] } }] },
                        { content: [] },
                      ]},
                    ],
                  },
                  startIndex: 0,
                  endIndex: 30,
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('| Field | Value |'));
      assert.ok(res.content[0].text!.includes('| Status |  |'));
    });

    it('escapes pipe characters in cell text', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with pipes in cells',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  table: {
                    tableRows: [
                      { tableCells: [
                        { content: [{ paragraph: { elements: [{ textRun: { content: 'Choice' }, startIndex: 1, endIndex: 7 }] } }] },
                      ]},
                      { tableCells: [
                        { content: [{ paragraph: { elements: [{ textRun: { content: 'Option A | Option B' }, startIndex: 8, endIndex: 27 }] } }] },
                      ]},
                    ],
                  },
                  startIndex: 0,
                  endIndex: 30,
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Option A \\| Option B'), 'pipe in cell text should be escaped');
      assert.ok(!text.includes('| Option A | Option B |'), 'unescaped pipe should not produce extra columns');
    });

    it('joins multi-paragraph cells with spaces', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1',
          title: 'Doc with multi-paragraph cell',
          tabs: [{
            tabProperties: { title: 'Main' },
            documentTab: {
              body: {
                content: [{
                  table: {
                    tableRows: [
                      { tableCells: [
                        { content: [{ paragraph: { elements: [{ textRun: { content: 'Header' }, startIndex: 1, endIndex: 7 }] } }] },
                      ]},
                      { tableCells: [
                        { content: [
                          { paragraph: { elements: [{ textRun: { content: 'Hello\n' }, startIndex: 8, endIndex: 14 }] } },
                          { paragraph: { elements: [{ textRun: { content: 'World\n' }, startIndex: 14, endIndex: 20 }] } },
                        ]},
                      ]},
                    ],
                  },
                  startIndex: 0,
                  endIndex: 25,
                }],
              },
            },
          }],
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Hello World'), 'multi-paragraph cell should join with space');
      assert.ok(!text.includes('HelloWorld'), 'paragraphs should not be concatenated without separator');
    });
  });

  // --- getGoogleDocImage ---
  describe('getGoogleDocImage', () => {
    const PNG_BYTES = Buffer.from('89504e470d0a1a0a', 'hex'); // PNG signature, stand-in bytes
    // Exact ArrayBuffer (Buffer.from(...).buffer is a shared/pooled 8KB buffer).
    const pngArrayBuffer = () => new Uint8Array(PNG_BYTES).buffer;
    const imageDocData = {
      documentId: 'doc-1', title: 'Doc with image',
      body: {
        content: [{
          paragraph: { elements: [{ inlineObjectElement: { inlineObjectId: 'obj-1' } }] },
        }],
      },
      inlineObjects: {
        'obj-1': {
          inlineObjectProperties: {
            embeddedObject: { imageProperties: { contentUri: 'https://lh3.googleusercontent.com/xyz' } },
          },
        },
      },
    };

    afterEach(() => {
      ctx.resetAuthRequest();
    });

    it('returns a native MCP image block by default', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: imageDocData }));
      let requestedUrl = '';
      ctx.setAuthRequest(async (opts: any) => {
        requestedUrl = opts.url;
        return { data: pngArrayBuffer(), headers: new Headers({ 'content-type': 'image/png' }) };
      });
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1' });
      assert.equal(res.isError, false);
      assert.equal(requestedUrl, 'https://lh3.googleusercontent.com/xyz');
      assert.equal(res.content[0].type, 'image');
      assert.equal(res.content[0].mimeType, 'image/png');
      assert.equal(res.content[0].data, PNG_BYTES.toString('base64'));
    });

    it('strips charset from the content-type header', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: imageDocData }));
      ctx.setAuthRequest(async () => ({ data: pngArrayBuffer(), headers: new Headers({ 'content-type': 'image/jpeg; charset=binary' }) }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1' });
      assert.equal(res.isError, false);
      assert.equal(res.content[0].mimeType, 'image/jpeg');
    });

    // gaxios 7 returns a Headers instance (see test/gaxios-contract.test.ts); the
    // doubles above use it. This one keeps the legacy plain-object shape covered,
    // since getResponseHeader() accepts both.
    it('reads a plain-object content-type header (pre-gaxios-7 shape)', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: imageDocData }));
      ctx.setAuthRequest(async () => ({ data: pngArrayBuffer(), headers: { 'content-type': 'image/gif' } }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1' });
      assert.equal(res.isError, false);
      assert.equal(res.content[0].mimeType, 'image/gif');
    });

    it('returns a base64 JSON envelope when outputFormat=base64', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: imageDocData }));
      ctx.setAuthRequest(async () => ({ data: pngArrayBuffer(), headers: new Headers({ 'content-type': 'image/png' }) }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1', outputFormat: 'base64' });
      assert.equal(res.isError, false);
      assert.equal(res.content[0].type, 'text');
      const envelope = JSON.parse(res.content[0].text!);
      assert.equal(envelope.inlineObjectId, 'obj-1');
      assert.equal(envelope.mimeType, 'image/png');
      assert.equal(envelope.byteLength, PNG_BYTES.byteLength);
      assert.equal(envelope.dataBase64, PNG_BYTES.toString('base64'));
    });

    it('resolves an inline object from a multi-tab document', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Multi-tab image doc',
          tabs: [
            { tabProperties: { tabId: 'tab-1', title: 'Tab1' }, documentTab: { body: { content: [] }, inlineObjects: {} } },
            {
              tabProperties: { tabId: 'tab-2', title: 'Tab2' },
              documentTab: {
                body: { content: [] },
                inlineObjects: {
                  'obj-2': {
                    inlineObjectProperties: {
                      embeddedObject: { imageProperties: { contentUri: 'https://lh3.googleusercontent.com/tab2' } },
                    },
                  },
                },
              },
            },
          ],
        },
      }));
      let requestedUrl = '';
      ctx.setAuthRequest(async (opts: any) => {
        requestedUrl = opts.url;
        return { data: pngArrayBuffer(), headers: new Headers({ 'content-type': 'image/png' }) };
      });
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-2' });
      assert.equal(res.isError, false);
      assert.equal(requestedUrl, 'https://lh3.googleusercontent.com/tab2');
    });

    it('errors when the inlineObjectId is not found', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: imageDocData }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'nope' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('not found'));
    });

    it('errors when the inline object has no fetchable image content', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with chart',
          body: { content: [] },
          inlineObjects: {
            'obj-1': { inlineObjectProperties: { embeddedObject: { title: 'A chart' } } },
          },
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('no fetchable image content'));
    });

    it('surfaces the external sourceUri instead of a misleading no-image error', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Doc with source-only image',
          body: { content: [] },
          inlineObjects: {
            'obj-1': {
              inlineObjectProperties: {
                embeddedObject: { imageProperties: { sourceUri: 'https://example.com/a.png' } },
              },
            },
          },
        },
      }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1' });
      assert.equal(res.isError, true);
      const text = res.content[0].text!;
      assert.ok(text.includes('https://example.com/a.png'), text);
      assert.ok(!text.includes('embedded chart or drawing'), 'should not use the no-image message when a sourceUri exists');
    });

    it('reports a non-contradictory decimal size when the image exceeds the 40 MB cap', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: imageDocData }));
      const oversized = new Uint8Array(42257613).buffer; // ~40.3 MB, just over the 40 MB cap
      ctx.setAuthRequest(async () => ({ data: oversized, headers: new Headers({ 'content-type': 'image/png' }) }));
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1', inlineObjectId: 'obj-1' });
      assert.equal(res.isError, true);
      const text = res.content[0].text!;
      assert.ok(text.includes('40.3 MB'), `expected decimal size, got: ${text}`);
      assert.ok(text.includes('limit 40 MB'), text);
      assert.ok(!text.includes('(40 MB,'), 'must not round to a self-contradictory "40 MB, limit 40 MB"');
    });

    it('validation error when inlineObjectId is missing', async () => {
      const res = await callTool(ctx.client, 'getGoogleDocImage', { documentId: 'doc-1' });
      assert.equal(res.isError, true);
    });
  });

  // --- readGoogleDocPaginated ---
  describe('readGoogleDocPaginated', () => {
    const longDoc = (text: string) => ({
      documentId: 'doc-1', title: 'Big Doc',
      body: { content: [{ paragraph: { elements: [{ textRun: { content: text } }] } }] },
    });

    it('returns first page with hasMore and nextOffset', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: longDoc('X'.repeat(120)) }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', offset: 0, limit: 50 });
      assert.equal(res.isError, false);
      const r = JSON.parse(res.content[0].text!);
      assert.equal(r.content.length, 50);
      assert.equal(r.outputLength, 120);
      assert.equal(r.documentLength, 120);
      assert.equal(r.hasMore, true);
      assert.equal(r.nextOffset, 50);
    });

    it('last page reports hasMore false and nextOffset at end', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: longDoc('Y'.repeat(40)) }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', offset: 0, limit: 50000 });
      const r = JSON.parse(res.content[0].text!);
      assert.equal(r.content.length, 40);
      assert.equal(r.hasMore, false);
      assert.equal(r.nextOffset, 40);
    });

    it('offset beyond document returns empty content', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: longDoc('Z'.repeat(30)) }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', offset: 9999, limit: 50 });
      const r = JSON.parse(res.content[0].text!);
      assert.equal(r.content, '');
      assert.equal(r.hasMore, false);
      assert.equal(r.nextOffset, 30);
    });

    it('markdown format includes the title in the first page', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: longDoc('Body text\n') }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', format: 'markdown', offset: 0, limit: 50000 });
      const r = JSON.parse(res.content[0].text!);
      assert.ok(r.content.startsWith('# Big Doc'), 'first page should start with the markdown title');
      assert.ok(r.outputLength > r.documentLength, 'outputLength includes the title prefix, documentLength does not');
    });

    it('does not split a markdown table across pages', async () => {
      const cell = (content: string) => ({ content: [{ paragraph: { elements: [{ textRun: { content } }] } }] });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Paged',
          body: {
            content: [
              { paragraph: { elements: [{ textRun: { content: 'Intro paragraph\n' } }] } },
              { table: { tableRows: [
                { tableCells: [cell('Owner'), cell('Role')] },
                { tableCells: [cell('Eero'), cell('CEO')] },
              ] } },
            ],
          },
        },
      }));
      // A limit that lands between the header row and the separator row.
      const first = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', format: 'markdown', offset: 0, limit: 50 });
      const p1 = JSON.parse(first.content[0].text!);
      // The whole table moves to the next page rather than leaving a headerless
      // fragment behind.
      assert.ok(!p1.content.includes('|'), JSON.stringify(p1.content));
      assert.equal(p1.hasMore, true);

      const second = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', format: 'markdown', offset: p1.nextOffset, limit: 50 });
      const p2 = JSON.parse(second.content[0].text!);
      assert.ok(p2.content.includes('| Owner | Role |\n| --- | --- |\n| Eero | CEO |'), JSON.stringify(p2.content));
    });

    it('still advances when a single table is larger than the page limit', async () => {
      const cell = (content: string) => ({ content: [{ paragraph: { elements: [{ textRun: { content } }] } }] });
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'T',
          body: {
            content: [
              { table: { tableRows: Array.from({ length: 8 }, (_, i) => ({ tableCells: [cell(`row ${i}`), cell('value')] })) } },
            ],
          },
        },
      }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', format: 'markdown', offset: 0, limit: 40 });
      const r = JSON.parse(res.content[0].text!);
      // No forward progress would be an infinite pagination loop.
      assert.ok(r.nextOffset > 0, JSON.stringify(r));
      assert.ok(r.content.length > 0, JSON.stringify(r));
    });

    it('reads a specific tab by tabId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: mockDocs.multiTab() }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', tabId: 'tab-2', offset: 0, limit: 50000 });
      const r = JSON.parse(res.content[0].text!);
      assert.ok(r.content.includes('Second tab'));
      assert.ok(!r.content.includes('First tab'));
    });

    it('returns error for unknown tabId', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: mockDocs.multiTab() }));
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', tabId: 'nope' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('not found'));
    });

    it('rejects the removed json format', async () => {
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', { documentId: 'doc-1', format: 'json' });
      assert.equal(res.isError, true);
    });

    it('validation error when documentId missing', async () => {
      const res = await callTool(ctx.client, 'readGoogleDocPaginated', {});
      assert.equal(res.isError, true);
    });
  });

  // --- getGoogleDocContentPaginated ---
  describe('getGoogleDocContentPaginated', () => {
    const indexedDoc = () => ({
      documentId: 'doc-1', title: 'Indexed Doc',
      body: { content: [
        { paragraph: { elements: [{ textRun: { content: 'Alpha\n' }, startIndex: 1, endIndex: 7 }] } },
        { paragraph: { elements: [{ textRun: { content: 'Bravo\n' }, startIndex: 7, endIndex: 13 }] } },
        { paragraph: { elements: [{ textRun: { content: 'Charlie\n' }, startIndex: 13, endIndex: 21 }] } },
      ] },
    });
    const oneLongLine = () => ({
      documentId: 'doc-1', title: 'One Long Line',
      body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Q'.repeat(200) + '\n' }, startIndex: 1, endIndex: 202 }] } }] },
    });

    it('never splits a page inside a table rendering', async () => {
      // Table rows are bare `| ... |` lines with no index of their own. A page
      // break between the <table ...> header and its rows opens the next page
      // with `| gamma | delta |` and nothing saying which table that is.
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({
        data: {
          documentId: 'doc-1', title: 'Table Doc',
          body: { content: [
            { paragraph: { elements: [{ textRun: { content: 'Intro\n' }, startIndex: 1, endIndex: 7 }] } },
            {
              startIndex: 7, endIndex: 20,
              table: { tableRows: [{ tableCells: [
                { content: [{ paragraph: { elements: [{ textRun: { content: 'gamma\n' }, startIndex: 9, endIndex: 15 }] } }] },
                { content: [{ paragraph: { elements: [{ textRun: { content: 'delta\n' }, startIndex: 16, endIndex: 22 }] } }] },
              ] }] },
            },
          ] },
        },
      }));

      // Walk every page boundary a caller could land on. Two invariants: a page
      // never ends on a table header with its rows pushed to the next page, and
      // when the whole block fits in the limit it is never split at all. A block
      // longer than the limit still takes a hard cut, so pagination advances.
      for (const limit of [40, 60, 100, 140, 172, 175, 190, 200, 300]) {
        let offset = 0;
        for (let page = 0; page < 30; page++) {
          const res = await callTool(ctx.client, 'getGoogleDocContentPaginated', {
            documentId: 'doc-1', offset, limit,
          });
          const r = JSON.parse(res.content[0].text!);
          // A page that ends right after a COMPLETE header line is the bad cut:
          // the next page then opens with bare `| a | b |` rows. A header split
          // mid-line is just ordinary long-line pagination and is fine.
          assert.ok(!/(^|\n)\[\d+-\d+\] <table[^\n]*\n$/.test(r.content),
            `limit=${limit}: page ends on a complete table header, orphaning its rows`);

          if (limit >= 300 && /\| gamma /.test(r.content)) {
            assert.match(r.content, /<table/, `limit=${limit}: rows without their header`);
            assert.match(r.content, /cells: r0c0/, `limit=${limit}: rows without their cell map`);
          }

          if (!r.hasMore) break;
          assert.ok(r.nextOffset > offset, 'pagination must advance');
          offset = r.nextOffset;
        }
      }
    });

    it('returns indexed content with hasMore and nextOffset', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: indexedDoc() }));
      const res = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'doc-1', offset: 0, limit: 50000 });
      assert.equal(res.isError, false);
      const r = JSON.parse(res.content[0].text!);
      assert.ok(r.content.includes('[1-6] Alpha'));
      assert.ok(r.content.includes('[7-12] Bravo'));
      assert.equal(r.hasMore, false);
      assert.equal(typeof r.outputLength, 'number');
      assert.equal(typeof r.documentLength, 'number');
    });

    it('snaps the page end to a line boundary so index prefixes are never split', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: indexedDoc() }));
      const page1 = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'doc-1', offset: 0, limit: 50 });
      const r1 = JSON.parse(page1.content[0].text!);
      assert.ok(r1.content.endsWith('\n'), 'snapped page must end on a newline');
      assert.ok(r1.content.includes('[1-6] Alpha'), 'the Alpha line must be whole');
      assert.ok(!r1.content.includes('[7-12'), 'the Bravo prefix must not be partially included');
      assert.equal(r1.hasMore, true);

      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: indexedDoc() }));
      const page2 = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'doc-1', offset: r1.nextOffset, limit: 50 });
      const r2 = JSON.parse(page2.content[0].text!);
      assert.ok(r2.content.startsWith('[7-12] Bravo'), 'next page must start at a clean index prefix');
    });

    it('makes forward progress when a single line exceeds the limit', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: oneLongLine() }));
      const page1 = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'doc-1', offset: 0, limit: 50 });
      const r1 = JSON.parse(page1.content[0].text!);

      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: oneLongLine() }));
      const page2 = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'doc-1', offset: r1.nextOffset, limit: 50 });
      const r2 = JSON.parse(page2.content[0].text!);
      assert.ok(r2.nextOffset > r1.nextOffset, 'pagination must advance even with no newline in the window');
      assert.ok(r2.content.length > 0, 'page must not be empty');
      assert.equal(r2.hasMore, true);
    });

    it('offset beyond document returns empty content', async () => {
      ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: indexedDoc() }));
      const res = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'doc-1', offset: 999999, limit: 50 });
      const r = JSON.parse(res.content[0].text!);
      assert.equal(r.content, '');
      assert.equal(r.hasMore, false);
      assert.equal(r.nextOffset, r.outputLength);
    });

    it('validation error when documentId missing', async () => {
      const res = await callTool(ctx.client, 'getGoogleDocContentPaginated', {});
      assert.equal(res.isError, true);
    });
  });

  describe('deleteComment', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'deleteComment', { documentId: 'doc-1', commentId: 'c1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('deleted'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'deleteComment', {});
      assert.equal(res.isError, true);
    });
  });

  describe('v1.6.0 docs tab/chip tools', () => {
    it('addDocumentTab happy path', async () => {
      const res = await callTool(ctx.client, 'addDocumentTab', { documentId: 'doc-1', title: 'New Tab' });
      assert.equal(res.isError, false);
    });

    it('renameDocumentTab happy path', async () => {
      const res = await callTool(ctx.client, 'renameDocumentTab', { documentId: 'doc-1', tabId: 'tab-1', title: 'Renamed' });
      assert.equal(res.isError, false);

      // tabId must live INSIDE tabProperties — Google rejects the payload if it's at the request root.
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const lastCall = calls[calls.length - 1];
      const requests = lastCall?.args?.[0]?.requestBody?.requests;
      assert.equal(requests?.length, 1);
      const req = requests[0].updateDocumentTabProperties;
      assert.equal(req.tabProperties.tabId, 'tab-1');
      assert.equal(req.tabProperties.title, 'Renamed');
      assert.equal(req.fields, 'title');
      assert.equal(req.tabId, undefined, 'tabId must not be at the request root');
    });

    it('insertSmartChip happy path', async () => {
      const res = await callTool(ctx.client, 'insertSmartChip', { documentId: 'doc-1', index: 1, chipType: 'person', personEmail: 'user@example.com' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('user@example.com'));

      // Verify the batchUpdate request uses insertPerson (not insertInlineObject)
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const lastCall = calls[calls.length - 1];
      const requests = lastCall?.args?.[0]?.requestBody?.requests;
      assert.ok(requests?.length === 1);
      assert.ok('insertPerson' in requests[0], 'request should use insertPerson');
      assert.equal(requests[0].insertPerson.personProperties.email, 'user@example.com');
    });

    it('insertSmartChip rejects missing email', async () => {
      const res = await callTool(ctx.client, 'insertSmartChip', { documentId: 'doc-1', index: 1, chipType: 'person' });
      assert.equal(res.isError, true);
    });

    it('readSmartChips happy path', async () => {
      const res = await callTool(ctx.client, 'readSmartChips', { documentId: 'doc-1' });
      assert.equal(res.isError, false);
    });
  });

  describe('createFootnote', () => {
    beforeEach(() => {
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => ({
        data: { replies: [{ createFootnote: { footnoteId: 'fn-123' } }] },
      }));
    });

    after(() => {
      ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    });

    it('creates footnote at index without content', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', index: 5 });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('fn-123'));
      assert.ok(res.content[0].text!.includes('at index 5'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 1);
      const req = calls[0].args[0].requestBody.requests[0];
      assert.ok('createFootnote' in req);
      assert.equal(req.createFootnote.location.index, 5);
    });

    it('creates footnote with content (two batchUpdate calls)', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', index: 3, content: 'See reference.' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Content inserted'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 2);

      // Second call should insertText into the footnote segment
      const secondReq = calls[1].args[0].requestBody.requests[0];
      assert.ok('insertText' in secondReq);
      assert.equal(secondReq.insertText.location.segmentId, 'fn-123');
      assert.equal(secondReq.insertText.text, 'See reference.');
    });

    it('creates footnote with endOfSegment', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', endOfSegment: true });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('end of document'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      const req = calls[0].args[0].requestBody.requests[0];
      assert.ok('createFootnote' in req);
      assert.deepEqual(req.createFootnote.endOfSegmentLocation, { segmentId: '' });
    });

    it('rejects when neither index nor endOfSegment provided', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1' });
      assert.equal(res.isError, true);
    });

    it('threads tabId into the footnote-reference location (#114)', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', index: 5, tabId: 'tab-2' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('in tab tab-2'));

      const req = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.equal(req.createFootnote.location.index, 5);
      assert.equal(req.createFootnote.location.tabId, 'tab-2');
    });

    it('threads tabId into endOfSegmentLocation (#114)', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', endOfSegment: true, tabId: 'tab-2' });
      assert.equal(res.isError, false);

      const req = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.createFootnote.endOfSegmentLocation, { segmentId: '', tabId: 'tab-2' });
    });

    it('threads tabId AND segmentId into the footnote-body insert (#114)', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', index: 3, content: 'See ref.', tabId: 'tab-2' });
      assert.equal(res.isError, false);

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 2);
      const secondReq = calls[1].args[0].requestBody.requests[0];
      assert.equal(secondReq.insertText.location.segmentId, 'fn-123');
      assert.equal(secondReq.insertText.location.tabId, 'tab-2');
    });

    it('omits tabId from locations when none is given (#114)', async () => {
      const res = await callTool(ctx.client, 'createFootnote', { documentId: 'doc-1', index: 5 });
      assert.equal(res.isError, false);
      assert.ok(!res.content[0].text!.includes('in tab'));

      const req = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.equal(req.createFootnote.location.tabId, undefined);
    });

    it('returns partial-success error when content insertion fails', async () => {
      let callCount = 0;
      ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => {
        callCount++;
        if (callCount === 1) {
          return { data: { replies: [{ createFootnote: { footnoteId: 'fn-orphan' } }] } };
        }
        throw new Error('Simulated Docs API failure');
      });

      const res = await callTool(ctx.client, 'createFootnote', {
        documentId: 'doc-1', index: 3, content: 'Some text',
      });

      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('fn-orphan'));
      assert.ok(res.content[0].text!.includes('failed to insert content'));
      assert.ok(res.content[0].text!.includes('Simulated Docs API failure'));

      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      assert.equal(calls.length, 2);
    });
  });

  // --- #114: tabId honored in table/format/smartchip editing handlers ---
  describe('tabId scoping for editing tools (#114)', () => {
    // A document content element holding a single 1x1 table at startIndex 5.
    const tableContent = () => [{
      startIndex: 5,
      table: { tableRows: [{ tableCells: [{ startIndex: 10, endIndex: 20, content: [] }] }] },
    }];

    after(() => {
      ctx.mocks.docs.service.documents.get._resetImpl();
      ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    });

    const lastRequests = () => {
      const calls = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
      return calls[calls.length - 1]?.args?.[0]?.requestBody?.requests;
    };

    describe('insertTable', () => {
      it('omits tabId from the location by default', async () => {
        const res = await callTool(ctx.client, 'insertTable', { documentId: 'doc-1', rows: 2, columns: 2, index: 1 });
        assert.equal(res.isError, false);
        assert.ok(!res.content[0].text!.includes('in tab'));
        assert.equal(lastRequests()[0].insertTable.location.tabId, undefined);
      });

      it('threads tabId into the location', async () => {
        const res = await callTool(ctx.client, 'insertTable', { documentId: 'doc-1', rows: 2, columns: 2, index: 1, tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));
        const req = lastRequests()[0].insertTable;
        assert.equal(req.location.tabId, 'tab-2');
        assert.equal(req.location.index, 1);
      });
    });

    describe('insertSmartChip', () => {
      it('omits tabId from the location by default', async () => {
        const res = await callTool(ctx.client, 'insertSmartChip', { documentId: 'doc-1', index: 1, chipType: 'person', personEmail: 'a@b.com' });
        assert.equal(res.isError, false);
        assert.equal(lastRequests()[0].insertPerson.location.tabId, undefined);
      });

      it('threads tabId into the location', async () => {
        const res = await callTool(ctx.client, 'insertSmartChip', { documentId: 'doc-1', index: 1, chipType: 'person', personEmail: 'a@b.com', tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));
        assert.equal(lastRequests()[0].insertPerson.location.tabId, 'tab-2');
      });
    });

    // The no-tabId path must stay byte-for-byte the cheap one: a narrow-field
    // GET (never includeTabsContent) and no tabId leaking into the range. These
    // guard the findTextRange/getParagraphRange tab/no-tab branching (#114).
    const lastGet = () => {
      const calls = ctx.mocks.docs.tracker.getCalls('documents.get');
      return calls[calls.length - 1]?.args?.[0];
    };
    const assertNoTabGets = () => {
      for (const c of ctx.mocks.docs.tracker.getCalls('documents.get')) {
        assert.ok(!c.args?.[0]?.includeTabsContent, 'default path must not use includeTabsContent');
      }
      assert.ok(typeof lastGet()?.fields === 'string', 'default path must use a narrow field mask');
    };

    describe('applyTextStyle', () => {
      it('uses the narrow-field GET and leaks no tabId by default (textToFind)', async () => {
        ctx.mocks.docs.service.documents.get._resetImpl(); // genuine default mock
        const res = await callTool(ctx.client, 'applyTextStyle', { documentId: 'doc-1', textToFind: 'Hello', bold: true });
        assert.equal(res.isError, false);
        assert.ok(!res.content[0].text!.includes('in tab'));
        assertNoTabGets();
        assert.equal(lastRequests()[0].updateTextStyle.range.tabId, undefined);
      });

      it('threads tabId into the range (explicit-index mode)', async () => {
        const res = await callTool(ctx.client, 'applyTextStyle', { documentId: 'doc-1', startIndex: 1, endIndex: 5, bold: true, tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));
        assert.equal(lastRequests()[0].updateTextStyle.range.tabId, 'tab-2');
      });

      it('resolves textToFind within the target tab', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [
            { tabProperties: { tabId: 'tab-1' }, documentTab: { body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Other\n' }, startIndex: 1, endIndex: 7 }] } }] } } },
            { tabProperties: { tabId: 'tab-2' }, documentTab: { body: { content: [{ paragraph: { elements: [{ textRun: { content: 'Find me here\n' }, startIndex: 1, endIndex: 14 }] } }] } } },
          ] },
        }));
        const res = await callTool(ctx.client, 'applyTextStyle', { documentId: 'doc-1', textToFind: 'Find me', bold: true, tabId: 'tab-2' });
        assert.equal(res.isError, false);

        const getCalls = ctx.mocks.docs.tracker.getCalls('documents.get');
        assert.equal(getCalls[getCalls.length - 1]?.args?.[0]?.includeTabsContent, true);
        assert.equal(lastRequests()[0].updateTextStyle.range.tabId, 'tab-2');
        ctx.mocks.docs.service.documents.get._resetImpl();
      });

      it('returns the standard not-found error for an unknown tabId (textToFind)', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [{ tabProperties: { tabId: 'tab-1' }, documentTab: { body: { content: [] } } }] },
        }));
        const before = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length;
        const res = await callTool(ctx.client, 'applyTextStyle', { documentId: 'doc-1', textToFind: 'x', bold: true, tabId: 'missing' });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('Tab with ID "missing" not found'));
        assert.ok(res.content[0].text!.includes('listDocumentTabs'));
        assert.equal(ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length, before);
        ctx.mocks.docs.service.documents.get._resetImpl();
      });

      it('sends baselineOffset in the updateTextStyle payload with a matching field mask', async () => {
        const res = await callTool(ctx.client, 'applyTextStyle', { documentId: 'doc-1', startIndex: 1, endIndex: 5, baselineOffset: 'SUBSCRIPT' });
        assert.equal(res.isError, false);
        const req = lastRequests()[0].updateTextStyle;
        assert.equal(req.textStyle.baselineOffset, 'SUBSCRIPT');
        assert.ok(req.fields.split(',').includes('baselineOffset'));
      });

      it('passes NONE through so existing super/subscript can be reset', async () => {
        const res = await callTool(ctx.client, 'applyTextStyle', { documentId: 'doc-1', startIndex: 1, endIndex: 5, baselineOffset: 'NONE' });
        assert.equal(res.isError, false);
        const req = lastRequests()[0].updateTextStyle;
        assert.equal(req.textStyle.baselineOffset, 'NONE');
        assert.ok(req.fields.split(',').includes('baselineOffset'));
      });

      it('appends baselineOffset to the field mask alongside other style options', async () => {
        // Guards the append path: with a single style option the mask is a
        // one-element string, so an overwriting mask builder would look correct.
        const res = await callTool(ctx.client, 'applyTextStyle', {
          documentId: 'doc-1', startIndex: 1, endIndex: 5, bold: true, baselineOffset: 'SUPERSCRIPT',
        });
        assert.equal(res.isError, false);
        const req = lastRequests()[0].updateTextStyle;
        assert.equal(req.textStyle.bold, true);
        assert.equal(req.textStyle.baselineOffset, 'SUPERSCRIPT');
        assert.deepEqual(req.fields.split(',').sort(), ['baselineOffset', 'bold']);
      });

      it('accepts baselineOffset through the formatGoogleDocText alias', async () => {
        const res = await callTool(ctx.client, 'formatGoogleDocText', {
          documentId: 'doc-1', startIndex: 1, endIndex: 5, baselineOffset: 'SUPERSCRIPT',
        });
        assert.equal(res.isError, false);
        const req = lastRequests()[0].updateTextStyle;
        assert.equal(req.textStyle.baselineOffset, 'SUPERSCRIPT');
        assert.ok(req.fields.split(',').includes('baselineOffset'));
      });
    });

    describe('applyParagraphStyle', () => {
      it('uses narrow-field GETs and leaks no tabId by default (textToFind)', async () => {
        ctx.mocks.docs.service.documents.get._resetImpl(); // genuine default mock
        const res = await callTool(ctx.client, 'applyParagraphStyle', { documentId: 'doc-1', textToFind: 'Hello', alignment: 'CENTER' });
        assert.equal(res.isError, false);
        assert.ok(!res.content[0].text!.includes('in tab'));
        assertNoTabGets();
        assert.equal(lastRequests()[0].updateParagraphStyle.range.tabId, undefined);
      });

      it('resolves indexWithinParagraph within the target tab', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [
            { tabProperties: { tabId: 'tab-2' }, documentTab: { body: { content: [{ startIndex: 1, endIndex: 14, paragraph: { elements: [] } }] } } },
          ] },
        }));
        const res = await callTool(ctx.client, 'applyParagraphStyle', { documentId: 'doc-1', indexWithinParagraph: 2, alignment: 'CENTER', tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));
        assert.equal(lastRequests()[0].updateParagraphStyle.range.tabId, 'tab-2');
        ctx.mocks.docs.service.documents.get._resetImpl();
      });

      it('resolves tab-scoped textToFind with a single GET (#114 follow-up)', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [
            { tabProperties: { tabId: 'tab-2' }, documentTab: { body: { content: [
              { startIndex: 1, endIndex: 14, paragraph: { elements: [{ textRun: { content: 'Find me here\n' }, startIndex: 1, endIndex: 14 }] } },
            ] } } },
          ] },
        }));
        const res = await callTool(ctx.client, 'applyParagraphStyle', { documentId: 'doc-1', textToFind: 'Find me', alignment: 'CENTER', tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));

        // The whole point of the optimization: range + enclosing-paragraph
        // resolution share one includeTabsContent fetch, not two.
        const getCalls = ctx.mocks.docs.tracker.getCalls('documents.get');
        assert.equal(getCalls.length, 1, 'tab-scoped textToFind must resolve from a single GET');
        assert.equal(getCalls[0].args[0].includeTabsContent, true);

        const range = lastRequests()[0].updateParagraphStyle.range;
        assert.equal(range.tabId, 'tab-2');
        assert.equal(range.startIndex, 1);
        assert.equal(range.endIndex, 14);
        ctx.mocks.docs.service.documents.get._resetImpl();
      });

      it('returns the standard not-found error and issues no batchUpdate for an unknown tabId (textToFind)', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [{ tabProperties: { tabId: 'tab-1' }, documentTab: { body: { content: [] } } }] },
        }));
        const before = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length;
        const res = await callTool(ctx.client, 'applyParagraphStyle', { documentId: 'doc-1', textToFind: 'x', alignment: 'CENTER', tabId: 'missing' });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('Tab with ID "missing" not found'));
        assert.ok(res.content[0].text!.includes('listDocumentTabs'));
        assert.equal(ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length, before);
        ctx.mocks.docs.service.documents.get._resetImpl();
      });
    });

    describe('createParagraphBullets', () => {
      it('uses the narrow-field GET and leaks no tabId by default (textToFind)', async () => {
        ctx.mocks.docs.service.documents.get._resetImpl(); // genuine default mock
        const res = await callTool(ctx.client, 'createParagraphBullets', { documentId: 'doc-1', textToFind: 'Hello', bulletPreset: 'BULLET_DISC_CIRCLE_SQUARE' });
        assert.equal(res.isError, false);
        assert.ok(!res.content[0].text!.includes('in tab'));
        assertNoTabGets();
        assert.equal(lastRequests()[0].createParagraphBullets.range.tabId, undefined);
      });

      it('threads tabId into the range (explicit-index mode)', async () => {
        const res = await callTool(ctx.client, 'createParagraphBullets', { documentId: 'doc-1', startIndex: 1, endIndex: 5, bulletPreset: 'BULLET_DISC_CIRCLE_SQUARE', tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));
        assert.equal(lastRequests()[0].createParagraphBullets.range.tabId, 'tab-2');
      });

      it('threads tabId into the range when removing bullets (NONE)', async () => {
        const res = await callTool(ctx.client, 'createParagraphBullets', { documentId: 'doc-1', startIndex: 1, endIndex: 5, bulletPreset: 'NONE', tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.equal(lastRequests()[0].deleteParagraphBullets.range.tabId, 'tab-2');
      });
    });

    describe('editTableCell', () => {
      it('operates on the default body and emits no tabId when none is given', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: { body: { content: tableContent() } } }));
        const res = await callTool(ctx.client, 'editTableCell', { documentId: 'doc-1', tableStartIndex: 5, rowIndex: 0, columnIndex: 0, textContent: 'Hi' });
        assert.equal(res.isError, false);
        assert.ok(!res.content[0].text!.includes('in tab'));
        const reqs = lastRequests();
        for (const r of reqs) {
          const inner = r.deleteContentRange ?? r.insertText;
          assert.equal((inner.range ?? inner.location).tabId, undefined);
        }
        ctx.mocks.docs.service.documents.get._resetImpl();
      });

      it('finds the table inside the target tab and threads tabId into every request', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [{ tabProperties: { tabId: 'tab-2' }, documentTab: { body: { content: tableContent() } } }] },
        }));
        const res = await callTool(ctx.client, 'editTableCell', { documentId: 'doc-1', tableStartIndex: 5, rowIndex: 0, columnIndex: 0, textContent: 'Hi', bold: true, alignment: 'CENTER', tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('in tab tab-2'));

        const getCalls = ctx.mocks.docs.tracker.getCalls('documents.get');
        assert.equal(getCalls[getCalls.length - 1]?.args?.[0]?.includeTabsContent, true);

        const reqs = lastRequests();
        assert.ok(reqs.length >= 3);
        assert.equal(reqs.find((r: any) => r.deleteContentRange).deleteContentRange.range.tabId, 'tab-2');
        assert.equal(reqs.find((r: any) => r.insertText).insertText.location.tabId, 'tab-2');
        assert.equal(reqs.find((r: any) => r.updateTextStyle).updateTextStyle.range.tabId, 'tab-2');
        assert.equal(reqs.find((r: any) => r.updateParagraphStyle).updateParagraphStyle.range.tabId, 'tab-2');
        ctx.mocks.docs.service.documents.get._resetImpl();
      });

      it('returns the standard not-found error and issues no batchUpdate for an unknown tabId', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({
          data: { tabs: [{ tabProperties: { tabId: 'tab-1' }, documentTab: { body: { content: tableContent() } } }] },
        }));
        const before = ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length;
        const res = await callTool(ctx.client, 'editTableCell', { documentId: 'doc-1', tableStartIndex: 5, rowIndex: 0, columnIndex: 0, textContent: 'Hi', tabId: 'missing' });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('Tab with ID "missing" not found'));
        assert.equal(ctx.mocks.docs.tracker.getCalls('documents.batchUpdate').length, before);
        ctx.mocks.docs.service.documents.get._resetImpl();
      });
    });
  });
  // -------------------------------------------------------------------------
  // Style reads
  // -------------------------------------------------------------------------
  describe('style reads', () => {
    const para = (startIndex: number, endIndex: number, text: string, paragraphStyle: any = {}) => ({
      startIndex,
      endIndex,
      paragraph: { paragraphStyle, elements: [{ startIndex, endIndex, textRun: { content: text, textStyle: {} } }] },
    });
    const RULE = { width: { magnitude: 1, unit: 'PT' }, dashStyle: 'SOLID' };

    // A header and a footer each restart indices at 0, separate from the body.
    // A walk that merged them answered describeGoogleDocRange(startIndex=1) with
    // three paragraphs carrying overlapping ranges and nothing to tell the
    // caller which one an index-taking write should target.
    const withHeaderAndFooter = () => ({
      documentId: 'doc-1',
      title: 'Report',
      revisionId: 'rev-9',
      body: { content: [para(1, 11, 'Body copy\n', { namedStyleType: 'NORMAL_TEXT' })] },
      headers: { 'h.1': { content: [para(0, 12, 'Header text\n', { namedStyleType: 'NORMAL_TEXT' })] } },
      footers: { 'f.1': { content: [para(0, 7, 'Footer\n', { namedStyleType: 'NORMAL_TEXT', alignment: 'CENTER' })] } },
    });

    const twoTabs = () => ({
      documentId: 'doc-1',
      title: 'Two Tabs',
      revisionId: 'rev-9',
      tabs: [
        { tabProperties: { tabId: 'tab-1', title: 'Overview' }, documentTab: { body: { content: [para(1, 9, 'First\n', { namedStyleType: 'HEADING_1', borderBottom: RULE })] } } },
        { tabProperties: { tabId: 'tab-2', title: 'Appendix' }, documentTab: { body: { content: [para(1, 9, 'Second\n', { namedStyleType: 'HEADING_2', borderBottom: RULE })] } } },
      ],
    });

    afterEach(() => ctx.mocks.docs.service.documents.get._resetImpl());

    describe('describeGoogleDocRange', () => {
      it('describes the body only, never the header or footer index space', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: withHeaderAndFooter() }));
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 1 });
        assert.equal(res.isError, false);
        const text = res.content[0].text!;
        assert.ok(text.includes('paragraph [1-11]'));
        assert.equal(text.includes('Header text'), false);
        assert.equal(text.includes('Footer'), false);
        // One paragraph reported, not three.
        assert.equal((text.match(/^paragraph \[/gm) ?? []).length, 1);
      });

      it('defaults to the first tab and reports only that tab on a two-tab document', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: twoTabs() }));
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 1, endIndex: 3 });
        assert.equal(res.isError, false);
        const text = res.content[0].text!;
        // Both tabs hold a paragraph at [1-9]; reporting both would print two
        // identical ranges with nothing to choose between them.
        assert.equal((text.match(/^paragraph \[/gm) ?? []).length, 1);
        assert.ok(text.includes('HEADING_1'));
        assert.equal(text.includes('HEADING_2'), false);
      });

      it('probes the named tab when given one', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: twoTabs() }));
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 1, tabId: 'tab-2' });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('HEADING_2'));
      });

      it('reports the revisionId from the same fetch the indices came from', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: withHeaderAndFooter() }));
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 1 });
        assert.ok(res.content[0].text!.startsWith('revisionId: rev-9\n'));
        // One read, not one for the range and another for the lock.
        assert.equal(ctx.mocks.docs.tracker.getCalls('documents.get').length, 1);
      });

      it('says the lock is unavailable rather than printing undefined', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => {
          const d: any = withHeaderAndFooter();
          delete d.revisionId;
          return { data: d };
        });
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 1 });
        assert.ok(res.content[0].text!.includes('revisionId: unavailable (no edit access)'));
      });

      it('refuses mixed targeting instead of silently ignoring endIndex', async () => {
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', textToFind: 'Body', endIndex: 5 });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('not both'));
      });

      it('refuses an endIndex that is not past startIndex', async () => {
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 5, endIndex: 5 });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('endIndex must be greater than startIndex'));
      });

      it('requires a targeting mode', async () => {
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1' });
        assert.equal(res.isError, true);
      });

      it('returns the standard not-found error for an unknown tabId', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: twoTabs() }));
        const res = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'doc-1', startIndex: 1, tabId: 'missing' });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('Tab with ID "missing" not found'));
      });
    });

    describe('getGoogleDocStyleSummary', () => {
      it('summarizes the body only, never the header or footer index space', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: withHeaderAndFooter() }));
        const res = await callTool(ctx.client, 'getGoogleDocStyleSummary', { documentId: 'doc-1' });
        assert.equal(res.isError, false);
        // Body holds one paragraph; the header and footer add two more that
        // would be counted by a merged walk.
        assert.ok(res.content[0].text!.includes('paragraphs: 1 '));
      });

      it('prefixes locations with the tab title across tabs and keeps counts whole-document', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: twoTabs() }));
        const res = await callTool(ctx.client, 'getGoogleDocStyleSummary', { documentId: 'doc-1' });
        const text = res.content[0].text!;
        assert.ok(text.includes('paragraphs: 2 '));
        assert.ok(text.includes('bordered paragraphs: 2 (at Overview:1, Appendix:1)'));
      });

      it('summarizes one tab when given a tabId, without tab prefixes', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: twoTabs() }));
        const res = await callTool(ctx.client, 'getGoogleDocStyleSummary', { documentId: 'doc-1', tabId: 'tab-2' });
        const text = res.content[0].text!;
        assert.ok(text.includes('paragraphs: 1 '));
        assert.ok(text.includes('bordered paragraphs: 1 (at 1)'));
        assert.equal(text.includes('Appendix:'), false);
      });

      it('reports the title and the revisionId', async () => {
        ctx.mocks.docs.service.documents.get._setImpl(async () => ({ data: withHeaderAndFooter() }));
        const res = await callTool(ctx.client, 'getGoogleDocStyleSummary', { documentId: 'doc-1' });
        const text = res.content[0].text!;
        assert.ok(text.includes('Style summary for "Report"'));
        assert.ok(text.includes('revisionId: rev-9'));
      });
    });
  });
});
