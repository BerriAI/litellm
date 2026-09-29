import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach, afterEach } from 'node:test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { PDFDocument } from 'pdf-lib';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';
import { setEnv } from '../helpers/env.js';

describe('Drive tools', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.drive.tracker.reset();
  });

  // --- search ---
  describe('search', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [{ id: 'f1', name: 'Report.pdf', mimeType: 'application/pdf' }] },
      }));
      const res = await callTool(ctx.client, 'search', { query: 'report' });
      assert.ok(res.content[0].text!.includes('Report.pdf'));
      assert.equal(res.isError, false);
    });

    it('validation error on empty args', async () => {
      const res = await callTool(ctx.client, 'search', {});
      assert.equal(res.isError, true);
    });

    it('passes corpora=allDrives to include shared drives', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [{ id: 'f1', name: 'SharedFile.txt', mimeType: 'text/plain' }] },
      }));
      await callTool(ctx.client, 'search', { query: 'shared' });

      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      assert.ok(listCalls.length >= 1);
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.corpora, 'allDrives');
      assert.equal(args.includeItemsFromAllDrives, true);
      assert.equal(args.supportsAllDrives, true);
    });

    it('propagates API error', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => { throw new Error('API quota exceeded'); });
      const res = await callTool(ctx.client, 'search', { query: 'test' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('API quota exceeded'));
      ctx.mocks.drive.service.files.list._resetImpl();
    });

    // --- rawQuery tests (PR #25) ---

    it('rawQuery passes query directly to Drive API', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      await callTool(ctx.client, 'search', {
        query: "mimeType = 'application/pdf'",
        rawQuery: true,
      });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.q, "mimeType = 'application/pdf' and trashed = false");
    });

    it('rawQuery preserves user trashed clause', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      await callTool(ctx.client, 'search', {
        query: "name contains 'test' and trashed = true",
        rawQuery: true,
      });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.q, "name contains 'test' and trashed = true");
    });

    it('rawQuery shows created/modified dates in output', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: {
          files: [{
            id: 'f1', name: 'Report.pdf', mimeType: 'application/pdf',
            createdTime: '2025-06-01T00:00:00Z', modifiedTime: '2025-06-15T00:00:00Z',
          }],
        },
      }));
      const res = await callTool(ctx.client, 'search', {
        query: "mimeType = 'application/pdf'",
        rawQuery: true,
      });
      assert.ok(res.content[0].text!.includes('created: 2025-06-01T00:00:00Z'));
      assert.ok(res.content[0].text!.includes('modified: 2025-06-15T00:00:00Z'));
      ctx.mocks.drive.service.files.list._resetImpl();
    });

    it('default search wraps in fullText contains', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      await callTool(ctx.client, 'search', { query: 'report' });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.q, "fullText contains 'report' and trashed = false");
    });

    // --- orderBy tests (issue #167) ---

    it('defaults to modifiedTime desc so results are newest-first', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      await callTool(ctx.client, 'search', { query: 'report' });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.orderBy, 'modifiedTime desc');
    });

    it('forwards an explicit orderBy', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      await callTool(ctx.client, 'search', { query: 'report', orderBy: 'name' });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.orderBy, 'name');
    });

    it('applies orderBy on the rawQuery path too', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      await callTool(ctx.client, 'search', {
        query: "mimeType = 'application/pdf'",
        rawQuery: true,
        orderBy: 'createdTime desc',
      });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.orderBy, 'createdTime desc');
    });

    it('rejects an unsupported orderBy instead of passing it to Drive', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [] },
      }));
      const res = await callTool(ctx.client, 'search', { query: 'report', orderBy: 'bogus' });
      assert.equal(res.isError, true);
      assert.equal(ctx.mocks.drive.tracker.getCalls('files.list').length, 0);
    });

    it('names the effective ordering in the response', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [{ id: 'f1', name: 'Report.pdf', mimeType: 'application/pdf' }] },
      }));
      const res = await callTool(ctx.client, 'search', { query: 'report' });
      assert.ok(res.content[0].text!.includes('(ordered by modifiedTime desc)'));

      const named = await callTool(ctx.client, 'search', { query: 'report', orderBy: 'name' });
      assert.ok(named.content[0].text!.includes('(ordered by name)'));
      ctx.mocks.drive.service.files.list._resetImpl();
    });

    it('repeats orderBy in the pagination hint so the next page keeps the ordering', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: {
          files: [{ id: 'f1', name: 'Report.pdf', mimeType: 'application/pdf' }],
          nextPageToken: 'tok-2',
        },
      }));
      const res = await callTool(ctx.client, 'search', { query: 'report', orderBy: 'name' });
      const text = res.content[0].text!;
      // orderBy defaults to `modifiedTime desc`, so a follow-up call carrying only
      // the token would re-sort page 2. The token also has to end its own line.
      assert.ok(text.includes('Use pageToken: tok-2\n'));
      assert.ok(text.includes('Pass orderBy: name again'));
      ctx.mocks.drive.service.files.list._resetImpl();
    });

    // --- Folder path resolution tests (PR #30) ---

    it('resolves folder paths in search results', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: {
          files: [{ id: 'f1', name: 'Doc.txt', mimeType: 'text/plain', parents: ['folder-2'] }],
        },
      }));
      ctx.mocks.drive.service.files.get._setImpl(async (params: any) => {
        if (params.fileId === 'folder-2') {
          return { data: { name: 'SubFolder', parents: ['folder-1'] } };
        }
        if (params.fileId === 'folder-1') {
          return { data: { name: 'RootFolder', parents: [] } };
        }
        return { data: { name: 'Unknown' } };
      });
      const res = await callTool(ctx.client, 'search', { query: 'doc' });
      assert.ok(res.content[0].text!.includes('path: RootFolder/SubFolder'));
      ctx.mocks.drive.service.files.list._resetImpl();
      ctx.mocks.drive.service.files.get._resetImpl();
    });

    it('caches folder paths (no duplicate API calls)', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: {
          files: [
            { id: 'f1', name: 'A.txt', mimeType: 'text/plain', parents: ['shared-parent'] },
            { id: 'f2', name: 'B.txt', mimeType: 'text/plain', parents: ['shared-parent'] },
          ],
        },
      }));
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { name: 'SharedFolder', parents: [] },
      }));
      await callTool(ctx.client, 'search', { query: 'test' });
      const getCalls = ctx.mocks.drive.tracker.getCalls('files.get');
      const parentLookups = getCalls.filter((c: any) => c.args[0].fileId === 'shared-parent');
      assert.equal(parentLookups.length, 1, 'Should only call files.get once for the same parent');
      ctx.mocks.drive.service.files.list._resetImpl();
      ctx.mocks.drive.service.files.get._resetImpl();
    });

    it('falls back to folder ID on resolution error', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: {
          files: [{ id: 'f1', name: 'Doc.txt', mimeType: 'text/plain', parents: ['bad-folder'] }],
        },
      }));
      ctx.mocks.drive.service.files.get._setImpl(async () => {
        throw new Error('Not found');
      });
      const res = await callTool(ctx.client, 'search', { query: 'doc' });
      assert.ok(res.content[0].text!.includes('path: bad-folder'));
      ctx.mocks.drive.service.files.list._resetImpl();
      ctx.mocks.drive.service.files.get._resetImpl();
    });
  });

  // --- createTextFile ---
  describe('createTextFile', () => {
    it('happy path', async () => {
      // checkFileExists returns no match
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      ctx.mocks.drive.service.files.create._setImpl(async () => ({
        data: { id: 'new-file', name: 'notes.txt' },
      }));
      const res = await callTool(ctx.client, 'createTextFile', { name: 'notes.txt', content: 'hello' });
      assert.ok(res.content[0].text!.includes('notes.txt'));
      assert.equal(res.isError, false);
    });

    it('validation error on missing required fields', async () => {
      const res = await callTool(ctx.client, 'createTextFile', {});
      assert.equal(res.isError, true);
    });
  });

  // --- updateTextFile ---
  describe('updateTextFile', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { mimeType: 'text/plain', name: 'notes.txt', parents: ['root'] },
      }));
      ctx.mocks.drive.service.files.update._setImpl(async () => ({
        data: { id: 'file-1', name: 'notes.txt' },
      }));
      const res = await callTool(ctx.client, 'updateTextFile', { fileId: 'file-1', content: 'updated' });
      assert.equal(res.isError, false);
    });

    it('accepts any text/* MIME type (e.g. text/csv)', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { mimeType: 'text/csv', name: 'data.csv', parents: ['root'] },
      }));
      const res = await callTool(ctx.client, 'updateTextFile', { fileId: 'file-1', content: 'a,b\n1,2\n' });
      assert.equal(res.isError, false);
    });

    it('rejects a non-text MIME type', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { mimeType: 'application/pdf', name: 'doc.pdf', parents: ['root'] },
      }));
      const res = await callTool(ctx.client, 'updateTextFile', { fileId: 'file-1', content: 'x' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('not a text file'));
    });

    it('validation error on missing required fields', async () => {
      const res = await callTool(ctx.client, 'updateTextFile', {});
      assert.equal(res.isError, true);
    });
  });

  // --- readTextFile ---
  describe('readTextFile', () => {
    // files.get is a single stub reached both for the metadata read and the
    // alt:'media' content download, so branch on params.alt.
    function stubTextFile(content: string, mimeType = 'text/plain', name = 'notes.txt') {
      ctx.mocks.drive.service.files.get._setImpl(async (p: any) =>
        p?.alt === 'media'
          ? { data: Readable.from(Buffer.from(content, 'utf-8')) }
          : { data: { id: 'file-1', name, mimeType, parents: ['root'] } });
    }

    afterEach(() => {
      ctx.mocks.drive.service.files.get._resetImpl();
    });

    it('happy path returns header + content with a code-point Length', async () => {
      stubTextFile('Hello World');
      const res = await callTool(ctx.client, 'readTextFile', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Length: 11 characters'));
      assert.ok(text.includes('Truncated: no'));
      assert.ok(text.endsWith('Hello World'));
    });

    it('reads content delivered as a web ReadableStream (gaxios 7 shape)', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async (p: any) =>
        p?.alt === 'media'
          ? {
              data: new ReadableStream<Uint8Array>({
                start(controller) {
                  controller.enqueue(new TextEncoder().encode('Hello Web'));
                  controller.close();
                },
              }),
            }
          : { data: { id: 'file-1', name: 'notes.txt', mimeType: 'text/plain', parents: ['root'] } });
      const res = await callTool(ctx.client, 'readTextFile', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.endsWith('Hello Web'));
    });

    it('reports code-point Length for astral characters (emoji counts as 1)', async () => {
      stubTextFile('😀😀😀');
      const res = await callTool(ctx.client, 'readTextFile', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      // 3 emoji = 3 code points (would be 6 if counting UTF-16 units).
      assert.ok(res.content[0].text!.includes('Length: 3 characters'));
    });

    it('truncates on a code-point boundary without garbling emoji', async () => {
      // 'ab😀cd': truncate to 3 code points → 'ab😀', never a lone surrogate.
      stubTextFile('ab😀cd');
      const res = await callTool(ctx.client, 'readTextFile', { fileId: 'file-1', maxLength: 3 });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.ok(text.includes('Length: 5 characters'));
      assert.ok(text.includes('Truncated: yes'));
      assert.ok(text.endsWith('ab😀'));
      assert.ok(!text.split('---\n')[1].includes('�'));
    });

    it('rejects a non-text MIME type', async () => {
      stubTextFile('irrelevant', 'application/vnd.google-apps.document', 'My Doc');
      const res = await callTool(ctx.client, 'readTextFile', { fileId: 'file-1' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('readGoogleDoc'));
    });

    it('validation error on missing fileId', async () => {
      const res = await callTool(ctx.client, 'readTextFile', {});
      assert.equal(res.isError, true);
    });
  });

  // --- createFolder ---
  describe('createFolder', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.create._setImpl(async () => ({
        data: { id: 'folder-1', name: 'New Folder' },
      }));
      const res = await callTool(ctx.client, 'createFolder', { name: 'New Folder' });
      assert.ok(res.content[0].text!.includes('New Folder'));
      assert.equal(res.isError, false);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'createFolder', {});
      assert.equal(res.isError, true);
    });
  });

  // --- listFolder ---
  describe('listFolder', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [{ id: 'f1', name: 'File1', mimeType: 'text/plain' }] },
      }));
      const res = await callTool(ctx.client, 'listFolder', {});
      assert.equal(res.isError, false);
    });

    it('lists shared-drive folder children via the two flags, without corpora=allDrives', async () => {
      // Parent-scoped ('<id>' in parents): the two flags surface Shared Drive
      // children, and corpora=allDrives is deliberately omitted so the listing
      // can never come back as an incompleteSearch partial result (#137).
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      await callTool(ctx.client, 'listFolder', { folderId: 'shared-folder-id' });
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      const args = listCalls[listCalls.length - 1].args[0];
      assert.equal(args.corpora, undefined);
      assert.equal(args.includeItemsFromAllDrives, true);
      assert.equal(args.supportsAllDrives, true);
      ctx.mocks.drive.service.files.list._resetImpl();
    });
  });

  // --- listSharedDrives ---
  describe('listSharedDrives', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.drives.list._setImpl(async () => ({
        data: { drives: [{ id: 'd1', name: 'Engineering Shared Drive', hidden: false }] },
      }));
      const res = await callTool(ctx.client, 'listSharedDrives', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Engineering Shared Drive'));
      assert.ok(res.content[0].text!.includes('d1'));
    });

    it('empty result', async () => {
      ctx.mocks.drive.service.drives.list._setImpl(async () => ({
        data: { drives: [] },
      }));
      const res = await callTool(ctx.client, 'listSharedDrives', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('No shared drives found'));
    });

    it('pagination token forwarded', async () => {
      ctx.mocks.drive.service.drives.list._setImpl(async () => ({
        data: { drives: [{ id: 'd1', name: 'Drive A', hidden: false }], nextPageToken: 'tok2' },
      }));
      const res = await callTool(ctx.client, 'listSharedDrives', { pageSize: 1 });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('tok2'));
    });
  });

  // --- deleteItem ---
  describe('deleteItem', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'deleteItem', { itemId: 'item-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('moved to trash'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'deleteItem', {});
      assert.equal(res.isError, true);
    });
  });

  // --- renameItem ---
  describe('renameItem', () => {
    it('happy path', async () => {
      // files.get returns a non-text mimeType so validateTextFileExtension is skipped
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { name: 'OldName', mimeType: 'application/vnd.google-apps.folder' },
      }));
      ctx.mocks.drive.service.files.update._setImpl(async () => ({
        data: { id: 'item-1', name: 'Renamed' },
      }));
      const res = await callTool(ctx.client, 'renameItem', { itemId: 'item-1', newName: 'Renamed' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Renamed'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'renameItem', {});
      assert.equal(res.isError, true);
    });
  });

  // --- listPermissions ---
  describe('listPermissions', () => {
    it('happy path includes inherited/direct marker', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({
        data: {
          permissions: [
            {
              id: 'perm-1',
              type: 'user',
              emailAddress: 'user@example.com',
              role: 'reader',
              permissionDetails: [{ inherited: true, inheritedFrom: 'folder-123', permissionType: 'file' }],
            },
            {
              id: 'perm-2',
              type: 'user',
              emailAddress: 'owner@example.com',
              role: 'owner',
              permissionDetails: [{ inherited: false, permissionType: 'file' }],
            },
          ],
        },
      }));

      const res = await callTool(ctx.client, 'listPermissions', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('[inherited from folder-123]'));
      assert.ok(res.content[0].text!.includes('[direct]'));

      const listCalls = ctx.mocks.drive.tracker.getCalls('permissions.list');
      assert.ok(listCalls.length >= 1);
      assert.ok(listCalls[0].args[0].fields.includes('permissionDetails(inherited,inheritedFrom,permissionType)'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'listPermissions', {});
      assert.equal(res.isError, true);
    });
  });

  // --- addPermission / shareFile ---
  describe('permission mutations', () => {
    it('addPermission happy path', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'reader', type: 'user',
      });
      assert.equal(res.isError, false);
    });

    it('addPermission type "anyone" needs no emailAddress and omits it from the request', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'anyone', role: 'reader',
      });
      assert.equal(res.isError, false);
      const createCalls = ctx.mocks.drive.tracker.getCalls('permissions.create');
      assert.ok(createCalls.length >= 1);
      const body = createCalls[0].args[0].requestBody;
      assert.equal(body.type, 'anyone');
      assert.equal('emailAddress' in body, false);
      assert.equal('domain' in body, false);
    });

    it('addPermission type "domain" sends domain, not emailAddress', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'domain', domain: 'example.com', role: 'reader',
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('permissions.create')[0].args[0].requestBody;
      assert.equal(body.type, 'domain');
      assert.equal(body.domain, 'example.com');
      assert.equal('emailAddress' in body, false);
    });

    it('addPermission verifies the applied role: upsert mismatch is corrected once and reported honestly', async () => {
      // Drive's upsert semantics: the principal already holds writer, so a
      // reader request comes back role=writer.
      ctx.mocks.drive.service.permissions.create._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer', emailAddress: 'user@example.com', type: 'user' },
      }));
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-9', role: 'reader' },
      }));
      try {
        const res = await callTool(ctx.client, 'addPermission', {
          fileId: 'file-1', emailAddress: 'user@example.com', role: 'reader', type: 'user',
        });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes("initially applied 'writer'"), res.content[0].text);
        assert.ok(res.content[0].text!.includes("changed to 'reader'"), res.content[0].text);
        const updateCalls = ctx.mocks.drive.tracker.getCalls('permissions.update');
        assert.equal(updateCalls[updateCalls.length - 1].args[0].requestBody.role, 'reader');
      } finally {
        ctx.mocks.drive.service.permissions.create._resetImpl();
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('addPermission reports a surviving mismatch as an error, never as success', async () => {
      ctx.mocks.drive.service.permissions.create._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer', emailAddress: 'user@example.com', type: 'user' },
      }));
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer' },
      }));
      try {
        const res = await callTool(ctx.client, 'addPermission', {
          fileId: 'file-1', emailAddress: 'user@example.com', role: 'reader', type: 'user',
        });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('MISMATCH'));
      } finally {
        ctx.mocks.drive.service.permissions.create._resetImpl();
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('updatePermission verifies the response role', async () => {
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-1', role: 'writer' },
      }));
      try {
        const res = await callTool(ctx.client, 'updatePermission', {
          fileId: 'file-1', permissionId: 'perm-1', role: 'commenter',
        });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('MISMATCH'));
      } finally {
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('addPermission reports a failed corrective update as an error that names the failure', async () => {
      ctx.mocks.drive.service.permissions.create._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer', emailAddress: 'user@example.com', type: 'user' },
      }));
      ctx.mocks.drive.service.permissions.update._setImpl(async () => {
        throw new Error('insufficientFilePermissions');
      });
      try {
        const res = await callTool(ctx.client, 'addPermission', {
          fileId: 'file-1', emailAddress: 'user@example.com', role: 'reader', type: 'user',
        });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('MISMATCH'), res.content[0].text);
        assert.ok(res.content[0].text!.includes('corrective update failed: insufficientFilePermissions'), res.content[0].text);
      } finally {
        ctx.mocks.drive.service.permissions.create._resetImpl();
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('addPermission treats a response without a role as a mismatch, like updatePermission does', async () => {
      ctx.mocks.drive.service.permissions.create._setImpl(async () => ({
        data: { id: 'perm-9', emailAddress: 'user@example.com', type: 'user' },
      }));
      try {
        const res = await callTool(ctx.client, 'addPermission', {
          fileId: 'file-1', emailAddress: 'user@example.com', role: 'reader', type: 'user',
        });
        // The default update mock echoes the requested role, so the correction succeeds.
        assert.equal(res.isError, false, res.content[0].text);
        assert.ok(res.content[0].text!.includes('initially applied no role'), res.content[0].text);
        const updateCalls = ctx.mocks.drive.tracker.getCalls('permissions.update');
        assert.equal(updateCalls.length, 1);
        assert.equal(updateCalls[0].args[0].requestBody.role, 'reader');
      } finally {
        ctx.mocks.drive.service.permissions.create._resetImpl();
      }
    });

    it('addPermission type "user" without emailAddress is rejected before any API call', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'user', role: 'reader',
      });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.toLowerCase().includes('emailaddress'));
      assert.equal(ctx.mocks.drive.tracker.getCalls('permissions.create').length, 0);
    });

    it('addPermission type "domain" without domain is rejected before any API call', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'domain', role: 'reader',
      });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.toLowerCase().includes('domain'));
      assert.equal(ctx.mocks.drive.tracker.getCalls('permissions.create').length, 0);
    });

    it('addPermission type "anyone" forwards allowFileDiscovery:true', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'anyone', role: 'reader', allowFileDiscovery: true,
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('permissions.create')[0].args[0].requestBody;
      assert.equal(body.allowFileDiscovery, true);
    });

    it('addPermission type "anyone" forwards allowFileDiscovery:false (not dropped as falsy)', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'anyone', role: 'reader', allowFileDiscovery: false,
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('permissions.create')[0].args[0].requestBody;
      assert.equal(body.allowFileDiscovery, false);
    });

    it('addPermission type "anyone" omits allowFileDiscovery when not provided', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'anyone', role: 'reader',
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('permissions.create')[0].args[0].requestBody;
      assert.equal('allowFileDiscovery' in body, false);
    });

    it('addPermission ignores allowFileDiscovery for type "user"', async () => {
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', type: 'user', emailAddress: 'user@example.com', role: 'reader', allowFileDiscovery: true,
      });
      assert.equal(res.isError, false);
      const body = ctx.mocks.drive.tracker.getCalls('permissions.create')[0].args[0].requestBody;
      assert.equal('allowFileDiscovery' in body, false);
    });

    it('shareFile happy path', async () => {
      const res = await callTool(ctx.client, 'shareFile', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'writer',
      });
      assert.equal(res.isError, false);
    });

    it('shareFile updates existing user permission (idempotent)', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({
        data: { permissions: [{ id: 'perm-1', type: 'user', emailAddress: 'user@example.com', role: 'reader' }] },
      }));

      const res = await callTool(ctx.client, 'shareFile', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'writer',
      });

      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Updated existing permission'));

      const createCalls = ctx.mocks.drive.tracker.getCalls('permissions.create');
      const updateCalls = ctx.mocks.drive.tracker.getCalls('permissions.update');
      assert.equal(createCalls.length, 0);
      assert.ok(updateCalls.length >= 1);
    });

    it('updatePermission happy path', async () => {
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-1', role: 'commenter', emailAddress: 'user@example.com', type: 'user' },
      }));
      try {
        const res = await callTool(ctx.client, 'updatePermission', {
          fileId: 'file-1', permissionId: 'perm-1', role: 'commenter',
        });
        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('perm-1 => commenter'), res.content[0].text);
      } finally {
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('removePermission happy path', async () => {
      const res = await callTool(ctx.client, 'removePermission', {
        fileId: 'file-1', permissionId: 'perm-1',
      });
      assert.equal(res.isError, false);
    });

    it('removePermission by email lookup', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({
        data: { permissions: [{ id: 'perm-1', type: 'user', emailAddress: 'user@example.com' }] },
      }));
      const res = await callTool(ctx.client, 'removePermission', {
        fileId: 'file-1', emailAddress: 'user@example.com',
      });
      assert.equal(res.isError, false);
    });

    it('shareFile no-op when role already matches', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({
        data: { permissions: [{ id: 'perm-1', type: 'user', emailAddress: 'user@example.com', role: 'writer' }] },
      }));

      const res = await callTool(ctx.client, 'shareFile', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'writer',
      });

      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('No changes needed'));
    });

    it('shareFile reports a mismatched role on the update path as an error', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({
        data: { permissions: [{ id: 'perm-1', type: 'user', emailAddress: 'user@example.com', role: 'reader' }] },
      }));
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-1', role: 'reader', emailAddress: 'user@example.com' },
      }));
      try {
        const res = await callTool(ctx.client, 'shareFile', {
          fileId: 'file-1', emailAddress: 'user@example.com', role: 'writer',
        });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('MISMATCH'), res.content[0].text);
        assert.ok(res.content[0].text!.includes("requested 'writer' but Drive reports 'reader'"), res.content[0].text);
      } finally {
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('shareFile corrects an upsert mismatch on the create path once and reports it', async () => {
      // No user-type match in the pre-check (e.g. the address is a group), so
      // shareFile creates — and Drive upserts onto the existing grant.
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({ data: { permissions: [] } }));
      ctx.mocks.drive.service.permissions.create._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer', emailAddress: 'team@example.com', type: 'group' },
      }));
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-9', role: 'reader' },
      }));
      try {
        const res = await callTool(ctx.client, 'shareFile', {
          fileId: 'file-1', emailAddress: 'team@example.com', role: 'reader',
        });
        assert.equal(res.isError, false, res.content[0].text);
        assert.ok(res.content[0].text!.includes("initially applied 'writer'"), res.content[0].text);
        assert.ok(res.content[0].text!.includes("changed to 'reader'"), res.content[0].text);
        const updateCalls = ctx.mocks.drive.tracker.getCalls('permissions.update');
        assert.equal(updateCalls.length, 1);
        assert.equal(updateCalls[0].args[0].permissionId, 'perm-9');
        assert.equal(updateCalls[0].args[0].requestBody.role, 'reader');
      } finally {
        ctx.mocks.drive.service.permissions.create._resetImpl();
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('shareFile reports a surviving mismatch on the create path as an error', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({ data: { permissions: [] } }));
      ctx.mocks.drive.service.permissions.create._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer', emailAddress: 'team@example.com', type: 'group' },
      }));
      ctx.mocks.drive.service.permissions.update._setImpl(async () => ({
        data: { id: 'perm-9', role: 'writer' },
      }));
      try {
        const res = await callTool(ctx.client, 'shareFile', {
          fileId: 'file-1', emailAddress: 'team@example.com', role: 'reader',
        });
        assert.equal(res.isError, true);
        assert.ok(res.content[0].text!.includes('MISMATCH'), res.content[0].text);
        assert.ok(res.content[0].text!.includes('even after a corrective update'), res.content[0].text);
      } finally {
        ctx.mocks.drive.service.permissions.create._resetImpl();
        ctx.mocks.drive.service.permissions.update._resetImpl();
      }
    });

    it('addPermission forwards emailMessage to permissions.create', async () => {
      ctx.mocks.drive.service.permissions.list._resetImpl();
      const res = await callTool(ctx.client, 'addPermission', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'reader', type: 'user',
        sendNotificationEmail: true, emailMessage: 'Welcome!',
      });
      assert.equal(res.isError, false);
      const createCalls = ctx.mocks.drive.tracker.getCalls('permissions.create');
      assert.ok(createCalls.length >= 1);
      assert.equal(createCalls[0].args[0].emailMessage, 'Welcome!');
    });

    it('shareFile forwards emailMessage to permissions.create', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({ data: { permissions: [] } }));
      const res = await callTool(ctx.client, 'shareFile', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'writer',
        emailMessage: 'Sharing this with you',
      });
      assert.equal(res.isError, false);
      const createCalls = ctx.mocks.drive.tracker.getCalls('permissions.create');
      assert.ok(createCalls.length >= 1);
      assert.equal(createCalls[0].args[0].emailMessage, 'Sharing this with you');
    });

    it('shareFile omits emailMessage when not provided', async () => {
      ctx.mocks.drive.service.permissions.list._setImpl(async () => ({ data: { permissions: [] } }));
      const res = await callTool(ctx.client, 'shareFile', {
        fileId: 'file-1', emailAddress: 'user@example.com', role: 'writer',
      });
      assert.equal(res.isError, false);
      const createCalls = ctx.mocks.drive.tracker.getCalls('permissions.create');
      assert.ok(createCalls.length >= 1);
      assert.equal('emailMessage' in createCalls[0].args[0], false);
    });
  });

  // --- auth diagnostics ---
  describe('auth diagnostics', () => {
    it('authGetStatus returns status payload', async () => {
      const res = await callTool(ctx.client, 'authGetStatus', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Auth status'));
    });

    it('authGetStatus reports the effective identity and oauth mode', async () => {
      const saved = setEnv({ GOOGLE_APPLICATION_CREDENTIALS: undefined, GOOGLE_DRIVE_MCP_ACCESS_TOKEN: undefined });
      ctx.mocks.drive.service.about.get._setImpl(async () => ({
        data: { user: { displayName: 'Ada L', emailAddress: 'ada@example.com' }, storageQuota: { limit: '100', usage: '1' } },
      }));
      try {
        const res = await callTool(ctx.client, 'authGetStatus', {});
        assert.equal(res.isError, false);
        const text = res.content[0].text!;
        assert.ok(text.includes('ada@example.com'), 'effective identity email is surfaced');
        assert.ok(/"authMode":\s*"oauth"/.test(text), 'active auth mode is oauth');
      } finally {
        ctx.mocks.drive.service.about.get._resetImpl();
        saved.restore();
      }
    });

    it('authGetStatus warns that tokens.json is IGNORED under GOOGLE_APPLICATION_CREDENTIALS', async () => {
      const dir = await mkdtemp(join(tmpdir(), 'gdmcp-tok-'));
      const tokenFile = join(dir, 'tokens.json');
      await writeFile(tokenFile, '{}');
      const saved = setEnv({
        GOOGLE_APPLICATION_CREDENTIALS: '/tmp/fake-service-account.json',
        GOOGLE_DRIVE_MCP_TOKEN_PATH: tokenFile,
      });
      try {
        const res = await callTool(ctx.client, 'authGetStatus', {});
        const text = res.content[0].text!;
        assert.ok(/"authMode":\s*"service_account"/.test(text), 'active auth mode is service_account');
        assert.ok(text.includes('IGNORED'), 'warns that the local token file is ignored');
        assert.ok(text.includes('GOOGLE_APPLICATION_CREDENTIALS'), 'names the overriding env var');
      } finally {
        saved.restore();
        await rm(dir, { recursive: true, force: true });
      }
    });

    it('authGetStatus reports needs_reauth when an oauth setup has no local token', async () => {
      // oauth mode with no token file is the actionable re-auth case; it must
      // outrank identity_error even though the live about.get fails for lack of
      // credentials (making about.get throw is exactly what the old ladder
      // mislabeled as identity_error). A non-existent token path makes
      // tokenFileExists false.
      const saved = setEnv({
        GOOGLE_APPLICATION_CREDENTIALS: undefined,
        GOOGLE_DRIVE_MCP_ACCESS_TOKEN: undefined,
        GOOGLE_DRIVE_MCP_TOKEN_PATH: '/tmp/gdmcp-nonexistent-token-path/tokens.json',
      });
      ctx.mocks.drive.service.about.get._setImpl(async () => { throw new Error('No refresh token is set.'); });
      try {
        const res = await callTool(ctx.client, 'authGetStatus', {});
        const text = res.content[0].text!;
        assert.ok(/"authMode":\s*"oauth"/.test(text), 'active auth mode is oauth');
        assert.ok(/Auth status \(needs_reauth\)/.test(text), 'status is needs_reauth, not identity_error');
      } finally {
        ctx.mocks.drive.service.about.get._resetImpl();
        saved.restore();
      }
    });

    it('authGetStatus surfaces an identity-resolution failure as identity_error', async () => {
      // Use service_account mode so the oauth-only needs_reauth branch is
      // skipped and a failing about.get genuinely surfaces as identity_error.
      const saved = setEnv({
        GOOGLE_APPLICATION_CREDENTIALS: '/tmp/fake-service-account.json',
        GOOGLE_DRIVE_MCP_ACCESS_TOKEN: undefined,
        GOOGLE_DRIVE_MCP_TOKEN_PATH: '/tmp/gdmcp-nonexistent-token-path/tokens.json',
      });
      ctx.mocks.drive.service.about.get._setImpl(async () => { throw new Error('Insufficient Permission'); });
      try {
        const res = await callTool(ctx.client, 'authGetStatus', {});
        const text = res.content[0].text!;
        assert.ok(/Auth status \(identity_error\)/.test(text), 'status is identity_error');
        assert.ok(text.includes('Insufficient Permission'), 'includes the underlying error message');
      } finally {
        ctx.mocks.drive.service.about.get._resetImpl();
        saved.restore();
      }
    });

    it('authListScopes returns scopes payload', async () => {
      const res = await callTool(ctx.client, 'authListScopes', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('requestedScopes'));
    });

    it('authTestFileAccess works without fileId', async () => {
      const res = await callTool(ctx.client, 'authTestFileAccess', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Auth access check OK'));
    });

    it('authTestFileAccess with specific fileId', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'file-1', name: 'TestDoc', mimeType: 'application/vnd.google-apps.document' },
      }));
      const res = await callTool(ctx.client, 'authTestFileAccess', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('"mode":"file"') || res.content[0].text!.includes('"mode": "file"'));
    });

  });

  // --- revisions ---
  describe('revisions', () => {
    it('getRevisions happy path', async () => {
      const res = await callTool(ctx.client, 'getRevisions', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Revisions for file file-1'));
    });

    it('restoreRevision requires confirmation', async () => {
      const res = await callTool(ctx.client, 'restoreRevision', { fileId: 'file-1', revisionId: '1' });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('confirm=true'));
    });

    it('restoreRevision happy path (workspace file) includes formatting warning', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({ data: { name: 'Doc', mimeType: 'application/vnd.google-apps.document' } }));
      ctx.mocks.drive.service.revisions.get._setImpl(async () => ({
        data: {
          id: '1',
          exportLinks: {
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'https://example.com/export.docx',
            'application/pdf': 'https://example.com/export.pdf',
          },
        },
      }));
      ctx.mocks.drive.service.files.update._setImpl(async () => ({ data: { id: 'file-1', name: 'Doc' } }));

      const res = await callTool(ctx.client, 'restoreRevision', { fileId: 'file-1', revisionId: '1', confirm: true });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Restored file file-1'));
      assert.ok(res.content[0].text!.includes('restored via export/import'), 'Should include workspace formatting warning');

      // Should use revisions.get for exportLinks, not files.export
      const revGetCalls = ctx.mocks.drive.tracker.getCalls('revisions.get');
      assert.ok(revGetCalls.length >= 1, 'Should use revisions.get to fetch exportLinks');
    });

    it('restoreRevision happy path (binary file) without workspace warning', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({ data: { name: 'photo.jpg', mimeType: 'image/jpeg' } }));
      ctx.mocks.drive.service.revisions.get._setImpl(async () => ({ data: Buffer.from('binary-content') }));
      ctx.mocks.drive.service.files.update._setImpl(async () => ({ data: { id: 'file-1', name: 'photo.jpg' } }));

      const res = await callTool(ctx.client, 'restoreRevision', { fileId: 'file-1', revisionId: '2', confirm: true });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Restored file file-1'));
      assert.ok(!res.content[0].text!.includes('export/import'), 'Should NOT include workspace warning for binary files');

      const revGetCalls = ctx.mocks.drive.tracker.getCalls('revisions.get');
      assert.ok(revGetCalls.length >= 1, 'Should use revisions.get for binary download');
    });
  });

  // --- moveItem ---
  describe('moveItem', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'item-1', name: 'File', parents: ['old-parent'] },
      }));
      ctx.mocks.drive.service.files.update._setImpl(async () => ({
        data: { id: 'item-1', name: 'File' },
      }));
      const res = await callTool(ctx.client, 'moveItem', { itemId: 'item-1', destinationFolderId: 'new-parent' });
      assert.equal(res.isError, false);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'moveItem', {});
      assert.equal(res.isError, true);
    });
  });

  // --- createShortcut ---
  describe('createShortcut', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'target-1', name: 'Report.pdf', mimeType: 'application/pdf' },
      }));
      ctx.mocks.drive.service.files.create._setImpl(async () => ({
        data: { id: 'shortcut-1', name: 'Report.pdf', webViewLink: 'https://drive.google.com/shortcut-1' },
      }));
      const res = await callTool(ctx.client, 'createShortcut', { targetFileId: 'target-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Shortcut created successfully'));
      assert.ok(res.content[0].text!.includes('Report.pdf'));

      const createCalls = ctx.mocks.drive.tracker.getCalls('files.create');
      assert.ok(createCalls.length >= 1);
      const createArgs = createCalls[createCalls.length - 1].args[0];
      assert.equal(createArgs.requestBody.mimeType, 'application/vnd.google-apps.shortcut');
      assert.equal(createArgs.requestBody.shortcutDetails.targetId, 'target-1');
      assert.equal(createArgs.supportsAllDrives, true);
    });

    it('uses custom shortcutName', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'target-1', name: 'Report.pdf', mimeType: 'application/pdf' },
      }));
      ctx.mocks.drive.service.files.create._setImpl(async () => ({
        data: { id: 'shortcut-1', name: 'My Link', webViewLink: 'https://drive.google.com/shortcut-1' },
      }));
      const res = await callTool(ctx.client, 'createShortcut', { targetFileId: 'target-1', shortcutName: 'My Link' });
      assert.ok(res.content[0].text!.includes('My Link'));

      const createCalls = ctx.mocks.drive.tracker.getCalls('files.create');
      const createArgs = createCalls[createCalls.length - 1].args[0];
      assert.equal(createArgs.requestBody.name, 'My Link');
    });

    it('validation error on missing targetFileId', async () => {
      const res = await callTool(ctx.client, 'createShortcut', {});
      assert.equal(res.isError, true);
    });
  });

  // --- lockFile ---
  describe('lockFile', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'file-1', name: 'Report.docx', contentRestrictions: [] },
      }));
      const res = await callTool(ctx.client, 'lockFile', { fileId: 'file-1', reason: 'Final version' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('File locked successfully'));
      assert.ok(res.content[0].text!.includes('Final version'));

      const updateCalls = ctx.mocks.drive.tracker.getCalls('files.update');
      assert.ok(updateCalls.length >= 1);
      const updateArgs = updateCalls[updateCalls.length - 1].args[0];
      assert.deepEqual(updateArgs.requestBody.contentRestrictions, [{ readOnly: true, reason: 'Final version', ownerRestricted: false }]);
      assert.equal(updateArgs.supportsAllDrives, true);
    });

    it('returns message when file is already locked', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'file-1', name: 'Report.docx', contentRestrictions: [{ readOnly: true, reason: 'Locked' }] },
      }));
      const res = await callTool(ctx.client, 'lockFile', { fileId: 'file-1' });
      assert.ok(res.content[0].text!.includes('already locked'));
    });

    it('uses default reason when none provided', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'file-1', name: 'Report.docx', contentRestrictions: [] },
      }));
      const res = await callTool(ctx.client, 'lockFile', { fileId: 'file-1' });
      assert.ok(res.content[0].text!.includes('Locked via MCP'));
    });

    it('validation error on missing fileId', async () => {
      const res = await callTool(ctx.client, 'lockFile', {});
      assert.equal(res.isError, true);
    });
  });

  // --- unlockFile ---
  describe('unlockFile', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'file-1', name: 'Report.docx', contentRestrictions: [{ readOnly: true, reason: 'Locked' }] },
      }));
      const res = await callTool(ctx.client, 'unlockFile', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('File unlocked successfully'));

      const updateCalls = ctx.mocks.drive.tracker.getCalls('files.update');
      assert.ok(updateCalls.length >= 1);
      const updateArgs = updateCalls[updateCalls.length - 1].args[0];
      assert.deepEqual(updateArgs.requestBody.contentRestrictions, [{ readOnly: false }]);
    });

    it('returns message when file is not locked', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { id: 'file-1', name: 'Report.docx', contentRestrictions: [] },
      }));
      const res = await callTool(ctx.client, 'unlockFile', { fileId: 'file-1' });
      assert.ok(res.content[0].text!.includes('not locked'));
    });

    it('validation error on missing fileId', async () => {
      const res = await callTool(ctx.client, 'unlockFile', {});
      assert.equal(res.isError, true);
    });
  });

  describe('v1.6.0 pdf conversion tools', () => {
    it('convertPdfToGoogleDoc happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({ data: { id: 'pdf-1', name: 'A.pdf', mimeType: 'application/pdf', parents: ['root'] } }));
      ctx.mocks.drive.service.files.copy._setImpl(async () => ({ data: { id: 'doc-1', name: 'A (Doc)', webViewLink: 'https://doc' } }));
      const res = await callTool(ctx.client, 'convertPdfToGoogleDoc', { fileId: 'pdf-1' });
      assert.equal(res.isError, false);
    });

    it('bulkConvertFolderPdfs happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [{ id: 'p1', name: 'X.pdf' }] } }));
      ctx.mocks.drive.service.files.copy._setImpl(async () => ({ data: { id: 'd1', name: 'X (Doc)' } }));
      const res = await callTool(ctx.client, 'bulkConvertFolderPdfs', { folderId: 'folder-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Success=1'));
    });

    it('uploadPdfWithSplit performs real split uploads', async () => {
      const tempDir = await mkdtemp(join(tmpdir(), 'gdrive-mcp-test-'));
      try {
        const pdfPath = join(tempDir, 'source.pdf');
        const pdf = await PDFDocument.create();
        pdf.addPage();
        pdf.addPage();
        pdf.addPage();
        const bytes = await pdf.save();
        await writeFile(pdfPath, bytes);

        let counter = 0;
        ctx.mocks.drive.service.files.create._setImpl(async ({ requestBody }: any) => {
          counter += 1;
          return { data: { id: `part-${counter}`, name: requestBody?.name } };
        });

        const res = await callTool(ctx.client, 'uploadPdfWithSplit', {
          localPath: pdfPath,
          split: true,
          maxPagesPerChunk: 2,
          namePrefix: 'invoice',
        });

        assert.equal(res.isError, false);
        assert.ok(res.content[0].text!.includes('Uploaded split PDF into 2 part(s)'));
        assert.ok(res.content[0].text!.includes('invoice-part-1.pdf'));
        assert.ok(res.content[0].text!.includes('invoice-part-2.pdf'));

        const createCalls = ctx.mocks.drive.tracker.getCalls('files.create');
        assert.equal(createCalls.length, 2);
      } finally {
        await rm(tempDir, { recursive: true, force: true });
      }
    });
  });
});
