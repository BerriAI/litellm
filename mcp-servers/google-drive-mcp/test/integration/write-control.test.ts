import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

// ifRevisionId threads WriteControl.requiredRevisionId through the Docs write
// tools, and the read tools surface the revisionId that makes it obtainable.
describe('ifRevisionId optimistic locking', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.docs.tracker.reset();
    ctx.mocks.drive.tracker.reset();
    ctx.mocks.docs.service.documents.get._resetImpl();
    ctx.mocks.docs.service.documents.batchUpdate._resetImpl();
    ctx.mocks.drive.service.files.get._resetImpl();
  });

  function batchCalls() {
    return ctx.mocks.docs.tracker.getCalls('documents.batchUpdate');
  }
  function lastWriteControl() {
    const calls = batchCalls();
    return calls[calls.length - 1]!.args[0].requestBody.writeControl;
  }
  function asGoogleDoc() {
    ctx.mocks.drive.service.files.get._setImpl(async () => ({
      data: { id: 'd', name: 'Doc', mimeType: 'application/vnd.google-apps.document' },
    }));
  }
  function asTextFile() {
    ctx.mocks.drive.service.files.get._setImpl(async () => ({
      data: { id: 'd', name: 'notes.txt', mimeType: 'text/plain' },
    }));
  }

  it('style tools and bullets pass the lock', async () => {
    await callTool(ctx.client, 'applyTextStyle', { documentId: 'd', startIndex: 1, endIndex: 5, bold: true, ifRevisionId: 'rev-a' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-a' });

    await callTool(ctx.client, 'applyParagraphStyle', { documentId: 'd', startIndex: 1, endIndex: 5, alignment: 'CENTER', ifRevisionId: 'rev-b' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-b' });

    await callTool(ctx.client, 'createParagraphBullets', { documentId: 'd', startIndex: 1, endIndex: 5, ifRevisionId: 'rev-c' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-c' });

    await callTool(ctx.client, 'createParagraphBullets', { documentId: 'd', startIndex: 1, endIndex: 5, bulletPreset: 'NONE', ifRevisionId: 'rev-d' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-d' });
  });

  it('table tools pass the lock', async () => {
    asGoogleDoc();
    await callTool(ctx.client, 'insertTable', { documentId: 'd', rows: 2, columns: 2, index: 1, ifRevisionId: 'rev-e' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-e' });

    ctx.mocks.docs.service.documents.get._setImpl(async () => ({
      data: {
        body: {
          content: [{
            startIndex: 5, endIndex: 40,
            table: { tableRows: [{ tableCells: [{ startIndex: 6, endIndex: 12, content: [] }] }] },
          }],
        },
      },
    }));
    await callTool(ctx.client, 'editTableCell', { documentId: 'd', tableStartIndex: 5, rowIndex: 0, columnIndex: 0, textContent: 'x', ifRevisionId: 'rev-f' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-f' });
  });

  it('smart chips and footnotes pass the lock on the creating call', async () => {
    await callTool(ctx.client, 'insertSmartChip', { documentId: 'd', index: 1, chipType: 'person', personEmail: 'a@b.com', ifRevisionId: 'rev-g' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-g' });

    ctx.mocks.docs.service.documents.batchUpdate._setImpl(async () => ({
      data: { replies: [{ createFootnote: { footnoteId: 'fn-1' } }] },
    }));
    await callTool(ctx.client, 'createFootnote', { documentId: 'd', index: 1, content: 'note', ifRevisionId: 'rev-h' });
    const calls = batchCalls();
    const creating = calls[calls.length - 2]!.args[0].requestBody;
    const inserting = calls[calls.length - 1]!.args[0].requestBody;
    assert.deepEqual(creating.writeControl, { requiredRevisionId: 'rev-h' });
    // The content insert targets the fresh footnote segment — never locked.
    assert.equal(inserting.writeControl, undefined);
  });

  it('insertText, deleteRange, and findAndReplaceInDoc pass the lock', async () => {
    asGoogleDoc();
    await callTool(ctx.client, 'insertText', { documentId: 'd', text: 'hi', index: 1, ifRevisionId: 'rev-k' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-k' });

    await callTool(ctx.client, 'deleteRange', { documentId: 'd', startIndex: 1, endIndex: 3, ifRevisionId: 'rev-l' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-l' });

    await callTool(ctx.client, 'findAndReplaceInDoc', { documentId: 'd', findText: 'Hello', replaceText: 'Goodbye', ifRevisionId: 'rev-m' });
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-m' });
  });

  it('every Docs tool whose handler takes ifRevisionId advertises it in its inputSchema', async () => {
    // A parameter the handler accepts but the schema omits is invisible to a
    // client that builds calls from listTools.
    const { tools } = await ctx.client.listTools();
    for (const name of ['insertText', 'deleteRange', 'findAndReplaceInDoc', 'applyTextStyle', 'applyParagraphStyle', 'updateGoogleDoc', 'formatGoogleDocText', 'formatGoogleDocParagraph', 'createParagraphBullets', 'insertTable', 'editTableCell', 'styleDocTable', 'insertSmartChip', 'createFootnote']) {
      const tool = tools.find((t) => t.name === name);
      assert.ok(tool, `${name} is registered`);
      assert.ok((tool!.inputSchema.properties as Record<string, unknown>).ifRevisionId, `${name} does not advertise ifRevisionId`);
    }
  });

  it('insertText and deleteRange refuse the lock on text files rather than ignoring it', async () => {
    asTextFile();
    const ins = await callTool(ctx.client, 'insertText', { documentId: 'd', text: 'hi', index: 0, ifRevisionId: 'rev-n' });
    assert.equal(ins.isError, true);
    assert.match(ins.content[0].text!, /only supported for Google Docs/);

    const del = await callTool(ctx.client, 'deleteRange', { documentId: 'd', startIndex: 0, endIndex: 2, ifRevisionId: 'rev-o' });
    assert.equal(del.isError, true);
    assert.match(del.content[0].text!, /only supported for Google Docs/);
  });

  it('updateGoogleDoc replaces in one locked batch, on both the tab and default paths', async () => {
    ctx.mocks.docs.service.documents.get._setImpl(async () => ({
      data: {
        title: 'Doc',
        tabs: [{ tabProperties: { tabId: 't1' }, documentTab: { body: { content: [{ endIndex: 30 }] } } }],
        body: { content: [{ endIndex: 30 }] },
      },
    }));

    await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'd', content: 'new', tabId: 't1', ifRevisionId: 'rev-i' });
    assert.equal(batchCalls().length, 1);
    assert.deepEqual(lastWriteControl(), { requiredRevisionId: 'rev-i' });

    ctx.mocks.docs.tracker.reset();
    await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'd', content: 'new', ifRevisionId: 'rev-j' });
    const calls = batchCalls();
    // One atomic batch, not a delete call followed by an unguarded insert.
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0]!.args[0].requestBody.writeControl, { requiredRevisionId: 'rev-j' });
    const kinds = calls[0]!.args[0].requestBody.requests.map((r: any) => Object.keys(r)[0]);
    assert.deepEqual(kinds, ['deleteContentRange', 'insertText', 'updateParagraphStyle']);
  });

  it('updateGoogleDoc locks an empty document, where there is no delete to carry it', async () => {
    ctx.mocks.docs.service.documents.get._setImpl(async () => ({
      data: { title: 'Empty', body: { content: [{ endIndex: 2 }] } },
    }));

    await callTool(ctx.client, 'updateGoogleDoc', { documentId: 'd', content: 'first words', ifRevisionId: 'rev-empty' });
    const calls = batchCalls();
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0]!.args[0].requestBody.writeControl, { requiredRevisionId: 'rev-empty' });
    const kinds = calls[0]!.args[0].requestBody.requests.map((r: any) => Object.keys(r)[0]);
    assert.deepEqual(kinds, ['insertText', 'updateParagraphStyle']);
  });

  it('omitting ifRevisionId sends no writeControl (back-compat)', async () => {
    await callTool(ctx.client, 'applyTextStyle', { documentId: 'd', startIndex: 1, endIndex: 5, bold: true });
    assert.equal(lastWriteControl(), undefined);
  });
});

describe('revisionId is obtainable from the reads', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.docs.tracker.reset();
    ctx.mocks.drive.tracker.reset();
    ctx.mocks.docs.service.documents.get._resetImpl();
    ctx.mocks.drive.service.files.get._resetImpl();
  });

  // The mock honours a `fields` mask with respect to revisionId, because the
  // API does. A projection that leaves revisionId out is the one change that
  // breaks every tool in this describe at once without touching any of their
  // logic: the field simply stops arriving. A mock that returns revisionId
  // regardless would report a lock the server could never actually obtain, so
  // masking the read would pass the whole suite and fail in production. Only
  // revisionId is modelled; the rest of the projection is not what these tests
  // are about.
  function suppliesRevisionId(fields: unknown): boolean {
    if (typeof fields !== 'string' || fields.trim() === '') return true;
    return /(^|[(,\s])revisionId([),\s]|$)/.test(fields);
  }

  function docWithRevision(revisionId?: string) {
    ctx.mocks.docs.service.documents.get._setImpl(async (params: any) => ({
      data: {
        documentId: 'doc-1',
        title: 'Test Doc',
        ...(revisionId && suppliesRevisionId(params?.fields) ? { revisionId } : {}),
        body: {
          content: [{
            startIndex: 0, endIndex: 13,
            paragraph: { elements: [{ textRun: { content: 'Hello World\n' }, startIndex: 1, endIndex: 13 }] },
          }],
        },
      },
    }));
  }

  it('readGoogleDoc leads text and markdown with the revisionId', async () => {
    docWithRevision('rev-read-1');

    const asText = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'd', format: 'text' });
    assert.ok(asText.content[0].text!.startsWith('revisionId: rev-read-1\n'));

    const asMarkdown = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'd', format: 'markdown' });
    assert.ok(asMarkdown.content[0].text!.startsWith('revisionId: rev-read-1\n'));
  });

  it('readGoogleDoc json output is the raw document, unchanged', async () => {
    docWithRevision('rev-read-2');
    const asJson = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'd', format: 'json' });
    const parsed = JSON.parse(asJson.content[0].text!);
    // No prefix line: the field is already part of the document payload.
    assert.equal(parsed.revisionId, 'rev-read-2');
    assert.equal(parsed.title, 'Test Doc');
  });

  it('readGoogleDoc says why the lock is unavailable to a read-only caller', async () => {
    docWithRevision(undefined);
    const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'd', format: 'text' });
    assert.ok(res.content[0].text!.startsWith('revisionId: unavailable (no edit access)\n'));
  });

  it('readGoogleDoc counts the revision line against maxLength', async () => {
    docWithRevision('rev-read-3');
    const res = await callTool(ctx.client, 'readGoogleDoc', { documentId: 'd', format: 'text', maxLength: 30 });
    const text = res.content[0].text!;
    assert.ok(text.startsWith('revisionId: rev-read-3\n'), 'lock stays usable on a truncated read');
    // The body is trimmed to fit the cap; only the truncation marker sits past it.
    assert.ok(text.replace('\n... (truncated)', '').length <= 30, `got ${text.length}: ${JSON.stringify(text)}`);
  });

  it('getGoogleDocContent and its paginated form report the revisionId', async () => {
    docWithRevision('rev-read-4');

    const plain = await callTool(ctx.client, 'getGoogleDocContent', { documentId: 'd' });
    assert.match(plain.content[0].text!, /\nrevisionId: rev-read-4$/);

    const paged = await callTool(ctx.client, 'getGoogleDocContentPaginated', { documentId: 'd', offset: 0, limit: 50 });
    assert.equal(JSON.parse(paged.content[0].text!).revisionId, 'rev-read-4');
  });

  it('getDocumentInfo reports the Docs revisionId distinctly from the Drive version', async () => {
    docWithRevision('rev-read-5');
    ctx.mocks.drive.service.files.get._setImpl(async () => ({
      data: { id: 'd', name: 'Doc', mimeType: 'application/vnd.google-apps.document', version: '42', webViewLink: 'https://example.com' },
    }));

    const res = await callTool(ctx.client, 'getDocumentInfo', { documentId: 'd' });
    const text = res.content[0].text!;
    assert.match(text, /\*\*Docs revisionId:\*\* rev-read-5/);
    assert.match(text, /\*\*Drive Version:\*\* 42/);
  });

  // These two shipped in #211 and lead their output with the revisionId, but
  // neither had a test saying so, which left their reads free to grow a
  // projection that drops the field.
  it('getGoogleDocStyleSummary and describeGoogleDocRange report the revisionId', async () => {
    docWithRevision('rev-read-6');

    const summary = await callTool(ctx.client, 'getGoogleDocStyleSummary', { documentId: 'd' });
    assert.equal(summary.isError, false);
    assert.match(summary.content[0].text!, /revisionId: rev-read-6/);

    const probe = await callTool(ctx.client, 'describeGoogleDocRange', { documentId: 'd', startIndex: 1, endIndex: 5 });
    assert.equal(probe.isError, false);
    assert.match(probe.content[0].text!, /revisionId: rev-read-6/);
  });

  it('getDocumentInfo still returns metadata when the revisionId lookup fails', async () => {
    ctx.mocks.docs.service.documents.get._setImpl(async () => { throw new Error('no edit access'); });
    ctx.mocks.drive.service.files.get._setImpl(async () => ({
      data: { id: 'd', name: 'Doc', mimeType: 'application/vnd.google-apps.document', version: '7', webViewLink: 'https://example.com' },
    }));

    const res = await callTool(ctx.client, 'getDocumentInfo', { documentId: 'd' });
    assert.equal(res.isError, false);
    assert.match(res.content[0].text!, /\*\*Docs revisionId:\*\* unavailable \(no edit access\)/);
  });
});
