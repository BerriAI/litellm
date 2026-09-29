import assert from 'node:assert/strict';
import { describe, it, before, after } from 'node:test';
import { setupTestServer, type TestContext } from '../helpers/setup-server.js';
import { TOOL_META } from '../../src/tools/toolMeta.js';

const EXPECTED_TOOL_COUNT = 130;

const EXPECTED_TOOLS = [
  'manage_accounts',
  'search', 'createTextFile', 'updateTextFile', 'readTextFile', 'createFolder', 'listFolder', 'listSharedDrives',
  'deleteItem', 'renameItem', 'moveItem',
  'createGoogleDoc', 'createDocFromHTML', 'updateGoogleDoc', 'insertText', 'deleteRange',
  'readGoogleDoc', 'readGoogleDocPaginated', 'listDocumentTabs', 'applyTextStyle', 'applyParagraphStyle', 'formatGoogleDocText', 'formatGoogleDocParagraph', 'createParagraphBullets', 'findAndReplaceInDoc',
  'listComments', 'getComment', 'addComment', 'replyToComment', 'deleteComment',
  'createGoogleSheet', 'updateGoogleSheet', 'batchUpdateGoogleSheetValues', 'updateGoogleSheetIfUnchanged', 'getGoogleSheetContent', 'getGoogleSheetCells',
  'formatGoogleSheetCells', 'formatGoogleSheetText', 'formatGoogleSheetNumbers',
  'setGoogleSheetBorders', 'mergeGoogleSheetCells', 'addGoogleSheetConditionalFormat',
  'getSpreadsheetInfo', 'appendSpreadsheetRows', 'addSpreadsheetSheet', 'addSheet', 'listSheets', 'renameSheet', 'deleteSheet', 'addDataValidation', 'protectRange', 'addNamedRange',
  'setColumnWidth', 'setRowHeight', 'autoResizeColumns', 'autoResizeRows', 'hideSheetDimension', 'showSheetDimension',
  'addDimensionGroup', 'deleteDimensionGroup', 'updateDimensionGroup', 'listDimensionGroups',
  'listGoogleSheets', 'copyFile',
  'createGoogleSlides', 'updateGoogleSlides',
  'getGoogleDocContent', 'getGoogleDocContentPaginated', 'getGoogleDocImage', 'getGoogleSlidesContent',
  'formatGoogleSlidesText', 'formatGoogleSlidesParagraph',
  'styleGoogleSlidesShape', 'setGoogleSlidesBackground',
  'createGoogleSlidesTextBox', 'createGoogleSlidesShape',
  'getGoogleSlidesSpeakerNotes', 'updateGoogleSlidesSpeakerNotes', 'deleteGoogleSlide', 'duplicateSlide', 'reorderSlides', 'replaceAllTextInSlides', 'exportSlideThumbnail',
  'insertSlidesImageFromUrl', 'insertSlidesLocalImage', 'moveSlideElement', 'deleteSlideElement', 'getSlideElementInfo',
  'createShortcut',
  'lockFile', 'unlockFile',
  'uploadFile', 'downloadFile', 'listPermissions', 'addPermission', 'updatePermission', 'removePermission', 'shareFile', 'getRevisions', 'restoreRevision', 'authGetStatus', 'authListScopes', 'authTestFileAccess',
  'listCalendars', 'getCalendarEvents', 'getCalendarEvent',
  'createCalendarEvent', 'updateCalendarEvent', 'deleteCalendarEvent',
  'insertTable', 'editTableCell', 'insertImageFromUrl', 'insertLocalImage',
  'listGoogleDocs', 'getDocumentInfo', 'addDocumentTab', 'renameDocumentTab', 'insertSmartChip', 'readSmartChips', 'createFootnote',
  'convertPdfToGoogleDoc', 'bulkConvertFolderPdfs', 'uploadPdfWithSplit',
];

describe('Tool Registry', () => {
  let ctx: TestContext;
  let tools: Array<{ name: string; inputSchema?: any; description?: string; annotations?: any }>;

  before(async () => {
    ctx = await setupTestServer();
    const result = await ctx.client.listTools();
    tools = result.tools as any;
  });

  after(async () => {
    await ctx.cleanup();
  });

  it(`registers exactly ${EXPECTED_TOOL_COUNT} tools`, () => {
    assert.equal(tools.length, EXPECTED_TOOL_COUNT, `Expected ${EXPECTED_TOOL_COUNT} tools, got ${tools.length}`);
  });

  it('has no duplicate tool names', () => {
    const names = tools.map((t) => t.name);
    const unique = new Set(names);
    assert.equal(unique.size, names.length, `Duplicate names: ${names.filter((n, i) => names.indexOf(n) !== i)}`);
  });

  it('every tool has a non-empty name and inputSchema', () => {
    for (const tool of tools) {
      assert.ok(tool.name, 'Tool name must be truthy');
      assert.ok(tool.inputSchema, `Tool "${tool.name}" is missing inputSchema`);
      assert.equal(tool.inputSchema.type, 'object', `Tool "${tool.name}" inputSchema.type must be "object"`);
    }
  });

  // The MCP spec defaults `destructiveHint` to true, so a server that annotates
  // nothing has every tool treated as destructive — `search` and `readGoogleDoc`
  // indistinguishable from `deleteItem`. `readOnlyHint` is projected from the
  // read/write/admin classification in TOOL_META, corrected at two edges that
  // are spelled out here independently of the server so a drift is visible:
  // `downloadFile` needs only a read scope but writes to the host filesystem,
  // and the auth diagnostics are admin-dispatched but only ever read.
  const HOST_MUTATING_READS = ['downloadFile'];
  const READ_ONLY_ADMIN_TOOLS = ['authGetStatus', 'authListScopes', 'authTestFileAccess'];

  it('marks exactly the read-only tools readOnlyHint, and no others', () => {
    const expectedReadOnly = Object.entries(TOOL_META)
      .filter(([name, meta]) => meta.opKind === 'read' && !HOST_MUTATING_READS.includes(name))
      .map(([name]) => name)
      .concat(READ_ONLY_ADMIN_TOOLS)
      .sort();

    const annotatedReadOnly = tools
      .filter((t) => t.annotations?.readOnlyHint === true)
      .map((t) => t.name)
      .sort();

    assert.notEqual(expectedReadOnly.length, 0, 'the server must expose read tools');
    assert.deepEqual(annotatedReadOnly, expectedReadOnly);
  });

  // The spec defines readOnlyHint as "does not modify its environment". A tool
  // that writes (or with `overwrite`, replaces) a local file must not carry it,
  // whatever Google scope it needs: a client honouring the hint would skip a
  // confirmation prompt on exactly the tool that can clobber a host file.
  it('never claims downloadFile is read-only', () => {
    const tool = tools.find((t) => t.name === 'downloadFile');
    assert.ok(tool, 'downloadFile must be registered');
    assert.equal(TOOL_META.downloadFile.opKind, 'read', 'scope classification is unchanged');
    assert.notEqual(tool.annotations?.readOnlyHint, true);
  });

  // destructiveHint defaults to true in the spec. Emitting it as false on a
  // tool that can in fact destroy data would suppress a confirmation prompt
  // that should have fired, so it stays unset until a deliberate per-tool pass.
  it('does not yet assert destructiveHint on any tool', () => {
    const claimed = tools.filter((t) => t.annotations?.destructiveHint !== undefined).map((t) => t.name);
    assert.deepEqual(claimed, []);
  });

  it('every expected tool is registered', () => {
    const names = new Set(tools.map((t) => t.name));
    for (const expected of EXPECTED_TOOLS) {
      assert.ok(names.has(expected), `Missing tool: ${expected}`);
    }
  });

  it('alias tools advertise the same parameters as the tools they alias', () => {
    // Alias definitions duplicate their primary's inputSchema by hand, so a
    // parameter added to only one of the pair leaves the alias advertising a
    // strictly smaller contract than the handler and Zod schema it shares.
    const ALIASES: Array<[string, string]> = [
      ['formatGoogleDocText', 'applyTextStyle'],
      ['formatGoogleDocParagraph', 'applyParagraphStyle'],
    ];
    for (const [alias, primary] of ALIASES) {
      const aliasTool = tools.find((t) => t.name === alias);
      const primaryTool = tools.find((t) => t.name === primary);
      assert.ok(aliasTool, `Missing alias tool: ${alias}`);
      assert.ok(primaryTool, `Missing primary tool: ${primary}`);
      assert.deepEqual(
        Object.keys(aliasTool.inputSchema.properties ?? {}).sort(),
        Object.keys(primaryTool.inputSchema.properties ?? {}).sort(),
        `"${alias}" inputSchema has drifted from "${primary}"`,
      );
      assert.deepEqual(
        [...(aliasTool.inputSchema.required ?? [])].sort(),
        [...(primaryTool.inputSchema.required ?? [])].sort(),
        `"${alias}" required fields have drifted from "${primary}"`,
      );
    }
  });

  it('every registered tool has a handler (does not return "Tool not found")', async () => {
    // Call each tool with empty args — should get a validation error, NOT "Tool not found"
    for (const tool of tools) {
      const result = await ctx.client.callTool({ name: tool.name, arguments: {} });
      const text = (result as any).content?.[0]?.text || '';
      assert.ok(
        !text.includes('Tool not found'),
        `Tool "${tool.name}" has no handler — got "Tool not found"`,
      );
    }
  });
});
