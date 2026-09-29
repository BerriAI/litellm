import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

describe('Sheets tools', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.drive.tracker.reset();
    ctx.mocks.sheets.tracker.reset();
  });

  // Helper to set up common Sheets mock that many formatting tests need
  function setupSheetsMock() {
    ctx.mocks.sheets.service.spreadsheets.get._setImpl(async () => ({
      data: {
        spreadsheetId: 'sheet-1',
        properties: { title: 'Test Sheet' },
        sheets: [{ properties: { sheetId: 0, title: 'Sheet1', gridProperties: { rowCount: 100, columnCount: 26 } } }],
      },
    }));
  }

  // --- createGoogleSheet ---
  describe('createGoogleSheet', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      ctx.mocks.sheets.service.spreadsheets.create._setImpl(async () => ({
        data: { spreadsheetId: 'sheet-new' },
      }));
      const res = await callTool(ctx.client, 'createGoogleSheet', {
        name: 'My Sheet', data: [['A', 'B'], ['1', '2']],
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('My Sheet'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'createGoogleSheet', {});
      assert.equal(res.isError, true);
    });
  });

  // --- updateGoogleSheet ---
  describe('updateGoogleSheet', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'updateGoogleSheet', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2', data: [['a', 'b']],
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Updated'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'updateGoogleSheet', {});
      assert.equal(res.isError, true);
    });
  });

  // --- batchUpdateGoogleSheetValues ---
  describe('batchUpdateGoogleSheetValues', () => {
    const twoUpdates = [
      { range: 'Sheet1!A1:B2', values: [['a', 'b'], ['1', '2']] },
      { range: 'Sheet2!C1', values: [['=SUM(A1:A10)']] },
    ];

    it('sends every range in one values.batchUpdate call', async () => {
      const res = await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        spreadsheetId: 'sheet-1', updates: twoUpdates,
      });
      assert.equal(res.isError, false);
      // The whole point of the tool: N ranges cost one API round trip, not N.
      assert.equal(ctx.mocks.sheets.tracker.getCalls('spreadsheets.values.update').length, 0);
      const calls = ctx.mocks.sheets.tracker.getCalls('spreadsheets.values.batchUpdate');
      assert.equal(calls.length, 1);
      assert.deepEqual(calls[0].args[0].requestBody.data, twoUpdates);
    });

    it('passes USER_ENTERED through so formulas land as formulas', async () => {
      await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        spreadsheetId: 'sheet-1', updates: twoUpdates, valueInputOption: 'USER_ENTERED',
      });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.values.batchUpdate')[0];
      assert.equal(call.args[0].requestBody.valueInputOption, 'USER_ENTERED');
    });

    it('defaults to RAW when valueInputOption is omitted', async () => {
      await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        spreadsheetId: 'sheet-1', updates: twoUpdates,
      });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.values.batchUpdate')[0];
      assert.equal(call.args[0].requestBody.valueInputOption, 'RAW');
    });

    it('reports the cell and range counts the API returned', async () => {
      ctx.mocks.sheets.service.spreadsheets.values.batchUpdate._setImpl(async () => ({
        data: {
          totalUpdatedCells: 7,
          responses: [{ updatedRange: 'Sheet1!A1:B2' }, { updatedRange: 'Sheet2!C1' }],
        },
      }));
      const res = await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        spreadsheetId: 'sheet-1', updates: twoUpdates,
      });
      assert.equal(res.isError, false);
      assert.match(res.content[0].text!, /7 cell/);
      assert.match(res.content[0].text!, /2 range/);
    });

    it('lists the first 10 ranges and summarises the rest', async () => {
      const responses = Array.from({ length: 12 }, (_, i) => ({ updatedRange: `Sheet1!A${i + 1}` }));
      ctx.mocks.sheets.service.spreadsheets.values.batchUpdate._setImpl(async () => ({
        data: { totalUpdatedCells: 12, responses },
      }));
      const res = await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        spreadsheetId: 'sheet-1', updates: twoUpdates,
      });
      assert.equal(res.isError, false);
      const text = res.content[0].text!;
      assert.match(text, /12 range/);
      assert.match(text, /Sheet1!A10, \+2 more/);
      assert.doesNotMatch(text, /Sheet1!A11/);
    });

    // Asserting on the message, not just isError: an unknown tool also reports
    // isError, so a bare flag check would pass before the tool even exists.
    it('rejects an empty updates array', async () => {
      const res = await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        spreadsheetId: 'sheet-1', updates: [],
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /at least one update/i);
    });

    it('rejects a missing spreadsheetId', async () => {
      const res = await callTool(ctx.client, 'batchUpdateGoogleSheetValues', {
        updates: twoUpdates,
      });
      assert.equal(res.isError, true);
      assert.match(res.content[0].text!, /spreadsheet id/i);
    });
  });

  // --- getGoogleSheetContent ---
  describe('getGoogleSheetContent', () => {
    it('happy path', async () => {
      ctx.mocks.sheets.service.spreadsheets.values.get._setImpl(async () => ({
        data: { values: [['Name', 'Age'], ['Alice', '30']] },
      }));
      const res = await callTool(ctx.client, 'getGoogleSheetContent', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Alice'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'getGoogleSheetContent', {});
      assert.equal(res.isError, true);
    });
  });

  // --- formatGoogleSheetCells ---
  describe('formatGoogleSheetCells', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'formatGoogleSheetCells', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2',
        horizontalAlignment: 'CENTER',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Formatted'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'formatGoogleSheetCells', {});
      assert.equal(res.isError, true);
    });
  });

  // --- formatGoogleSheetText ---
  describe('formatGoogleSheetText', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'formatGoogleSheetText', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2', bold: true,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('text formatting'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'formatGoogleSheetText', {});
      assert.equal(res.isError, true);
    });
  });

  // --- formatGoogleSheetNumbers ---
  describe('formatGoogleSheetNumbers', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'formatGoogleSheetNumbers', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2', pattern: '#,##0.00',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('number formatting'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'formatGoogleSheetNumbers', {});
      assert.equal(res.isError, true);
    });
  });

  // --- setGoogleSheetBorders ---
  describe('setGoogleSheetBorders', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'setGoogleSheetBorders', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2', style: 'SOLID',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('borders'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'setGoogleSheetBorders', {});
      assert.equal(res.isError, true);
    });
  });

  // --- mergeGoogleSheetCells ---
  describe('mergeGoogleSheetCells', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'mergeGoogleSheetCells', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:C3', mergeType: 'MERGE_ALL',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Merged'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'mergeGoogleSheetCells', {});
      assert.equal(res.isError, true);
    });
  });

  // --- addGoogleSheetConditionalFormat ---
  describe('addGoogleSheetConditionalFormat', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'addGoogleSheetConditionalFormat', {
        spreadsheetId: 'sheet-1',
        range: 'Sheet1!A1:B2',
        condition: { type: 'NUMBER_GREATER', value: '100' },
        format: { backgroundColor: { red: 1, green: 0, blue: 0 } },
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('conditional formatting'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'addGoogleSheetConditionalFormat', {});
      assert.equal(res.isError, true);
    });
  });

  // --- getSpreadsheetInfo ---
  describe('getSpreadsheetInfo', () => {
    it('happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'getSpreadsheetInfo', { spreadsheetId: 'sheet-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Test Sheet'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'getSpreadsheetInfo', {});
      assert.equal(res.isError, true);
    });
  });

  // --- appendSpreadsheetRows ---
  describe('appendSpreadsheetRows', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'appendSpreadsheetRows', {
        spreadsheetId: 'sheet-1', range: 'A1', values: [['x', 'y']],
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('appended'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'appendSpreadsheetRows', {});
      assert.equal(res.isError, true);
    });
  });

  // --- addSpreadsheetSheet ---
  describe('addSpreadsheetSheet', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'addSpreadsheetSheet', {
        spreadsheetId: 'sheet-1', sheetTitle: 'Sheet2',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Sheet2'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'addSpreadsheetSheet', {});
      assert.equal(res.isError, true);
    });
  });

  // --- addSheet (alias) ---
  describe('addSheet', () => {
    it('happy path', async () => {
      ctx.mocks.sheets.service.spreadsheets.batchUpdate._setImpl(async () => ({
        data: { replies: [{ addSheet: { properties: { title: 'AliasSheet', sheetId: 7 } } }] },
      }));
      const res = await callTool(ctx.client, 'addSheet', {
        spreadsheetId: 'sheet-1', title: 'AliasSheet',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('AliasSheet'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'addSheet', {});
      assert.equal(res.isError, true);
    });
  });

  // --- listSheets / renameSheet / deleteSheet ---
  describe('sheet lifecycle tools', () => {
    it('listSheets happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'listSheets', { spreadsheetId: 'sheet-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Sheet1'));
    });

    it('renameSheet happy path', async () => {
      const res = await callTool(ctx.client, 'renameSheet', { spreadsheetId: 'sheet-1', sheetId: 0, newTitle: 'Renamed' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Renamed'));
    });

    it('deleteSheet happy path', async () => {
      const res = await callTool(ctx.client, 'deleteSheet', { spreadsheetId: 'sheet-1', sheetId: 0 });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Deleted sheet'));
    });

    it('deleteSheet validation error', async () => {
      const res = await callTool(ctx.client, 'deleteSheet', {});
      assert.equal(res.isError, true);
    });
  });

  // --- governance helpers ---
  describe('validation/protection/named-range tools', () => {
    it('addDataValidation happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'addDataValidation', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:A10', conditionType: 'ONE_OF_LIST', values: ['A', 'B'],
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Added data validation'));
    });

    it('addDataValidation ONE_OF_RANGE happy path prepends "="', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'addDataValidation', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:A10', conditionType: 'ONE_OF_RANGE', values: ['Reference!A2:A50'],
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Added data validation (ONE_OF_RANGE)'));
      const calls = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate');
      const rule = calls[0].args[0].requestBody.requests[0].setDataValidation.rule;
      assert.equal(rule.condition.type, 'ONE_OF_RANGE');
      assert.deepEqual(rule.condition.values, [{ userEnteredValue: '=Reference!A2:A50' }]);
    });

    it('addDataValidation ONE_OF_RANGE keeps explicit "="', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'addDataValidation', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:A10', conditionType: 'ONE_OF_RANGE', values: ['=Reference!A2:A50'],
      });
      assert.equal(res.isError, false);
      const calls = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate');
      const rule = calls[0].args[0].requestBody.requests[0].setDataValidation.rule;
      assert.deepEqual(rule.condition.values, [{ userEnteredValue: '=Reference!A2:A50' }]);
    });

    it('addDataValidation ONE_OF_RANGE rejects multiple values', async () => {
      const res = await callTool(ctx.client, 'addDataValidation', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:A10', conditionType: 'ONE_OF_RANGE', values: ['Ref!A2:A50', 'Ref!B2:B50'],
      });
      assert.equal(res.isError, true);
    });

    it('addDataValidation requires values', async () => {
      const res = await callTool(ctx.client, 'addDataValidation', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:A10', conditionType: 'NUMBER_GREATER',
      });
      assert.equal(res.isError, true);
    });

    it('protectRange happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'protectRange', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B10', description: 'Lock critical',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Protected range'));
    });

    it('addNamedRange happy path', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'addNamedRange', {
        spreadsheetId: 'sheet-1', name: 'InputRange', range: 'Sheet1!A1:B10',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Added named range'));
    });
  });

  // --- dimension tools (setColumnWidth, setRowHeight, autoResize*, hide/showSheetDimension) ---
  describe('dimension tools', () => {
    it('setColumnWidth happy path', async () => {
      const res = await callTool(ctx.client, 'setColumnWidth', {
        spreadsheetId: 'sheet-1', sheetId: 0, startColumn: 0, endColumn: 3, pixelSize: 120,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Set column width'));
      const req = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.updateDimensionProperties.range, { sheetId: 0, dimension: 'COLUMNS', startIndex: 0, endIndex: 3 });
      assert.deepEqual(req.updateDimensionProperties.properties, { pixelSize: 120 });
      assert.equal(req.updateDimensionProperties.fields, 'pixelSize');
    });

    it('setColumnWidth rejects negative pixelSize', async () => {
      const res = await callTool(ctx.client, 'setColumnWidth', {
        spreadsheetId: 'sheet-1', sheetId: 0, startColumn: 0, endColumn: 3, pixelSize: -1,
      });
      assert.equal(res.isError, true);
    });

    it('setColumnWidth rejects reversed range', async () => {
      const res = await callTool(ctx.client, 'setColumnWidth', {
        spreadsheetId: 'sheet-1', sheetId: 0, startColumn: 5, endColumn: 2, pixelSize: 120,
      });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('startColumn must be less than endColumn'));
      assert.equal(ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate').length, 0);
    });

    it('setColumnWidth validation error', async () => {
      const res = await callTool(ctx.client, 'setColumnWidth', {});
      assert.equal(res.isError, true);
    });

    it('setRowHeight happy path', async () => {
      const res = await callTool(ctx.client, 'setRowHeight', {
        spreadsheetId: 'sheet-1', sheetId: 0, startRow: 0, endRow: 5, pixelSize: 30,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Set row height'));
      const req = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.updateDimensionProperties.range, { sheetId: 0, dimension: 'ROWS', startIndex: 0, endIndex: 5 });
      assert.deepEqual(req.updateDimensionProperties.properties, { pixelSize: 30 });
      assert.equal(req.updateDimensionProperties.fields, 'pixelSize');
    });

    it('setRowHeight rejects negative pixelSize', async () => {
      const res = await callTool(ctx.client, 'setRowHeight', {
        spreadsheetId: 'sheet-1', sheetId: 0, startRow: 0, endRow: 5, pixelSize: -1,
      });
      assert.equal(res.isError, true);
    });

    it('setRowHeight rejects reversed range', async () => {
      const res = await callTool(ctx.client, 'setRowHeight', {
        spreadsheetId: 'sheet-1', sheetId: 0, startRow: 5, endRow: 2, pixelSize: 30,
      });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('startRow must be less than endRow'));
      assert.equal(ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate').length, 0);
    });

    it('setRowHeight validation error', async () => {
      const res = await callTool(ctx.client, 'setRowHeight', {});
      assert.equal(res.isError, true);
    });

    it('autoResizeColumns happy path', async () => {
      const res = await callTool(ctx.client, 'autoResizeColumns', {
        spreadsheetId: 'sheet-1', sheetId: 0, startColumn: 0, endColumn: 3,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Auto-resized columns'));
      const req = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.autoResizeDimensions.dimensions, { sheetId: 0, dimension: 'COLUMNS', startIndex: 0, endIndex: 3 });
    });

    it('autoResizeColumns validation error', async () => {
      const res = await callTool(ctx.client, 'autoResizeColumns', {});
      assert.equal(res.isError, true);
    });

    it('autoResizeRows happy path', async () => {
      const res = await callTool(ctx.client, 'autoResizeRows', {
        spreadsheetId: 'sheet-1', sheetId: 0, startRow: 0, endRow: 5,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Auto-resized rows'));
      const req = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.autoResizeDimensions.dimensions, { sheetId: 0, dimension: 'ROWS', startIndex: 0, endIndex: 5 });
    });

    it('autoResizeRows validation error', async () => {
      const res = await callTool(ctx.client, 'autoResizeRows', {});
      assert.equal(res.isError, true);
    });

    it('hideSheetDimension happy path (columns)', async () => {
      const res = await callTool(ctx.client, 'hideSheetDimension', {
        spreadsheetId: 'sheet-1', sheetId: 0, dimension: 'COLUMNS', startIndex: 2, endIndex: 4,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Hid columns'));
      const req = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.updateDimensionProperties.range, { sheetId: 0, dimension: 'COLUMNS', startIndex: 2, endIndex: 4 });
      assert.deepEqual(req.updateDimensionProperties.properties, { hiddenByUser: true });
      assert.equal(req.updateDimensionProperties.fields, 'hiddenByUser');
    });

    it('showSheetDimension happy path (rows)', async () => {
      const res = await callTool(ctx.client, 'showSheetDimension', {
        spreadsheetId: 'sheet-1', sheetId: 0, dimension: 'ROWS', startIndex: 2, endIndex: 4,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Showed rows'));
      const req = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0].args[0].requestBody.requests[0];
      assert.deepEqual(req.updateDimensionProperties.range, { sheetId: 0, dimension: 'ROWS', startIndex: 2, endIndex: 4 });
      assert.deepEqual(req.updateDimensionProperties.properties, { hiddenByUser: false });
      assert.equal(req.updateDimensionProperties.fields, 'hiddenByUser');
    });

    it('hideSheetDimension validation error', async () => {
      const res = await callTool(ctx.client, 'hideSheetDimension', {});
      assert.equal(res.isError, true);
    });

    it('hideSheetDimension rejects invalid dimension enum', async () => {
      const res = await callTool(ctx.client, 'hideSheetDimension', {
        spreadsheetId: 'sheet-1', sheetId: 0, dimension: 'CELLS', startIndex: 0, endIndex: 1,
      });
      assert.equal(res.isError, true);
    });

    it('hideSheetDimension rejects reversed range', async () => {
      const res = await callTool(ctx.client, 'hideSheetDimension', {
        spreadsheetId: 'sheet-1', sheetId: 0, dimension: 'COLUMNS', startIndex: 4, endIndex: 2,
      });
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('startIndex must be less than endIndex'));
      assert.equal(ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate').length, 0);
    });
  });

  // --- listGoogleSheets ---
  describe('listGoogleSheets', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({
        data: { files: [{ id: 'sheet-1', name: 'Budget', modifiedTime: '2025-01-01', webViewLink: 'https://link', owners: [{ displayName: 'Owner' }] }] },
      }));
      const res = await callTool(ctx.client, 'listGoogleSheets', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Budget'));
    });

    it('defaults to modifiedTime desc, not oldest-first (issue #167)', async () => {
      ctx.mocks.drive.service.files.list._setImpl(async () => ({ data: { files: [] } }));
      await callTool(ctx.client, 'listGoogleSheets', {});
      const listCalls = ctx.mocks.drive.tracker.getCalls('files.list');
      assert.equal(listCalls[listCalls.length - 1].args[0].orderBy, 'modifiedTime desc');

      await callTool(ctx.client, 'listGoogleSheets', { orderBy: 'name' });
      const afterExplicit = ctx.mocks.drive.tracker.getCalls('files.list');
      assert.equal(afterExplicit[afterExplicit.length - 1].args[0].orderBy, 'name');
      ctx.mocks.drive.service.files.list._resetImpl();
    });
  });

  // --- copyFile ---
  describe('copyFile', () => {
    it('happy path', async () => {
      ctx.mocks.drive.service.files.get._setImpl(async () => ({
        data: { name: 'Original', parents: ['root'] },
      }));
      const res = await callTool(ctx.client, 'copyFile', { fileId: 'file-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('copied'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'copyFile', {});
      assert.equal(res.isError, true);
    });
  });

  // --- getGoogleSheetContent: valueRenderOption ---
  describe('getGoogleSheetContent valueRenderOption', () => {
    it('requests formulas when valueRenderOption is FORMULA', async () => {
      ctx.mocks.sheets.service.spreadsheets.values.get._setImpl(async () => ({
        data: { values: [['=SUM(A1:A10)']] },
      }));
      const res = await callTool(ctx.client, 'getGoogleSheetContent', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!B1', valueRenderOption: 'FORMULA',
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.values.get')[0];
      assert.equal(call.args[0].valueRenderOption, 'FORMULA');
      assert.ok(res.content[0].text!.includes('=SUM(A1:A10)'));
    });

    it('defaults to FORMATTED_VALUE when omitted', async () => {
      await callTool(ctx.client, 'getGoogleSheetContent', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2',
      });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.values.get')[0];
      assert.equal(call.args[0].valueRenderOption, 'FORMATTED_VALUE');
    });

    it('rejects an unknown render option', async () => {
      const res = await callTool(ctx.client, 'getGoogleSheetContent', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A1:B2', valueRenderOption: 'FORMULAS',
      });
      assert.equal(res.isError, true);
    });

    // A comma-space separator cannot be told apart from a comma-space inside a
    // cell, so rows are tab-separated under every render option.
    it('keeps a formula containing ", " in one cell', async () => {
      ctx.mocks.sheets.service.spreadsheets.values.get._setImpl(async () => ({
        data: { values: [['=IF(A1>0, "yes", "no")', '=SUM(A1, B1)']] },
      }));
      const res = await callTool(ctx.client, 'getGoogleSheetContent', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!B1:C1', valueRenderOption: 'FORMULA',
      });
      assert.equal(res.isError, false);
      const row = res.content[0].text!.split('\n').find(l => l.startsWith('Row 1: '))!;
      assert.deepEqual(row.slice('Row 1: '.length).split('\t'), [
        '=IF(A1>0, "yes", "no")',
        '=SUM(A1, B1)',
      ]);
    });

    it('keeps a prose cell containing ", " in one cell under the default render option', async () => {
      ctx.mocks.sheets.service.spreadsheets.values.get._setImpl(async () => ({
        data: { values: [['North', '$1,234.56', 'Up, but slowing']] },
      }));
      const res = await callTool(ctx.client, 'getGoogleSheetContent', {
        spreadsheetId: 'sheet-1', range: 'Sheet1!A2:C2',
      });
      assert.equal(res.isError, false);
      const row = res.content[0].text!.split('\n').find(l => l.startsWith('Row 1: '))!;
      assert.deepEqual(row.slice('Row 1: '.length).split('\t'), [
        'North',
        '$1,234.56',
        'Up, but slowing',
      ]);
    });
  });

  // --- addDimensionGroup ---
  describe('addDimensionGroup', () => {
    it('groups rows over the 0-based half-open range it was given', async () => {
      const res = await callTool(ctx.client, 'addDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 4, endIndex: 12,
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0];
      assert.deepEqual(call.args[0].requestBody.requests[0], {
        addDimensionGroup: {
          range: { sheetId: 0, dimension: 'ROWS', startIndex: 4, endIndex: 12 },
        },
      });
    });

    it('groups columns on a non-zero sheet id', async () => {
      await callTool(ctx.client, 'addDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 7,
        dimension: 'COLUMNS', startIndex: 2, endIndex: 6,
      });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0];
      assert.deepEqual(call.args[0].requestBody.requests[0].addDimensionGroup.range, {
        sheetId: 7, dimension: 'COLUMNS', startIndex: 2, endIndex: 6,
      });
    });

    it('rejects an empty range', async () => {
      const res = await callTool(ctx.client, 'addDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 5, endIndex: 5,
      });
      assert.equal(res.isError, true);
      assert.equal(ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate').length, 0);
    });

    it('rejects a negative start index', async () => {
      const res = await callTool(ctx.client, 'addDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: -1, endIndex: 5,
      });
      assert.equal(res.isError, true);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'addDimensionGroup', {});
      assert.equal(res.isError, true);
    });
  });

  // --- deleteDimensionGroup ---
  describe('deleteDimensionGroup', () => {
    it('sends a deleteDimensionGroup request for the range', async () => {
      const res = await callTool(ctx.client, 'deleteDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 4, endIndex: 12,
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0];
      assert.deepEqual(call.args[0].requestBody.requests[0], {
        deleteDimensionGroup: {
          range: { sheetId: 0, dimension: 'ROWS', startIndex: 4, endIndex: 12 },
        },
      });
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'deleteDimensionGroup', {});
      assert.equal(res.isError, true);
    });
  });

  // --- updateDimensionGroup ---
  describe('updateDimensionGroup', () => {
    it('collapses a group at a given depth', async () => {
      const res = await callTool(ctx.client, 'updateDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 4, endIndex: 12, depth: 1, collapsed: true,
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0];
      assert.deepEqual(call.args[0].requestBody.requests[0], {
        updateDimensionGroup: {
          dimensionGroup: {
            range: { sheetId: 0, dimension: 'ROWS', startIndex: 4, endIndex: 12 },
            depth: 1,
            collapsed: true,
          },
          fields: 'collapsed',
        },
      });
    });

    it('expands a group when collapsed is false', async () => {
      await callTool(ctx.client, 'updateDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 4, endIndex: 12, depth: 2, collapsed: false,
      });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0];
      const group = call.args[0].requestBody.requests[0].updateDimensionGroup.dimensionGroup;
      assert.equal(group.collapsed, false);
      assert.equal(group.depth, 2);
    });

    it('defaults to depth 1 when depth is omitted', async () => {
      await callTool(ctx.client, 'updateDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 4, endIndex: 12, collapsed: true,
      });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.batchUpdate')[0];
      assert.equal(call.args[0].requestBody.requests[0].updateDimensionGroup.dimensionGroup.depth, 1);
    });

    it('validation error when collapsed is missing', async () => {
      const res = await callTool(ctx.client, 'updateDimensionGroup', {
        spreadsheetId: 'sheet-1', sheetId: 0,
        dimension: 'ROWS', startIndex: 4, endIndex: 12, depth: 1,
      });
      assert.equal(res.isError, true);
    });
  });

  // --- listDimensionGroups ---
  describe('listDimensionGroups', () => {
    function setupGroupsMock() {
      ctx.mocks.sheets.service.spreadsheets.get._setImpl(async () => ({
        data: {
          spreadsheetId: 'sheet-1',
          sheets: [
            {
              properties: { sheetId: 0, title: 'Sheet1' },
              rowGroups: [
                { range: { startIndex: 4, endIndex: 12 }, depth: 1, collapsed: true },
                { range: { startIndex: 6, endIndex: 9 }, depth: 2, collapsed: false },
              ],
              columnGroups: [
                { range: { startIndex: 2, endIndex: 6 }, depth: 1, collapsed: false },
              ],
            },
          ],
        },
      }));
    }

    it('reports group ranges as 0-based half-open intervals', async () => {
      setupGroupsMock();
      const res = await callTool(ctx.client, 'listDimensionGroups', { spreadsheetId: 'sheet-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('rows [4, 12)'), res.content[0].text!);
      assert.ok(res.content[0].text!.includes('columns [2, 6)'), res.content[0].text!);
    });

    it('reports depth and collapsed state', async () => {
      setupGroupsMock();
      const res = await callTool(ctx.client, 'listDimensionGroups', { spreadsheetId: 'sheet-1' });
      assert.ok(res.content[0].text!.includes('depth 2'), res.content[0].text!);
      assert.ok(/collapsed/i.test(res.content[0].text!));
    });

    it('reports the sheet id so the range can be fed back to the group tools', async () => {
      setupGroupsMock();
      const res = await callTool(ctx.client, 'listDimensionGroups', { spreadsheetId: 'sheet-1' });
      assert.ok(res.content[0].text!.includes('sheetId: 0'), res.content[0].text!);
    });

    it('asks the API for the group fields', async () => {
      setupGroupsMock();
      await callTool(ctx.client, 'listDimensionGroups', { spreadsheetId: 'sheet-1' });
      const call = ctx.mocks.sheets.tracker.getCalls('spreadsheets.get')[0];
      assert.match(call.args[0].fields, /rowGroups/);
      assert.match(call.args[0].fields, /columnGroups/);
    });

    it('limits the listing to one sheet id', async () => {
      setupGroupsMock();
      const res = await callTool(ctx.client, 'listDimensionGroups', {
        spreadsheetId: 'sheet-1', sheetId: 0,
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Sheet1'));
    });

    it('says so when a sheet has no groups', async () => {
      setupSheetsMock();
      const res = await callTool(ctx.client, 'listDimensionGroups', { spreadsheetId: 'sheet-1' });
      assert.equal(res.isError, false);
      assert.ok(/no groups/i.test(res.content[0].text!), res.content[0].text!);
    });

    it('reports an unknown sheet id', async () => {
      setupGroupsMock();
      const res = await callTool(ctx.client, 'listDimensionGroups', {
        spreadsheetId: 'sheet-1', sheetId: 42,
      });
      assert.equal(res.isError, true);
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'listDimensionGroups', {});
      assert.equal(res.isError, true);
    });
  });
});
