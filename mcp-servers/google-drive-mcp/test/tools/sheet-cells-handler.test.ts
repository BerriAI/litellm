import assert from 'node:assert/strict';
import test from 'node:test';

import { handleTool, toolDefinitions } from '../../src/tools/sheets.js';

const fakeCtx = (captured: Record<string, unknown>) => ({
  authClient: {},
  google: { sheets: () => ({ spreadsheets: { get: async (params: Record<string, unknown>) => {
    Object.assign(captured, params);
    return { data: {
      properties: { title: 'Model' },
      sheets: [{ properties: { sheetId: 7, title: 'Probe' },
        data: [{ startRow: 51, startColumn: 2, rowData: [{ values: [
          { userEnteredValue: { formulaValue: '=IF(C$27<>"",0,1)' },
            effectiveValue: { numberValue: 0.1 }, formattedValue: '10.00%' },
        ] }] }] }],
    } };
  } } }) },
} as unknown as Parameters<typeof handleTool>[2]);

test('getGoogleSheetCells is registered with ranges and fields', () => {
  const def = toolDefinitions.find(d => d.name === 'getGoogleSheetCells');
  assert.ok(def, 'tool must be in toolDefinitions');
  const props = (def!.inputSchema as { properties: Record<string, unknown> }).properties;
  assert.ok(props.ranges && props.fields && props.sheetMetadata);
});

test('a formula and its result arrive together, addressed absolutely', async () => {
  const captured: Record<string, unknown> = {};
  const result = await handleTool('getGoogleSheetCells',
    { spreadsheetId: 'abc', ranges: ['Probe!C52'] }, fakeCtx(captured));
  // handleTool returns ToolResult | null; without this guard it will not compile
  assert.ok(result, 'handleTool must claim the tool name');
  assert.equal(result.isError, false);

  const payload = JSON.parse(result.content[0].text as string);
  assert.deepEqual(payload.results[0].cells[0], {
    a1: 'C52',
    userEnteredValue: { formulaValue: '=IF(C$27<>"",0,1)' },
    effectiveValue: { numberValue: 0.1 },
    formattedValue: '10.00%',
  });
  assert.equal(payload.results[0].sheetId, 7);
  assert.equal(payload.truncated, false);
});

test('the request carries the ranges and a field mask, never includeGridData', async () => {
  const captured: Record<string, unknown> = {};
  await handleTool('getGoogleSheetCells', { spreadsheetId: 'abc', ranges: ['Probe!C52'] }, fakeCtx(captured));
  assert.deepEqual(captured.ranges, ['Probe!C52']);
  assert.match(String(captured.fields), /data\(startRow,startColumn/);
  assert.equal(captured.includeGridData, undefined);
});

test('the grid offsets are requested even when no cell fields are, since every A1 address depends on them', async () => {
  const captured: Record<string, unknown> = {};
  await handleTool('getGoogleSheetCells', { spreadsheetId: 'abc', ranges: ['Probe!C52'], fields: [] }, fakeCtx(captured));
  // buildFieldMask([], []) === 'properties.title,sheets(properties(sheetId,title),data(startRow,startColumn))'
  assert.match(String(captured.fields), /data\(startRow,startColumn\)/);
  assert.doesNotMatch(String(captured.fields), /rowData\.values/);
});

test('an unknown field name is rejected before any network call', async () => {
  const result = await handleTool('getGoogleSheetCells',
    { spreadsheetId: 'abc', ranges: ['A1'], fields: ['nope'] }, fakeCtx({}));
  assert.ok(result);
  assert.equal(result.isError, true);
});
