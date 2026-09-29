import assert from 'node:assert/strict';
import test from 'node:test';

import { handleTool, toolDefinitions } from '../../src/tools/sheets.js';
import { convertA1ToGridRange } from '../../src/utils.js';
import { splitRange } from '../../src/tools/sheetCells.js';

/** A call-counting double: dryRun must not write at all. */
function fakeCtx(grid: unknown, opts: { writeResponse?: unknown } = {}) {
  const calls = { get: 0, write: 0, lastWriteBody: null as any };
  const ctx = {
    authClient: {},
    google: {
      sheets: () => ({
        spreadsheets: {
          get: async () => { calls.get++; return { data: grid }; },
          values: {
            batchUpdate: async (p: any) => {
              calls.write++; calls.lastWriteBody = p.requestBody;
              return { data: opts.writeResponse ?? { responses: [] } };
            },
          },
        },
      }),
    },
  } as unknown as Parameters<typeof handleTool>[2];
  return { ctx, calls };
}

const gridWith = (uev: Record<string, unknown>) => ({
  properties: { title: 'Model' },
  sheets: [{ properties: { sheetId: 0, title: 'Probe' },
    data: [{ startRow: 0, startColumn: 0, rowData: [{ values: [{ userEnteredValue: uev }] }] }] }],
});

const dryFingerprint = async (grid: unknown) => {
  const { ctx } = fakeCtx(grid);
  const r = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], dryRun: true }, ctx);
  return JSON.parse(r!.content[0].text as string).fingerprint as string;
};

const writeOk = { responses: [{ updatedData: { range: 'Probe!A1', values: [['1']] } }] };

test('the tool is registered and requires spreadsheetId and updates', () => {
  const def = toolDefinitions.find(d => d.name === 'updateGoogleSheetIfUnchanged');
  assert.ok(def, 'tool must be in toolDefinitions');
  const schema = def!.inputSchema as { properties: Record<string, unknown>; required: string[] };
  assert.ok(schema.properties.expectedFingerprint && schema.properties.guardRanges && schema.properties.dryRun);
  assert.deepEqual([...schema.required].sort(), ['spreadsheetId', 'updates']);
});

test('dryRun returns a fingerprint and writes nothing at all', async () => {
  const { ctx, calls } = fakeCtx(gridWith({ formulaValue: '=A2' }));
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], dryRun: true }, ctx);
  assert.ok(result); assert.equal(result.isError, false);
  const payload = JSON.parse(result.content[0].text as string);
  assert.match(payload.fingerprint, /^v1:[0-9a-f]{64}$/);
  assert.equal(calls.write, 0, 'dryRun must not call the write API');
});

test('a fingerprint mismatch refuses, writes nothing, and returns the current contents', async () => {
  const { ctx, calls } = fakeCtx(gridWith({ formulaValue: '=A2' }));
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }],
      expectedFingerprint: 'v1:' + '0'.repeat(64) }, ctx);
  assert.ok(result); assert.equal(result.isError, true);
  assert.equal(calls.write, 0, 'a refused write must not reach the API');
  const payload = JSON.parse(result.content[0].text as string);
  assert.equal(payload.refused, 'fingerprint-mismatch');
  assert.ok(payload.actualFingerprint && payload.guardContents);
});

test('a matching fingerprint writes once and returns the pre-image and postFingerprint', async () => {
  const grid = gridWith({ formulaValue: '=A2' });
  const fp = await dryFingerprint(grid);
  const { ctx, calls } = fakeCtx(grid, { writeResponse: writeOk });
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);

  assert.ok(result); assert.equal(result.isError, false);
  assert.equal(calls.write, 1);
  assert.equal(calls.get, 1, 'exactly one read: no verification round trip');
  const payload = JSON.parse(result.content[0].text as string);
  assert.deepEqual(payload.preImage, [{ range: 'Probe!A1', values: [['=A2']] }]);
  assert.match(payload.postFingerprint, /^v1:[0-9a-f]{64}$/);
  assert.deepEqual(payload.hazards, []);
});

test('the write asks the API to return what it wrote, so no second read is needed', async () => {
  const grid = gridWith({ formulaValue: '=A2' });
  const fp = await dryFingerprint(grid);
  const { ctx, calls } = fakeCtx(grid, { writeResponse: writeOk });
  await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  assert.equal(calls.lastWriteBody.includeValuesInResponse, true);
  assert.equal(calls.lastWriteBody.responseValueRenderOption, 'FORMULA');
});

test('text that looks like a formula is reported with its native form', async () => {
  const grid = gridWith({ stringValue: '=notaformula' });
  const fp = await dryFingerprint(grid);
  const { ctx } = fakeCtx(grid, { writeResponse: writeOk });
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  const payload = JSON.parse(result!.content[0].text as string);
  assert.equal(payload.hazards.length, 1);
  assert.equal(payload.hazards[0].a1, 'A1');
  assert.deepEqual(payload.hazards[0].userEnteredValue, { stringValue: '=notaformula' });
});

test('a write without expectedFingerprint is rejected before any API call', async () => {
  const { ctx, calls } = fakeCtx(gridWith({ formulaValue: '=A2' }));
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }] }, ctx);
  assert.ok(result); assert.equal(result.isError, true);
  assert.equal(calls.get, 0);
  assert.equal(calls.write, 0);
});

test('a guardRanges set that differs from the write range requests both, guard first, when actually writing', async () => {
  // Distinct from the write range, so the handler must NOT collapse this to
  // one read: two grid entries, one per requested range in guard-then-write
  // order. This is the ACTUAL WRITE path - preImage/hazards need the write
  // range read too. (The dryRun path is deliberately different, as the test
  // below pins: it never touches writeMatches, so it must read only the
  // guard range.)
  const grid = {
    properties: { title: 'Model' },
    sheets: [{ properties: { sheetId: 0, title: 'Probe' },
      data: [
        { startRow: 0, startColumn: 1, rowData: [{ values: [{ userEnteredValue: { numberValue: 9 } }] }] },
        { startRow: 0, startColumn: 0, rowData: [{ values: [{ userEnteredValue: { formulaValue: '=A2' } }] }] },
      ] }],
  };
  const captured: { ranges?: string[] } = {};
  const ctx = {
    authClient: {},
    google: { sheets: () => ({ spreadsheets: {
      get: async (params: any) => { captured.ranges = params.ranges; return { data: grid }; },
      values: { batchUpdate: async () => ({ data: { responses: [{ updatedData: { range: 'Probe!A1', values: [['1']] } }] } }) },
    } }) },
  } as unknown as Parameters<typeof handleTool>[2];

  const dry = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }],
      guardRanges: ['Probe!B1'], dryRun: true }, ctx);
  const fp = JSON.parse(dry!.content[0].text as string).fingerprint as string;

  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }],
      guardRanges: ['Probe!B1'], expectedFingerprint: fp }, ctx);
  assert.ok(result); assert.equal(result.isError, false, JSON.stringify(JSON.parse(result!.content[0].text as string)));
  assert.deepEqual(captured.ranges, ['Probe!B1', 'Probe!A1']);
});

test('a dryRun with guardRanges distinct from the write range requests only the guard ranges - the write targets are never fetched', async () => {
  // The dryRun branch never touches writeMatches at all, so
  // concatenating writeRanges onto readRanges pulled the write targets out of
  // Google for nothing - defeating the documented "guard a source block while
  // writing a summary elsewhere" pattern, where the write target may not even
  // hold anything worth reading yet.
  const grid = {
    properties: { title: 'Model' },
    sheets: [{ properties: { sheetId: 0, title: 'Probe' },
      data: [{ startRow: 0, startColumn: 1, rowData: [{ values: [{ userEnteredValue: { numberValue: 9 } }] }] }] }],
  };
  const captured: { ranges?: string[] } = {};
  const ctx = {
    authClient: {},
    google: { sheets: () => ({ spreadsheets: { get: async (params: any) => {
      captured.ranges = params.ranges;
      return { data: grid };
    } } }) },
  } as unknown as Parameters<typeof handleTool>[2];

  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }],
      guardRanges: ['Probe!B1'], dryRun: true }, ctx);
  assert.ok(result); assert.equal(result.isError, false);
  assert.deepEqual(captured.ranges, ['Probe!B1'], 'only the guard range should be read - not the write range too');
});

test('guardContents is absent from the success payload - it is only ever needed by dryRun and the refusal path', async () => {
  // guardContents pays a full JSON.stringify + Buffer.byteLength
  // per cell unconditionally, even though the success path never reads it.
  const grid = gridWith({ formulaValue: '=A2' });
  const fp = await dryFingerprint(grid);
  const { ctx } = fakeCtx(grid, { writeResponse: writeOk });
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  assert.equal(result!.isError, false);
  const payload = JSON.parse(result!.content[0].text as string);
  assert.equal('guardContents' in payload, false);
});

test('a successful guarded write with the default guard set projects each match only once', async () => {
  // Both the fingerprint pass (blockFromMatch) and extractRanges (behind
  // guardContents) read match.grid.rowData exactly once per match they
  // actually process. Counting reads on a shared getter therefore catches
  // EITHER blockFromMatch being recomputed for the identical guard/write
  // match or guardContents being computed even though it is never read: the
  // only read of rowData on this success path is the single
  // blockFromMatch call whose block is reused for both the fingerprint and
  // the pre-image/hazards.
  let rowDataReads = 0;
  const rows = [{ values: [{ userEnteredValue: { formulaValue: '=A2' } }] }];
  const grid = {
    properties: { title: 'Model' },
    sheets: [{ properties: { sheetId: 0, title: 'Probe' }, data: [{
      startRow: 0, startColumn: 0,
      get rowData() { rowDataReads++; return rows; },
    }] }],
  };
  const fp = await dryFingerprint(grid); // its own guard read is not part of what's being measured below
  rowDataReads = 0;

  const { ctx } = fakeCtx(grid, { writeResponse: writeOk });
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  assert.equal(result!.isError, false);
  assert.equal(rowDataReads, 1, 'rowData must be read exactly once - not twice for the same guard/write match');
});

test('an absurdly large declared guard range is refused with a clear error, not materialized', async () => {
  const grid = {
    properties: { title: 'Model' },
    sheets: [{ properties: { sheetId: 0, title: 'Probe' },
      data: [{ startRow: 0, startColumn: 0, rowData: [{ values: [{ userEnteredValue: { numberValue: 1 } }] }] }] }],
  };
  const { ctx, calls } = fakeCtx(grid);
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }],
      guardRanges: ['Probe!A1:Z100000'], dryRun: true }, ctx);
  assert.ok(result);
  assert.equal(result.isError, true);
  assert.match(result.content[0].text as string, /exceeds the 100000-cell/);
  assert.equal(calls.write, 0);
});

test('a batchUpdate that throws after the request was sent still returns preImage and hazards, flagging the outcome as unknown', async () => {
  const grid = gridWith({ formulaValue: '=A2' });
  const fp = await dryFingerprint(grid);
  const { ctx } = fakeCtx(grid);
  (ctx as any).google.sheets = () => ({
    spreadsheets: {
      get: async () => ({ data: grid }),
      values: { batchUpdate: async () => { throw new Error('socket hang up'); } },
    },
  });

  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  assert.ok(result);
  assert.equal(result.isError, true);
  const payload = JSON.parse(result.content[0].text as string);
  assert.equal(payload.error, 'write-outcome-unknown');
  assert.match(payload.message, /socket hang up/);
  assert.match(payload.message, /UNKNOWN/);
  assert.deepEqual(payload.preImage, [{ range: 'Probe!A1', values: [['=A2']] }]);
  assert.deepEqual(payload.hazards, []);
});

test('a guardRanges set equal to the write range collapses to a single read of exactly that one range', async () => {
  // When the guard set equals the write set, the handler must read it once,
  // not twice.
  const grid = gridWith({ formulaValue: '=A2' });
  const captured: { ranges?: string[] } = {};
  const ctx = {
    authClient: {},
    google: { sheets: () => ({ spreadsheets: { get: async (params: any) => {
      captured.ranges = params.ranges;
      return { data: grid };
    } } }) },
  } as unknown as Parameters<typeof handleTool>[2];

  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], dryRun: true }, ctx);
  assert.ok(result); assert.equal(result.isError, false);
  assert.deepEqual(captured.ranges, ['Probe!A1']);
});

test('a write response missing updatedData.range fails loudly instead of yielding a meaningless postFingerprint', async () => {
  const grid = gridWith({ formulaValue: '=A2' });
  const fp = await dryFingerprint(grid);
  const { ctx, calls } = fakeCtx(grid, { writeResponse: { responses: [{}] } });
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  assert.ok(result);
  assert.equal(result.isError, true, 'the write happened but the post-state is unknown, so this must fail loudly');
  assert.equal(calls.write, 1, 'the write itself must still have gone through - only the fingerprint is unavailable');
  const payload = JSON.parse(result.content[0].text as string);
  assert.equal(payload.error, 'post-write-fingerprint-unavailable');
  assert.match(payload.message, /succeeded/i);
  assert.ok(payload.preImage, 'preImage must still be reported so a manual rollback stays possible');
});

// ---------------------------------------------------------------------------
// A stateful fake spreadsheet: enough of spreadsheets.get / values.batchUpdate
// to run the tool against itself across several calls and see the ACTUAL
// resulting sheet state, not just what one call reports about itself. Ranges
// used against it are kept to plain, single-sheet A1 rectangles (no
// open-ended ranges) - all this needs to model.
// ---------------------------------------------------------------------------

type RawCell = Record<string, unknown>;

function makeStatefulCtx(sheetTitle: string, sheetId: number, initial: Record<string, RawCell>) {
  const key = (row: number, col: number) => `${row},${col}`;
  const cells = new Map<string, RawCell>();
  for (const [a1, value] of Object.entries(initial)) {
    const g = convertA1ToGridRange(a1, sheetId);
    cells.set(key(g.startRowIndex!, g.startColumnIndex!), value);
  }
  const calls = { get: 0, write: 0 };

  function rectangleOf(range: string) {
    const { cellRange } = splitRange(range);
    const g = convertA1ToGridRange(cellRange!, sheetId);
    return {
      startRow: g.startRowIndex!, startColumn: g.startColumnIndex!,
      endRow: g.endRowIndex!, endColumn: g.endColumnIndex!,
    };
  }

  // Mirrors the real API's own trimming: rows/columns are trimmed from the
  // TRAILING edge only, down to the last row/column actually holding a value
  // within the requested rectangle; an all-empty rectangle yields no rowData
  // at all - exactly the shape blockFromMatch's padding is meant to survive.
  function gridDataFor(range: string) {
    const { startRow, startColumn, endRow, endColumn } = rectangleOf(range);
    let lastRow = -1;
    for (let r = startRow; r < endRow; r++) {
      for (let c = startColumn; c < endColumn; c++) {
        if (cells.has(key(r, c))) { lastRow = r; break; }
      }
    }
    if (lastRow === -1) return { startRow, startColumn };
    const rowData = [];
    for (let r = startRow; r <= lastRow; r++) {
      let lastCol = -1;
      for (let c = startColumn; c < endColumn; c++) if (cells.has(key(r, c))) lastCol = c;
      const values = [];
      for (let c = startColumn; c <= lastCol; c++) values.push({ userEnteredValue: cells.get(key(r, c)) ?? {} });
      rowData.push({ values });
    }
    return { startRow, startColumn, rowData };
  }

  function applyWrite(range: string, values: string[][], valueInputOption: string) {
    const { startRow, startColumn } = rectangleOf(range);
    values.forEach((row, r) => row.forEach((raw, c) => {
      const k = key(startRow + r, startColumn + c);
      if (raw === '') { cells.delete(k); return; }
      if (valueInputOption === 'USER_ENTERED') {
        if (raw.startsWith('=')) { cells.set(k, { formulaValue: raw }); return; }
        if (/^(?:true|false)$/i.test(raw)) { cells.set(k, { boolValue: raw.toLowerCase() === 'true' }); return; }
        if (raw.trim() !== '' && Number.isFinite(Number(raw))) { cells.set(k, { numberValue: Number(raw) }); return; }
      }
      cells.set(k, { stringValue: raw });
    }));
  }

  function updatedDataFor(range: string) {
    const grid = gridDataFor(range) as { rowData?: { values: { userEnteredValue: RawCell }[] }[] };
    const rows = grid.rowData ?? [];
    const values = rows.map(row => row.values.map(v => {
      const uev = v.userEnteredValue;
      if (typeof uev.formulaValue === 'string') return uev.formulaValue;
      if (typeof uev.numberValue === 'number') return uev.numberValue;
      if (typeof uev.stringValue === 'string') return uev.stringValue;
      if (typeof uev.boolValue === 'boolean') return uev.boolValue;
      return '';
    }));
    return { range, values };
  }

  const ctx = {
    authClient: {},
    google: {
      sheets: () => ({
        spreadsheets: {
          get: async (params: any) => {
            calls.get++;
            const data = (params.ranges as string[]).map(r => gridDataFor(r));
            return { data: { properties: { title: 'Model' },
              sheets: [{ properties: { sheetId, title: sheetTitle }, data }] } };
          },
          values: {
            batchUpdate: async (p: any) => {
              calls.write++;
              const updates = p.requestBody.data as { range: string; values: string[][] }[];
              updates.forEach(u => applyWrite(u.range, u.values, p.requestBody.valueInputOption));
              const responses = updates.map(u => ({ updatedData: updatedDataFor(u.range) }));
              const totalUpdatedCells = updates.reduce((n, u) => n + u.values.reduce((m, row) => m + row.length, 0), 0);
              return { data: { responses, totalUpdatedCells } };
            },
          },
        },
      }),
    },
  } as unknown as Parameters<typeof handleTool>[2];

  return { ctx, calls };
}

test('a real round trip: writing into a previously empty cell and a formula cell, then applying the returned preImage, restores exactly the original state', async () => {
  const { ctx, calls } = makeStatefulCtx('Probe', 0, {
    // A1 starts out with no entry at all - genuinely empty. A2 holds a
    // formula.
    A2: { formulaValue: '=SUM(B1:B2)' },
  });

  const updates = [
    { range: 'Probe!A1', values: [['NEW']] },
    { range: 'Probe!A2', values: [['NEW2']] },
  ];

  const dry = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates, dryRun: true }, ctx);
  const originalFingerprint = JSON.parse(dry!.content[0].text as string).fingerprint as string;

  const write = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates, expectedFingerprint: originalFingerprint }, ctx);
  assert.equal(write!.isError, false);
  const writePayload = JSON.parse(write!.content[0].text as string);

  // The empty cell's pre-image must be an empty string a write can apply,
  // not an under-covering `[]`.
  assert.deepEqual(writePayload.preImage, [
    { range: 'Probe!A1', values: [['']] },
    { range: 'Probe!A2', values: [['=SUM(B1:B2)']] },
  ]);

  // Exactly what the description asks for, and nothing else: guardRanges is
  // not passed, because it defaults to the ranges in updates and the preImage
  // already carries the ranges that were written.
  const undo = await handleTool('updateGoogleSheetIfUnchanged',
    {
      spreadsheetId: 'x',
      updates: writePayload.preImage,
      expectedFingerprint: writePayload.postFingerprint,
    }, ctx);
  assert.equal(undo!.isError, false, JSON.stringify(JSON.parse(undo!.content[0].text as string)));

  // Round trip complete: fingerprinting the same ranges again matches the
  // ORIGINAL fingerprint exactly - the sheet is back to what it was before
  // either write, empty cell and formula alike.
  const after = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates, dryRun: true }, ctx);
  const restoredFingerprint = JSON.parse(after!.content[0].text as string).fingerprint as string;
  assert.equal(restoredFingerprint, originalFingerprint);
  assert.equal(calls.write, 2);
});

test('the post-block geometry matches the guard geometry for the same declared range, when the write covers only part of it', async () => {
  const { ctx } = makeStatefulCtx('Probe', 0, {});
  const range = 'Probe!A1:B2'; // declared 2x2; only A1 is ever actually written.

  const dry = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range, values: [['X']] }], dryRun: true }, ctx);
  const fp = JSON.parse(dry!.content[0].text as string).fingerprint as string;

  const write = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range, values: [['X']] }], expectedFingerprint: fp }, ctx);
  const payload = JSON.parse(write!.content[0].text as string);

  // Re-guard the identical declared range: its fingerprint must agree with
  // postFingerprint bit for bit - both are canonical blocks padded to the
  // same A1:B2 rectangle, even though only A1 was ever actually written.
  const reguard = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range, values: [['X']] }], dryRun: true }, ctx);
  const reguardFp = JSON.parse(reguard!.content[0].text as string).fingerprint as string;

  assert.equal(payload.postFingerprint, reguardFp);
});

// ---------------------------------------------------------------------------
// Write ranges must name a bounded rectangle. Both of the tool's promises
// about a write rest on the declared rectangle being the rectangle actually
// written: preImage is padded out to it, and postFingerprint is taken over the
// one Google echoes in updatedData.range. An open-ended write range breaks
// both - the pre-image is trimmed to the data that happened to be there, and a
// later guard read of the same range observes down to the last data row, a
// different geometry - so it is refused, and refused before the read, since no
// API call can make it work.
// ---------------------------------------------------------------------------

test('an open-ended write range is refused before any API call, with a narrow-the-range message', async () => {
  const { ctx, calls } = fakeCtx(gridWith({ numberValue: 1 }));
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A2:C', values: [['1', '2', '3']] }],
      expectedFingerprint: 'v1:' + '0'.repeat(64) }, ctx);
  assert.ok(result);
  assert.equal(result.isError, true);
  assert.match(result.content[0].text as string, /does not name a bounded rectangle/);
  assert.match(result.content[0].text as string, /Probe!A2:C/);
  assert.match(result.content[0].text as string, /Narrow the range/);
  assert.equal(calls.get, 0, 'a request that cannot be honoured must cost no read');
  assert.equal(calls.write, 0);
});

test('every open-ended write form is refused, including a bare sheet name', async () => {
  for (const range of ['Probe!A5:A', 'Probe!A:C', 'Probe!5:9', 'Probe']) {
    const { ctx, calls } = fakeCtx(gridWith({ numberValue: 1 }));
    const result = await handleTool('updateGoogleSheetIfUnchanged',
      { spreadsheetId: 'x', updates: [{ range, values: [['1']] }],
        expectedFingerprint: 'v1:' + '0'.repeat(64) }, ctx);
    assert.equal(result!.isError, true, `"${range}" must be refused`);
    assert.match(result!.content[0].text as string, /bounded rectangle/);
    assert.equal(calls.get, 0);
  }
});

test('a dryRun over an open-ended write range is refused too - its fingerprint could only ever be spent on a refused write', async () => {
  const { ctx, calls } = fakeCtx(gridWith({ numberValue: 1 }));
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A2:C', values: [['1']] }], dryRun: true }, ctx);
  assert.ok(result);
  assert.equal(result.isError, true);
  assert.match(result.content[0].text as string, /bounded rectangle/);
  assert.equal(calls.get, 0);
});

test('an open-ended GUARD range is still accepted - only the ranges being written must be bounded', async () => {
  const { ctx, calls } = fakeCtx(gridWith({ numberValue: 1 }));
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }],
      guardRanges: ['Probe!A1:C'], dryRun: true }, ctx);
  assert.ok(result);
  assert.equal(result.isError, false, result.content[0].text as string);
  assert.equal(calls.get, 1);
});

test('a Google API error keeps its stack instead of being flattened into a local errorResponse', async () => {
  // The cell-cap refusal is caught around the two blockFromMatch passes and
  // nowhere else. A catch-all over the whole handler would turn this into the
  // same errorResponse the dispatcher produces, minus the dispatcher's log()
  // call - the only record such a failure leaves anywhere.
  const { ctx } = fakeCtx(gridWith({ numberValue: 1 }));
  (ctx as any).google.sheets = () => ({
    spreadsheets: { get: async () => { throw new Error('Request had insufficient authentication scopes'); } },
  });
  await assert.rejects(
    () => handleTool('updateGoogleSheetIfUnchanged',
      { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], dryRun: true }, ctx),
    /insufficient authentication scopes/,
  );
});

test('a post-write failure still hands back preImage and hazards rather than losing them', async () => {
  // Everything after the batchUpdate runs with the write already applied, so a
  // throw there must not escape with the pre-image: it is the only way back.
  // Forced here by an echoed updatedData.range far past the cell cap.
  const grid = gridWith({ formulaValue: '=A2' });
  const fp = await dryFingerprint(grid);
  const { ctx, calls } = fakeCtx(grid, {
    writeResponse: { responses: [{ updatedData: { range: 'Probe!A1:Z100000', values: [['1']] } }], totalUpdatedCells: 1 },
  });
  const result = await handleTool('updateGoogleSheetIfUnchanged',
    { spreadsheetId: 'x', updates: [{ range: 'Probe!A1', values: [['1']] }], expectedFingerprint: fp }, ctx);
  assert.ok(result);
  assert.equal(result.isError, true);
  assert.equal(calls.write, 1, 'the write itself did happen');
  const payload = JSON.parse(result.content[0].text as string);
  assert.equal(payload.error, 'post-write-fingerprint-unavailable');
  assert.match(payload.message, /exceeds the 100000-cell/);
  assert.deepEqual(payload.preImage, [{ range: 'Probe!A1', values: [['=A2']] }]);
  assert.deepEqual(payload.hazards, []);
});

test('the description tells callers to undo with preImage and postFingerprint alone', async () => {
  const def = toolDefinitions.find(d => d.name === 'updateGoogleSheetIfUnchanged')!;
  assert.match(def.description, /To undo a write, call again with updates set to the returned preImage and expectedFingerprint set to the returned postFingerprint/);
  assert.doesNotMatch(def.description, /guardRanges set to the ranges you wrote/);
});
