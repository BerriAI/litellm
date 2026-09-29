import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { buildFieldMask, cellA1, columnLetter, extractRanges, GridDataLike, matchRangesToGridData, rangeSheetTitle, remainderRange, collectSheetMetadata, splitRange } from '../../src/tools/sheetCells.js';

/** Count top-level sheets(...) roots in a field mask. Returns 2+ if malformed. */
function countSheetsRoots(mask: string): number {
  return mask.split('sheets(').length - 1;
}

test('columnLetter covers the single-, double- and triple-letter rollovers', () => {
  assert.equal(columnLetter(0), 'A');
  assert.equal(columnLetter(25), 'Z');
  assert.equal(columnLetter(26), 'AA');
  assert.equal(columnLetter(131), 'EB');   // the right edge of a real-world C6:EB13
  assert.equal(columnLetter(701), 'ZZ');
  assert.equal(columnLetter(702), 'AAA');
});

test('cellA1 turns 0-based grid coordinates into 1-based A1', () => {
  assert.equal(cellA1(0, 0), 'A1');
  assert.equal(cellA1(51, 2), 'C52');
});

test('buildFieldMask always asks for the offsets the addresses are computed from', () => {
  const mask = buildFieldMask(['userEnteredValue', 'effectiveValue'], []);
  assert.match(mask, /data\(startRow,startColumn/);
  assert.match(mask, /rowData\.values\(effectiveValue,userEnteredValue\)/);
  assert.match(mask, /properties\(sheetId,title\)/);
  // Exactly one top-level sheets root to prevent Google's mask rejection.
  assert.equal(countSheetsRoots(mask), 1);
});

test('buildFieldMask folds frozen counts into the single properties sub-mask', () => {
  const mask = buildFieldMask(['formattedValue'], ['frozen']);
  // Single sheets(...) root containing properties with gridProperties.
  assert.match(mask, /sheets\(properties\(.*gridProperties\(frozenRowCount,frozenColumnCount\)/);
  assert.equal(countSheetsRoots(mask), 1);
});

test('buildFieldMask asks for row and column metadata only when it was requested', () => {
  assert.doesNotMatch(buildFieldMask(['effectiveValue'], []), /rowMetadata/);
  assert.match(buildFieldMask(['effectiveValue'], ['hiddenRows']), /rowMetadata\(hiddenByUser,hiddenByFilter\)/);
  assert.match(buildFieldMask(['effectiveValue'], ['dimensionSizes']), /columnMetadata\(pixelSize\)/);
});

test('buildFieldMask includes merges inside the single sheets group when requested', () => {
  const mask = buildFieldMask([], ['merges']);
  assert.match(mask, /sheets\(.*,merges,/);
  assert.equal(countSheetsRoots(mask), 1);
});

test('buildFieldMask includes rowGroups and columnGroups inside the single sheets group when dimensionGroups requested', () => {
  const mask = buildFieldMask([], ['dimensionGroups']);
  assert.match(mask, /sheets\(.*,rowGroups,columnGroups,/);
  assert.equal(countSheetsRoots(mask), 1);
});

test('buildFieldMask with dimensionGroups + frozen + hiddenRows + fields nests everything under one sheets root', () => {
  const mask = buildFieldMask(['effectiveValue', 'userEnteredValue'], ['dimensionGroups', 'frozen', 'hiddenRows']);
  // Regression: the mask must have exactly one `sheets(...)` root.
  assert.equal(countSheetsRoots(mask), 1);
  assert.match(mask, /properties\(.*gridProperties\(frozenRowCount,frozenColumnCount\)/);
  assert.match(mask, /rowGroups,columnGroups/);
  assert.match(mask, /rowMetadata\(hiddenByUser,hiddenByFilter\)/);
  assert.match(mask, /rowData\.values\(effectiveValue,userEnteredValue\)/);
});

test('buildFieldMask with hiddenColumns + dimensionSizes puts both in columnMetadata', () => {
  const mask = buildFieldMask([], ['hiddenColumns', 'dimensionSizes']);
  assert.match(mask, /columnMetadata\(hiddenByUser,hiddenByFilter,pixelSize\)/);
  assert.equal(countSheetsRoots(mask), 1);
});

test('rangeSheetTitle handles quoting, embedded quotes and bare ranges', () => {
  assert.equal(rangeSheetTitle("'Entity Assumptions'!B52:E52"), 'Entity Assumptions');
  assert.equal(rangeSheetTitle('Sheet1!A1:C10'), 'Sheet1');
  assert.equal(rangeSheetTitle("'Bob''s Model'!A1"), "Bob's Model");
  assert.equal(rangeSheetTitle("'Whole Sheet'"), 'Whole Sheet');
  assert.equal(rangeSheetTitle('A1:C10'), null);   // null means the document's first sheet
});

test('splitRange treats an unquoted bare sheet name as a whole-sheet reference, not a range on the first sheet', () => {
  assert.deepEqual(splitRange('Probe!$A$1:$C$5'), { sheetTitle: 'Probe', cellRange: '$A$1:$C$5' });
  assert.deepEqual(splitRange('Probe!a1:b2'), { sheetTitle: 'Probe', cellRange: 'a1:b2' });
  assert.deepEqual(splitRange('Probe'), { sheetTitle: 'Probe', cellRange: null });
  assert.deepEqual(splitRange("'Probe'"), { sheetTitle: 'Probe', cellRange: null });
  assert.deepEqual(splitRange('C6:E9'), { sheetTitle: null, cellRange: 'C6:E9' });
  assert.deepEqual(splitRange('A1'), { sheetTitle: null, cellRange: 'A1' });
});

test('rangeSheetTitle resolves an unquoted bare sheet name to itself instead of defaulting to the first sheet', () => {
  assert.equal(rangeSheetTitle('Probe'), 'Probe');
});

test('month, year and quarter tab names are sheet names, not bare ranges', () => {
  // 'Probe' only ever passed because it has five letters. A single-part bare
  // range must carry BOTH letters and digits to be a cell, so letters alone
  // ('Jan', 'Dec', 'Log', 'Tax') and digits alone ('2024') are sheet names.
  assert.deepEqual(splitRange('Jan'), { sheetTitle: 'Jan', cellRange: null });
  assert.deepEqual(splitRange('Jan!A1:C5'), { sheetTitle: 'Jan', cellRange: 'A1:C5' });
  assert.deepEqual(splitRange('Dec'), { sheetTitle: 'Dec', cellRange: null });
  assert.deepEqual(splitRange('Log'), { sheetTitle: 'Log', cellRange: null });
  assert.deepEqual(splitRange('Tax'), { sheetTitle: 'Tax', cellRange: null });
  assert.deepEqual(splitRange('2024'), { sheetTitle: '2024', cellRange: null });
  assert.deepEqual(splitRange('2024!A1:C5'), { sheetTitle: '2024', cellRange: 'A1:C5' });
  assert.deepEqual(splitRange('Q1!A1:C5'), { sheetTitle: 'Q1', cellRange: 'A1:C5' });
  assert.equal(rangeSheetTitle('Jan'), 'Jan');
  assert.equal(rangeSheetTitle('2024'), '2024');
  // 'Q1' unqualified stays a range: it is a well-formed cell (column Q, row 1)
  // and Google resolves it against the first sheet, so a tab called Q1 has to
  // be named with a '!' or quotes, exactly as in Sheets itself.
  assert.deepEqual(splitRange('Q1'), { sheetTitle: null, cellRange: 'Q1' });
  assert.deepEqual(splitRange("'Q1'"), { sheetTitle: 'Q1', cellRange: null });
});

test('the bare-range grammar still admits the real A1 forms', () => {
  assert.deepEqual(splitRange('C6:E9'), { sheetTitle: null, cellRange: 'C6:E9' });
  assert.deepEqual(splitRange('C:EB'), { sheetTitle: null, cellRange: 'C:EB' });   // column to column
  assert.deepEqual(splitRange('6:9'), { sheetTitle: null, cellRange: '6:9' });     // row to row
  assert.deepEqual(splitRange('A8:9'), { sheetTitle: null, cellRange: 'A8:9' });   // cell to row
  assert.deepEqual(splitRange('$A$1:$C$5'), { sheetTitle: null, cellRange: '$A$1:$C$5' });
});

const sheets = [
  { properties: { sheetId: 1, title: 'Alpha' },
    data: [ { startRow: 5, startColumn: 2, rowData: [] }, { startRow: 50, startColumn: 0, rowData: [] } ] },
  { properties: { sheetId: 2, title: 'Beta Two' },
    data: [ { startColumn: 3, rowData: [] } ] },
];

test('two ranges on one sheet keep their requested order', () => {
  const out = matchRangesToGridData(['Alpha!C6:E9', "'Beta Two'!D1:D2", 'Alpha!A51:B60'], sheets);
  assert.ok('matches' in out);
  assert.deepEqual(out.matches.map(m => [m.range, m.grid.startRow ?? 0]), [
    ['Alpha!C6:E9', 5],
    ["'Beta Two'!D1:D2", 0],   // the protocol omits startRow when it is 0
    ['Alpha!A51:B60', 50],
  ]);
});

test('an unqualified range resolves to the first sheet of the document', () => {
  const out = matchRangesToGridData(['C6:E9'], sheets);
  assert.ok('matches' in out);
  assert.equal(out.matches[0].sheetTitle, 'Alpha');
});

test('a bare unquoted sheet name resolves to that sheet even mixed with a range naming a different sheet, not to the first sheet of the document', () => {
  // 'Beta Two' is sheets[0] here on purpose - if the bare name fell back to
  // "first sheet" it would silently match Beta Two instead of Gamma.
  const sheetsWithGamma = [
    { properties: { sheetId: 2, title: 'Beta Two' }, data: [{ startRow: 0, startColumn: 0, rowData: [] }] },
    { properties: { sheetId: 3, title: 'Gamma' }, data: [{ startRow: 0, startColumn: 0, rowData: [] }] },
  ];
  const out = matchRangesToGridData(['Gamma', "'Beta Two'!A1:B2"], sheetsWithGamma);
  assert.ok('matches' in out);
  assert.deepEqual(out.matches.map(m => m.sheetTitle), ['Gamma', 'Beta Two']);
});

test('a sheet title is matched case-insensitively, the way Google resolves it', () => {
  // 'jan!A1' is answered by Google with the data of the sheet named 'Jan', so
  // the lookup must find it - and the reply must carry the sheet's own casing.
  const janSheets = [
    { properties: { sheetId: 4, title: 'Jan' },
      data: [{ startRow: 0, startColumn: 0, rowData: [] }, { startRow: 9, startColumn: 0, rowData: [] }] },
  ];
  const out = matchRangesToGridData(['jan!A1', 'JAN!A10'], janSheets);
  assert.ok('matches' in out, ('error' in out) ? out.error : undefined);
  if (!('matches' in out)) return;
  assert.deepEqual(out.matches.map(m => m.sheetTitle), ['Jan', 'Jan']);
  // One case-folded cursor, so the second range takes the SECOND grid.
  assert.deepEqual(out.matches.map(m => m.grid.startRow ?? 0), [0, 9]);
});

test('a range naming an absent sheet is reported, not silently mismatched', () => {
  const out = matchRangesToGridData(['Nope!A1'], sheets);
  assert.ok('error' in out);
  assert.match(out.error, /Nope/);
});

test('more ranges than returned grids is reported rather than yielding undefined', () => {
  const out = matchRangesToGridData(['Alpha!A1', 'Alpha!A2', 'Alpha!A3'], sheets);
  assert.ok('error' in out);
  assert.match(out.error, /Alpha/);
});

test('an entirely empty range still occupies its slot in data[], keeping the per-sheet cursor in step', () => {
  // Verified live against the API: requesting 'Probe!Z90:AA95' (empty) and
  // 'Probe!A1:B2' together returns TWO elements - the empty one carries
  // startRow/startColumn and no rowData, and the second elides both offsets
  // because they are 0. The element is not dropped, so the cursor cannot slip.
  const sheetsWithEmpty = [
    { properties: { sheetId: 9, title: 'Probe' },
      data: [
        { startRow: 89, startColumn: 25 },
        { rowData: [{ values: [{ effectiveValue: { stringValue: 'a1' } }] }] },
      ] },
  ];
  const out = matchRangesToGridData(['Probe!Z90:AA95', 'Probe!A1:B2'], sheetsWithEmpty);
  assert.ok('matches' in out, ('error' in out) ? out.error : undefined);
  if (!('matches' in out)) return;
  assert.equal(out.matches[0].grid.rowData, undefined);
  assert.equal(out.matches[1].grid.startRow, undefined);

  const extracted = extractRanges(out.matches, ['effectiveValue'],
    { maxCells: 100, maxBytes: 100_000, includeEmpty: false });
  assert.deepEqual(extracted.results[0].cells, []);                 // the empty range, in order
  assert.deepEqual(extracted.results[1].cells.map(c => c.a1), ['A1']);  // A1, not Z90
});

const cell = (v: unknown) => ({ effectiveValue: { numberValue: v } });
const grid = (startRow: number, rows: number, cols: number) => ({
  startRow, startColumn: 2,
  rowData: Array.from({ length: rows }, () => ({ values: Array.from({ length: cols }, (_, c) => cell(c)) })),
});
// sheetTitle mirrors what matchRangesToGridData actually guarantees: it is
// always the range's own resolved sheet, never a fixed stand-in - remainderRange
// depends on that invariant to rebuild a correct prefix.
const match = (range: string, g: GridDataLike) =>
  ({ range, sheetId: 1, sheetTitle: rangeSheetTitle(range) ?? 'Alpha', grid: g, sheet: {} });

test('cells carry absolute A1 addresses built from the grid offsets', () => {
  const out = extractRanges([match('Alpha!C6:D7', grid(5, 2, 2))], ['effectiveValue'],
    { maxCells: 100, maxBytes: 100_000, includeEmpty: false });
  assert.deepEqual(out.results[0].cells.map((c: any) => c.a1), ['C6', 'D6', 'C7', 'D7']);
  assert.equal(out.truncated, false);
});

test('a missing startRow means row 0, not an unknown offset', () => {
  const out = extractRanges([match('Alpha!C1', { startColumn: 2, rowData: [{ values: [cell(1)] }] })],
    ['effectiveValue'], { maxCells: 100, maxBytes: 100_000, includeEmpty: false });
  assert.equal(out.results[0].cells[0].a1, 'C1');
});

test('cells with none of the requested fields are dropped unless includeEmpty', () => {
  const sparse = { startRow: 0, startColumn: 0, rowData: [{ values: [cell(1), {}, cell(3)] }] };
  const lean = extractRanges([match('Alpha!A1:C1', sparse)], ['effectiveValue'],
    { maxCells: 100, maxBytes: 100_000, includeEmpty: false });
  assert.deepEqual(lean.results[0].cells.map((c: any) => c.a1), ['A1', 'C1']);

  const full = extractRanges([match('Alpha!A1:C1', sparse)], ['effectiveValue'],
    { maxCells: 100, maxBytes: 100_000, includeEmpty: true });
  assert.deepEqual(full.results[0].cells.map((c: any) => c.a1), ['A1', 'B1', 'C1']);
  assert.equal(full.results[0].cells[1].empty, true);
});

test('the cell budget cuts on a row boundary and reports the unread remainder', () => {
  const out = extractRanges([match('Alpha!C6:D9', grid(5, 4, 2))], ['effectiveValue'],
    { maxCells: 5, maxBytes: 100_000, includeEmpty: false });
  assert.equal(out.results[0].cells.length, 4);          // two whole rows, not five cells
  assert.equal(out.results[0].truncated, true);
  assert.equal(out.truncated, true);
  assert.deepEqual(out.nextRanges, ['Alpha!C8:D9']);
});

test('ranges never reached are passed through to nextRanges verbatim', () => {
  const out = extractRanges(
    [match('Alpha!C6:D9', grid(5, 4, 2)), match("'Beta Two'!A1:B2", grid(0, 2, 2))],
    ['effectiveValue'], { maxCells: 2, maxBytes: 100_000, includeEmpty: false });
  assert.deepEqual(out.nextRanges, ['Alpha!C7:D9', "'Beta Two'!A1:B2"]);
});

test('a single row larger than the whole budget is still returned, so paging can progress', () => {
  const out = extractRanges([match('Alpha!C6:F7', grid(5, 2, 4))], ['effectiveValue'],
    { maxCells: 1, maxBytes: 10, includeEmpty: false });
  assert.equal(out.results[0].cells.length, 4);
  assert.equal(out.truncated, true);
  assert.deepEqual(out.nextRanges, ['Alpha!C7:F7']);
});

test('the first row of a LATER range is not admitted once the budget is spent', () => {
  // Four ranges of one five-cell row each under maxCells: 1. The per-range
  // progress rule returned all 20 cells with truncated:false; the guarantee is
  // per response, so only the first range may overshoot.
  const row = (startRow: number) => ({
    startRow, startColumn: 0,
    rowData: [{ values: Array.from({ length: 5 }, (_, c) => cell(c)) }],
  });
  const out = extractRanges(
    [match('Alpha!A1:E1', row(0)), match('Alpha!A3:E3', row(2)),
     match('Alpha!A5:E5', row(4)), match('Alpha!A7:E7', row(6))],
    ['effectiveValue'], { maxCells: 1, maxBytes: 1_000_000, includeEmpty: false });

  assert.equal(out.results.reduce((n, r) => n + r.cells.length, 0), 5);  // the first row only
  assert.equal(out.returned.cells, 5);
  assert.equal(out.truncated, true);
  // The three untouched ranges come back verbatim, ready to be passed straight in.
  assert.deepEqual(out.nextRanges, ['Alpha!A3:E3', 'Alpha!A5:E5', 'Alpha!A7:E7']);
});

test('a range that exactly fills the budget stops the next range rather than granting it a free row', () => {
  // Ten cells fill maxCells: 10 exactly, then a one-row range follows. That
  // second range used to be admitted whole, for 15 cells and truncated:false.
  const ten = { startRow: 0, startColumn: 0,
    rowData: Array.from({ length: 2 }, () => ({ values: Array.from({ length: 5 }, (_, c) => cell(c)) })) };
  const one = { startRow: 4, startColumn: 0,
    rowData: [{ values: Array.from({ length: 5 }, (_, c) => cell(c)) }] };
  const out = extractRanges([match('Alpha!A1:E2', ten), match('Alpha!A5:E5', one)],
    ['effectiveValue'], { maxCells: 10, maxBytes: 1_000_000, includeEmpty: false });

  assert.equal(out.returned.cells, 10);
  assert.equal(out.results.length, 1);            // the untouched range gets no result block
  assert.equal(out.truncated, true);
  assert.deepEqual(out.nextRanges, ['Alpha!A5:E5']);
});

test('the byte budget is measured in bytes, not UTF-16 code units', () => {
  // Non-Latin text costs more bytes than it has code units; counting .length
  // let a CJK sheet overrun maxBytes by more than 2x.
  const cjk = { startRow: 0, startColumn: 0, rowData: [
    { values: [{ effectiveValue: { stringValue: '営業利益の見通し' } }] },
    { values: [{ effectiveValue: { stringValue: '営業利益の見通し' } }] },
  ] };
  const out = extractRanges([match('Alpha!A1:A2', cjk)], ['effectiveValue'],
    { maxCells: 100, maxBytes: 1_000_000, includeEmpty: false });

  const serialized = out.results[0].cells.map(c => Buffer.byteLength(JSON.stringify(c)) + 1);
  assert.equal(out.returned.bytes, serialized.reduce((a, b) => a + b, 0));
  // And it is genuinely the byte count, strictly above the code-unit count the
  // old .length produced for this same content.
  const codeUnits = out.results[0].cells.map(c => JSON.stringify(c).length + 1)
    .reduce((a, b) => a + b, 0);
  assert.ok(out.returned.bytes > codeUnits,
    `expected bytes (${out.returned.bytes}) to exceed code units (${codeUnits})`);
});

test('a maxBytes that code-unit counting would have fitted twice over truncates after one row', () => {
  const text = '営業利益の見通し';
  const oneCell = { a1: 'A1', effectiveValue: { stringValue: text } };
  const codeUnits = JSON.stringify(oneCell).length + 1;
  const bytes = Buffer.byteLength(JSON.stringify(oneCell)) + 1;
  const cjk = { startRow: 0, startColumn: 0, rowData: Array.from({ length: 3 }, () => (
    { values: [{ effectiveValue: { stringValue: text } }] })) };

  // A budget that holds exactly two rows measured in code units, but only one
  // measured in bytes - so it pins which of the two the tool is charging for.
  const maxBytes = codeUnits * 2;
  assert.ok(bytes <= maxBytes && maxBytes < bytes * 2, 'the budget must separate the two readings');

  const out = extractRanges([match('Alpha!A1:A3', cjk)], ['effectiveValue'],
    { maxCells: 100, maxBytes, includeEmpty: false });
  assert.deepEqual(out.results[0].cells.map(c => c.a1), ['A1']);
  assert.equal(out.truncated, true);
  assert.deepEqual(out.nextRanges, ['Alpha!A2:A3']);
});

test('remainderRange rebuilds the start anchor and keeps the requested end anchor', () => {
  assert.equal(remainderRange(match("'Mgmt Fees Calc'!C6:EB13", grid(5, 8, 2)), 3, 2), "'Mgmt Fees Calc'!C9:EB13");
  assert.equal(remainderRange(match('Alpha!C:EB', grid(5, 8, 2)), 3, 2), 'Alpha!C9:EB');
});

test('a row-only range continues as a cell-to-row range, the form the API accepts', () => {
  // Verified live: 'Probe!A8:9' is accepted and answered with startRow: 7, no
  // startColumn, and 2 rows - so the cell-to-row anchor this produces for a
  // row-only request such as 'Alpha!6:9' pages correctly on the second call.
  const rows = { startRow: 5, startColumn: 0,
    rowData: Array.from({ length: 4 }, () => ({ values: [cell(1), cell(2)] })) };
  assert.equal(remainderRange(match('Alpha!6:9', rows), 2, 2), 'Alpha!A8:9');
  // And that continuation round-trips back through the bare-range grammar.
  assert.deepEqual(splitRange('Alpha!A8:9'), { sheetTitle: 'Alpha', cellRange: 'A8:9' });
});

test('remainderRange keeps the sheet prefix for a bare (unquoted) whole-sheet range', () => {
  // 'Probe' names no cell range at all, so there is no original end anchor
  // to reuse — the continuation is bounded to the observed width instead,
  // with the row end left open (no digit), mirroring the C:EB idiom above.
  const wholeSheetMatch = { range: 'Probe', sheetId: 1, sheetTitle: 'Probe', sheet: {}, grid: { startRow: 0, startColumn: 0 } };
  assert.equal(remainderRange(wholeSheetMatch, 3, 4), 'Probe!A4:D');
});

test('remainderRange quotes the sheet title when reconstructing the prefix for a range that needs it', () => {
  const wholeSheetMatch = { range: "'Mgmt Fees Calc'", sheetId: 1, sheetTitle: 'Mgmt Fees Calc', sheet: {}, grid: { startRow: 0, startColumn: 2 } };
  assert.equal(remainderRange(wholeSheetMatch, 2, 3), "'Mgmt Fees Calc'!C3:E");
});

const sheetWith = (extra: Record<string, unknown>) => ({
  properties: { sheetId: 1, title: 'Alpha', gridProperties: { frozenRowCount: 2, frozenColumnCount: 1 } },
  ...extra,
});

test('merges are reported at full extent even when they leave the requested range', () => {
  const sheet = sheetWith({ merges: [
    { sheetId: 1, startRowIndex: 3, endRowIndex: 4, startColumnIndex: 1, endColumnIndex: 4 },   // B4:D4, overlaps
    { sheetId: 1, startRowIndex: 90, endRowIndex: 91, startColumnIndex: 0, endColumnIndex: 2 }, // A91:B91, does not overlap
  ] });
  const matchObj = { range: 'Alpha!A1:B10', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: { startRow: 0, startColumn: 0 } };
  const meta = collectSheetMetadata([matchObj], ['merges']);
  assert.deepEqual(meta.Alpha.merges, ['B4:D4']);
});

// Regression coverage for the crash utils.parseA1Range/convertA1ToGridRange
// used to hit here on inputs Google itself accepts without complaint:
// $-absolute refs, lowercase refs, and a bare (whole-sheet) reference. Each
// must now resolve to the SAME merge, not throw and not silently miss it.
test('collectSheetMetadata does not crash on a $-absolute range and still finds its merge', () => {
  const sheet = sheetWith({ merges: [{ sheetId: 1, startRowIndex: 3, endRowIndex: 4, startColumnIndex: 1, endColumnIndex: 4 }] }); // B4:D4
  const matchObj = { range: 'Alpha!$A$1:$C$5', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: { startRow: 0, startColumn: 0 } };
  assert.deepEqual(collectSheetMetadata([matchObj], ['merges']).Alpha.merges, ['B4:D4']);
});

test('collectSheetMetadata does not crash on a lowercase range and still finds its merge', () => {
  const sheet = sheetWith({ merges: [{ sheetId: 1, startRowIndex: 3, endRowIndex: 4, startColumnIndex: 1, endColumnIndex: 4 }] }); // B4:D4
  const matchObj = { range: 'Alpha!a1:c5', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: { startRow: 0, startColumn: 0 } };
  assert.deepEqual(collectSheetMetadata([matchObj], ['merges']).Alpha.merges, ['B4:D4']);
});

test('collectSheetMetadata treats a bare whole-sheet range as an unbounded window, finding a merge far outside any plausible narrow guess', () => {
  // The merge sits at row 901 - nowhere near a small window a half-fix might
  // produce - so this only passes if the window is genuinely unbounded.
  const sheet = sheetWith({ merges: [{ sheetId: 1, startRowIndex: 900, endRowIndex: 901, startColumnIndex: 0, endColumnIndex: 2 }] }); // A901:B901
  const matchObj = { range: 'Alpha', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: { startRow: 0, startColumn: 0 } };
  assert.deepEqual(collectSheetMetadata([matchObj], ['merges']).Alpha.merges, ['A901:B901']);
});

test('a bare whole-sheet range mixed with a range naming a different sheet keeps each sheet\'s merge window separate', () => {
  const alphaSheet = sheetWith({ merges: [{ sheetId: 1, startRowIndex: 900, endRowIndex: 901, startColumnIndex: 0, endColumnIndex: 2 }] }); // A901:B901
  const betaSheet = {
    properties: { sheetId: 2, title: 'Beta', gridProperties: {} },
    merges: [
      { sheetId: 2, startRowIndex: 0, endRowIndex: 1, startColumnIndex: 0, endColumnIndex: 2 },  // A1:B1 - inside Beta!A1:B2
      { sheetId: 2, startRowIndex: 50, endRowIndex: 51, startColumnIndex: 0, endColumnIndex: 1 }, // A51 - outside Beta!A1:B2
    ],
  };
  const matches = [
    { range: 'Alpha', sheetId: 1, sheetTitle: 'Alpha', sheet: alphaSheet, grid: { startRow: 0, startColumn: 0 } },
    { range: 'Beta!A1:B2', sheetId: 2, sheetTitle: 'Beta', sheet: betaSheet, grid: { startRow: 0, startColumn: 0 } },
  ];
  const meta = collectSheetMetadata(matches, ['merges']);
  assert.deepEqual(meta.Alpha.merges, ['A901:B901']);  // whole sheet: the distant merge is still found
  assert.deepEqual(meta.Beta.merges, ['A1:B1']);        // Beta is still bounded by its own A1:B2 window
});

test('hidden rows and columns come back as absolute 1-based rows and column letters', () => {
  const grid2 = {
    startRow: 5, startColumn: 2,
    rowMetadata: [{}, { hiddenByUser: true }, { hiddenByFilter: true }],
    columnMetadata: [{ pixelSize: 120 }, { hiddenByUser: true, pixelSize: 40 }],
  };
  const matchObj = { range: 'Alpha!C6:D8', sheetId: 1, sheetTitle: 'Alpha', sheet: sheetWith({}), grid: grid2 };
  const meta = collectSheetMetadata([matchObj], ['hiddenRows', 'hiddenColumns', 'dimensionSizes']);
  assert.deepEqual(meta.Alpha.hiddenRows, [7, 8]);
  assert.deepEqual(meta.Alpha.hiddenColumns, ['D']);
  assert.deepEqual(meta.Alpha.dimensionSizes, { columnWidths: { C: 120, D: 40 }, rowHeights: {} });
});

test('frozen counts and dimension groups keep the 0-based half-open convention of the sibling tools', () => {
  const sheet = sheetWith({ rowGroups: [{ range: { startIndex: 9, endIndex: 14 }, depth: 1, collapsed: true }] });
  const matchObj = { range: 'Alpha!A1', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: {} };
  const meta = collectSheetMetadata([matchObj], ['frozen', 'dimensionGroups']);
  assert.deepEqual(meta.Alpha.frozen, { rows: 2, columns: 1 });
  assert.deepEqual(meta.Alpha.dimensionGroups, { rows: [{ start: 9, end: 14, depth: 1, collapsed: true }], columns: [] });
});

test('one sheet hit by two ranges is described once', () => {
  const sheet = sheetWith({ merges: [] });
  const mk = (range: string) => ({ range, sheetId: 1, sheetTitle: 'Alpha', sheet, grid: { startRow: 0, startColumn: 0 } });
  const meta = collectSheetMetadata([mk('Alpha!A1:B2'), mk('Alpha!D1:E2')], ['frozen']);
  assert.deepEqual(Object.keys(meta), ['Alpha']);
});

test('accumulating fields merge across multiple ranges on the same sheet', () => {
  // Grid 1: startRow=0, startColumn=0
  // - rowMetadata[1] is hidden → row 0+1+1 = 2
  // - columnMetadata[0] is hidden and has pixelSize 100 → column A (columnLetter(0+0) = 'A')
  const grid1 = {
    startRow: 0, startColumn: 0,
    rowMetadata: [{}, { hiddenByUser: true }],
    columnMetadata: [{ hiddenByUser: true, pixelSize: 100 }],
  };

  // Grid 2: startRow=10, startColumn=5
  // - rowMetadata[0] is hidden → row 10+0+1 = 11
  // - columnMetadata[1] is hidden and has pixelSize 150 → column G (columnLetter(5+1) = 'G')
  const grid2 = {
    startRow: 10, startColumn: 5,
    rowMetadata: [{ hiddenByUser: true }],
    columnMetadata: [{}, { hiddenByUser: true, pixelSize: 150 }],
  };

  const sheet = sheetWith({});
  const matches = [
    { range: 'Alpha!A1:C2', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: grid1 },
    { range: 'Alpha!F11:H20', sheetId: 1, sheetTitle: 'Alpha', sheet, grid: grid2 },
  ];

  const meta = collectSheetMetadata(matches, ['hiddenRows', 'hiddenColumns', 'dimensionSizes']);

  // Verify accumulation: both hidden rows are present (not just the last one)
  assert.deepEqual(meta.Alpha.hiddenRows, [2, 11]);
  // Verify accumulation: both hidden columns are present (not just the last one)
  assert.deepEqual(meta.Alpha.hiddenColumns, ['A', 'G']);
  // Verify accumulation: both pixel sizes are present
  assert.deepEqual(meta.Alpha.dimensionSizes, { columnWidths: { A: 100, G: 150 }, rowHeights: {} });
});

// This module's only test against a real, captured Sheets API response
// (every other test above builds a small synthetic double instead) - runs
// the actual pipeline (matchRangesToGridData -> extractRanges,
// collectSheetMetadata) over test/fixtures/sheet-cells-grid.json. Every
// value asserted below was read out of that fixture by hand; the fixture is
// checked in, so each one can be confirmed against the JSON directly.
test('the real captured spreadsheets.get fixture flows through matchRangesToGridData, extractRanges and collectSheetMetadata correctly', () => {
  const fixturePath = new URL('../fixtures/sheet-cells-grid.json', import.meta.url);
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

  const ranges = ['Probe!A1:F5', 'Probe!A7:F14', "'Probe 2'!A1:B2"];
  const matched = matchRangesToGridData(ranges, fixture.sheets);
  assert.ok('matches' in matched, ('error' in matched) ? (matched as { error: string }).error : undefined);
  if (!('matches' in matched)) return;

  const fields = ['userEnteredValue', 'effectiveValue', 'formattedValue', 'note', 'dataValidation'];
  const extracted = extractRanges(matched.matches, fields, { maxCells: 1000, maxBytes: 1_000_000, includeEmpty: true });
  assert.equal(extracted.truncated, false);
  const [probeBlock0, probeBlock1, probe2Block] = extracted.results;

  const byA1 = (result: { cells: { a1: string }[] }, a1: string) =>
    result.cells.find(c => c.a1 === a1) as any;

  // Probe!A1:F5, row 1: formula + its computed result, a checkbox with its
  // dataValidation, and a note - all hand-read from the fixture's row 0.
  assert.equal(byA1(probeBlock0, 'A1').userEnteredValue.formulaValue, '=IF(C$27<>"",0,SUM(B1:B3))');
  assert.equal(byA1(probeBlock0, 'A1').effectiveValue.numberValue, 504);
  assert.equal(byA1(probeBlock0, 'B1').effectiveValue.boolValue, true);
  assert.equal(byA1(probeBlock0, 'B1').dataValidation.condition.type, 'BOOLEAN');
  assert.equal(byA1(probeBlock0, 'C1').effectiveValue.stringValue, '28206');
  assert.equal(byA1(probeBlock0, 'C1').note, 'probe note on C1');

  // Row 4 (A4) is the deliberately empty cell in the middle of a row.
  assert.equal(byA1(probeBlock0, 'A4').empty, true);
  assert.equal(byA1(probeBlock0, 'B4').effectiveValue.numberValue, 402);

  // Row 5 (A5): formatted-vs-effective divergence (0.1 shown as "10.00%").
  assert.equal(byA1(probeBlock0, 'A5').effectiveValue.numberValue, 0.1);
  assert.equal(byA1(probeBlock0, 'A5').formattedValue, '10.00%');

  // Probe!A7:F14: response has startRow=6, so this block's own row 0 is
  // real row 7 - the nonzero-startRow case, where a range-relative reading
  // would report these cells six rows too high.
  assert.equal(probeBlock1.sheetTitle, 'Probe');
  assert.equal(byA1(probeBlock1, 'A7').effectiveValue.numberValue, 701);
  assert.equal(byA1(probeBlock1, 'D8').effectiveValue.numberValue, 804);

  // 'Probe 2'!A1:B2: the quoted, space-containing sheet name.
  assert.equal(probe2Block.sheetTitle, 'Probe 2');
  assert.equal(byA1(probe2Block, 'A1').effectiveValue.stringValue, 'p2-a1');
  assert.equal(byA1(probe2Block, 'B1').effectiveValue.stringValue, 'p2-b1');

  // merges: reported at full extent (D1:F1) - a separate pass from the
  // fields/includeEmpty extraction above.
  const meta = collectSheetMetadata(matched.matches, ['merges']);
  assert.deepEqual(meta.Probe.merges, ['D1:F1']);
});
