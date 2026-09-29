import assert from 'node:assert/strict';
import test from 'node:test';

import { fingerprintOf, projectCellData, projectResponseValue } from '../../src/tools/sheetGuard.js';
import { buildPreImage, findHazards, renderForWrite } from '../../src/tools/sheetGuard.js';
import { blockFromMatch, guardRangesFor, padToDeclaredRange } from '../../src/tools/sheetGuard.js';
import { checkWriteRanges } from '../../src/tools/sheetGuard.js';

test('projectCellData reduces CellData to one comparable scalar', () => {
  assert.equal(projectCellData({ formulaValue: '=SUM(A1:A3)' }), '=SUM(A1:A3)');
  assert.equal(projectCellData({ numberValue: 504 }), 504);
  assert.equal(projectCellData({ stringValue: 'plain' }), 'plain');
  assert.equal(projectCellData({ boolValue: true }), 'TRUE');
  assert.equal(projectCellData({ boolValue: false }), 'FALSE');
  assert.equal(projectCellData(undefined), null);
});

test('projectResponseValue maps the write response onto the same scalars', () => {
  // The write response renders formulas as strings and may encode booleans
  // either way; both must land on the token projectCellData produces.
  assert.equal(projectResponseValue('=SUM(A1:A3)'), '=SUM(A1:A3)');
  assert.equal(projectResponseValue(504), 504);
  assert.equal(projectResponseValue('plain'), 'plain');
  assert.equal(projectResponseValue(true), 'TRUE');
  assert.equal(projectResponseValue('TRUE'), 'TRUE');
  assert.equal(projectResponseValue(''), null);
  assert.equal(projectResponseValue(undefined), null);
});

const block = (cells: (string | number | boolean | null)[][]) => ({
  sheetTitle: 'Probe', startRow: 5, startColumn: 2,
  rows: cells.length, columns: cells[0].length, cells,
});

test('fingerprintOf is deterministic and carries the v1 prefix', () => {
  const a = fingerprintOf([block([['=A1', 1], [null, 'x']])]);
  const b = fingerprintOf([block([['=A1', 1], [null, 'x']])]);
  assert.equal(a, b);
  assert.match(a, /^v1:[0-9a-f]{64}$/);
});

test('fingerprintOf separates an empty cell from one holding an empty string', () => {
  const withEmpty = fingerprintOf([block([['=A1', null]])]);
  const withText  = fingerprintOf([block([['=A1', '']])]);
  assert.notEqual(withEmpty, withText);
});

test('fingerprintOf reacts to the block dimensions, not just the cells', () => {
  const tall = fingerprintOf([{ ...block([['a'], ['b']]), rows: 2, columns: 1 }]);
  const wide = fingerprintOf([{ ...block([['a', 'b']]), rows: 1, columns: 2 }]);
  assert.notEqual(tall, wide);
});

test('fingerprintOf reacts to the block origin, so a shifted range is a different block', () => {
  const here  = fingerprintOf([block([['a']])]);
  const there = fingerprintOf([{ ...block([['a']]), startRow: 6 }]);
  assert.notEqual(here, there);
});

test('the fingerprint ignores everything except what the user entered', () => {
  // Structural, but it is the whole premise: a recalculating model must not
  // invalidate a guard, and neither must a repaint.
  const plain = projectCellData({ formulaValue: '=A1' });
  const noisy = projectCellData({ formulaValue: '=A1', effectiveValue: { numberValue: 99 },
    formattedValue: '99', userEnteredFormat: { backgroundColor: { red: 1 } } } as Record<string, unknown>);
  assert.equal(plain, noisy);
  assert.equal(
    fingerprintOf([block([[plain]])]),
    fingerprintOf([block([[noisy]])]),
  );
});

test('several guard ranges hash to ONE fingerprint, order-sensitively', () => {
  const one = block([['a']]);
  const two = { ...block([['b']]), sheetTitle: 'Probe 2' };
  assert.notEqual(fingerprintOf([one, two]), fingerprintOf([two, one]));
});

test('fingerprintOf regression: a sheet title cannot forge another block\'s newline-joined encoding', () => {
  // Two genuinely different blocks...
  const twoBlocks = [
    { sheetTitle: 'Foo', startRow: 1, startColumn: 1, rows: 1, columns: 1, cells: [[1]] },
    { sheetTitle: 'Bar', startRow: 99, startColumn: 99, rows: 0, columns: 0, cells: [] },
  ];
  // ...versus one crafted block whose title embeds the exact newline-joined
  // lines the pair above would have produced under raw (unescaped)
  // interpolation of the sheet title. Before the fix these hashed identically:
  // a sheet renamed to contain a newline could make an edited sheet's
  // fingerprint equal an unrelated, unedited one.
  const crafted = [
    { sheetTitle: 'Foo!1,1+1x1\n0,0=1\nBar', startRow: 99, startColumn: 99, rows: 0, columns: 0, cells: [] },
  ];
  assert.notEqual(fingerprintOf(twoBlocks), fingerprintOf(crafted));
});

test('fingerprintOf regression: a title cannot forge the empty-cell marker either', () => {
  // A real empty cell inside one block, versus another block whose actual
  // cells are NOT empty but whose title text spells out what the first
  // block's header+marker line would look like when joined. Before the fix
  // these collided for the same reason as above.
  const withRealEmptyCell = [
    { sheetTitle: 'Foo', startRow: 1, startColumn: 1, rows: 1, columns: 1, cells: [[null]] },
    { sheetTitle: 'Bar', startRow: 9, startColumn: 9, rows: 0, columns: 0, cells: [] },
  ];
  const titleSpellsOutTheMarker = [
    { sheetTitle: 'Foo!1,1+1x1\n0,0=<empty>\nBar', startRow: 9, startColumn: 9, rows: 0, columns: 0, cells: [] },
  ];
  assert.notEqual(fingerprintOf(withRealEmptyCell), fingerprintOf(titleSpellsOutTheMarker));
});

test('fingerprintOf handles a title containing "!" as ordinary text', () => {
  const withBang = { ...block([['x']]), sheetTitle: 'Sheet!Q1' };
  const withoutBang = block([['x']]);
  const a = fingerprintOf([withBang]);
  assert.notEqual(a, fingerprintOf([withoutBang]));
  assert.equal(a, fingerprintOf([{ ...block([['x']]), sheetTitle: 'Sheet!Q1' }]));  // deterministic
});

test('projectCellData and projectResponseValue agree on text that merely reads "true"/"false"', () => {
  assert.equal(projectCellData({ stringValue: 'true' }), 'true');
  assert.equal(projectResponseValue('true'), 'true');
  assert.equal(projectCellData({ stringValue: 'True' }), 'True');
  assert.equal(projectResponseValue('True'), 'True');
  assert.equal(projectCellData({ stringValue: 'false' }), 'false');
  assert.equal(projectResponseValue('false'), 'false');
});

test('renderForWrite turns canonical cells back into USER_ENTERED input', () => {
  assert.equal(renderForWrite('=SUM(A1:A3)'), '=SUM(A1:A3)');
  assert.equal(renderForWrite(504), '504');
  assert.equal(renderForWrite('plain'), 'plain');
  assert.equal(renderForWrite('TRUE'), 'TRUE');
  assert.equal(renderForWrite(null), '');   // clears the cell
});

test('buildPreImage produces a payload the tool can be re-fed for rollback', () => {
  const pre = buildPreImage("'Entity Assumptions'!C52:D52", [['=B52*1.1', 0.12]]);
  assert.deepEqual(pre, { range: "'Entity Assumptions'!C52:D52", values: [['=B52*1.1', '0.12']] });
});

test('findHazards flags text that would come back as a formula, with its address', () => {
  const raw = [
    [{ formulaValue: '=SUM(A1:A3)' }, { stringValue: '=notaformula' }],
    [{ stringValue: 'plain' }, undefined],
  ];
  const hazards = findHazards(0, 0, raw);
  assert.equal(hazards.length, 1);
  assert.equal(hazards[0].a1, 'B1');
  assert.equal(hazards[0].reason, 'text-looks-like-formula');
  assert.deepEqual(hazards[0].userEnteredValue, { stringValue: '=notaformula' });
});

test('findHazards reports addresses against the grid origin, not the block', () => {
  const raw = [[{ stringValue: '=x' }]];
  assert.equal(findHazards(51, 2, raw)[0].a1, 'C52');
});

test('a real formula is never a hazard', () => {
  assert.deepEqual(findHazards(0, 0, [[{ formulaValue: '=A2' }]]), []);
});

test('findHazards flags text that would come back as a number', () => {
  const hazards = findHazards(0, 0, [[{ stringValue: '504' }]]);
  assert.equal(hazards.length, 1);
  assert.equal(hazards[0].a1, 'A1');
  assert.equal(hazards[0].reason, 'text-looks-like-number');
});

test('findHazards flags text that would come back as a boolean, case-insensitively', () => {
  const hazards = findHazards(0, 0, [[{ stringValue: 'TRUE' }, { stringValue: 'false' }]]);
  assert.equal(hazards.length, 2);
  assert.equal(hazards[0].a1, 'A1');
  assert.equal(hazards[0].reason, 'text-looks-like-boolean');
  assert.equal(hazards[1].a1, 'B1');
  assert.equal(hazards[1].reason, 'text-looks-like-boolean');
});

test('findHazards does not mistake blank text for a number', () => {
  assert.deepEqual(findHazards(0, 0, [[{ stringValue: '   ' }]]), []);
});

test('a real number cell is never a hazard', () => {
  assert.deepEqual(findHazards(0, 0, [[{ numberValue: 504 }]]), []);
});

test('guardRangesFor defaults to exactly the ranges being written, in order', () => {
  const updates = [{ range: 'Probe!A1:B2' }, { range: "'Probe 2'!C1" }];
  assert.deepEqual(guardRangesFor(updates), ['Probe!A1:B2', "'Probe 2'!C1"]);
});

test('guardRangesFor keeps a declared guard set verbatim, duplicates included', () => {
  const updates = [{ range: 'Probe!A1' }];
  // Deliberate: the read must request exactly what the fingerprint is computed
  // over, so de-duplicating here would make the hash irreproducible.
  assert.deepEqual(guardRangesFor(updates, ['Probe!A1:H90', 'Probe!A1:H90']),
    ['Probe!A1:H90', 'Probe!A1:H90']);
});

test('blockFromMatch projects a grid into one canonical block and keeps the raw cells', () => {
  const match = {
    range: 'Probe!C6:D7', sheetId: 0, sheetTitle: 'Probe', sheet: {},
    grid: {
      startRow: 5, startColumn: 2,
      rowData: [
        { values: [{ userEnteredValue: { formulaValue: '=A1' } }, { userEnteredValue: { numberValue: 2 } }] },
        { values: [{ userEnteredValue: { stringValue: 'x' } }] },
      ],
    },
  } as any;
  const { block, raw } = blockFromMatch(match);
  assert.equal(block.sheetTitle, 'Probe');
  assert.equal(block.startRow, 5);
  assert.equal(block.startColumn, 2);
  assert.equal(block.rows, 2);
  assert.equal(block.columns, 2);            // widest row wins, short rows pad
  assert.deepEqual(block.cells, [['=A1', 2], ['x', null]]);
  assert.equal(raw[1][1], undefined);
});

test('blockFromMatch treats an absent offset as zero, per proto3', () => {
  const match = { range: 'Probe!A1', sheetId: 0, sheetTitle: 'Probe', sheet: {},
    grid: { rowData: [{ values: [{ userEnteredValue: { numberValue: 1 } }] }] } } as any;
  const { block } = blockFromMatch(match);
  assert.equal(block.startRow, 0);
  assert.equal(block.startColumn, 0);
});

test('blockFromMatch on an entirely empty range pads out to the DECLARED rectangle, not a zero-sized block', () => {
  // Google omits rowData entirely for an all-empty range, but the declared
  // range Z90:AA95 is still 6 rows x 2 columns: a block sized to the
  // OBSERVED (here: zero) extent
  // under-covers the guard, and a pre-image built from it would restore
  // nothing while reporting success.
  const match = { range: 'Probe!Z90:AA95', sheetId: 0, sheetTitle: 'Probe', sheet: {},
    grid: { startRow: 89, startColumn: 25 } } as any;
  const { block } = blockFromMatch(match);
  assert.equal(block.startRow, 89);
  assert.equal(block.startColumn, 25);
  assert.equal(block.rows, 6);
  assert.equal(block.columns, 2);
  assert.deepEqual(block.cells, [
    [null, null], [null, null], [null, null], [null, null], [null, null], [null, null],
  ]);
});

test('padToDeclaredRange pads an under-covering observed block out to the declared rectangle, missing cells as null', () => {
  const block = padToDeclaredRange('Probe!A1:B2', 'Probe', [['x']], 0, 0);
  assert.deepEqual(block, {
    sheetTitle: 'Probe', startRow: 0, startColumn: 0, rows: 2, columns: 2,
    cells: [['x', null], [null, null]],
  });
});

test('padToDeclaredRange is a no-op in shape when the observed block already fills the declared rectangle', () => {
  const block = padToDeclaredRange('Probe!A1:B1', 'Probe', [['x', 'y']], 0, 0);
  assert.deepEqual(block, {
    sheetTitle: 'Probe', startRow: 0, startColumn: 0, rows: 1, columns: 2,
    cells: [['x', 'y']],
  });
});

test('padToDeclaredRange takes its origin from the declared range, not from the caller-supplied fallback, when bounded', () => {
  // C6:D7 -> row 5, column 2 (0-based) is the declared origin; the fallback
  // arguments here are deliberately wrong, to prove they are ignored whenever
  // a bounded rectangle exists.
  const block = padToDeclaredRange('Probe!C6:D7', 'Probe', [['x', 'y'], ['z']], 999, 999);
  assert.equal(block.startRow, 5);
  assert.equal(block.startColumn, 2);
  assert.deepEqual(block.cells, [['x', 'y'], ['z', null]]);
});

test('padToDeclaredRange falls back to the observed extent for an open-ended range (no declared rectangle)', () => {
  const block = padToDeclaredRange('Probe!A:C', 'Probe', [['x', 'y']], 4, 0);
  assert.equal(block.startRow, 4);
  assert.equal(block.startColumn, 0);
  assert.equal(block.rows, 1);
  assert.equal(block.columns, 2);
  assert.deepEqual(block.cells, [['x', 'y']]);
});

test('padToDeclaredRange falls back to the observed extent for a bare whole-sheet reference', () => {
  const block = padToDeclaredRange('Probe', 'Probe', [['x']], 7, 1);
  assert.equal(block.startRow, 7);
  assert.equal(block.startColumn, 1);
  assert.equal(block.rows, 1);
  assert.equal(block.columns, 1);
  assert.deepEqual(block.cells, [['x']]);
});

// ---------------------------------------------------------------------------
// convertA1ToGridRange FABRICATES a missing end row (defaulting it
// to startRow+1, as though a lone anchor were a 1-cell range), so "A1:C" and
// "A5:A" were previously treated as bounded 1-row rectangles instead of
// open-ended ranges. That silently discarded every row past the first out of
// a guard fingerprint and a rollback pre-image alike. declaredRectangle now
// has its own strict parser and must report NO rectangle for these forms,
// so padToDeclaredRange falls back to the observed extent and keeps every
// observed row.
// ---------------------------------------------------------------------------

test('padToDeclaredRange keeps every observed row for "A1:C" (open row end) instead of fabricating a 1-row rectangle', () => {
  const observed = [['a1', 'b1', 'c1'], ['a2', 'b2', 'c2'], ['a3', 'b3', 'c3']];
  const block = padToDeclaredRange('Probe!A1:C', 'Probe', observed, 0, 0);
  assert.equal(block.rows, 3, 'all three observed rows must survive, not just the first');
  assert.equal(block.columns, 3);
  assert.deepEqual(block.cells, observed);
});

test('padToDeclaredRange keeps every observed row for "A5:A" (open row end on a single column) instead of fabricating a 1-row rectangle', () => {
  const observed = [['x'], ['y'], ['z']];
  const block = padToDeclaredRange('Probe!A5:A', 'Probe', observed, 4, 0);
  assert.equal(block.startRow, 4);
  assert.equal(block.rows, 3);
  assert.equal(block.columns, 1);
  assert.deepEqual(block.cells, observed);
});

test('padToDeclaredRange still falls back to the observed extent for "A:C" and "1:3" (genuinely open ranges)', () => {
  const byColumn = padToDeclaredRange('Probe!A:C', 'Probe', [['a', 'b']], 0, 0);
  assert.equal(byColumn.rows, 1);
  assert.equal(byColumn.columns, 2);

  const byRow = padToDeclaredRange('Probe!1:3', 'Probe', [['a', 'b', 'c']], 0, 0);
  assert.equal(byRow.rows, 1);
  assert.equal(byRow.columns, 3);
});

// ---------------------------------------------------------------------------
// convertA1ToGridRange's regex is uppercase-only and does not strip '$', so
// a lowercase or absolute-reference spelling of a perfectly bounded range
// threw and fell back to the (trimming-dependent) observed extent - exactly
// the geometry padding was added to eliminate.
// declaredRectangle now normalizes ($ stripped, upper-cased) before parsing,
// matching collectSheetMetadata's existing precedent.
// ---------------------------------------------------------------------------

test('padToDeclaredRange recognizes a lowercase declared range as the same bounded rectangle as its uppercase spelling', () => {
  const observed = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']];
  const lower = padToDeclaredRange('Probe!a1:c3', 'Probe', observed, 0, 0);
  const upper = padToDeclaredRange('Probe!A1:C3', 'Probe', observed, 0, 0);
  assert.deepEqual(lower, upper);
  assert.equal(lower.rows, 3);
  assert.equal(lower.columns, 3);
});

test('padToDeclaredRange recognizes a $-absolute declared range as the same bounded rectangle as its plain spelling', () => {
  const observed = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']];
  const dollared = padToDeclaredRange('Probe!A$1:C$3', 'Probe', observed, 0, 0);
  const plain = padToDeclaredRange('Probe!A1:C3', 'Probe', observed, 0, 0);
  assert.deepEqual(dollared, plain);
  assert.equal(dollared.rows, 3);
  assert.equal(dollared.columns, 3);
});

// ---------------------------------------------------------------------------
// The declared rectangle must be capped so that an absurd declared area is
// refused with a clear error instead of being materialized - `Probe!A1:Z100000`
// is twelve characters of input for 2.6 million cells, almost all of which
// Google trims away in the read but which padToDeclaredRange would otherwise
// allocate in full.
// ---------------------------------------------------------------------------

test('padToDeclaredRange refuses an absurdly large declared rectangle instead of materializing it', () => {
  assert.throws(
    () => padToDeclaredRange('Probe!A1:Z100000', 'Probe', [['x']], 0, 0),
    /exceeds the 100000-cell/,
  );
});

test('padToDeclaredRange accepts a large-but-ordinary declared rectangle right at the cap', () => {
  // 1000 rows x 100 columns = 100,000 cells - at the cap, must not throw.
  const block = padToDeclaredRange('Probe!A1:CV1000', 'Probe', [['x']], 0, 0);
  assert.equal(block.rows, 1000);
  assert.equal(block.columns, 100);
});

// ---------------------------------------------------------------------------
// Google reads "C3:A1" as "A1:C3", so a reversed range comes back from a write
// as a bounded, ascending rectangle. Dropping it here as "no rectangle" would
// leave the guard side on the observed extent while the write echo sat on the
// declared one - the same range with two geometries, which is exactly what the
// padding exists to prevent. The two anchors are corners, not a start and an
// end, so they are normalized by min/max just as spelling is normalized above.
// ---------------------------------------------------------------------------

// Each of these observes a single cell where the declared rectangle is 3x3:
// treated as no rectangle, the block would stay the 1x1 observed extent, so
// the padding is what is actually being pinned, not an accidental agreement
// between two equally-sized blocks.
test('padToDeclaredRange normalizes fully reversed anchors ("C3:A1") to the same rectangle as "A1:C3"', () => {
  const reversed = padToDeclaredRange('Probe!C3:A1', 'Probe', [['x']], 0, 0);
  assert.deepEqual(reversed, padToDeclaredRange('Probe!A1:C3', 'Probe', [['x']], 0, 0));
  assert.equal(reversed.startRow, 0);
  assert.equal(reversed.startColumn, 0);
  assert.equal(reversed.rows, 3);
  assert.equal(reversed.columns, 3);
  assert.deepEqual(reversed.cells[2], [null, null, null]);
});

test('padToDeclaredRange normalizes a row-reversed range ("A3:C1") to the same rectangle as "A1:C3"', () => {
  const rowReversed = padToDeclaredRange('Probe!A3:C1', 'Probe', [['x']], 0, 0);
  assert.deepEqual(rowReversed, padToDeclaredRange('Probe!A1:C3', 'Probe', [['x']], 0, 0));
  assert.equal(rowReversed.rows, 3);
  assert.equal(rowReversed.columns, 3);
});

test('padToDeclaredRange normalizes a column-reversed range ("C1:A3") to the same rectangle as "A1:C3"', () => {
  const colReversed = padToDeclaredRange('Probe!C1:A3', 'Probe', [['x']], 0, 0);
  assert.deepEqual(colReversed, padToDeclaredRange('Probe!A1:C3', 'Probe', [['x']], 0, 0));
  assert.equal(colReversed.startColumn, 0);
  assert.equal(colReversed.columns, 3);
});

test('a reversed range away from the origin takes its origin from the upper-left corner, not from the fallback', () => {
  // The fallback anchors the block wherever the observation says; the declared
  // rectangle of "C5:A3" starts at A3 whichever way round it was written.
  const block = padToDeclaredRange('Probe!C5:A3', 'Probe', [['x']], 0, 0);
  assert.equal(block.startRow, 2, 'row 3, zero-based');
  assert.equal(block.startColumn, 0);
  assert.equal(block.rows, 3);
  assert.equal(block.columns, 3);
});

// ---------------------------------------------------------------------------
// Write ranges, unlike guard ranges, must name a bounded rectangle: preImage
// covers the declared rectangle and postFingerprint is taken over it, so an
// open-ended write range yields a pre-image trimmed to whatever data was there
// before the write and a fingerprint no later guard read can reproduce.
// ---------------------------------------------------------------------------

test('checkWriteRanges accepts bounded write ranges, in every spelling declaredRectangle understands', () => {
  assert.equal(checkWriteRanges(['Probe!A1']), null);
  assert.equal(checkWriteRanges(['Probe!A2:C50', "'Entity Assumptions'!C52:E52"]), null);
  assert.equal(checkWriteRanges(['Probe!a$1:c$3', 'Probe!C3:A1', 'A1:B2']), null);
});

test('checkWriteRanges refuses every open-ended write range, naming it and asking for a narrower one', () => {
  for (const range of ['Probe!A2:C', 'Probe!A5:A', 'Probe!A:C', 'Probe!5:9', 'A:C', '5:9', 'Probe']) {
    const message = checkWriteRanges([range]);
    assert.ok(message, `"${range}" must be refused as a write range`);
    assert.match(message, /does not name a bounded rectangle/);
    assert.ok(message.includes(`"${range}"`), 'the message must name the offending range');
    assert.match(message, /Narrow the range/);
  }
});

test('checkWriteRanges reports the first offending range out of several, and passes a wholly bounded set', () => {
  const message = checkWriteRanges(['Probe!A1:B2', 'Probe!D2:F', 'Probe!H1']);
  assert.ok(message);
  assert.ok(message.includes('"Probe!D2:F"'));
  assert.equal(checkWriteRanges(['Probe!A1:B2', 'Probe!D2:F9', 'Probe!H1']), null);
});

test('checkWriteRanges surfaces the cell cap for a bounded but absurd write range, rather than throwing', () => {
  const message = checkWriteRanges(['Probe!A1:Z100000']);
  assert.ok(message);
  assert.match(message, /exceeds the 100000-cell/);
});
