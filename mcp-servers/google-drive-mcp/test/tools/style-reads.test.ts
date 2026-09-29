import assert from 'node:assert/strict';
import test from 'node:test';

import { describeRangeStyles, summarizeDocumentStyles } from '../../src/tools/styleProjection.js';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function para(startIndex: number, endIndex: number, text: string, paragraphStyle: any = {}, textStyle: any = {}) {
  return {
    startIndex,
    endIndex,
    paragraph: {
      paragraphStyle,
      elements: [{ startIndex, endIndex, textRun: { content: text, textStyle } }],
    },
  };
}

const SOLID_RULE = { width: { magnitude: 1.5, unit: 'PT' }, dashStyle: 'SOLID', color: { color: { rgbColor: { red: 0.8, green: 0, blue: 0 } } } };
// Docs stamps these on most paragraphs; only a positive width renders.
const EMPTY_BORDER = { width: { magnitude: 0, unit: 'PT' } };

const bodyContent = [
  para(1, 16, 'Quarterly plan\n', { namedStyleType: 'TITLE' }, { weightedFontFamily: { fontFamily: 'Georgia' }, fontSize: { magnitude: 20 } }),
  para(16, 30, 'Revenue\n', { namedStyleType: 'HEADING_1', borderBottom: SOLID_RULE }, { weightedFontFamily: { fontFamily: 'Georgia' }, fontSize: { magnitude: 14 } }),
  para(30, 52, 'Body copy here.\n', { namedStyleType: 'NORMAL_TEXT', borderTop: EMPTY_BORDER }, { weightedFontFamily: { fontFamily: 'Arial' }, fontSize: { magnitude: 11 } }),
  {
    startIndex: 52,
    endIndex: 80,
    table: {
      tableRows: [
        { tableCells: [{ content: [para(54, 62, 'Header A\n', { namedStyleType: 'NORMAL_TEXT', alignment: 'CENTER' })] }, { content: [para(63, 71, 'Header B\n')] }] },
      ],
    },
  },
];

// ---------------------------------------------------------------------------
// summarizeDocumentStyles
// ---------------------------------------------------------------------------

test('summary reports counts, fonts, borders and the table span from one body', () => {
  const out = summarizeDocumentStyles([{ content: bodyContent }]);

  // Five paragraphs: three at body level plus one in each of the two cells,
  // since the walk recurses into table cells.
  assert.match(out, /^paragraphs: 5 /m);
  assert.match(out, /1x TITLE/);
  assert.match(out, /1x HEADING_1/);
  // Fonts sort by character volume, and sizes form a ladder per family.
  assert.match(out, /fonts: Georgia \(14\/20pt, \d+ chars\); Arial \(11pt, \d+ chars\)/);
  // Only the positive-width border counts; the width-0 one at index 30 does not.
  assert.equal(out.includes('bordered paragraphs: 1 (at 16)'), true);
  assert.match(out, /tables: 1 - \[52-80\] 1x2/);
  assert.match(out, /outline:\n {2}\[1\] TITLE "Quarterly plan"\n {2}\[16\] HEADING_1 "Revenue"/);
});

test('summary does not print shading for a shading object that carries no color', () => {
  const out = summarizeDocumentStyles([{
    content: [para(1, 10, 'x\n', { shading: { backgroundColor: {} } })],
  }]);
  assert.match(out, /shaded paragraphs: 0/);
  assert.equal(out.includes('null'), false);
});

test('summary prefixes every location with the tab title on a multi-tab document', () => {
  // Index spaces restart in each tab, so both tabs hold a bordered paragraph at
  // 16 and a table at 52. Bare numbers would be indistinguishable.
  const out = summarizeDocumentStyles([
    { label: 'Overview', content: bodyContent },
    { label: 'Appendix', content: bodyContent },
  ]);

  assert.match(out, /bordered paragraphs: 2 \(at Overview:16, Appendix:16\)/);
  assert.match(out, /tables: 2 - \[Overview:52-80\] 1x2, \[Appendix:52-80\] 1x2/);
  assert.match(out, /\[Overview:1\] TITLE/);
  assert.match(out, /\[Appendix:1\] TITLE/);
  // Counts still aggregate across tabs.
  assert.match(out, /^paragraphs: 10 /m);
});

test('summary truncates every long list with an explicit remainder count', () => {
  const many = Array.from({ length: 30 }, (_, i) =>
    para(1 + i * 10, 10 + i * 10, `Heading ${i}\n`, { namedStyleType: 'HEADING_2', borderTop: SOLID_RULE }));
  const out = summarizeDocumentStyles([{ content: many }]);

  // 12 shown for the location lists, 20 for the outline, each saying how many
  // it held back — a silent cut reads exactly like a short document.
  assert.match(out, /bordered paragraphs: 30 \(at 1, 11, .*, 111, … \(\+18 more\)\)/);
  assert.equal((out.match(/^ {2}\[\d+\] HEADING_2/gm) ?? []).length, 20);
  assert.match(out, /… \(\+10 more\)/);
});

// ---------------------------------------------------------------------------
// describeRangeStyles
// ---------------------------------------------------------------------------

test('describe reports the paragraph and only the runs overlapping the range', () => {
  const out = describeRangeStyles(bodyContent, 16, 17);
  assert.match(out, /paragraph \[16-30\]: HEADING_1, borderBottom\(#cc0000 1\.5PT SOLID\)/);
  assert.match(out, /run \[16-30\] "Revenue⏎" - Georgia, 14pt/);
  // The title and body paragraphs do not overlap [16,17).
  assert.equal(out.includes('TITLE'), false);
  assert.equal(out.includes('Body copy'), false);
});

test('describe always names the style, including NORMAL_TEXT, and flags in-table', () => {
  const out = describeRangeStyles(bodyContent, 54, 55);
  assert.match(out, /paragraph \[54-62\]: NORMAL_TEXT, center, in-table/);
});

test('describe appends indent, spacing and bullet bits after the shared style bits', () => {
  const out = describeRangeStyles([
    para(1, 10, 'item\n', {
      namedStyleType: 'NORMAL_TEXT',
      indentStart: { magnitude: 36, unit: 'PT' },
      spaceAbove: { magnitude: 6, unit: 'PT' },
    }),
  ], 1, 2);
  assert.match(out, /indentStart=36PT/);
  assert.match(out, /spaceAbove=6PT/);
});

test('describe reports text-run styling flags, colors and links', () => {
  const out = describeRangeStyles([
    para(1, 10, 'linked\n', {}, {
      bold: true,
      italic: true,
      foregroundColor: { color: { rgbColor: { red: 0, green: 0, blue: 1 } } },
      link: { url: 'https://example.com' },
    }),
  ], 1, 5);
  assert.match(out, /run \[1-10\] "linked⏎" - bold, italic, #0000ff, link https:\/\/example\.com/);
});

test('describe says so plainly when nothing overlaps', () => {
  assert.equal(describeRangeStyles(bodyContent, 900, 901), 'no paragraphs overlap [900-901)');
});

test('describe tolerates a body with no content', () => {
  assert.equal(describeRangeStyles(undefined, 1, 2), 'no paragraphs overlap [1-2)');
});
