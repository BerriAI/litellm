import assert from 'node:assert/strict';
import test from 'node:test';

import { buildDocFormattedContent } from '../../src/tools/docs.js';

function run(content: string, startIndex: number) {
  return { textRun: { content }, startIndex, endIndex: startIndex + content.length };
}
function tableEl(startIndex: number, endIndex: number, cellTexts: string[][], cellStart: number) {
  let idx = cellStart;
  return {
    startIndex,
    endIndex,
    table: {
      tableRows: cellTexts.map((row) => ({
        tableCells: row.map((text) => {
          const cell = { content: [{ paragraph: { elements: [run(text + '\n', idx)] } }] };
          idx += text.length + 2;
          return cell;
        }),
      })),
    },
  };
}

// Two tables. The old code fabricated per-row ranges from markdown line
// lengths, starting at each table's own real start index and running FORWARD
// past its end into whatever followed — colliding with the content after the
// table, which may itself be a later table. They never reached backward.
const doc = {
  body: {
    content: [
      { paragraph: { elements: [run('Intro\n', 1)] } },
      tableEl(7, 20, [['alpha', 'beta']], 9),
      { paragraph: { elements: [run('After\n', 20)] } },
      tableEl(26, 40, [['gamma', 'delta']], 28),
    ],
  },
};

test('tables emit exactly one real-span line each; no fabricated per-row ranges', () => {
  const { formattedContent, totalLength } = buildDocFormattedContent(doc, false);
  const bracketLines = formattedContent.split('\n').filter((l) => /^\[\d+-\d+\]/.test(l));

  const tableLines = bracketLines.filter((l) => l.includes('<table'));
  assert.equal(tableLines.length, 2);
  assert.match(tableLines[0], /^\[7-20\]/);
  assert.match(tableLines[1], /^\[26-40\]/);

  // Every emitted range must be a real doc range: within [1, 90] and, for
  // non-table lines, derived from paragraph indices. The spans here are
  // deliberately tighter than the markdown rendering, so the old
  // length-arithmetic ranges run past the document end and fail this loop.
  for (const line of bracketLines) {
    const m = line.match(/^\[(\d+)-(\d+)\]/)!;
    const [s, e] = [Number(m[1]), Number(m[2])];
    assert.ok(s >= 1 && e <= 40, `range [${s}-${e}] escapes the document`);
  }

  // The rendering is still present for humans.
  assert.match(formattedContent, /alpha.*beta/s);
  assert.equal(totalLength, 40);
});

test('a table prints its cells\' real index ranges so text inside a cell stays targetable', () => {
  const { formattedContent } = buildDocFormattedContent(doc, false);

  // Dropping the fabricated per-row ranges must not drop every index for text
  // inside a table: applyTextStyle/formatGoogleDocText need a range, and
  // editTableCell can only style a whole cell.
  const cellMaps = formattedContent.split('\n').filter((l) => l.startsWith('cells:'));
  assert.equal(cellMaps.length, 2, 'one cell map per table');

  // Fixture: table 1's cells start at 9 ('alpha\n') and 16 ('beta\n').
  assert.equal(cellMaps[0], 'cells: r0c0 [9-15], r0c1 [16-21]');
  assert.equal(cellMaps[1], 'cells: r0c0 [28-34], r0c1 [35-41]');
});

test('on a multi-tab doc the table hint names tabId, and the tab header supplies it', () => {
  // Index spaces restart per tab, so two tabs can hold a table at the same
  // start index. A hint naming only tableStartIndex sends editTableCell — which
  // searches the first tab when given no tabId — at the wrong table.
  const twoTabs = {
    tabs: [
      {
        tabProperties: { tabId: 'tab-a', title: 'Alpha' },
        documentTab: { body: { content: [tableEl(7, 20, [['one', 'two']], 9)] } },
      },
      {
        tabProperties: { tabId: 'tab-b', title: 'Beta' },
        documentTab: { body: { content: [tableEl(7, 20, [['three', 'four']], 9)] } },
      },
    ],
  };

  const { formattedContent } = buildDocFormattedContent(twoTabs, false);
  const hints = formattedContent.split('\n').filter((l) => l.includes('<table'));

  assert.equal(hints.length, 2);
  assert.match(hints[0], /tableStartIndex=7, tabId=tab-a/);
  assert.match(hints[1], /tableStartIndex=7, tabId=tab-b/);

  // The hint must name a value the caller can actually obtain.
  assert.match(formattedContent, /=== Tab: Alpha \(tabId=tab-a\) ===/);
  assert.match(formattedContent, /=== Tab: Beta \(tabId=tab-b\) ===/);
});
