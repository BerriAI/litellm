import assert from 'node:assert/strict';
import test from 'node:test';

import { collectDocPlainText, countOccurrences, diagnoseZeroMatch, findOccurrenceRanges } from '../../src/tools/findDiagnostics.js';

function para(text: string) {
  return { paragraph: { elements: [{ textRun: { content: text } }] } };
}

const fixtureDoc = {
  body: {
    content: [
      para('Hello world\n'),
      {
        table: {
          tableRows: [
            { tableCells: [{ content: [para('cell world\n')] }, { content: [para('fee 500\n')] }] },
            { tableCells: [{ content: [para('total 500\n')] }] },
          ],
        },
      },
    ],
  },
  headers: { h1: { content: [para('header world\n')] } },
  footers: { f1: { content: [para('page footer\n')] } },
};

const tabbedDoc = {
  tabs: [
    {
      tabProperties: { tabId: 't.0' },
      documentTab: { body: { content: [para('alpha target\n')] } },
      childTabs: [
        {
          tabProperties: { tabId: 't.1' },
          documentTab: { body: { content: [para('beta target\n')] } },
        },
      ],
    },
  ],
};

test('collectDocPlainText walks body, nested tables, headers, and footers', () => {
  const text = collectDocPlainText(fixtureDoc);
  assert.equal(countOccurrences(text, 'world', true), 3);
  assert.equal(countOccurrences(text, '500', true), 2);
  assert.equal(countOccurrences(text, 'page footer', true), 1);
});

test('collectDocPlainText scopes to one tab and reaches child tabs', () => {
  assert.equal(countOccurrences(collectDocPlainText(tabbedDoc), 'target', true), 2);
  assert.equal(countOccurrences(collectDocPlainText(tabbedDoc, 't.1'), 'target', true), 1);
  assert.equal(countOccurrences(collectDocPlainText(tabbedDoc, 't.1'), 'alpha', true), 0);
});

test('countOccurrences is literal (no regex) and honors matchCase', () => {
  assert.equal(countOccurrences('a.b a.b aXb', 'a.b', true), 2);
  assert.equal(countOccurrences('Foo foo FOO', 'foo', false), 3);
  assert.equal(countOccurrences('Foo foo FOO', 'foo', true), 1);
  assert.equal(countOccurrences('anything', '', true), 0);
});

test('diagnoseZeroMatch names non-breaking spaces (letterspaced wordmarks)', () => {
  const hint = diagnoseZeroMatch('S & E wordmark', 'S & E', true);
  assert.match(hint ?? '', /U\+00A0/);
});

test('diagnoseZeroMatch names curly-quote mismatches', () => {
  const hint = diagnoseZeroMatch('it’s here', "it's here", true);
  assert.match(hint ?? '', /curly quotes/);
});

test('diagnoseZeroMatch spots pre-existing &amp; entities', () => {
  const hint = diagnoseZeroMatch('Production &amp; Publishing', 'Production & Publishing', true);
  assert.match(hint ?? '', /&amp;/);
});

test('diagnoseZeroMatch suggests matchCase relaxation, and stays null with no lookalikes', () => {
  assert.match(diagnoseZeroMatch('HELLO', 'hello', true) ?? '', /case-insensitive/);
  assert.equal(diagnoseZeroMatch('completely different', 'missing text', false), null);
});

// ---------------------------------------------------------------------------
// findOccurrenceRanges — exact doc-index ranges
// ---------------------------------------------------------------------------

/** A paragraph whose first text run starts at `startIndex` in the doc index space. */
function indexedPara(text: string, startIndex: number) {
  return { paragraph: { elements: [{ startIndex, textRun: { content: text } }] } };
}

test('findOccurrenceRanges maps case-insensitive matches through a length-changing lowercase', () => {
  // U+0130 (LATIN CAPITAL LETTER I WITH DOT ABOVE) lowercases to two UTF-16
  // units, so a lowercased haystack is longer than the source text. The match
  // position must be mapped back to the SOURCE offset, not used directly.
  assert.equal('İ'.toLowerCase().length, 2, 'precondition: U+0130 expands when lowercased');

  const doc = { body: { content: [indexedPara('İtarget', 9)] } };
  const ranges = findOccurrenceRanges(doc, 'TARGET', false);

  assert.equal(ranges.length, 1);
  assert.equal(ranges[0].startIndex, 10, 'startIndex must point at the "t" of target');
  assert.equal(ranges[0].endIndex, 16, 'endIndex must be one past the final "t"');
});

test('findOccurrenceRanges is exact for plain ASCII case-insensitive matches', () => {
  const doc = { body: { content: [indexedPara('Hello World\n', 1)] } };
  const ranges = findOccurrenceRanges(doc, 'world', false);

  assert.deepEqual(ranges, [{ startIndex: 7, endIndex: 12 }]);
});
