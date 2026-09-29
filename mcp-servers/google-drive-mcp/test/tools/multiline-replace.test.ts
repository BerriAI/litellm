import assert from 'node:assert/strict';
import test from 'node:test';

import { buildMultilineReplaceRequests, findOccurrenceRanges } from '../../src/tools/findDiagnostics.js';

function run(content: string, startIndex: number) {
  return { textRun: { content }, startIndex, endIndex: startIndex + content.length };
}

const doc = {
  body: {
    content: [
      // "Hello world here" split across two runs mid-word (a bold split)
      { paragraph: { elements: [run('Hello wo', 1), run('rld here\n', 9)] } },
      {
        table: {
          tableRows: [{ tableCells: [{ content: [{ paragraph: { elements: [run('cell target\n', 25)] } }] }] }],
        },
      },
      { paragraph: { elements: [run('another target\n', 50)] } },
    ],
  },
  headers: { h9: { headerId: 'h9', content: [{ paragraph: { elements: [run('head target\n', 0)] } }] } },
};

test('findOccurrenceRanges maps matches that span split runs to exact doc indices', () => {
  const ranges = findOccurrenceRanges(doc, 'world', true);
  assert.deepEqual(ranges, [{ startIndex: 7, endIndex: 12 }]);
});

test('findOccurrenceRanges reaches table cells and headers, stamping segmentId', () => {
  const ranges = findOccurrenceRanges(doc, 'target', true);
  assert.deepEqual(ranges, [
    { startIndex: 30, endIndex: 36 },
    { startIndex: 58, endIndex: 64 },
    { startIndex: 5, endIndex: 11, segmentId: 'h9' },
  ]);
});

test('findOccurrenceRanges refuses newline-bearing findText and stamps tabId on tabbed docs', () => {
  assert.deepEqual(findOccurrenceRanges(doc, 'world\nhere', true), []);
  const tabbed = {
    tabs: [
      {
        tabProperties: { tabId: 't.0' },
        documentTab: { body: { content: [{ paragraph: { elements: [run('tab target\n', 1)] } }] } },
      },
    ],
  };
  assert.deepEqual(findOccurrenceRanges(tabbed, 'target', true), [{ startIndex: 5, endIndex: 11, tabId: 't.0' }]);
});

test('buildMultilineReplaceRequests emits descending delete+insert pairs with segment/tab threading', () => {
  const requests = buildMultilineReplaceRequests(
    [
      { startIndex: 10, endIndex: 16 },
      { startIndex: 40, endIndex: 46, tabId: 't.1' },
      { startIndex: 25, endIndex: 31, segmentId: 'h9' },
    ],
    'line one\nline two',
  ) as any[];
  assert.equal(requests.length, 6);
  // Descending by startIndex: 40, 25, 10 — each delete immediately followed by its insert.
  assert.deepEqual(requests[0], { deleteContentRange: { range: { startIndex: 40, endIndex: 46, tabId: 't.1' } } });
  assert.deepEqual(requests[1], { insertText: { location: { index: 40, tabId: 't.1' }, text: 'line one\nline two' } });
  assert.deepEqual(requests[2].deleteContentRange.range, { startIndex: 25, endIndex: 31, segmentId: 'h9' });
  assert.equal(requests[3].insertText.location.segmentId, 'h9');
  assert.deepEqual(requests[4].deleteContentRange.range, { startIndex: 10, endIndex: 16 });
});

test('empty replaceText compiles to deletes only', () => {
  const requests = buildMultilineReplaceRequests([{ startIndex: 5, endIndex: 8 }], '');
  assert.equal(requests.length, 1);
});
