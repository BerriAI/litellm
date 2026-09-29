import assert from 'node:assert/strict';
import test from 'node:test';

import { buildDocFormattedContent } from '../../src/tools/docs.js';
import { paragraphMetaBits } from '../../src/tools/styleProjection.js';

const styledDoc = {
  body: {
    content: [
      {
        startIndex: 1,
        endIndex: 20,
        paragraph: {
          paragraphStyle: {
            namedStyleType: 'HEADING_2',
            borderBottom: { width: { magnitude: 1.5, unit: 'PT' }, dashStyle: 'SOLID', color: { color: { rgbColor: { red: 0.8, green: 0.1, blue: 0.1 } } } },
            shading: { backgroundColor: { color: { rgbColor: { red: 1, green: 1, blue: 1 } } } },
          },
          elements: [{ startIndex: 1, endIndex: 20, textRun: { content: 'Section heading\n', textStyle: {} } }],
        },
      },
      {
        startIndex: 20,
        endIndex: 32,
        paragraph: {
          paragraphStyle: { namedStyleType: 'NORMAL_TEXT' },
          elements: [{ startIndex: 20, endIndex: 32, textRun: { content: 'plain body\n', textStyle: {} } }],
        },
      },
    ],
  },
};

test('includeFormatting emits one ¶ meta line per non-default paragraph over its real span', () => {
  const { formattedContent } = buildDocFormattedContent(styledDoc, true);
  const metaLines = formattedContent.split('\n').filter((l) => l.includes('¶'));
  assert.equal(metaLines.length, 1);
  assert.match(metaLines[0], /^\[1-20\] ¶ HEADING_2, borderBottom\(#cc1a1a 1\.5PT SOLID\), shading\(#ffffff\)$/);
  // The plain paragraph's content is still present, unannotated.
  assert.ok(formattedContent.includes('plain body'));
});

test('paragraph meta stays out of plain reads (withFormatting=false)', () => {
  const { formattedContent } = buildDocFormattedContent(styledDoc, false);
  assert.ok(!formattedContent.includes('¶'));
});

test('paragraphMetaBits skips defaults and invisible borders', () => {
  assert.deepEqual(paragraphMetaBits({ paragraphStyle: { namedStyleType: 'NORMAL_TEXT', alignment: 'START' } }), []);
  assert.deepEqual(
    paragraphMetaBits({ paragraphStyle: { borderTop: { width: { magnitude: 0 } }, alignment: 'CENTER' } }),
    ['center'],
  );
});

test('paragraph meta never leaks into a table cell rendering', () => {
  // getCellText splices every segment pushed during the cell walk and joins
  // their text, so a styled cell paragraph would put its `¶` meta line inside
  // the pipe row: `| ¶ center Header A | ¶ HEADING_3 Header B |`. That is a
  // regression in getGoogleDocContent, not merely a gap in the style tools.
  const cell = (text: string, startIndex: number, paragraphStyle: Record<string, unknown>) => ({
    content: [{
      startIndex,
      endIndex: startIndex + text.length,
      paragraph: {
        paragraphStyle,
        elements: [{ startIndex, endIndex: startIndex + text.length, textRun: { content: text, textStyle: {} } }],
      },
    }],
  });

  const doc = {
    body: {
      content: [{
        startIndex: 7,
        endIndex: 51,
        table: {
          tableRows: [{
            tableCells: [
              cell('Header A\n', 9, { alignment: 'CENTER' }),
              cell('Header B\n', 20, { namedStyleType: 'HEADING_3' }),
            ],
          }],
        },
      }],
    },
  };

  const { formattedContent } = buildDocFormattedContent(doc, true);
  // Match on content, not line prefix: the rendering carries an index prefix
  // in some read paths and not others, but a pipe row always has the cells.
  const pipeRows = formattedContent.split('\n').filter((l: string) => l.includes('| Header A'));

  assert.ok(pipeRows.length > 0, 'the table still renders');
  for (const row of pipeRows) {
    assert.ok(!row.includes('¶'), `paragraph meta leaked into a table row: ${row}`);
  }
  assert.match(formattedContent, /\| Header A \| Header B \|/);
});

test('an empty shading color is omitted rather than printed as shading(null)', () => {
  // The presence check was on the backgroundColor object, but rgbColorToHex() returns
  // null for a color object carrying no rgbColor — printing the literal
  // "shading(null)". Guard on the resolved hex, the way borders guard on width.
  const emptyShading = paragraphMetaBits({
    paragraphStyle: { namedStyleType: 'NORMAL_TEXT', shading: { backgroundColor: {} } },
  });
  assert.deepEqual(emptyShading.filter((b) => b.startsWith('shading')), []);

  const realShading = paragraphMetaBits({
    paragraphStyle: {
      namedStyleType: 'NORMAL_TEXT',
      shading: { backgroundColor: { color: { rgbColor: { red: 1, green: 0, blue: 0 } } } },
    },
  });
  assert.deepEqual(realShading.filter((b) => b.startsWith('shading')), ['shading(#ff0000)']);
});
