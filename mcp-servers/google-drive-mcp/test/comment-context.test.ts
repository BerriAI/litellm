import assert from 'node:assert/strict';
import test from 'node:test';
import JSZip from 'jszip';

import {
  buildFlatTextFromDoc,
  extractRowCells,
  resolveContextFromDocx,
  matchDocxToDriveComments,
} from '../src/tools/docs.js';

import type { CommentContext, DocxContextResult } from '../src/tools/docs.js';

// ---------------------------------------------------------------------------
// buildFlatTextFromDoc
// ---------------------------------------------------------------------------

test('buildFlatTextFromDoc: extracts paragraphs with correct offsets', () => {
  const docData = {
    body: {
      content: [
        {
          paragraph: {
            elements: [
              { textRun: { content: 'Hello ' }, startIndex: 1 },
              { textRun: { content: 'world' }, startIndex: 7 },
            ],
          },
        },
      ],
    },
  };
  const { flatText, offsetMap } = buildFlatTextFromDoc(docData);
  assert.equal(flatText, 'Hello world');
  assert.equal(offsetMap[0], 1);  // 'H' at Docs index 1
  assert.equal(offsetMap[6], 7);  // 'w' at Docs index 7
  assert.equal(offsetMap.length, 11);
});

test('buildFlatTextFromDoc: extracts table cell text', () => {
  const docData = {
    body: {
      content: [
        {
          table: {
            tableRows: [
              {
                tableCells: [
                  {
                    content: [
                      {
                        paragraph: {
                          elements: [
                            { textRun: { content: 'Cell A' }, startIndex: 5 },
                          ],
                        },
                      },
                    ],
                  },
                  {
                    content: [
                      {
                        paragraph: {
                          elements: [
                            { textRun: { content: 'Cell B' }, startIndex: 20 },
                          ],
                        },
                      },
                    ],
                  },
                ],
              },
            ],
          },
        },
      ],
    },
  };
  const { flatText, offsetMap } = buildFlatTextFromDoc(docData);
  assert.equal(flatText, 'Cell ACell B');
  assert.equal(offsetMap[0], 5);
  assert.equal(offsetMap[6], 20);
});

test('buildFlatTextFromDoc: handles multi-tab docs', () => {
  const docData = {
    tabs: [
      {
        documentTab: {
          body: {
            content: [
              {
                paragraph: {
                  elements: [
                    { textRun: { content: 'Tab 1' }, startIndex: 1 },
                  ],
                },
              },
            ],
          },
        },
      },
      {
        documentTab: {
          body: {
            content: [
              {
                paragraph: {
                  elements: [
                    { textRun: { content: 'Tab 2' }, startIndex: 10 },
                  ],
                },
              },
            ],
          },
        },
      },
    ],
  };
  const { flatText } = buildFlatTextFromDoc(docData);
  assert.equal(flatText, 'Tab 1Tab 2');
});

test('buildFlatTextFromDoc: returns empty for empty doc', () => {
  const { flatText, offsetMap } = buildFlatTextFromDoc({});
  assert.equal(flatText, '');
  assert.equal(offsetMap.length, 0);
});

test('buildFlatTextFromDoc: skips elements without startIndex', () => {
  const docData = {
    body: {
      content: [
        {
          paragraph: {
            elements: [
              { textRun: { content: 'visible' }, startIndex: 0 },
              { textRun: { content: 'hidden' } }, // no startIndex
            ],
          },
        },
      ],
    },
  };
  const { flatText } = buildFlatTextFromDoc(docData);
  assert.equal(flatText, 'visible');
});

// ---------------------------------------------------------------------------
// extractRowCells
// ---------------------------------------------------------------------------

test('extractRowCells: extracts text from simple row', () => {
  const rowXml =
    '<w:tr><w:tc><w:p><w:r><w:t>Alpha</w:t></w:r></w:p></w:tc>' +
    '<w:tc><w:p><w:r><w:t>Beta</w:t></w:r></w:p></w:tc></w:tr>';
  const cells = extractRowCells(rowXml);
  assert.deepEqual(cells, ['Alpha', 'Beta']);
});

test('extractRowCells: handles w:tc with attributes', () => {
  const rowXml =
    '<w:tr><w:tc w:val="foo"><w:p><w:r><w:t>One</w:t></w:r></w:p></w:tc>' +
    '<w:tc><w:p><w:r><w:t>Two</w:t></w:r></w:p></w:tc></w:tr>';
  const cells = extractRowCells(rowXml);
  assert.deepEqual(cells, ['One', 'Two']);
});

test('extractRowCells: concatenates multiple w:t in one cell', () => {
  const rowXml =
    '<w:tr><w:tc><w:p><w:r><w:t>Hello </w:t><w:t>World</w:t></w:r></w:p></w:tc></w:tr>';
  const cells = extractRowCells(rowXml);
  assert.deepEqual(cells, ['Hello World']);
});

test('extractRowCells: returns empty array for row with no text', () => {
  const rowXml = '<w:tr><w:tc><w:p></w:p></w:tc></w:tr>';
  const cells = extractRowCells(rowXml);
  assert.deepEqual(cells, []);
});

test('extractRowCells: handles w:t with xml:space attribute', () => {
  const rowXml =
    '<w:tr><w:tc><w:p><w:r><w:t xml:space="preserve"> spaced </w:t></w:r></w:p></w:tc></w:tr>';
  const cells = extractRowCells(rowXml);
  assert.deepEqual(cells, [' spaced ']);
});

// ---------------------------------------------------------------------------
// resolveContextFromDocx
// ---------------------------------------------------------------------------

async function buildDocxBuffer(commentsXml: string, documentXml: string): Promise<ArrayBuffer> {
  const zip = new JSZip();
  zip.file('word/comments.xml', commentsXml);
  zip.file('word/document.xml', documentXml);
  const buf = await zip.generateAsync({ type: 'arraybuffer' });
  return buf;
}

test('resolveContextFromDocx: returns null when comments.xml is missing', async () => {
  const zip = new JSZip();
  zip.file('word/document.xml', '<w:document></w:document>');
  const buf = await zip.generateAsync({ type: 'arraybuffer' });
  const result = await resolveContextFromDocx(buf);
  assert.equal(result, null);
});

test('resolveContextFromDocx: returns null when document.xml is missing', async () => {
  const zip = new JSZip();
  zip.file('word/comments.xml', '<w:comments></w:comments>');
  const buf = await zip.generateAsync({ type: 'arraybuffer' });
  const result = await resolveContextFromDocx(buf);
  assert.equal(result, null);
});

test('resolveContextFromDocx: parses comment author and date', async () => {
  const commentsXml = `<?xml version="1.0"?>
    <w:comments>
      <w:comment w:id="1" w:author="Alice" w:date="2026-01-15T10:00:00Z">
        <w:p><w:r><w:t>Nice work</w:t></w:r></w:p>
      </w:comment>
    </w:comments>`;
  const documentXml = `<?xml version="1.0"?>
    <w:document><w:body>
      <w:p><w:r><w:t>Some text</w:t></w:r></w:p>
    </w:body></w:document>`;
  const buf = await buildDocxBuffer(commentsXml, documentXml);
  const result = await resolveContextFromDocx(buf);
  assert.ok(result);
  const comment = result.docxComments.get(1);
  assert.ok(comment);
  assert.equal(comment.author, 'Alice');
  assert.equal(comment.date, '2026-01-15T10:00:00Z');
  assert.equal(comment.content, 'Nice work');
});

test('resolveContextFromDocx: extracts table row context', async () => {
  const commentsXml = `<?xml version="1.0"?>
    <w:comments>
      <w:comment w:id="2" w:author="Bob" w:date="2026-01-20T12:00:00Z">
        <w:p><w:r><w:t>Check this</w:t></w:r></w:p>
      </w:comment>
    </w:comments>`;
  const documentXml = `<?xml version="1.0"?>
    <w:document><w:body>
      <w:tbl><w:tr>
        <w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:commentRangeStart w:id="2"/><w:r><w:t>Value</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Notes</w:t></w:r></w:p></w:tc>
      </w:tr></w:tbl>
    </w:body></w:document>`;
  const buf = await buildDocxBuffer(commentsXml, documentXml);
  const result = await resolveContextFromDocx(buf);
  assert.ok(result);
  // Comment is on "Value" (cell index 1), so before = [Name, Value], after = [Value, Notes]
  const before = result.contextsBefore.get(2);
  const after = result.contextsAfter.get(2);
  assert.ok(before);
  assert.ok(after);
  assert.ok(before.includes('Name'));
  assert.ok(before.includes('Value'));
  assert.ok(after.includes('Value'));
  assert.ok(after.includes('Notes'));
});

test('resolveContextFromDocx: comment in last cell grabs next row for after context', async () => {
  const commentsXml = `<?xml version="1.0"?>
    <w:comments>
      <w:comment w:id="3" w:author="Carol" w:date="2026-02-01T08:00:00Z">
        <w:p><w:r><w:t>Review</w:t></w:r></w:p>
      </w:comment>
    </w:comments>`;
  const documentXml = `<?xml version="1.0"?>
    <w:document><w:body>
      <w:tbl>
        <w:tr>
          <w:tc><w:p><w:r><w:t>A1</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:commentRangeStart w:id="3"/><w:r><w:t>B1</w:t></w:r></w:p></w:tc>
        </w:tr>
        <w:tr>
          <w:tc><w:p><w:r><w:t>A2</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>B2</w:t></w:r></w:p></w:tc>
        </w:tr>
      </w:tbl>
    </w:body></w:document>`;
  const buf = await buildDocxBuffer(commentsXml, documentXml);
  const result = await resolveContextFromDocx(buf);
  assert.ok(result);
  const after = result.contextsAfter.get(3);
  assert.ok(after);
  // After context should include next row cells (A2, B2)
  assert.ok(after.includes('A2'));
  assert.ok(after.includes('B2'));
});

test('resolveContextFromDocx: paragraph fallback for non-table docs', async () => {
  const commentsXml = `<?xml version="1.0"?>
    <w:comments>
      <w:comment w:id="4" w:author="Dave" w:date="2026-03-01T09:00:00Z">
        <w:p><w:r><w:t>Suggestion</w:t></w:r></w:p>
      </w:comment>
    </w:comments>`;
  const documentXml = `<?xml version="1.0"?>
    <w:document><w:body>
      <w:p w:rsidR="001"><w:commentRangeStart w:id="4"/><w:r><w:t>This is a paragraph with a comment on it.</w:t></w:r></w:p>
    </w:body></w:document>`;
  const buf = await buildDocxBuffer(commentsXml, documentXml);
  const result = await resolveContextFromDocx(buf);
  assert.ok(result);
  const before = result.contextsBefore.get(4);
  assert.ok(before);
  assert.ok(before.includes('This is a paragraph'));
});

// ---------------------------------------------------------------------------
// matchDocxToDriveComments
// ---------------------------------------------------------------------------

test('matchDocxToDriveComments: matches by author and stripped timestamp', () => {
  const driveComments = [
    {
      id: 'drive-1',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
      quotedFileContent: { value: 'hello' },
    },
  ];
  const docxResult: DocxContextResult = {
    docxComments: new Map([
      [1, { author: 'Alice', date: '2026-01-15T10:00:00Z', content: 'Nice' }],
    ]),
    contextsBefore: new Map([[1, 'before text']]),
    contextsAfter: new Map([[1, 'after text']]),
    rowCells: new Map(),
  };
  const contextMap = new Map<string, CommentContext>();

  matchDocxToDriveComments(driveComments, docxResult, contextMap, '', []);

  assert.ok(contextMap.has('drive-1'));
  assert.equal(contextMap.get('drive-1')!.contextBefore, 'before text');
  assert.equal(contextMap.get('drive-1')!.contextAfter, 'after text');
});

test('matchDocxToDriveComments: skips comments with existing Tier 1 context', () => {
  const driveComments = [
    {
      id: 'drive-1',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
      quotedFileContent: { value: 'hello' },
    },
  ];
  const docxResult: DocxContextResult = {
    docxComments: new Map([
      [1, { author: 'Alice', date: '2026-01-15T10:00:00Z', content: 'Nice' }],
    ]),
    contextsBefore: new Map([[1, 'docx before']]),
    contextsAfter: new Map([[1, 'docx after']]),
    rowCells: new Map(),
  };
  const contextMap = new Map<string, CommentContext>();
  contextMap.set('drive-1', { contextBefore: 'tier1 before' });

  matchDocxToDriveComments(driveComments, docxResult, contextMap, '', []);

  // Tier 1 context should be preserved
  assert.equal(contextMap.get('drive-1')!.contextBefore, 'tier1 before');
});

test('matchDocxToDriveComments: skips resolved comments', () => {
  const driveComments = [
    {
      id: 'drive-1',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
      resolved: true,
    },
  ];
  const docxResult: DocxContextResult = {
    docxComments: new Map([
      [1, { author: 'Alice', date: '2026-01-15T10:00:00Z', content: 'Nice' }],
    ]),
    contextsBefore: new Map([[1, 'context']]),
    contextsAfter: new Map(),
    rowCells: new Map(),
  };
  const contextMap = new Map<string, CommentContext>();

  matchDocxToDriveComments(driveComments, docxResult, contextMap, '', []);

  assert.equal(contextMap.size, 0);
});

test('matchDocxToDriveComments: resolves character offsets when flatText matches', () => {
  // flatText: "Hello world" with offsetMap [1,2,3,4,5,6,7,8,9,10,11]
  const flatText = 'col1\ncol2\ncol3';
  const offsetMap = Array.from({ length: flatText.length }, (_, i) => i + 1);

  const driveComments = [
    {
      id: 'drive-1',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
      quotedFileContent: { value: 'col2' },
    },
  ];
  const docxResult: DocxContextResult = {
    docxComments: new Map([
      [1, { author: 'Alice', date: '2026-01-15T10:00:00Z', content: 'Note' }],
    ]),
    // contextBefore joins with ' | ', so when split and joined with '\n' it becomes "col1\ncol2"
    contextsBefore: new Map([[1, 'col1 | col2']]),
    contextsAfter: new Map([[1, 'col2 | col3']]),
    rowCells: new Map(),
  };
  const contextMap = new Map<string, CommentContext>();

  matchDocxToDriveComments(driveComments, docxResult, contextMap, flatText, offsetMap);

  const ctx = contextMap.get('drive-1');
  assert.ok(ctx);
  assert.ok(ctx.startIndex !== undefined, 'startIndex should be resolved');
  assert.ok(ctx.endIndex !== undefined, 'endIndex should be resolved');
});

test('matchDocxToDriveComments: no match when author differs', () => {
  const driveComments = [
    {
      id: 'drive-1',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
    },
  ];
  const docxResult: DocxContextResult = {
    docxComments: new Map([
      [1, { author: 'Bob', date: '2026-01-15T10:00:00Z', content: 'Note' }],
    ]),
    contextsBefore: new Map([[1, 'context']]),
    contextsAfter: new Map(),
    rowCells: new Map(),
  };
  const contextMap = new Map<string, CommentContext>();

  matchDocxToDriveComments(driveComments, docxResult, contextMap, '', []);

  assert.equal(contextMap.size, 0);
});

test('matchDocxToDriveComments: removes matched DOCX comment to prevent double-matching', () => {
  const driveComments = [
    {
      id: 'drive-1',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
      quotedFileContent: { value: 'text' },
    },
    {
      id: 'drive-2',
      author: { displayName: 'Alice' },
      createdTime: '2026-01-15T10:00:00.123Z',
      quotedFileContent: { value: 'other' },
    },
  ];
  // Only one DOCX comment with the same author+date
  const docxResult: DocxContextResult = {
    docxComments: new Map([
      [1, { author: 'Alice', date: '2026-01-15T10:00:00Z', content: 'Note' }],
    ]),
    contextsBefore: new Map([[1, 'context']]),
    contextsAfter: new Map(),
    rowCells: new Map(),
  };
  const contextMap = new Map<string, CommentContext>();

  matchDocxToDriveComments(driveComments, docxResult, contextMap, '', []);

  // Only the first should match; the second should not double-match
  assert.ok(contextMap.has('drive-1'));
  assert.ok(!contextMap.has('drive-2'));
});

// ---------------------------------------------------------------------------
// buildFlatTextFromDoc: offsetMap bounds
// ---------------------------------------------------------------------------

test('buildFlatTextFromDoc: offsetMap length matches flatText length', () => {
  const docData = {
    body: {
      content: [
        {
          paragraph: {
            elements: [
              { textRun: { content: 'abc' }, startIndex: 100 },
            ],
          },
        },
      ],
    },
  };
  const { flatText, offsetMap } = buildFlatTextFromDoc(docData);
  assert.equal(flatText.length, offsetMap.length);
  assert.equal(offsetMap[0], 100);
  assert.equal(offsetMap[2], 102);
});
