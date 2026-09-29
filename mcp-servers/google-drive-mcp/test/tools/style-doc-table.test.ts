import assert from 'node:assert/strict';
import test from 'node:test';

import { buildStyleDocTableRequests } from '../../src/tools/docs.js';

test('whole-table unbox compiles to width-0 borders via tableStartLocation', () => {
  const { requests, applied } = buildStyleDocTableRequests({ tableStartIndex: 42, removeBorders: ['all'] });
  assert.equal(requests.length, 1);
  const req = requests[0].updateTableCellStyle;
  assert.deepEqual(req.tableStartLocation, { index: 42 });
  assert.equal(req.tableRange, undefined);
  for (const field of ['borderTop', 'borderBottom', 'borderLeft', 'borderRight']) {
    assert.deepEqual(req.tableCellStyle[field].width, { magnitude: 0, unit: 'PT' });
  }
  assert.deepEqual(
    req.fields.split(',').sort(),
    ['borderBottom', 'borderLeft', 'borderRight', 'borderTop'],
  );
  assert.ok(applied.some((line) => line.includes('borders cleared')));
});

test('cell-subset targeting builds a tableRange with spans defaulting to 1', () => {
  const { requests } = buildStyleDocTableRequests({
    tableStartIndex: 10,
    rowIndex: 1,
    columnIndex: 2,
    backgroundColor: '#FFFFFF',
  });
  const req = requests[0].updateTableCellStyle;
  assert.deepEqual(req.tableRange, {
    tableCellLocation: { tableStartLocation: { index: 10 }, rowIndex: 1, columnIndex: 2 },
    rowSpan: 1,
    columnSpan: 1,
  });
  assert.equal(req.tableStartLocation, undefined);
  assert.deepEqual(req.tableCellStyle.backgroundColor.color.rgbColor, { red: 1, green: 1, blue: 1 });
});

test('padding expands to all four edges; contentAlignment and explicit borders carry through', () => {
  const { requests } = buildStyleDocTableRequests({
    tableStartIndex: 5,
    padding: 4,
    contentAlignment: 'MIDDLE',
    borderBottom: { color: '#CC1111', width: 2, dashStyle: 'DASH' },
  });
  const req = requests[0].updateTableCellStyle;
  for (const edge of ['paddingTop', 'paddingBottom', 'paddingLeft', 'paddingRight']) {
    assert.deepEqual(req.tableCellStyle[edge], { magnitude: 4, unit: 'PT' });
  }
  assert.equal(req.tableCellStyle.contentAlignment, 'MIDDLE');
  assert.equal(req.tableCellStyle.borderBottom.dashStyle, 'DASH');
  assert.deepEqual(req.tableCellStyle.borderBottom.width, { magnitude: 2, unit: 'PT' });
});

test('column width and row height compile to their own requests with empty indices = all', () => {
  const { requests, applied } = buildStyleDocTableRequests({
    tableStartIndex: 7,
    columnWidth: 90,
    minRowHeight: 24,
  });
  assert.equal(requests.length, 2);
  assert.deepEqual(requests[0].updateTableColumnProperties, {
    tableStartLocation: { index: 7 },
    columnIndices: [],
    tableColumnProperties: { widthType: 'FIXED_WIDTH', width: { magnitude: 90, unit: 'PT' } },
    fields: 'width,widthType',
  });
  assert.deepEqual(requests[1].updateTableRowStyle, {
    tableStartLocation: { index: 7 },
    rowIndices: [],
    tableRowStyle: { minRowHeight: { magnitude: 24, unit: 'PT' } },
    fields: 'minRowHeight',
  });
  assert.equal(applied.length, 2);
});

test('scoped column/row updates keep their indices and tabId threads into locations', () => {
  const { requests } = buildStyleDocTableRequests({
    tableStartIndex: 7,
    tabId: 't.0',
    columnIndices: [0, 2],
    columnWidth: 60,
    rowIndices: [1],
    minRowHeight: 18,
  });
  assert.deepEqual(requests[0].updateTableColumnProperties.tableStartLocation, { index: 7, tabId: 't.0' });
  assert.deepEqual(requests[0].updateTableColumnProperties.columnIndices, [0, 2]);
  assert.deepEqual(requests[1].updateTableRowStyle.rowIndices, [1]);
});

test('refuses ambiguous or conflicting input', () => {
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, removeBorders: ['top'], borderTop: {} }),
    /both set and remove/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, backgroundColor: '#FFF', removeBackground: true }),
    /both set and remove/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, rowIndex: 0, backgroundColor: '#FFF' }),
    /needs both rowIndex and columnIndex/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, rowIndex: 0, columnIndex: 0, columnWidth: 40 }),
    /scope cell styles only/,
  );
  // Same refusal when a cell style IS present: the cell request would be
  // scoped to one cell while the width silently widened to every column.
  assert.throws(
    () => buildStyleDocTableRequests({
      tableStartIndex: 1, rowIndex: 0, columnIndex: 0, backgroundColor: '#FFFFFF', columnWidth: 40,
    }),
    /scope cell styles only/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({
      tableStartIndex: 1, rowIndex: 0, columnIndex: 0, backgroundColor: '#FFFFFF', minRowHeight: 20,
    }),
    /scope cell styles only/,
  );
  // A span with no anchoring cell would otherwise fall through to the
  // whole-table path and be dropped without a word.
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, rowSpan: 2, backgroundColor: '#FFFFFF' }),
    /needs both rowIndex and columnIndex/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, columnWidth: 3 }),
    /at least 5 points/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, columnIndices: [0] }),
    /columnIndices given without columnWidth/,
  );
  assert.throws(
    () => buildStyleDocTableRequests({ tableStartIndex: 1, rowIndices: [0] }),
    /rowIndices given without minRowHeight/,
  );
});
