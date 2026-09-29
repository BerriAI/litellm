import assert from 'node:assert/strict';
import test from 'node:test';

import { buildUpdateParagraphStyleRequest } from '../../src/tools/docs.js';

test('builds a rendering border from a sparse param (color/width/dashStyle defaults)', () => {
  const result = buildUpdateParagraphStyleRequest(1, 50, { borderBottom: {} });
  assert.ok(result);
  const { updateParagraphStyle } = result.request;
  assert.deepEqual(updateParagraphStyle.paragraphStyle.borderBottom, {
    color: { color: { rgbColor: { red: 0, green: 0, blue: 0 } } },
    width: { magnitude: 1, unit: 'PT' },
    // The API refuses a border without padding (UNIT_UNSPECIFIED) — always defaulted.
    padding: { magnitude: 1, unit: 'PT' },
    dashStyle: 'SOLID',
  });
  assert.equal(updateParagraphStyle.fields, 'borderBottom');
  assert.deepEqual(updateParagraphStyle.range, { startIndex: 1, endIndex: 50 });
});

test('honors explicit border subfields including padding', () => {
  const result = buildUpdateParagraphStyleRequest(1, 10, {
    borderTop: { color: '#FF0000', width: 2.5, padding: 3, dashStyle: 'DOT' },
  });
  assert.ok(result);
  const border = result.request.updateParagraphStyle.paragraphStyle.borderTop;
  assert.deepEqual(border.color.color.rgbColor, { red: 1, green: 0, blue: 0 });
  assert.deepEqual(border.width, { magnitude: 2.5, unit: 'PT' });
  assert.deepEqual(border.padding, { magnitude: 3, unit: 'PT' });
  assert.equal(border.dashStyle, 'DOT');
});

test('removeBorders names the fields while leaving them unset (FieldMask reset)', () => {
  const result = buildUpdateParagraphStyleRequest(1, 500, { removeBorders: ['all'] });
  assert.ok(result);
  const { updateParagraphStyle } = result.request;
  assert.deepEqual(
    updateParagraphStyle.fields.split(',').sort(),
    ['borderBetween', 'borderBottom', 'borderLeft', 'borderRight', 'borderTop'],
  );
  assert.deepEqual(updateParagraphStyle.paragraphStyle, {});
});

test('removeBorders accepts individual edges without touching the others', () => {
  const result = buildUpdateParagraphStyleRequest(1, 20, {
    removeBorders: ['bottom'],
    borderTop: { width: 0.5 },
  });
  assert.ok(result);
  const { updateParagraphStyle } = result.request;
  assert.deepEqual(updateParagraphStyle.fields.split(',').sort(), ['borderBottom', 'borderTop']);
  assert.equal(updateParagraphStyle.paragraphStyle.borderBottom, undefined);
  assert.ok(updateParagraphStyle.paragraphStyle.borderTop);
});

test('setting and removing the same border edge is refused', () => {
  assert.throws(
    () => buildUpdateParagraphStyleRequest(1, 10, { removeBorders: ['top'], borderTop: {} }),
    /both set and remove/,
  );
});

test('shading compiles to a paragraph background and removeShading resets it', () => {
  const set = buildUpdateParagraphStyleRequest(1, 10, { shading: '#F1F3F4' });
  assert.ok(set);
  assert.deepEqual(set.request.updateParagraphStyle.paragraphStyle.shading, {
    backgroundColor: { color: { rgbColor: { red: 0xf1 / 255, green: 0xf3 / 255, blue: 0xf4 / 255 } } },
  });
  assert.equal(set.request.updateParagraphStyle.fields, 'shading');

  const removed = buildUpdateParagraphStyleRequest(1, 10, { removeShading: true });
  assert.ok(removed);
  assert.equal(removed.request.updateParagraphStyle.fields, 'shading');
  assert.deepEqual(removed.request.updateParagraphStyle.paragraphStyle, {});

  assert.throws(
    () => buildUpdateParagraphStyleRequest(1, 10, { shading: '#FFFFFF', removeShading: true }),
    /both set and remove/,
  );
});

test('exposes the remaining ParagraphStyle siblings', () => {
  const result = buildUpdateParagraphStyleRequest(1, 10, {
    indentFirstLine: 18,
    keepLinesTogether: true,
    avoidWidowAndOrphan: false,
    pageBreakBefore: true,
  });
  assert.ok(result);
  const { paragraphStyle, fields } = result.request.updateParagraphStyle;
  assert.deepEqual(paragraphStyle.indentFirstLine, { magnitude: 18, unit: 'PT' });
  assert.equal(paragraphStyle.keepLinesTogether, true);
  assert.equal(paragraphStyle.avoidWidowAndOrphan, false);
  assert.equal(paragraphStyle.pageBreakBefore, true);
  assert.deepEqual(
    fields.split(',').sort(),
    ['avoidWidowAndOrphan', 'indentFirstLine', 'keepLinesTogether', 'pageBreakBefore'],
  );
});

test('rejects invalid hex colors', () => {
  assert.throws(() => buildUpdateParagraphStyleRequest(1, 10, { borderTop: { color: 'red' } }), /Invalid border hex color/);
  assert.throws(() => buildUpdateParagraphStyleRequest(1, 10, { shading: 'gray' }), /Invalid shading hex color/);
});

test('keeps prior behavior: existing params, tabId threading, and null on empty style', () => {
  const result = buildUpdateParagraphStyleRequest(2, 8, { alignment: 'CENTER', spaceBelow: 6 }, 't.0');
  assert.ok(result);
  const { updateParagraphStyle } = result.request;
  assert.equal(updateParagraphStyle.paragraphStyle.alignment, 'CENTER');
  assert.deepEqual(updateParagraphStyle.range, { startIndex: 2, endIndex: 8, tabId: 't.0' });

  assert.equal(buildUpdateParagraphStyleRequest(1, 10, {}), null);
});

test('rejects hex colors that parseInt would silently truncate', () => {
  // parseInt('12345G', 16) === 0x12345, so before the strict check this
  // passed the length test and became #012345.
  assert.throws(() => buildUpdateParagraphStyleRequest(1, 10, { shading: '#12345G' }), /Invalid shading hex color/);
  assert.throws(() => buildUpdateParagraphStyleRequest(1, 10, { borderTop: { color: '#GGGGGG' } }), /Invalid border hex color/);
  assert.throws(() => buildUpdateParagraphStyleRequest(1, 10, { borderTop: { color: '#12G' } }), /Invalid border hex color/);

  // Well-formed 3- and 6-digit forms, with or without '#', still parse.
  const short = buildUpdateParagraphStyleRequest(1, 10, { shading: 'fff' });
  assert.ok(short);
  assert.deepEqual(short.request.updateParagraphStyle.paragraphStyle.shading.backgroundColor.color.rgbColor, { red: 1, green: 1, blue: 1 });
  const long = buildUpdateParagraphStyleRequest(1, 10, { borderTop: { color: '#00ff00' } });
  assert.ok(long);
  assert.deepEqual(long.request.updateParagraphStyle.paragraphStyle.borderTop.color.color.rgbColor, { red: 0, green: 1, blue: 0 });
});
