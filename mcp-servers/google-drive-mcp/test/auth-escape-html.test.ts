import assert from 'node:assert/strict';
import test from 'node:test';

import { escapeHtml } from '../src/auth/html.js';

test('escapeHtml neutralizes script injection', () => {
  assert.equal(
    escapeHtml('<script>alert(1)</script>'),
    '&lt;script&gt;alert(1)&lt;/script&gt;'
  );
});

test('escapeHtml escapes & first (no double-escaping)', () => {
  assert.equal(escapeHtml('a & <b>'), 'a &amp; &lt;b&gt;');
});

test('escapeHtml escapes quotes for attribute context', () => {
  assert.equal(escapeHtml(`" '`), '&quot; &#39;');
});
