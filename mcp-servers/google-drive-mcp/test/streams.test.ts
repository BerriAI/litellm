import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import test from 'node:test';

import { toNodeReadable, getResponseHeader } from '../src/utils/streams.js';

async function collect(stream: Readable): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf-8');
}

test('toNodeReadable passes a Node Readable through unchanged', async () => {
  const source = Readable.from(['node-', 'payload']);
  const out = toNodeReadable(source);
  assert.equal(out, source);
  assert.equal(await collect(out), 'node-payload');
});

test('toNodeReadable wraps buffered bodies as a single chunk', async () => {
  assert.equal(await collect(toNodeReadable(Buffer.from('buffer-payload'))), 'buffer-payload');
  assert.equal(await collect(toNodeReadable('string-payload')), 'string-payload');
});

test('toNodeReadable converts a web ReadableStream to a Node Readable', async () => {
  const source = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('web-'));
      controller.enqueue(new TextEncoder().encode('payload'));
      controller.close();
    },
  });
  const out = toNodeReadable(source);
  assert.ok(out instanceof Readable);
  assert.equal(await collect(out), 'web-payload');
});

// ---------------------------------------------------------------------------
// getResponseHeader
//
// gaxios 7 returns a `Headers` instance where gaxios 6 returned a plain object,
// so `headers['content-type']` silently became undefined — which made
// getGoogleDocImage label every image application/octet-stream.
// ---------------------------------------------------------------------------

test('getResponseHeader reads from a gaxios-7 Headers instance', () => {
  const h = new Headers({ 'content-type': 'image/png' });
  assert.equal(getResponseHeader(h, 'content-type'), 'image/png');
  assert.equal(getResponseHeader(h, 'Content-Type'), 'image/png', 'must be case-insensitive');
  assert.equal(getResponseHeader(h, 'x-absent'), undefined);
});

test('getResponseHeader still reads a gaxios-6 style plain object', () => {
  const h = { 'content-type': 'image/jpeg' };
  assert.equal(getResponseHeader(h, 'content-type'), 'image/jpeg');
  assert.equal(getResponseHeader(h, 'CONTENT-TYPE'), 'image/jpeg', 'must be case-insensitive');
  assert.equal(getResponseHeader(h, 'x-absent'), undefined);
});

test('getResponseHeader handles missing/!string headers without throwing', () => {
  assert.equal(getResponseHeader(undefined, 'content-type'), undefined);
  assert.equal(getResponseHeader(null, 'content-type'), undefined);
  assert.equal(getResponseHeader({ 'content-type': 12345 }, 'content-type'), undefined);
});
