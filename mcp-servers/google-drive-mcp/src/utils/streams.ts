import { Readable } from 'node:stream';

/**
 * Normalize a media-download response body to a Node Readable.
 *
 * gaxios 7 (native fetch) returns a web ReadableStream for
 * `responseType: 'stream'`, while older clients and test doubles produce Node
 * Readables or already-buffered bodies (Buffer/string). Downstream code relies
 * on Node stream semantics (`.on()` listeners, upload media bodies), so accept
 * any of those shapes here.
 */
export function toNodeReadable(data: unknown): Readable {
  if (data instanceof Readable) return data;
  if (typeof data === 'string' || data instanceof Uint8Array) {
    // Wrap in an array so Readable.from emits one chunk instead of iterating
    // per character / per byte.
    return Readable.from([data]);
  }
  return Readable.fromWeb(data as import('node:stream/web').ReadableStream);
}

/**
 * Read a single header from a gaxios response.
 *
 * gaxios 7 (native fetch) exposes `response.headers` as a `Headers` instance,
 * where bracket access silently yields `undefined` — gaxios 6 handed back a
 * plain object, so `headers['content-type']` used to work. Both shapes are
 * accepted here so test doubles that pass plain objects keep working.
 * Header names are case-insensitive in both.
 */
export function getResponseHeader(headers: unknown, name: string): string | undefined {
  if (!headers) return undefined;

  if (typeof (headers as Headers).get === 'function') {
    return (headers as Headers).get(name) ?? undefined;
  }

  const lower = name.toLowerCase();
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    if (key.toLowerCase() === lower) {
      return typeof value === 'string' ? value : undefined;
    }
  }
  return undefined;
}
