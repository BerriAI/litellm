import type { drive_v3 } from 'googleapis';
import { toNodeReadable } from '../utils/streams.js';

// ---------------------------------------------------------------------------
// Shared read/write helpers for raw text/* files in Drive.
//
// Used by readTextFile (drive.ts) and the text-file branches of
// insertText / deleteRange (docs.ts) so the media-download and write-back
// plumbing lives in exactly one place.
// ---------------------------------------------------------------------------

/**
 * Download a Drive file's raw bytes and decode them as UTF-8.
 * Streams the media response and concatenates chunks.
 */
export async function downloadTextContent(
  drive: drive_v3.Drive,
  fileId: string,
): Promise<string> {
  const mediaResponse = await drive.files.get(
    { fileId, alt: 'media', supportsAllDrives: true },
    { responseType: 'stream' },
  );

  const chunks: Buffer[] = [];
  await new Promise<void>((resolve, reject) => {
    toNodeReadable(mediaResponse.data)
      .on('data', (chunk: Buffer) => chunks.push(chunk))
      .on('end', () => resolve())
      .on('error', (err: Error) => reject(err));
  });

  return Buffer.concat(chunks).toString('utf-8');
}

/**
 * Overwrite a Drive file's content with `text`.
 *
 * The body is passed as a Buffer (not a raw string): googleapis skips media
 * upload when `media.body` is falsy, and the empty string `''` is falsy — so a
 * full-content deletion written as a string silently becomes a no-op metadata
 * PATCH. An empty Buffer is truthy, so the (possibly empty) content is always
 * uploaded.
 */
export async function writeTextContent(
  drive: drive_v3.Drive,
  fileId: string,
  mimeType: string,
  text: string,
): Promise<void> {
  await drive.files.update({
    fileId,
    media: { mimeType, body: Buffer.from(text, 'utf-8') },
    supportsAllDrives: true,
  });
}
