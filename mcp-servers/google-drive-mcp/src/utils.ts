// -----------------------------------------------------------------------------
// Pure utility functions extracted from index.ts for testability
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Drive list params
// ---------------------------------------------------------------------------

/**
 * The two flags that surface Shared Drive items for a *parent-scoped* query
 * (`'<id>' in parents`). Spread into any listing that is already narrowed to a
 * folder: these flags alone reveal Shared Drive children, and crucially they do
 * NOT set `corpora: 'allDrives'` — the only corpus that can return an
 * `incompleteSearch` partial result. A parent-scoped listing must never be
 * partial, or it silently omits items (issue #137).
 */
export const PARENT_SCOPED_LIST_PARAMS = {
  includeItemsFromAllDrives: true,
  supportsAllDrives: true,
} as const;

/**
 * The `files.list` params that surface Shared Drive items across *all* drives
 * the user can reach. Spread into broad (non-parent-scoped) listings so every
 * such endpoint stays consistent and a new one can't silently omit a flag and
 * re-hide Shared Drive results (issue #137). Adds `corpora: 'allDrives'` to the
 * parent-scoped flags; use {@link PARENT_SCOPED_LIST_PARAMS} instead whenever a
 * query is already scoped to a folder, since `allDrives` is the only corpus
 * that can return `incompleteSearch` partial results.
 */
export const ALL_DRIVES_LIST_PARAMS = {
  corpora: 'allDrives',
  ...PARENT_SCOPED_LIST_PARAMS,
} as const;

/**
 * Sort keys accepted by the `files.list` `orderBy` param, ascending and descending.
 * A deliberately small subset of the keys Drive documents, so a caller (usually an
 * LLM) picks a valid value first try. Drive sorts every key ascending unless the
 * `desc` modifier is present — `modifiedTime` alone means *oldest* first (#167).
 *
 * `name` sorts 1, 12, 2, 22 where `name_natural` sorts 1, 2, 12, 22; `recency` is
 * Drive's composite of created/modified/viewed. Drive warns against `createdTime`
 * on very large collections — prefer `modifiedTime desc` there.
 */
export const DRIVE_ORDER_BY_VALUES = [
  'modifiedTime desc',
  'modifiedTime',
  'createdTime desc',
  'createdTime',
  'recency desc',
  'recency',
  'name',
  'name_natural',
] as const;

// ---------------------------------------------------------------------------
// Calendar helpers
// ---------------------------------------------------------------------------

export interface CalendarEventOverrides {
  summary?: string;
  description?: string;
  location?: string;
  start?: any;
  end?: any;
  attendees?: string[];
  attachments?: { fileUrl: string; title?: string; mimeType?: string }[];
  [key: string]: any;
}

/**
 * Build the event resource for an updateCalendarEvent call by merging user
 * overrides onto the existing event while excluding read-only fields.
 *
 * Rules:
 * - User overrides win when explicitly provided (even if empty string / empty array)
 * - Existing values are preserved when the override is `undefined`
 * - Attendees are mapped from `string[]` → `{email}[]`
 * - Attachments are preserved as-is when not overridden; passing them through
 *   (with the read-only fileId/iconLink the API returned) is required so an
 *   update doesn't wipe an event's existing attachments. Caller must set
 *   `supportsAttachments: true` on the update request for this to take effect.
 * - Only mutable fields are included; read-only fields (id, kind, etag, htmlLink,
 *   iCalUID, creator, organizer, sequence, …) are never forwarded
 */
export function buildCalendarEventUpdate(existing: any, overrides: CalendarEventOverrides): any {
  return {
    summary:     overrides.summary     !== undefined ? overrides.summary     : existing.summary,
    description: overrides.description !== undefined ? overrides.description : existing.description,
    location:    overrides.location    !== undefined ? overrides.location    : existing.location,
    start:       overrides.start  || existing.start,
    end:         overrides.end    || existing.end,
    attendees:   overrides.attendees !== undefined
      ? overrides.attendees.map((email: string) => ({ email }))
      : existing.attendees,
    attachments: overrides.attachments !== undefined
      ? overrides.attachments
      : existing.attachments,
    recurrence:  existing.recurrence,
    visibility:  existing.visibility,
    reminders:   existing.reminders,
  };
}

/**
 * Get file extension from a filename (lowercase).
 */
export function getExtensionFromFilename(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() || '';
}

export const TEXT_MIME_TYPES: Record<string, string> = {
  txt: 'text/plain',
  md: 'text/markdown',
};

/**
 * Get the MIME type for a text file from its filename.
 * Falls back to 'text/plain' for unknown extensions.
 */
export function getMimeTypeFromFilename(filename: string): string {
  const ext = getExtensionFromFilename(filename);
  return TEXT_MIME_TYPES[ext] || 'text/plain';
}

/**
 * Whether a MIME type denotes a raw text file that the readTextFile /
 * insertText / deleteRange / updateTextFile tools can read and edit in place.
 * Covers all `text/*` types (text/plain, text/markdown, text/csv, text/html, …).
 * Note: this is intentionally broader than TEXT_MIME_TYPES, which is the
 * extension→MIME allowlist used for naming new/renamed text files.
 */
export function isTextMime(mimeType: string): boolean {
  return mimeType.startsWith('text/');
}

/**
 * Escape a string for use in a Google Drive API query.
 * Escapes backslashes and single quotes.
 */
export function escapeDriveQuery(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

/**
 * Parse a Sheets A1 range reference (e.g. "'My Sheet'!A1:B2") into its
 * sheet name and cell range components.
 *
 * - Strips surrounding single quotes from the sheet name
 * - Defaults to 'Sheet1' when no sheet prefix is present
 */
export function parseA1Range(range: string): { sheetName: string; cellRange: string } {
  if (range.includes('!')) {
    const sheetName = range.split('!')[0].replace(/^'+|'+$/g, '');
    const cellRange = range.split('!')[1];
    return { sheetName, cellRange };
  }
  return { sheetName: 'Sheet1', cellRange: range };
}

/**
 * Convert column letters to a zero-based index (A=0, B=1, ... Z=25, AA=26).
 */
export function colToIndex(col: string): number {
  let num = 0;
  for (let i = 0; i < col.length; i++) {
    num = num * 26 + (col.charCodeAt(i) - 'A'.charCodeAt(0) + 1);
  }
  return num - 1;
}

export interface GridRange {
  sheetId: number;
  startColumnIndex?: number;
  startRowIndex?: number;
  endColumnIndex?: number;
  endRowIndex?: number;
}

/**
 * Convert an A1 notation string (e.g. "A1:C5") into a Sheets GridRange object.
 * Supports ranges, single cells, full-row ("1:3"), and full-column ("A:C") notation.
 */
export function convertA1ToGridRange(a1Notation: string, sheetId: number): GridRange {
  const rangeRegex = /^([A-Z]*)([0-9]*)(:([A-Z]*)([0-9]*))?$/;
  const match = a1Notation.match(rangeRegex);

  if (!match) {
    throw new Error(`Invalid A1 notation: ${a1Notation}`);
  }

  const [, startCol, startRow, , endCol, endRow] = match;

  const gridRange: GridRange = { sheetId };

  if (startCol) gridRange.startColumnIndex = colToIndex(startCol);
  if (startRow) gridRange.startRowIndex = parseInt(startRow) - 1;

  if (endCol) {
    gridRange.endColumnIndex = colToIndex(endCol) + 1;
  } else if (startCol && !endCol) {
    gridRange.endColumnIndex = gridRange.startColumnIndex! + 1;
  }

  if (endRow) {
    gridRange.endRowIndex = parseInt(endRow);
  } else if (startRow && !endRow) {
    gridRange.endRowIndex = gridRange.startRowIndex! + 1;
  }

  return gridRange;
}
