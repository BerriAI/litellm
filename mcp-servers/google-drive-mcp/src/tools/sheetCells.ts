// The pure, network-free half of getGoogleSheetCells: all of the address
// arithmetic, the field-mask builder, the matching of requested ranges onto
// the returned GridData, and truncation. Kept out of sheets.ts so it can be
// tested without stubbing googleapis - the same split batchPassthrough.ts uses.

import { convertA1ToGridRange } from '../utils.js';

/** 0-based column index -> A1 letters (0 -> A, 26 -> AA). */
export function columnLetter(index: number): string {
  let n = index + 1;
  let out = '';
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

/** 0-based grid coordinates -> an A1 address. */
export function cellA1(rowIndex: number, colIndex: number): string {
  return `${columnLetter(colIndex)}${rowIndex + 1}`;
}

/** The CellData fields this tool is willing to return. The names deliberately
 *  match the Sheets API's own field names, so the argument documents itself. */
export const CELL_FIELDS = [
  'userEnteredValue', 'effectiveValue', 'formattedValue', 'note',
  'dataValidation', 'userEnteredFormat', 'effectiveFormat', 'hyperlink', 'textFormatRuns',
] as const;

export const SHEET_METADATA = [
  'merges', 'hiddenRows', 'hiddenColumns', 'frozen', 'dimensionGroups', 'dimensionSizes',
] as const;

/** Fields returned when the caller doesn't specify `fields` explicitly. The
 *  single source of truth for that default: sheets.ts's zod schema reads it
 *  rather than repeating the literal. */
export const DEFAULT_CELL_FIELDS = ['userEnteredValue', 'effectiveValue', 'formattedValue'] as const;

/** The field mask for spreadsheets.get. Note: when a mask is supplied the
 *  includeGridData parameter is ignored - the grid comes back exactly when the
 *  mask asks for it, which is why startRow/startColumn are always requested. */
export function buildFieldMask(fields: string[], sheetMetadata: string[]): string {
  // Build sheets.properties inner part.
  const props = ['sheetId', 'title'];
  if (sheetMetadata.includes('frozen')) {
    props.push('gridProperties(frozenRowCount,frozenColumnCount)');
  }
  const propertiesPart = `properties(${props.join(',')})`;

  // Build sheets.data inner part.
  const dataParts = ['startRow', 'startColumn'];
  if (fields.length > 0) {
    dataParts.push(`rowData.values(${[...fields].sort().join(',')})`);
  }

  // Add row/column metadata.
  const rowMeta: string[] = [];
  const colMeta: string[] = [];
  if (sheetMetadata.includes('hiddenRows')) rowMeta.push('hiddenByUser', 'hiddenByFilter');
  if (sheetMetadata.includes('hiddenColumns')) colMeta.push('hiddenByUser', 'hiddenByFilter');
  if (sheetMetadata.includes('dimensionSizes')) { rowMeta.push('pixelSize'); colMeta.push('pixelSize'); }
  if (rowMeta.length > 0) dataParts.push(`rowMetadata(${rowMeta.join(',')})`);
  if (colMeta.length > 0) dataParts.push(`columnMetadata(${colMeta.join(',')})`);

  const dataPart = `data(${dataParts.join(',')})`;

  // Build the single sheets(...) root containing everything.
  const sheetsParts = [propertiesPart];
  if (sheetMetadata.includes('merges')) sheetsParts.push('merges');
  if (sheetMetadata.includes('dimensionGroups')) {
    sheetsParts.push('rowGroups', 'columnGroups');
  }
  sheetsParts.push(dataPart);

  return `properties.title,sheets(${sheetsParts.join(',')})`;
}

// Grammar for a bare (sheet-less) A1 range. Used only to tell a bare RANGE
// ("C6:E9", "A1", "C:EB") apart from a bare, unquoted SHEET NAME ("Probe",
// "Jan", "2024", "Q1") that also has no '!' of its own — Google itself
// resolves this exact ambiguity by grammar (a string valid as a range is a
// range on the first sheet; anything else is a sheet name), so this mirrors
// that rule rather than inventing a different one.
//
// A single-part form must be a full cell: both letters and digits. Letters
// alone ("Jan") or digits alone ("2024") are NOT ranges, and treating them as
// such routed month and year tab names to the first sheet. A two-part form
// needs each side to be a column, a row, or a cell.
const COL = '\\$?[A-Za-z]{1,3}';
const ROW = '\\$?[0-9]+';
const CELL = `${COL}${ROW}`;
const SIDE = `(?:${CELL}|${COL}|${ROW})`;
const BARE_RANGE_RE = new RegExp(`^(?:${CELL}|${SIDE}:${SIDE})$`);

function isBareCellRange(s: string): boolean {
  return s.length > 0 && /[A-Za-z0-9]/.test(s) && BARE_RANGE_RE.test(s);
}

export interface ParsedRange {
  /** Explicit sheet name, or null when the range names no sheet (first sheet applies). */
  sheetTitle: string | null;
  /** Bare A1 cell range, or null when the range is a whole-sheet reference
   *  (quoted or unquoted bare sheet name, no cell range of its own). */
  cellRange: string | null;
}

/** Splits a range into its sheet-qualifier and bare A1 part. Own parser, not
 *  utils.parseA1Range: that one is wrong for this purpose — it substitutes
 *  the literal string 'Sheet1' where the first sheet of the document is
 *  actually meant, and it doesn't distinguish a whole-sheet reference from a
 *  cell range at all. */
export function splitRange(range: string): ParsedRange {
  const trimmed = range.trim();
  if (trimmed.startsWith("'")) {
    let title = '';
    for (let i = 1; i < trimmed.length; i++) {
      if (trimmed[i] !== "'") { title += trimmed[i]; continue; }
      if (trimmed[i + 1] === "'") { title += "'"; i++; continue; }  // '' is an escaped quote
      const rest = trimmed.slice(i + 1);
      return rest.startsWith('!')
        ? { sheetTitle: title, cellRange: rest.slice(1) || null }
        : { sheetTitle: title, cellRange: null };  // the whole sheet
    }
    return { sheetTitle: null, cellRange: null };  // unterminated quote - let Google reject the range
  }
  const bang = trimmed.indexOf('!');
  if (bang !== -1) return { sheetTitle: trimmed.slice(0, bang), cellRange: trimmed.slice(bang + 1) || null };
  // No '!' and no quotes: either a bare range on the first sheet ("C6:E9") or
  // a bare whole-sheet name ("Probe"). Google tells the two apart by grammar;
  // see BARE_RANGE_RE above.
  return isBareCellRange(trimmed) ? { sheetTitle: null, cellRange: trimmed } : { sheetTitle: trimmed, cellRange: null };
}

/** The sheet name in an A1 range, or null when it names none (= first sheet). */
export function rangeSheetTitle(range: string): string | null {
  return splitRange(range).sheetTitle;
}

/** Quotes a sheet title back into A1 form only when needed — mirrors
 *  splitRange's own quoting rule so a title this module resolved round-trips
 *  back through splitRange unchanged. */
function quoteSheetTitle(title: string): string {
  const needsQuote = !/^[A-Za-z0-9_]+$/.test(title) || isBareCellRange(title);
  return needsQuote ? `'${title.replace(/'/g, "''")}'` : title;
}

export interface GridDataLike {
  startRow?: number | null;
  startColumn?: number | null;
  rowData?: Array<{ values?: Array<Record<string, unknown>> }> | null;
  rowMetadata?: Array<Record<string, unknown>> | null;
  columnMetadata?: Array<Record<string, unknown>> | null;
}

export interface SheetLike {
  properties?: { sheetId?: number | null; title?: string | null; gridProperties?: Record<string, unknown> | null } | null;
  merges?: Array<Record<string, unknown>> | null;
  rowGroups?: unknown[] | null;
  columnGroups?: unknown[] | null;
  data?: GridDataLike[] | null;
}

export interface RangeMatch {
  range: string;
  sheetId: number;
  sheetTitle: string;
  grid: GridDataLike;
  sheet: SheetLike;
}

/** Rebuilds the requested-range -> GridData correspondence. The response is
 *  grouped by sheet and the request order is preserved only WITHIN a sheet,
 *  so each sheet gets its own cursor. */
export function matchRangesToGridData(
  ranges: string[],
  sheets: SheetLike[],
): { matches: RangeMatch[] } | { error: string } {
  // Google resolves sheet titles case-insensitively - 'jan!A1' is answered
  // with the data of a sheet named 'Jan' - so the lookup and the per-sheet
  // cursors are keyed on a case-folded title. The sheet's own spelling, as the
  // API reported it, is what goes into sheetTitle.
  const fold = (title: string) => title.toLowerCase();
  const byTitle = new Map<string, SheetLike>();
  for (const sheet of sheets) {
    const title = sheet.properties?.title;
    if (typeof title === 'string') byTitle.set(fold(title), sheet);
  }
  const cursors = new Map<string, number>();
  const matches: RangeMatch[] = [];

  for (const range of ranges) {
    const requested = rangeSheetTitle(range);
    const title = requested ?? sheets[0]?.properties?.title ?? null;
    if (title === null) return { error: `Cannot resolve a sheet for range "${range}"` };

    const key = fold(title);
    const sheet = byTitle.get(key);
    if (!sheet) return { error: `Range "${range}" names sheet "${title}", which is not in the response` };

    const index = cursors.get(key) ?? 0;
    const grid = sheet.data?.[index];
    if (!grid) return { error: `Sheet "${title}" returned fewer grids than the ranges requested from it` };
    cursors.set(key, index + 1);

    matches.push({
      range,
      sheetId: sheet.properties?.sheetId ?? -1,
      sheetTitle: sheet.properties?.title ?? title,
      grid,
      sheet,
    });
  }
  return { matches };
}

export interface CellOut { a1: string; empty?: true; [field: string]: unknown }
export interface RangeResult { range: string; sheetId: number; sheetTitle: string; cells: CellOut[]; truncated: boolean }
export interface Budget { maxCells: number; maxBytes: number; includeEmpty: boolean }

/** The unread tail of a range, in A1, ready to be passed straight back in
 *  `ranges`. The start anchor is rebuilt from the response's own offsets
 *  (absolute and reliable); the end is reused from the request verbatim, so
 *  open-ended ranges like C:EB, which have no end row at all, survive.
 *
 *  The sheet prefix is always built from match.sheetTitle (already resolved by
 *  matchRangesToGridData) rather than re-sliced out of match.range: a range
 *  like 'Probe' (a bare sheet name, no '!') contains no '!' to slice on, and
 *  slicing dropped the sheet prefix entirely, so the continuation pointed at
 *  the document's first sheet instead of the requested one. */
export function remainderRange(match: RangeMatch, rowsConsumed: number, width: number): string {
  const startRow = match.grid.startRow ?? 0;
  const startColumn = match.grid.startColumn ?? 0;
  const anchor = `${columnLetter(startColumn)}${startRow + rowsConsumed + 1}`;
  const prefix = `${quoteSheetTitle(match.sheetTitle)}!`;

  const { cellRange } = splitRange(match.range);
  if (cellRange === null) {
    // A bare whole-sheet reference: the original request had no end to reuse.
    // Google returns rows spanning the sheet's real width, so the observed
    // width is that width; the end row is left open (no digit), the same trick
    // used for C:EB-style ranges below.
    return `${prefix}${anchor}:${columnLetter(startColumn + Math.max(width, 1) - 1)}`;
  }

  const colon = cellRange.indexOf(':');
  return colon === -1 ? `${prefix}${anchor}` : `${prefix}${anchor}:${cellRange.slice(colon + 1)}`;
}

export function extractRanges(
  matches: RangeMatch[],
  fields: string[],
  budget: Budget,
): { results: RangeResult[]; truncated: boolean; nextRanges: string[]; returned: { cells: number; bytes: number } } {
  const results: RangeResult[] = [];
  const nextRanges: string[] = [];
  let usedCells = 0;
  let usedBytes = 0;
  let exhausted = false;

  for (const match of matches) {
    if (exhausted) { nextRanges.push(match.range); continue; }

    const startRow = match.grid.startRow ?? 0;
    const startColumn = match.grid.startColumn ?? 0;
    const rows = match.grid.rowData ?? [];
    // The rectangle's width is the widest row returned: the API trims trailing
    // empty cells, and includeEmpty has to produce an even rectangle.
    const width = rows.reduce((max, row) => Math.max(max, row.values?.length ?? 0), 0);

    const cells: CellOut[] = [];
    let rowsConsumed = 0;
    let truncated = false;
    let untouched = false;

    for (let r = 0; r < rows.length; r++) {
      const values = rows[r].values ?? [];
      const rowCells: CellOut[] = [];
      let rowBytes = 0;

      for (let c = 0; c < width; c++) {
        const source = values[c] ?? {};
        const out: CellOut = { a1: cellA1(startRow + r, startColumn + c) };
        let present = false;
        for (const field of fields) {
          if (source[field] !== undefined) { out[field] = source[field]; present = true; }
        }
        if (!present) {
          if (!budget.includeEmpty) continue;
          out.empty = true;
        }
        rowCells.push(out);
        // Bytes, not UTF-16 code units: .length under-counts non-Latin text by
        // more than 2x, so a sheet of CJK content would overrun maxBytes.
        rowBytes += Buffer.byteLength(JSON.stringify(out)) + 1;
      }

      const wouldExceed =
        usedCells + rowCells.length > budget.maxCells || usedBytes + rowBytes > budget.maxBytes;
      // The guarantee is that the RESPONSE makes progress, not that every range
      // does. An over-budget first row goes through only when nothing has been
      // returned yet - otherwise a range whose single row does not fit would
      // never advance and the client would loop. Once something has been
      // returned, a range that cannot start is left untouched for the
      // continuation; admitting its first row anyway let a multi-range read
      // return many times maxCells with truncated:false.
      if (wouldExceed && rowsConsumed === 0 && usedCells > 0) { untouched = true; break; }
      if (wouldExceed && rowsConsumed > 0) { truncated = true; break; }

      cells.push(...rowCells);
      usedCells += rowCells.length;
      usedBytes += rowBytes;
      rowsConsumed = r + 1;
      if (wouldExceed) { truncated = r + 1 < rows.length; break; }
    }

    // Nothing of this range was read: it goes to the continuation verbatim,
    // exactly like a range the loop never reached, with no result block of its
    // own to suggest it was looked at.
    if (untouched) { nextRanges.push(match.range); exhausted = true; continue; }

    results.push({ range: match.range, sheetId: match.sheetId, sheetTitle: match.sheetTitle, cells, truncated });
    if (truncated) {
      nextRanges.push(remainderRange(match, rowsConsumed, width));
      exhausted = true;
    }
  }

  return {
    results,
    truncated: nextRanges.length > 0,
    nextRanges,
    returned: { cells: usedCells, bytes: usedBytes },
  };
}

interface Span { startRowIndex?: number; endRowIndex?: number; startColumnIndex?: number; endColumnIndex?: number }

function overlaps(a: Span, b: Span): boolean {
  const rowsMiss = a.endRowIndex !== undefined && b.startRowIndex !== undefined && a.endRowIndex <= b.startRowIndex;
  const rowsMiss2 = b.endRowIndex !== undefined && a.startRowIndex !== undefined && b.endRowIndex <= a.startRowIndex;
  const colsMiss = a.endColumnIndex !== undefined && b.startColumnIndex !== undefined && a.endColumnIndex <= b.startColumnIndex;
  const colsMiss2 = b.endColumnIndex !== undefined && a.startColumnIndex !== undefined && b.endColumnIndex <= a.startColumnIndex;
  return !(rowsMiss || rowsMiss2 || colsMiss || colsMiss2);
}

function spanToA1(span: Span): string {
  const start = `${columnLetter(span.startColumnIndex ?? 0)}${(span.startRowIndex ?? 0) + 1}`;
  const end = `${columnLetter((span.endColumnIndex ?? 1) - 1)}${span.endRowIndex ?? 1}`;
  return start === end ? start : `${start}:${end}`;
}

/** Sheet-level metadata: one block per sheet rather than a copy inside every
 *  range. Merges are returned at their FULL extent even when they reach outside
 *  the request - clipping them would hide the very fact that a merge is wider
 *  than the window that was read. */
export function collectSheetMetadata(
  matches: RangeMatch[],
  requested: string[],
): Record<string, Record<string, unknown>> {
  if (requested.length === 0) return {};
  const out: Record<string, Record<string, unknown>> = {};

  // The requested windows, per sheet - needed only to filter merges.
  // convertA1ToGridRange (via utils.parseA1Range) does not understand
  // $-absolute references or lowercase letters, and does not recognise a
  // whole-sheet reference at all, yet Google accepts all three without
  // complaint. So they are normalized here ($ stripped, letters upper-cased),
  // and anything still unparseable after that (a bare whole-sheet reference
  // included) is treated as an UNBOUNDED window: an empty Span constrains
  // nothing, so overlaps() considers it to intersect every merge on the sheet.
  // Neither throwing (a failed request) nor silently narrowing the window
  // (hidden merged cells) is acceptable - a window covering the whole sheet is
  // the only correct answer for "the whole sheet" and a safe fallback for
  // "could not parse".
  const windows = new Map<string, Span[]>();
  for (const match of matches) {
    const { cellRange } = splitRange(match.range);
    let span: Span;
    if (cellRange === null) {
      span = {};
    } else {
      const normalized = cellRange.replace(/\$/g, '').toUpperCase();
      try {
        span = convertA1ToGridRange(normalized, match.sheetId) as Span;
      } catch {
        span = {};
      }
    }
    const list = windows.get(match.sheetTitle) ?? [];
    list.push(span);
    windows.set(match.sheetTitle, list);
  }

  for (const match of matches) {
    const block = out[match.sheetTitle] ?? (out[match.sheetTitle] = {});
    const grid = match.grid;
    const startRow = grid.startRow ?? 0;
    const startColumn = grid.startColumn ?? 0;

    if (requested.includes('merges') && block.merges === undefined) {
      const spans = windows.get(match.sheetTitle) ?? [];
      block.merges = (match.sheet.merges ?? [])
        .filter(merge => spans.some(span => overlaps(merge as Span, span)))
        .map(merge => spanToA1(merge as Span));
    }
    if (requested.includes('frozen') && block.frozen === undefined) {
      const grid2 = match.sheet.properties?.gridProperties ?? {};
      block.frozen = {
        rows: (grid2 as Record<string, number>).frozenRowCount ?? 0,
        columns: (grid2 as Record<string, number>).frozenColumnCount ?? 0,
      };
    }
    if (requested.includes('dimensionGroups') && block.dimensionGroups === undefined) {
      const shape = (groups: unknown[] | null | undefined) => (groups ?? []).map(raw => {
        const group = raw as { range?: { startIndex?: number; endIndex?: number }; depth?: number; collapsed?: boolean };
        return { start: group.range?.startIndex ?? 0, end: group.range?.endIndex ?? 0,
                 depth: group.depth ?? 0, collapsed: group.collapsed === true };
      });
      block.dimensionGroups = { rows: shape(match.sheet.rowGroups), columns: shape(match.sheet.columnGroups) };
    }

    // Deliberate asymmetry with merges/frozen/dimensionGroups above: those come
    // from the sheet-level lists and describe the whole sheet, while
    // hiddenRows, hiddenColumns and dimensionSizes are read out of the grid's
    // rowMetadata/columnMetadata, which the API scopes to the requested window.
    // An empty hiddenRows therefore means "no hidden rows inside the ranges
    // that were read", not "this sheet has no hidden rows". The tool
    // description says the same thing (see sheetMetadata in sheets.ts).
    if (requested.includes('hiddenRows')) {
      const hidden = (block.hiddenRows as number[]) ?? [];
      (grid.rowMetadata ?? []).forEach((meta, i) => {
        if (meta.hiddenByUser === true || meta.hiddenByFilter === true) hidden.push(startRow + i + 1);
      });
      block.hiddenRows = [...new Set(hidden)].sort((a, b) => a - b);
    }
    if (requested.includes('hiddenColumns')) {
      const hidden = (block.hiddenColumns as string[]) ?? [];
      (grid.columnMetadata ?? []).forEach((meta, i) => {
        if (meta.hiddenByUser === true || meta.hiddenByFilter === true) hidden.push(columnLetter(startColumn + i));
      });
      block.hiddenColumns = [...new Set(hidden)];
    }
    if (requested.includes('dimensionSizes')) {
      const sizes = (block.dimensionSizes as { columnWidths: Record<string, number>; rowHeights: Record<string, number> })
        ?? { columnWidths: {}, rowHeights: {} };
      (grid.columnMetadata ?? []).forEach((meta, i) => {
        if (typeof meta.pixelSize === 'number') sizes.columnWidths[columnLetter(startColumn + i)] = meta.pixelSize;
      });
      (grid.rowMetadata ?? []).forEach((meta, i) => {
        if (typeof meta.pixelSize === 'number') sizes.rowHeights[String(startRow + i + 1)] = meta.pixelSize;
      });
      block.dimensionSizes = sizes;
    }
  }
  return out;
}
