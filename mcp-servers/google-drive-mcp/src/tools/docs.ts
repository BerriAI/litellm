import { Readable } from 'stream';
import { z } from 'zod';
import JSZip from 'jszip';
import type { ToolDefinition, ToolContext, ToolResult } from '../types.js';
import { errorResponse } from '../types.js';
import { escapeDriveQuery, isTextMime, ALL_DRIVES_LIST_PARAMS, DRIVE_ORDER_BY_VALUES } from '../utils.js';
import { downloadTextContent, writeTextContent } from './text-content.js';
import { uploadImageToDrive } from '../utils/driveImageUpload.js';
import { withRetry } from '../utils/retry.js';
import { getResponseHeader } from '../utils/streams.js';
import { describeRangeStyles, paragraphMetaBits, rgbColorToHex, summarizeDocumentStyles, type StyleSummarySegment } from './styleProjection.js';
import { collectDocPlainText, countOccurrences, diagnoseZeroMatch, findOccurrenceRanges, buildMultilineReplaceRequests } from './findDiagnostics.js';

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

// Upper bound on an inline image returned by getGoogleDocImage. Enforced up
// front via the request's maxContentLength (so an oversized body is aborted
// mid-download) and re-checked against the buffer as a backstop.
const MAX_IMAGE_BYTES = 40 * 1024 * 1024;

// Pure helper – no context needed
function hexToRgbColor(hex: string): { red: number; green: number; blue: number } | null {
  if (!hex) return null;
  let hexClean = hex.startsWith('#') ? hex.slice(1) : hex;

  if (hexClean.length === 3) {
    hexClean = hexClean[0] + hexClean[0] + hexClean[1] + hexClean[1] + hexClean[2] + hexClean[2];
  }
  // Require every character to be a hex digit. parseInt stops at the first
  // non-hex character, so "#12345G" would otherwise pass the length check and
  // silently become #012345.
  if (!/^[0-9a-fA-F]{6}$/.test(hexClean)) return null;
  const bigint = parseInt(hexClean, 16);

  const r = ((bigint >> 16) & 255) / 255;
  const g = ((bigint >> 8) & 255) / 255;
  const b = (bigint & 255) / 255;

  return { red: r, green: g, blue: b };
}

// Helper to recursively collect all tabs with their nesting level
function collectAllTabsWithLevel(tabs: any[], level: number = 0): Array<{ tab: any; level: number }> {
  const result: Array<{ tab: any; level: number }> = [];
  for (const tab of tabs) {
    result.push({ tab, level });
    if (tab.childTabs && tab.childTabs.length > 0) {
      result.push(...collectAllTabsWithLevel(tab.childTabs, level + 1));
    }
  }
  return result;
}

// Helper to recursively find a tab by ID in the tab tree
function findTabById(tabs: any[], targetId: string): any | null {
  for (const tab of tabs) {
    if (tab.tabProperties?.tabId === targetId) {
      return tab;
    }
    if (tab.childTabs && tab.childTabs.length > 0) {
      const found = findTabById(tab.childTabs, targetId);
      if (found) return found;
    }
  }
  return null;
}

// Escape square brackets so bracket-delimited placeholders / markdown links
// remain unambiguous when the payload itself contains `[` or `]`.
function escapeBrackets(s: string): string {
  return s.replace(/[\[\]]/g, '\\$&');
}

// Percent-encode the characters that would break a Markdown link destination
// `![alt](uri "title")` — spaces and parentheses (plus any stray control chars).
function escapeMarkdownUri(uri: string): string {
  return uri.replace(/[\s()]/g, (c) =>
    c === '(' ? '%28' : c === ')' ? '%29' : encodeURIComponent(c)
  );
}

// Normalized view of an inline image, resolved from the document's inlineObjects
// map. Returns null when the map is absent or has no entry for the id (callers
// fall back to a bare `[image]` placeholder).
interface InlineImageMeta {
  objectId: string;
  contentUri?: string;
  sourceUri?: string;
  altText?: string;
  width?: number;
  height?: number;
  unit?: string;
}

function getInlineImageMeta(inlineObjectId: string, inlineObjects?: any): InlineImageMeta | null {
  const obj = inlineObjects?.[inlineObjectId];
  if (!obj) return null;
  const embedded = obj.inlineObjectProperties?.embeddedObject;
  const meta: InlineImageMeta = { objectId: inlineObjectId };
  const img = embedded?.imageProperties;
  if (img?.contentUri) meta.contentUri = img.contentUri;
  if (img?.sourceUri) meta.sourceUri = img.sourceUri;
  const alt = embedded?.description || embedded?.title;
  // Collapse newlines/CR (and surrounding whitespace) to a single space so the
  // single-line placeholder invariant holds even for multi-line alt text.
  if (alt) meta.altText = String(alt).replace(/\s*[\r\n]+\s*/g, ' ');
  const w = embedded?.size?.width?.magnitude;
  const h = embedded?.size?.height?.magnitude;
  if (w != null) meta.width = Math.round(w);
  if (h != null) meta.height = Math.round(h);
  const unit = embedded?.size?.width?.unit || embedded?.size?.height?.unit;
  if (unit) meta.unit = String(unit).toLowerCase();
  return meta;
}

// Locate an inline object's embeddedObject by id across the whole document —
// every tab's inlineObjects map plus the legacy single-body map. Returns the
// embeddedObject (which carries imageProperties) or null when not found.
function findInlineObjectById(doc: any, inlineObjectId: string): any | null {
  const fromMap = (m: any) => m?.[inlineObjectId]?.inlineObjectProperties?.embeddedObject;
  const tabs = doc.tabs as any[] | undefined;
  if (tabs && tabs.length > 0) {
    for (const { tab } of collectAllTabsWithLevel(tabs)) {
      const embedded = fromMap(tab.documentTab?.inlineObjects);
      if (embedded) return embedded;
    }
  }
  return fromMap(doc.inlineObjects) || null;
}

// Single-line, self-describing image placeholder used by the text and indexed
// readers. Deliberately contains no newline (keeps index/pagination math intact)
// and no unescaped `|` (keeps table-cell escaping from mangling it).
function renderImagePlaceholder(meta: InlineImageMeta | null): string {
  if (!meta) return '[image]';
  const parts: string[] = [`objectId=${meta.objectId}`];
  if (meta.altText) parts.push(`alt="${escapeBrackets(meta.altText).replace(/"/g, '\\"')}"`);
  // Bracket-escape the URIs too: a raw `]` in a source/content URL would
  // otherwise prematurely close the `[image: ...]` delimiter.
  if (meta.contentUri) parts.push(`contentUri=${escapeBrackets(meta.contentUri)}`);
  if (meta.sourceUri) parts.push(`sourceUri=${escapeBrackets(meta.sourceUri)}`);
  if (meta.width != null && meta.height != null) {
    parts.push(`size=${meta.width}x${meta.height}${meta.unit || 'pt'}`);
  }
  return `[image: ${parts.join(' ')}]`;
}

// Markdown rendering: a real `![alt](uri)` carrying the durable objectId in the
// title slot so the caller can pass it to getGoogleDocImage. Falls back to the
// single-line placeholder when no fetchable URI is available.
function renderImageMarkdown(meta: InlineImageMeta | null): string {
  if (!meta) return '[image]';
  // Prefer the durable original source URL. The Google contentUri is ephemeral
  // (expires ~30 min), so embedding it in persisted markdown would 404; fall back
  // to it only when there is no source URL. The durable objectId stays in the
  // title slot for reliable re-fetch via getGoogleDocImage.
  const uri = meta.sourceUri || meta.contentUri;
  if (!uri) return renderImagePlaceholder(meta);
  const alt = meta.altText ? escapeBrackets(meta.altText) : '';
  return `![${alt}](${escapeMarkdownUri(uri)} "objectId=${meta.objectId}")`;
}

// Render a non-textRun paragraph element (person chip, rich link, image,
// footnote reference, horizontal rule) as text, or null when the element type
// carries no text. Shared by the markdown reader and the indexed reader; only
// image rendering differs between them.
function resolveInlineElementText(
  el: any,
  inlineObjects?: any,
  format: 'text' | 'markdown' = 'text'
): string | null {
  if (el.person?.personProperties) {
    const p = el.person.personProperties;
    if (p.name && p.email) return `@${p.name} (${p.email})`;
    return `@${p.name || p.email || ''}`;
  }
  if (el.richLink?.richLinkProperties) {
    const rl = el.richLink.richLinkProperties;
    const title = escapeBrackets(rl.title || rl.uri || '');
    const uri = rl.uri;
    return title && uri ? `[${title}](${uri})` : title || null;
  }
  if (el.inlineObjectElement?.inlineObjectId) {
    const meta = getInlineImageMeta(el.inlineObjectElement.inlineObjectId, inlineObjects);
    return format === 'markdown' ? renderImageMarkdown(meta) : renderImagePlaceholder(meta);
  }
  if (el.footnoteReference) {
    return `[^${el.footnoteReference.footnoteNumber || ''}]`;
  }
  if (el.horizontalRule) {
    return '---\n';
  }
  return null;
}

// Google Docs named paragraph styles that map onto ATX headings. TITLE and
// SUBTITLE have no exact ATX equivalent, but leaving them unmapped drops the
// most prominent heading in the document from the outline entirely.
const MARKDOWN_HEADING_PREFIX: Record<string, string> = {
  TITLE: '#',
  SUBTITLE: '##',
  HEADING_1: '#',
  HEADING_2: '##',
  HEADING_3: '###',
  HEADING_4: '####',
  HEADING_5: '#####',
  HEADING_6: '######',
};

// `# ` / `## ` / … for a heading paragraph, '' for body text.
function markdownHeadingPrefix(paragraph: any): string {
  const hashes = MARKDOWN_HEADING_PREFIX[paragraph?.paragraphStyle?.namedStyleType];
  return hashes ? `${hashes} ` : '';
}

// `- ` / `1. ` for a list paragraph, indented to its parent item's content
// column. The indent has to be measured, not assumed: a fixed two spaces per
// level is short of the three an ordered `1. ` parent needs, and CommonMark
// reads an under-indented child as a sibling list rather than a nested one.
// Marker widths are recorded per list as the levels are encountered, so a
// child always knows how wide its actual ancestors were.
function createListPrefixer(lists?: any) {
  const markerWidths = new Map<string, number[]>();

  return (paragraph: any): string => {
    const bullet = paragraph?.bullet;
    if (!bullet) return '';
    const nestingLevel = bullet.nestingLevel ?? 0;
    const level = lists?.[bullet.listId]?.listProperties?.nestingLevels?.[nestingLevel];
    const glyphType = level?.glyphType;
    const ordered = !!glyphType && glyphType !== 'NONE' && glyphType !== 'GLYPH_TYPE_UNSPECIFIED';
    // Every ordered item is emitted as `1.` — markdown renderers renumber
    // automatically, and the Docs model has no reliable per-item ordinal here.
    const marker = ordered ? '1. ' : '- ';

    const widths = markerWidths.get(bullet.listId) ?? [];
    let indent = 0;
    for (let i = 0; i < nestingLevel; i++) indent += widths[i] ?? 2;
    widths[nestingLevel] = marker.length;
    // Levels deeper than this one are stale once the list comes back up.
    widths.length = nestingLevel + 1;
    markerWidths.set(bullet.listId, widths);

    return `${' '.repeat(indent)}${marker}`;
  };
}

type EmphasisKey = 'bold' | 'italic' | 'strikethrough';

// Ordered outermost-first, so the markers a run shares with its neighbours are
// always opened in the same sequence and can stay open across run boundaries.
const EMPHASIS_MARKERS: Array<{ key: EmphasisKey; marker: string }> = [
  { key: 'bold', marker: '**' },
  { key: 'italic', marker: '*' },
  { key: 'strikethrough', marker: '~~' },
];

interface MarkdownRun {
  text: string;
  keys: EmphasisKey[];
  // Already-rendered markdown (an image, chip or footnote marker) — never
  // escaped, never emphasised.
  literal?: boolean;
}

function emphasisKeys(textStyle: any): EmphasisKey[] {
  return EMPHASIS_MARKERS.filter(m => textStyle?.[m.key]).map(m => m.key);
}

// Escape the inline metacharacters that would otherwise change how the
// document's own text renders. A literal `*` inside a bold run is the worst
// case: it pairs with one of the surrounding markers and mangles the span.
// `_` is only emphasis at a word boundary in CommonMark, so snake_case stays
// readable.
function escapeMarkdownText(text: string): string {
  return text.replace(/[\\`*~[\]]|(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])/g, m => `\\${m}`);
}

// Emphasis markers must sit tight against visible text, so whitespace at a
// styled run's edges — including the paragraph's trailing newline — is pulled
// out into unstyled runs. Neighbouring runs that share a style set are then
// merged: Docs splits a visually continuous span at every style boundary, and
// wrapping each fragment separately produces fused delimiter runs such as
// `**re*****ally***`, which renders as literal asterisks.
function normalizeMarkdownRuns(runs: MarkdownRun[]): MarkdownRun[] {
  const mergeAdjacent = (input: MarkdownRun[]): MarkdownRun[] => {
    const out: MarkdownRun[] = [];
    for (const run of input) {
      const prev = out[out.length - 1];
      const mergeable =
        prev && !prev.literal && !run.literal &&
        prev.keys.length === run.keys.length &&
        prev.keys.every((k, i) => k === run.keys[i]);
      if (mergeable) {
        prev.text += run.text;
      } else {
        out.push({ ...run });
      }
    }
    return out;
  };

  // Coalesce before measuring edges: a span Docs fragmented mid-word has to be
  // one run first, or the whitespace between the fragments reads as an edge and
  // the span is split back apart.
  const split: MarkdownRun[] = [];
  for (const run of mergeAdjacent(runs)) {
    if (run.literal) {
      split.push(run);
      continue;
    }
    if (run.keys.length === 0 || run.text.trim() === '') {
      split.push({ text: run.text, keys: [] });
      continue;
    }
    const [, lead, core, trail] = /^(\s*)([\s\S]*?)(\s*)$/.exec(run.text) ?? [];
    if (lead) split.push({ text: lead, keys: [] });
    split.push({ text: core, keys: run.keys });
    if (trail) split.push({ text: trail, keys: [] });
  }
  return mergeAdjacent(split);
}

// Render a paragraph's runs, opening and closing emphasis markers across run
// boundaries so spans nest properly instead of being reopened per run.
function renderMarkdownRuns(runs: MarkdownRun[]): string {
  const markerFor = (key: EmphasisKey) => EMPHASIS_MARKERS.find(m => m.key === key)!.marker;
  let result = '';
  let open: EmphasisKey[] = [];

  for (const run of normalizeMarkdownRuns(runs)) {
    const keys = run.literal ? [] : run.keys;
    // Close markers that are no longer active, plus everything opened after
    // them — markers only nest, so they close in reverse order.
    const stale = open.findIndex(k => !keys.includes(k));
    if (stale !== -1) {
      for (let i = open.length - 1; i >= stale; i--) result += markerFor(open[i]);
      open = open.slice(0, stale);
    }
    for (const m of EMPHASIS_MARKERS) {
      if (keys.includes(m.key) && !open.includes(m.key)) {
        result += m.marker;
        open.push(m.key);
      }
    }
    result += run.literal ? run.text : escapeMarkdownText(run.text);
  }

  for (let i = open.length - 1; i >= 0; i--) result += markerFor(open[i]);
  return result;
}

// Assemble a GFM pipe table from already-escaped cell text.
//
// GFM has no colspan, and it sizes the whole table from the header row: a
// merged first-row cell would otherwise define a one-column table and every
// body column past the first would be dropped from the rendering. Rows are
// padded to the widest row so no cell is lost.
function renderPipeTable(rows: string[][]): string {
  if (rows.length === 0) return '';
  const width = Math.max(...rows.map(r => r.length));
  const line = (cells: string[]) =>
    `| ${Array.from({ length: width }, (_, i) => cells[i] ?? '').join(' | ')} |`;
  const lines = [line(rows[0]), `| ${Array.from({ length: width }, () => '---').join(' | ')} |`];
  for (const row of rows.slice(1)) lines.push(line(row));
  return lines.join('\n');
}

// One entry per column the row occupies: a cell merged across N columns
// contributes its text plus N-1 empty cells, keeping later rows aligned.
function expandRowCells(row: any, renderCell: (cell: any) => string): string[] {
  const cells: string[] = [];
  for (const cell of row.tableCells || []) {
    cells.push(renderCell(cell));
    const span = cell.tableCellStyle?.columnSpan ?? 1;
    for (let i = 1; i < span; i++) cells.push('');
  }
  return cells;
}

// A newline would terminate the table row, so multi-paragraph cell content is
// joined with `<br>` rather than flattened into one run-on line. `|` has to be
// escaped or it reads as a column separator.
function escapeTableCell(text: string): string {
  return text
    .replace(/\|/g, '\\|')
    .replace(/[ \t]*\n[ \t]*/g, '<br>')
    .replace(/^(?:<br>)+|(?:<br>)+$/g, '')
    .trim();
}

// Extract text from body content (paragraphs + tables). Inline images are
// rendered from `inlineObjects` (markdown `![alt](uri)` or a single-line
// placeholder) instead of being dropped.
//
// In markdown mode the document's structure is carried over too: heading
// paragraphs get their ATX prefix, list paragraphs a `- `/`1. ` marker indented
// by nesting level, styled runs their emphasis markers, and tables render as
// pipe tables. Text mode is unchanged — plain runs, tab-separated table cells.
function extractText(
  bodyContent: any[],
  inlineObjects?: any,
  format: 'text' | 'markdown' = 'text',
  lists?: any
): string {
  if (format === 'markdown') {
    return extractMarkdown(bodyContent, inlineObjects, lists);
  }

  const renderImage = (id: string): string =>
    renderImagePlaceholder(getInlineImageMeta(id, inlineObjects));

  let result = '';
  for (const element of bodyContent) {
    if (element.paragraph?.elements) {
      for (const elem of element.paragraph.elements) {
        if (elem.textRun?.content) {
          result += elem.textRun.content;
        } else if (elem.inlineObjectElement?.inlineObjectId) {
          result += renderImage(elem.inlineObjectElement.inlineObjectId);
        }
      }
    } else if (element.table) {
      for (const row of element.table.tableRows || []) {
        for (const cell of row.tableCells || []) {
          for (const cellContent of cell.content || []) {
            if (cellContent.paragraph?.elements) {
              for (const elem of cellContent.paragraph.elements) {
                if (elem.textRun?.content) {
                  result += elem.textRun.content;
                } else if (elem.inlineObjectElement?.inlineObjectId) {
                  result += renderImage(elem.inlineObjectElement.inlineObjectId);
                }
              }
            }
          }
          result += '\t';
        }
        result += '\n';
      }
    }
  }
  return result;
}

// Markdown rendering of body content, assembled as discrete blocks rather than
// by string concatenation: markdown separates blocks with a blank line, and a
// single newline between two paragraphs is only a soft break — the renderer
// would merge them back into one paragraph and the document's most basic
// structure would be lost. Consecutive list items stay one newline apart so a
// list renders tight rather than as a series of loose paragraphs.
function extractMarkdown(bodyContent: any[], inlineObjects?: any, lists?: any): string {
  const listPrefix = createListPrefixer(lists);
  const blocks: Array<{ text: string; list: boolean }> = [];

  const paragraphRuns = (paragraph: any): MarkdownRun[] => {
    const runs: MarkdownRun[] = [];
    for (const elem of paragraph.elements || []) {
      if (elem.textRun?.content) {
        runs.push({ text: elem.textRun.content, keys: emphasisKeys(elem.textRun.textStyle) });
      } else {
        const inline = resolveInlineElementText(elem, inlineObjects, 'markdown');
        if (inline) runs.push({ text: inline, keys: [], literal: true });
      }
    }
    return runs;
  };

  // The paragraph's own trailing newline is dropped — block separation is the
  // joiner's job below.
  const paragraphBody = (paragraph: any): string =>
    renderMarkdownRuns(paragraphRuns(paragraph)).replace(/\n+$/, '');

  const renderCell = (cell: any): string => {
    // A fresh prefixer per cell: list nesting inside a cell is independent of
    // the surrounding document's lists.
    const cellPrefix = createListPrefixer(lists);
    const lines: string[] = [];
    for (const cellContent of cell.content || []) {
      if (!cellContent.paragraph?.elements) continue;
      const body = paragraphBody(cellContent.paragraph);
      if (body.trim() === '') continue;
      lines.push((cellContent.paragraph.bullet ? cellPrefix(cellContent.paragraph) : '') + body);
    }
    return escapeTableCell(lines.join('\n'));
  };

  for (const element of bodyContent) {
    if (element.paragraph?.elements) {
      const body = paragraphBody(element.paragraph);
      // Empty paragraphs carry no content and would only emit a bare `## ` or
      // `- `; block separation already reproduces the vertical space.
      if (body.trim() === '') continue;
      const bullet = element.paragraph.bullet;
      // A bulleted paragraph reads as a list item even when it also carries a
      // heading style, so the list prefix wins.
      const prefix = bullet
        ? listPrefix(element.paragraph)
        : markdownHeadingPrefix(element.paragraph);
      blocks.push({ text: prefix + body, list: !!bullet });
    } else if (element.table?.tableRows) {
      const rows = element.table.tableRows.map((row: any) => expandRowCells(row, renderCell));
      const table = renderPipeTable(rows);
      if (table) blocks.push({ text: table, list: false });
    }
  }

  let result = '';
  blocks.forEach((block, i) => {
    if (i > 0) result += block.list && blocks[i - 1].list ? '\n' : '\n\n';
    result += block.text;
  });
  return result === '' ? '' : `${result}\n`;
}

// Prepend the document title as the markdown H1. A TITLE-styled paragraph in
// the body renders as `# Title` in its own right, so when the two say the same
// thing the prefix is dropped rather than giving the document two identical H1s.
function withMarkdownTitle(title: string | null | undefined, body: string): string {
  const heading = `# ${title}`;
  if (body === heading || body.startsWith(`${heading}\n`)) return body;
  return `${heading}\n\n${body}`;
}

// Extract full text from a Google Doc, handling tabs and optional tabId filtering
function extractDocText(
  doc: any,
  tabId?: string,
  format: 'text' | 'markdown' = 'text'
): { text: string; error?: string } {
  let text = '';
  const tabs = doc.tabs as any[] | undefined;

  if (tabs && tabs.length > 0) {
    if (tabId) {
      const tab = findTabById(tabs, tabId);
      if (!tab) {
        return { text: '', error: `Tab with ID "${tabId}" not found. Use listDocumentTabs to see available tabs.` };
      }
      const bodyContent = tab.documentTab?.body?.content;
      if (bodyContent) {
        text = extractText(bodyContent, tab.documentTab?.inlineObjects, format, tab.documentTab?.lists);
      }
    } else {
      const allTabs = collectAllTabsWithLevel(tabs);
      const isMultiTab = allTabs.length > 1;
      for (const { tab, level } of allTabs) {
        const bodyContent = tab.documentTab?.body?.content;
        if (isMultiTab) {
          const title = tab.tabProperties?.title || 'Untitled';
          const indent = '  '.repeat(level);
          // The blank line keeps the header its own markdown block; without it
          // the tab's first paragraph folds into the header line.
          text += `${indent}=== Tab: ${title} ===\n${format === 'markdown' ? '\n' : ''}`;
        }
        if (bodyContent) {
          text += extractText(bodyContent, tab.documentTab?.inlineObjects, format, tab.documentTab?.lists);
        }
        if (isMultiTab) {
          text += '\n';
        }
      }
    }
  } else {
    const body = doc.body;
    if (body?.content) {
      text = extractText(body.content, doc.inlineObjects, format, doc.lists);
    }
  }

  return { text };
}

// The Docs API `revisionId` is what `WriteControl.requiredRevisionId` expects.
// It is a different identifier from the Drive `version` integer that
// getDocumentInfo reports and from the Drive revision ids getRevisions lists;
// neither of those is accepted as a lock, so the read tools have to surface the
// Docs one explicitly. Docs populates it only for callers with edit access, so
// a read-only caller is told why the lock is unavailable rather than silently
// getting no value and no explanation.
//
// Two limits worth knowing, both from the Docs API reference: a revisionId is
// only guaranteed valid for 24 hours, and it cannot be shared across users. The
// second matters in this server's multi-account mode — a revisionId read under
// one `manage_accounts` alias will not lock a write issued under another, so
// read and write under the same account.
const REVISION_UNAVAILABLE = 'unavailable (no edit access)';

function revisionIdOf(doc: { revisionId?: string | null } | null | undefined): string {
  return doc?.revisionId || REVISION_UNAVAILABLE;
}

function writeControlFor(ifRevisionId?: string): { writeControl?: { requiredRevisionId: string } } {
  return ifRevisionId ? { writeControl: { requiredRevisionId: ifRevisionId } } : {};
}

// Execute batch update for Google Docs
async function executeBatchUpdate(ctx: ToolContext, documentId: string, requests: any[], ifRevisionId?: string): Promise<any> {
  if (!requests || requests.length === 0) {
    return {};
  }

  const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

  try {
    const response = await docs.documents.batchUpdate({
      documentId: documentId,
      requestBody: { requests, ...writeControlFor(ifRevisionId) },
    });
    return response.data;
  } catch (error: any) {
    ctx.log('Google Docs batchUpdate error:', error.message);
    // gaxios 7 only sets a numeric `code` for AIP-193-shaped errors; `status`
    // is the reliable field. Keep `code` for gaxios-6-shaped callers/mocks.
    if ((error.status ?? error.code) === 404) throw new Error(`Document not found (ID: ${documentId})`);
    if ((error.status ?? error.code) === 403) throw new Error(`Permission denied for document (ID: ${documentId})`);
    throw new Error(`Google Docs API Error: ${error.message}`);
  }
}

// Resolve the body segments of an already-fetched document into distinct index
// spaces. Every tab restarts indices at 1, so a caller that merges them gets
// several paragraphs with identical ranges and no way to tell which one an
// index-taking write should target. `label` is set only when the document has
// more than one tab, so a single-tab document keeps reporting bare indices.
//
// Takes the document data rather than fetching: the indices a caller acts on
// must come from the same read as the revisionId they lock against.
function resolveBodySegments(
  docData: any,
  tabId?: string,
): { segments: StyleSummarySegment[] } | { error: string } {
  const tabs = docData?.tabs as any[] | undefined;

  if (!tabs || tabs.length === 0) {
    if (tabId) {
      return { error: `Tab with ID "${tabId}" not found. Use listDocumentTabs to see available tabs.` };
    }
    return { segments: [{ content: docData?.body?.content }] };
  }

  if (tabId) {
    const tab = findTabById(tabs, tabId);
    if (!tab) {
      return { error: `Tab with ID "${tabId}" not found. Use listDocumentTabs to see available tabs.` };
    }
    return { segments: [{ content: tab.documentTab?.body?.content }] };
  }

  const all = collectAllTabsWithLevel(tabs);
  const isMultiTab = all.length > 1;
  return {
    segments: all.map(({ tab }) => ({
      label: isMultiTab ? (tab.tabProperties?.title || 'Untitled') : undefined,
      content: tab.documentTab?.body?.content,
    })),
  };
}

// Resolve a specific tab's body content array. A narrow `fields` projection
// that names only `body(...)` would exclude `tabs` from the response even with
// includeTabsContent, so we omit `fields` here (same precedent as
// updateGoogleDoc / readGoogleDoc). Returns the standard not-found error when
// the tabId can't be resolved (consistent with updateGoogleDoc's tab path).
async function getTabBodyContent(
  ctx: ToolContext,
  documentId: string,
  tabId: string,
): Promise<{ content?: any[]; error?: string }> {
  const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
  const res = await docs.documents.get({ documentId, includeTabsContent: true });
  const tabs = (res.data as any).tabs as any[] | undefined;
  const tab = tabs ? findTabById(tabs, tabId) : null;
  if (!tab) {
    return { error: `Tab with ID "${tabId}" not found. Use listDocumentTabs to see available tabs.` };
  }
  return { content: tab.documentTab?.body?.content ?? [] };
}

// Thread an optional tabId into a Docs API range/location object. One shared
// spelling for "scope this to a tab iff one was given", so every editing
// handler stays consistent (#114).
function withTab<T extends object>(target: T, tabId?: string): T & { tabId?: string } {
  return tabId ? { ...target, tabId } : target;
}

// Find text in a document and return the range indices.
// Pass `preResolvedContent` to reuse an already-fetched body content array
// (e.g. a tab body resolved once by the caller) and skip the internal GET —
// see applyParagraphStyle's tab-scoped textToFind path (#114 follow-up).
// `segmentEndIndex` is the end index of the searched body/tab segment (the
// index just past its final paragraph break). Callers that delete the match or
// insert after it use it to stay inside the segment: Docs rejects deleting the
// final newline and inserting at the segment end (#183 follow-up).
async function findTextRange(ctx: ToolContext, documentId: string, textToFind: string, instance: number = 1, tabId?: string, preResolvedContent?: any[]): Promise<{ startIndex: number; endIndex: number; segmentEndIndex?: number } | { error: string } | null> {
  const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

  try {
    let content: any[];
    if (preResolvedContent !== undefined) {
      content = preResolvedContent;
    } else if (tabId) {
      const resolved = await getTabBodyContent(ctx, documentId, tabId);
      if (resolved.error) return { error: resolved.error };
      content = resolved.content!;
    } else {
      const res = await docs.documents.get({
        documentId,
        fields: 'body(content(paragraph(elements(startIndex,endIndex,textRun(content))),table,startIndex,endIndex))',
      });
      if (!res.data.body?.content) {
        return null;
      }
      content = res.data.body.content;
    }

    // Collect all text segments with their positions
    let fullText = '';
    const segments: { text: string; start: number; end: number }[] = [];

    const collectTextFromContent = (content: any[]) => {
      content.forEach(element => {
        if (element.paragraph?.elements) {
          element.paragraph.elements.forEach((pe: any) => {
            if (pe.textRun?.content && pe.startIndex !== undefined && pe.endIndex !== undefined) {
              const text = pe.textRun.content;
              fullText += text;
              segments.push({ text, start: pe.startIndex, end: pe.endIndex });
            }
          });
        }

        // Handle tables recursively
        if (element.table?.tableRows) {
          element.table.tableRows.forEach((row: any) => {
            if (row.tableCells) {
              row.tableCells.forEach((cell: any) => {
                if (cell.content) {
                  collectTextFromContent(cell.content);
                }
              });
            }
          });
        }
      });
    };

    collectTextFromContent(content);
    segments.sort((a, b) => a.start - b.start);

    // The segment end is the last structural element's endIndex. Fall back to
    // the furthest text run when the structural index is absent (the body's
    // closing paragraph always sorts last, after any tables).
    const lastElement = content.length ? content[content.length - 1] : undefined;
    const segmentEndIndex: number | undefined = typeof lastElement?.endIndex === 'number'
      ? lastElement.endIndex
      : (segments.length ? segments[segments.length - 1].end : undefined);

    // Find the specified instance
    let foundCount = 0;
    let searchStartIndex = 0;

    while (foundCount < instance) {
      const currentIndex = fullText.indexOf(textToFind, searchStartIndex);
      if (currentIndex === -1) break;

      foundCount++;

      if (foundCount === instance) {
        const targetStartInFullText = currentIndex;
        const targetEndInFullText = currentIndex + textToFind.length;
        let currentPosInFullText = 0;
        let startIndex = -1;
        let endIndex = -1;

        for (const seg of segments) {
          const segStartInFullText = currentPosInFullText;
          const segEndInFullText = segStartInFullText + seg.text.length;

          if (startIndex === -1 && targetStartInFullText >= segStartInFullText && targetStartInFullText < segEndInFullText) {
            startIndex = seg.start + (targetStartInFullText - segStartInFullText);
          }

          if (targetEndInFullText > segStartInFullText && targetEndInFullText <= segEndInFullText) {
            endIndex = seg.start + (targetEndInFullText - segStartInFullText);
            break;
          }

          currentPosInFullText = segEndInFullText;
        }

        if (startIndex !== -1 && endIndex !== -1) {
          return segmentEndIndex !== undefined ? { startIndex, endIndex, segmentEndIndex } : { startIndex, endIndex };
        }
      }

      searchStartIndex = currentIndex + 1;
    }

    return null;
  } catch (error: any) {
    ctx.log('Error finding text in document:', error.message);
    if ((error.status ?? error.code) === 404) throw new Error(`Document not found (ID: ${documentId})`);
    throw new Error(`Failed to search document: ${error.message}`);
  }
}

// Get paragraph range containing a specific index.
// `preResolvedContent` works as in findTextRange: supply an already-fetched
// body content array to skip the internal GET (#114 follow-up).
async function getParagraphRange(ctx: ToolContext, documentId: string, indexWithin: number, tabId?: string, preResolvedContent?: any[]): Promise<{ startIndex: number; endIndex: number } | { error: string } | null> {
  const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

  try {
    let content: any[];
    if (preResolvedContent !== undefined) {
      content = preResolvedContent;
    } else if (tabId) {
      const resolved = await getTabBodyContent(ctx, documentId, tabId);
      if (resolved.error) return { error: resolved.error };
      content = resolved.content!;
    } else {
      const res = await docs.documents.get({
        documentId,
        fields: 'body(content(startIndex,endIndex,paragraph,table))',
      });
      if (!res.data.body?.content) {
        return null;
      }
      content = res.data.body.content;
    }

    const findParagraphInContent = (content: any[]): { startIndex: number; endIndex: number } | null => {
      for (const element of content) {
        if (element.startIndex !== undefined && element.endIndex !== undefined) {
          if (indexWithin >= element.startIndex && indexWithin < element.endIndex) {
            if (element.paragraph) {
              return { startIndex: element.startIndex, endIndex: element.endIndex };
            }

            // Check table cells recursively
            if (element.table?.tableRows) {
              for (const row of element.table.tableRows) {
                if (row.tableCells) {
                  for (const cell of row.tableCells) {
                    if (cell.content) {
                      const result = findParagraphInContent(cell.content);
                      if (result) return result;
                    }
                  }
                }
              }
            }
          }
        }
      }
      return null;
    };

    return findParagraphInContent(content);
  } catch (error: any) {
    ctx.log('Error getting paragraph range:', error.message);
    throw new Error(`Failed to find paragraph: ${error.message}`);
  }
}

// Pure helper – build text style update request
function buildUpdateTextStyleRequest(
  startIndex: number,
  endIndex: number,
  style: {
    bold?: boolean;
    italic?: boolean;
    underline?: boolean;
    strikethrough?: boolean;
    fontSize?: number;
    fontFamily?: string;
    foregroundColor?: string;
    backgroundColor?: string;
    linkUrl?: string;
    baselineOffset?: 'SUPERSCRIPT' | 'SUBSCRIPT' | 'NONE';
  },
  tabId?: string
): { request: any; fields: string[] } | null {
  const textStyle: any = {};
  const fieldsToUpdate: string[] = [];

  if (style.bold !== undefined) { textStyle.bold = style.bold; fieldsToUpdate.push('bold'); }
  if (style.italic !== undefined) { textStyle.italic = style.italic; fieldsToUpdate.push('italic'); }
  if (style.underline !== undefined) { textStyle.underline = style.underline; fieldsToUpdate.push('underline'); }
  if (style.strikethrough !== undefined) { textStyle.strikethrough = style.strikethrough; fieldsToUpdate.push('strikethrough'); }
  if (style.fontSize !== undefined) { textStyle.fontSize = { magnitude: style.fontSize, unit: 'PT' }; fieldsToUpdate.push('fontSize'); }
  if (style.fontFamily !== undefined) { textStyle.weightedFontFamily = { fontFamily: style.fontFamily }; fieldsToUpdate.push('weightedFontFamily'); }

  if (style.foregroundColor !== undefined) {
    const rgbColor = hexToRgbColor(style.foregroundColor);
    if (!rgbColor) throw new Error(`Invalid foreground hex color: ${style.foregroundColor}`);
    textStyle.foregroundColor = { color: { rgbColor } };
    fieldsToUpdate.push('foregroundColor');
  }

  if (style.backgroundColor !== undefined) {
    const rgbColor = hexToRgbColor(style.backgroundColor);
    if (!rgbColor) throw new Error(`Invalid background hex color: ${style.backgroundColor}`);
    textStyle.backgroundColor = { color: { rgbColor } };
    fieldsToUpdate.push('backgroundColor');
  }

  if (style.linkUrl !== undefined) {
    textStyle.link = { url: style.linkUrl };
    fieldsToUpdate.push('link');
  }

  if (style.baselineOffset !== undefined) {
    textStyle.baselineOffset = style.baselineOffset;
    fieldsToUpdate.push('baselineOffset');
  }

  if (fieldsToUpdate.length === 0) return null;

  return {
    request: {
      updateTextStyle: {
        range: withTab({ startIndex, endIndex }, tabId),
        textStyle,
        fields: fieldsToUpdate.join(','),
      }
    },
    fields: fieldsToUpdate
  };
}

// Paragraph border edges as exposed on applyParagraphStyle. "between" is the
// rule Docs draws between consecutive paragraphs that share identical border
// settings.
const PARAGRAPH_BORDER_EDGES = ['top', 'bottom', 'left', 'right', 'between'] as const;
type ParagraphBorderEdge = (typeof PARAGRAPH_BORDER_EDGES)[number];

export interface ParagraphBorderInput {
  color?: string;
  width?: number;
  padding?: number;
  dashStyle?: 'SOLID' | 'DOT' | 'DASH';
}

const PARAGRAPH_BORDER_KEYS = {
  top: 'borderTop',
  bottom: 'borderBottom',
  left: 'borderLeft',
  right: 'borderRight',
  between: 'borderBetween',
} as const satisfies Record<ParagraphBorderEdge, string>;

// Compile a border param into the API's ParagraphBorder. The API refuses a
// partially-specified border ("Unsupported dimension unit: UNIT_UNSPECIFIED"
// when padding is omitted), so every subfield gets a default: black, 1pt
// wide, 1pt padding, SOLID.
function buildParagraphBorder(input: ParagraphBorderInput): any {
  const rgbColor = hexToRgbColor(input.color ?? '#000000');
  if (!rgbColor) throw new Error(`Invalid border hex color: ${input.color}`);
  return {
    color: { color: { rgbColor } },
    width: { magnitude: input.width ?? 1, unit: 'PT' },
    padding: { magnitude: input.padding ?? 1, unit: 'PT' },
    dashStyle: input.dashStyle ?? 'SOLID',
  };
}

// Pure helper – build paragraph style update request
export function buildUpdateParagraphStyleRequest(
  startIndex: number,
  endIndex: number,
  style: {
    alignment?: 'START' | 'END' | 'CENTER' | 'JUSTIFIED';
    indentStart?: number;
    indentEnd?: number;
    indentFirstLine?: number;
    spaceAbove?: number;
    spaceBelow?: number;
    namedStyleType?: string;
    keepWithNext?: boolean;
    keepLinesTogether?: boolean;
    avoidWidowAndOrphan?: boolean;
    pageBreakBefore?: boolean;
    borderTop?: ParagraphBorderInput;
    borderBottom?: ParagraphBorderInput;
    borderLeft?: ParagraphBorderInput;
    borderRight?: ParagraphBorderInput;
    borderBetween?: ParagraphBorderInput;
    removeBorders?: Array<ParagraphBorderEdge | 'all'>;
    shading?: string;
    removeShading?: boolean;
  },
  tabId?: string
): { request: any; fields: string[] } | null {
  const paragraphStyle: any = {};
  const fieldsToUpdate: string[] = [];

  if (style.alignment !== undefined) { paragraphStyle.alignment = style.alignment; fieldsToUpdate.push('alignment'); }
  if (style.indentStart !== undefined) { paragraphStyle.indentStart = { magnitude: style.indentStart, unit: 'PT' }; fieldsToUpdate.push('indentStart'); }
  if (style.indentEnd !== undefined) { paragraphStyle.indentEnd = { magnitude: style.indentEnd, unit: 'PT' }; fieldsToUpdate.push('indentEnd'); }
  if (style.indentFirstLine !== undefined) { paragraphStyle.indentFirstLine = { magnitude: style.indentFirstLine, unit: 'PT' }; fieldsToUpdate.push('indentFirstLine'); }
  if (style.spaceAbove !== undefined) { paragraphStyle.spaceAbove = { magnitude: style.spaceAbove, unit: 'PT' }; fieldsToUpdate.push('spaceAbove'); }
  if (style.spaceBelow !== undefined) { paragraphStyle.spaceBelow = { magnitude: style.spaceBelow, unit: 'PT' }; fieldsToUpdate.push('spaceBelow'); }
  if (style.namedStyleType !== undefined) { paragraphStyle.namedStyleType = style.namedStyleType; fieldsToUpdate.push('namedStyleType'); }
  if (style.keepWithNext !== undefined) { paragraphStyle.keepWithNext = style.keepWithNext; fieldsToUpdate.push('keepWithNext'); }
  if (style.keepLinesTogether !== undefined) { paragraphStyle.keepLinesTogether = style.keepLinesTogether; fieldsToUpdate.push('keepLinesTogether'); }
  if (style.avoidWidowAndOrphan !== undefined) { paragraphStyle.avoidWidowAndOrphan = style.avoidWidowAndOrphan; fieldsToUpdate.push('avoidWidowAndOrphan'); }
  if (style.pageBreakBefore !== undefined) { paragraphStyle.pageBreakBefore = style.pageBreakBefore; fieldsToUpdate.push('pageBreakBefore'); }

  // Removal rides the API's FieldMask reset semantics: naming a border field in
  // `fields` while leaving it unset in paragraphStyle clears the paragraph-level
  // border, so the paragraph falls back to whatever its named style defines.
  const removals = new Set<ParagraphBorderEdge>(
    (style.removeBorders ?? []).flatMap((edge) => (edge === 'all' ? PARAGRAPH_BORDER_EDGES : [edge]))
  );
  for (const edge of PARAGRAPH_BORDER_EDGES) {
    const field = PARAGRAPH_BORDER_KEYS[edge];
    const input = style[field];
    if (removals.has(edge)) {
      if (input !== undefined) throw new Error(`Cannot both set and remove the ${edge} border in one call`);
      fieldsToUpdate.push(field);
    } else if (input !== undefined) {
      paragraphStyle[field] = buildParagraphBorder(input);
      fieldsToUpdate.push(field);
    }
  }

  if (style.removeShading) {
    if (style.shading !== undefined) throw new Error('Cannot both set and remove shading in one call');
    fieldsToUpdate.push('shading');
  } else if (style.shading !== undefined) {
    const rgbColor = hexToRgbColor(style.shading);
    if (!rgbColor) throw new Error(`Invalid shading hex color: ${style.shading}`);
    paragraphStyle.shading = { backgroundColor: { color: { rgbColor } } };
    fieldsToUpdate.push('shading');
  }

  if (fieldsToUpdate.length === 0) return null;

  return {
    request: {
      updateParagraphStyle: {
        range: withTab({ startIndex, endIndex }, tabId),
        paragraphStyle,
        fields: fieldsToUpdate.join(','),
      }
    },
    fields: fieldsToUpdate
  };
}

// ---------------------------------------------------------------------------
// styleDocTable – compile table styling params into updateTableCellStyle /
// updateTableColumnProperties / updateTableRowStyle requests (one atomic batch)
// ---------------------------------------------------------------------------

function borderFieldName(edge: string): string {
  return `border${edge.charAt(0).toUpperCase()}${edge.slice(1)}`;
}

const TABLE_BORDER_EDGES = ['top', 'bottom', 'left', 'right'] as const;
type TableBorderEdge = (typeof TABLE_BORDER_EDGES)[number];

export interface TableCellBorderInput {
  color?: string;
  width?: number;
  dashStyle?: 'SOLID' | 'DOT' | 'DASH';
}

function buildTableCellBorder(input: TableCellBorderInput): any {
  const rgbColor = hexToRgbColor(input.color ?? '#000000');
  if (!rgbColor) throw new Error(`Invalid border hex color: ${input.color}`);
  return {
    color: { color: { rgbColor } },
    width: { magnitude: input.width ?? 1, unit: 'PT' },
    dashStyle: input.dashStyle ?? 'SOLID',
  };
}

export interface StyleDocTableArgs {
  tableStartIndex: number;
  tabId?: string;
  rowIndex?: number;
  columnIndex?: number;
  rowSpan?: number;
  columnSpan?: number;
  backgroundColor?: string;
  removeBackground?: boolean;
  contentAlignment?: 'TOP' | 'MIDDLE' | 'BOTTOM';
  padding?: number;
  borderTop?: TableCellBorderInput;
  borderBottom?: TableCellBorderInput;
  borderLeft?: TableCellBorderInput;
  borderRight?: TableCellBorderInput;
  removeBorders?: Array<TableBorderEdge | 'all'>;
  columnIndices?: number[];
  columnWidth?: number;
  rowIndices?: number[];
  minRowHeight?: number;
}

export function buildStyleDocTableRequests(a: StyleDocTableArgs): { requests: any[]; applied: string[] } {
  const requests: any[] = [];
  const applied: string[] = [];
  const tableStartLocation = withTab({ index: a.tableStartIndex }, a.tabId);

  // Cell targeting scopes the cell-style request only. updateTableColumnProperties
  // and updateTableRowStyle address the table through columnIndices/rowIndices,
  // so a caller who passes rowIndex/columnIndex alongside columnWidth would get
  // the cell style scoped to one cell and the width applied to every column —
  // not what "this cell" reads as. Refuse rather than silently widen.
  const cellTargeted = a.rowIndex !== undefined || a.columnIndex !== undefined
    || a.rowSpan !== undefined || a.columnSpan !== undefined;
  if (cellTargeted && (a.columnWidth !== undefined || a.minRowHeight !== undefined)) {
    throw new Error('rowIndex/columnIndex/rowSpan/columnSpan scope cell styles only — columnWidth and minRowHeight always apply at table scope, so issue them in a separate call (scoped with columnIndices/rowIndices)');
  }

  // ---- updateTableCellStyle -----------------------------------------------
  const tableCellStyle: any = {};
  const cellFields: string[] = [];

  if (a.removeBackground) {
    if (a.backgroundColor !== undefined) throw new Error('Cannot both set and remove backgroundColor in one call');
    cellFields.push('backgroundColor');
    applied.push('background cleared');
  } else if (a.backgroundColor !== undefined) {
    const rgbColor = hexToRgbColor(a.backgroundColor);
    if (!rgbColor) throw new Error(`Invalid background hex color: ${a.backgroundColor}`);
    tableCellStyle.backgroundColor = { color: { rgbColor } };
    cellFields.push('backgroundColor');
    applied.push(`background ${a.backgroundColor}`);
  }

  if (a.contentAlignment !== undefined) {
    tableCellStyle.contentAlignment = a.contentAlignment;
    cellFields.push('contentAlignment');
    applied.push(`contentAlignment ${a.contentAlignment}`);
  }

  if (a.padding !== undefined) {
    for (const edge of ['paddingTop', 'paddingBottom', 'paddingLeft', 'paddingRight']) {
      tableCellStyle[edge] = { magnitude: a.padding, unit: 'PT' };
      cellFields.push(edge);
    }
    applied.push(`padding ${a.padding}pt`);
  }

  // Unlike paragraph borders, a cleared table border is expressed as an
  // explicit width-0 border (the Docs UI's "0 pt" unbox), not a FieldMask
  // reset — cells always carry a border object.
  const removals = new Set<TableBorderEdge>(
    (a.removeBorders ?? []).flatMap((edge) => (edge === 'all' ? TABLE_BORDER_EDGES : [edge]))
  );
  for (const edge of TABLE_BORDER_EDGES) {
    const field = borderFieldName(edge);
    const input = a[field as 'borderTop'];
    if (removals.has(edge)) {
      if (input !== undefined) throw new Error(`Cannot both set and remove the ${edge} border in one call`);
      tableCellStyle[field] = buildTableCellBorder({ width: 0 });
      cellFields.push(field);
    } else if (input !== undefined) {
      tableCellStyle[field] = buildTableCellBorder(input);
      cellFields.push(field);
    }
  }
  if (removals.size > 0) applied.push(`borders cleared (${[...removals].join(', ')})`);
  const setEdges = TABLE_BORDER_EDGES.filter((e) => a[borderFieldName(e) as 'borderTop'] !== undefined);
  if (setEdges.length > 0) applied.push(`borders set (${setEdges.join(', ')})`);

  if (cellFields.length > 0) {
    const isSubset = cellTargeted;
    if (isSubset && (a.rowIndex === undefined || a.columnIndex === undefined)) {
      throw new Error('Cell subset targeting needs both rowIndex and columnIndex (spans optional)');
    }
    const target = isSubset
      ? {
          tableRange: {
            tableCellLocation: { tableStartLocation, rowIndex: a.rowIndex, columnIndex: a.columnIndex },
            rowSpan: a.rowSpan ?? 1,
            columnSpan: a.columnSpan ?? 1,
          },
        }
      : { tableStartLocation };
    requests.push({
      updateTableCellStyle: { ...target, tableCellStyle, fields: cellFields.join(',') },
    });
  } else if (cellTargeted) {
    // Cell targeting with no cell style to apply: nothing would be sent.
    throw new Error('rowIndex/columnIndex/rowSpan/columnSpan target cell styles only — pass a cell style param, or use columnIndices/rowIndices for column and row updates');
  }

  // ---- updateTableColumnProperties ----------------------------------------
  if (a.columnWidth !== undefined) {
    if (a.columnWidth < 5) throw new Error('columnWidth must be at least 5 points');
    requests.push({
      updateTableColumnProperties: {
        tableStartLocation,
        columnIndices: a.columnIndices ?? [],
        tableColumnProperties: { widthType: 'FIXED_WIDTH', width: { magnitude: a.columnWidth, unit: 'PT' } },
        fields: 'width,widthType',
      },
    });
    applied.push(`column width ${a.columnWidth}pt (${a.columnIndices?.length ? `columns ${a.columnIndices.join(', ')}` : 'all columns'})`);
  } else if (a.columnIndices !== undefined) {
    throw new Error('columnIndices given without columnWidth');
  }

  // ---- updateTableRowStyle ------------------------------------------------
  if (a.minRowHeight !== undefined) {
    requests.push({
      updateTableRowStyle: {
        tableStartLocation,
        rowIndices: a.rowIndices ?? [],
        tableRowStyle: { minRowHeight: { magnitude: a.minRowHeight, unit: 'PT' } },
        fields: 'minRowHeight',
      },
    });
    applied.push(`min row height ${a.minRowHeight}pt (${a.rowIndices?.length ? `rows ${a.rowIndices.join(', ')}` : 'all rows'})`);
  } else if (a.rowIndices !== undefined) {
    throw new Error('rowIndices given without minRowHeight');
  }

  return { requests, applied };
}

// Insert an inline image from a URL
async function insertInlineImageHelper(
  ctx: ToolContext,
  documentId: string,
  imageUrl: string,
  index: number,
  width?: number,
  height?: number
): Promise<any> {
  // Validate URL format
  try {
    new URL(imageUrl);
  } catch (_e) {
    throw new Error(`Invalid image URL format: ${imageUrl}`);
  }

  const request: any = {
    insertInlineImage: {
      location: { index },
      uri: imageUrl
    }
  };

  if (width && height) {
    request.insertInlineImage.objectSize = {
      height: { magnitude: height, unit: 'PT' },
      width: { magnitude: width, unit: 'PT' }
    };
  }

  return executeBatchUpdate(ctx, documentId, [request]);
}

// Image upload moved to ../utils/driveImageUpload.ts.

// ---------------------------------------------------------------------------
// Comment context extraction helpers
// ---------------------------------------------------------------------------

/** Context extracted for a single comment (keyed by Drive API comment ID) */
export interface CommentContext {
  contextBefore?: string;
  contextAfter?: string;
  startIndex?: number;
  endIndex?: number;
}

/** A segment of text with its Docs API startIndex */
interface TextSegment {
  text: string;
  startIndex: number;
}

/** Result of building flat text from a Google Doc */
interface FlatTextResult {
  flatText: string;
  offsetMap: number[];
}

// Guard against matching XML elements from distant/unrelated tables or paragraphs
const MAX_ROW_XML_DISTANCE = 100_000;
const MAX_PARAGRAPH_XML_DISTANCE = 50_000;
const MAX_PARAGRAPH_CONTEXT_LENGTH = 300;

/**
 * Build flat text from a Google Doc, tracking each character's Docs API startIndex.
 * Handles paragraphs, tables (including nested), and multi-tab docs.
 */
export function buildFlatTextFromDoc(docData: any): FlatTextResult {
  function extractSegments(bodyContent: any[]): TextSegment[] {
    const segs: TextSegment[] = [];
    function fromElements(elements: any[]) {
      for (const el of elements) {
        if (el.textRun?.content && el.startIndex != null) {
          segs.push({ text: el.textRun.content, startIndex: el.startIndex });
        }
      }
    }
    for (const el of bodyContent) {
      if (el.paragraph?.elements) {
        fromElements(el.paragraph.elements);
      } else if (el.table) {
        for (const row of el.table.tableRows || []) {
          for (const cell of row.tableCells || []) {
            for (const cc of cell.content || []) {
              if (cc.paragraph?.elements) fromElements(cc.paragraph.elements);
              if (cc.table) {
                const nested = extractSegments([cc]);
                segs.push(...nested);
              }
            }
          }
        }
      }
    }
    return segs;
  }

  const allSegments: TextSegment[] = [];
  const tabs = (docData as any).tabs as any[] | undefined;
  if (tabs && tabs.length > 0) {
    for (const tab of tabs) {
      const bc = tab.documentTab?.body?.content;
      if (bc) allSegments.push(...extractSegments(bc));
    }
  } else if (docData.body?.content) {
    allSegments.push(...extractSegments(docData.body.content));
  }

  let flatText = '';
  const offsetMap: number[] = [];
  for (const seg of allSegments) {
    for (let i = 0; i < seg.text.length; i++) {
      offsetMap.push(seg.startIndex + i);
      flatText += seg.text[i];
    }
  }

  return { flatText, offsetMap };
}

/** Extract cell texts from a DOCX table row XML string */
export function extractRowCells(rowXml: string): string[] {
  const cells: string[] = [];
  let searchFrom = 0;
  while (true) {
    const tcStart1 = rowXml.indexOf('<w:tc>', searchFrom);
    const tcStart2 = rowXml.indexOf('<w:tc ', searchFrom);
    const tcStart = (tcStart1 === -1 && tcStart2 === -1) ? -1 :
      (tcStart1 === -1) ? tcStart2 : (tcStart2 === -1) ? tcStart1 : Math.min(tcStart1, tcStart2);
    if (tcStart === -1) break;
    const tcEnd = rowXml.indexOf('</w:tc>', tcStart);
    if (tcEnd === -1) break;
    const cellXml = rowXml.substring(tcStart, tcEnd);
    const tTexts: string[] = [];
    const tRegex = /<w:t[^>]*>([^<]*)<\/w:t>/g;
    let t: RegExpExecArray | null;
    while ((t = tRegex.exec(cellXml)) !== null) tTexts.push(t[1]);
    if (tTexts.length > 0) cells.push(tTexts.join(''));
    searchFrom = tcEnd + 7;
  }
  return cells;
}

/** DOCX comment info parsed from word/comments.xml */
export interface DocxComment {
  author: string;
  date: string;
  content: string;
}

/** Context extracted from DOCX comment ranges in document.xml */
export interface DocxContextResult {
  docxComments: Map<number, DocxComment>;
  contextsBefore: Map<number, string>;
  contextsAfter: Map<number, string>;
  rowCells: Map<number, string[]>;
}

/**
 * Build formatted content with indices from a Google Doc document data object.
 * Returns the formatted string and total character length.
 */
export function buildDocFormattedContent(
  docData: any,
  withFormatting: boolean
): { formattedContent: string; totalLength: number } {
  interface Segment {
    text: string;
    startIndex: number;
    endIndex: number;
    // True for inline replacements (images, persons, rich links, footnotes,
    // rules) whose rendered text length is unrelated to their real doc span, so
    // the displayed edit range must use [startIndex, endIndex), not text length.
    atomic?: boolean;
    // True for multi-line renderings (tables) whose text is a *rendering*, not
    // index-mapped content: emit one real [startIndex-endIndex) span for the
    // whole element and never derive per-line ranges from rendered lengths.
    // The old arithmetic started at the table's own real start index and ran
    // FORWARD past its end into whatever followed, so a table's ranges
    // collided with the content after it (sometimes a later table); they never
    // reached backward into an earlier one.
    spanOnly?: boolean;
    // Real per-cell index ranges, carried alongside the rendering so text
    // inside a cell is still addressable by applyTextStyle/formatGoogleDocText.
    // editTableCell only styles a whole cell.
    cellSpans?: Array<{ row: number; column: number; startIndex: number; endIndex: number }>;
    // Paragraph-level style annotation: a one-line rendering of the
    // paragraph's non-default styles over its real span; carries no content.
    paraMeta?: boolean;
    fontFamily?: string;
    fontSize?: number;
    bold?: boolean;
    italic?: boolean;
    underline?: boolean;
    strikethrough?: boolean;
    foregroundColor?: string;
    backgroundColor?: string;
    // Only set for SUPERSCRIPT/SUBSCRIPT. The API reports NONE for ordinary
    // text, which carries no information and would force a meta line onto
    // every unformatted run.
    baselineOffset?: 'SUPERSCRIPT' | 'SUBSCRIPT';
  }

  function extractSegments(bodyContent: any[], inlineObjects?: any): Segment[] {
    const segments: Segment[] = [];

    function getCellText(cellContent: any[]): { text: string; startIndex?: number; endIndex?: number } {
      const before = segments.length;
      processContent(cellContent);
      const cellSegs = segments.splice(before);
      // Paragraph meta is a one-line annotation over a paragraph's span, not
      // content. Joined into a cell it would render inside the pipe row as
      // `| ¶ center Header A |` — a regression in the formatted read itself.
      // Filtered from the joined text only; the spans still feed the cell range.
      const text = cellSegs
        .filter(c => !c.paraMeta)
        .map(s => s.text.replace(/\n$/g, ''))
        .join(' ')
        .replace(/\|/g, '\\|')
        .trim();
      // The spliced segments carry the cell's true indices from the API. They
      // used to be discarded, which left text inside a table with no range at
      // all once the fabricated per-row ranges were removed.
      if (cellSegs.length === 0) return { text };
      return {
        text,
        startIndex: Math.min(...cellSegs.map(c => c.startIndex)),
        endIndex: Math.max(...cellSegs.map(c => c.endIndex)),
      };
    }

    function processContent(content: any[]) {
      for (const element of content) {
        if (element.paragraph?.elements) {
          if (withFormatting && element.startIndex != null && element.endIndex != null) {
            const bits = paragraphMetaBits(element.paragraph);
            if (bits.length) {
              segments.push({
                text: `\u00b6 ${bits.join(', ')}`,
                startIndex: element.startIndex,
                endIndex: element.endIndex,
                paraMeta: true,
              });
            }
          }
          for (const textElement of element.paragraph.elements) {
            if (textElement.textRun?.content && textElement.startIndex != null && textElement.endIndex != null) {
              const seg: Segment = {
                text: textElement.textRun.content,
                startIndex: textElement.startIndex,
                endIndex: textElement.endIndex,
              };
              if (withFormatting) {
                const ts = textElement.textRun.textStyle;
                if (ts) {
                  if (ts.weightedFontFamily?.fontFamily) seg.fontFamily = ts.weightedFontFamily.fontFamily;
                  if (ts.fontSize?.magnitude != null) seg.fontSize = ts.fontSize.magnitude;
                  if (ts.bold) seg.bold = true;
                  if (ts.italic) seg.italic = true;
                  if (ts.underline) seg.underline = true;
                  if (ts.strikethrough) seg.strikethrough = true;
                  const fg = rgbColorToHex(ts.foregroundColor);
                  const bg = rgbColorToHex(ts.backgroundColor);
                  if (fg) seg.foregroundColor = fg;
                  if (bg) seg.backgroundColor = bg;
                  if (ts.baselineOffset === 'SUPERSCRIPT' || ts.baselineOffset === 'SUBSCRIPT') {
                    seg.baselineOffset = ts.baselineOffset;
                  }
                }
              }
              segments.push(seg);
            } else {
              const inlineText = resolveInlineElementText(textElement, inlineObjects);
              if (inlineText && textElement.startIndex != null && textElement.endIndex != null) {
                segments.push({
                  text: inlineText,
                  startIndex: textElement.startIndex,
                  endIndex: textElement.endIndex,
                  atomic: true,
                });
              }
            }
          }
        } else if (element.table?.tableRows) {
          const rows: string[][] = [];
          const cellSpans: Array<{ row: number; column: number; startIndex: number; endIndex: number }> = [];
          let rowIndex = 0;
          for (const row of element.table.tableRows) {
            if (!row.tableCells) continue;
            let columnIndex = 0;
            rows.push(expandRowCells(row, (cell: any) => {
              const got = cell.content ? getCellText(cell.content) : { text: '' };
              if (got.startIndex != null && got.endIndex != null) {
                cellSpans.push({ row: rowIndex, column: columnIndex, startIndex: got.startIndex, endIndex: got.endIndex });
              }
              columnIndex++;
              return got.text;
            }));
            rowIndex++;
          }
          const md = renderPipeTable(rows) + '\n\n';
          if (element.startIndex != null && element.endIndex != null) {
            segments.push({
              text: md,
              startIndex: element.startIndex,
              endIndex: element.endIndex,
              spanOnly: true,
              ...(cellSpans.length ? { cellSpans } : {}),
            });
          }
        } else if (element.tableOfContents?.content) {
          processContent(element.tableOfContents.content);
        }
      }
    }

    processContent(bodyContent);
    return segments;
  }

  function formatSegments(segments: Segment[], tabId?: string): string {
    // A table hint that names only tableStartIndex is wrong on a multi-tab
    // document: index spaces restart per tab, so two tabs can each hold a
    // table starting at the same index, and editTableCell without a tabId
    // only searches the first tab. Name the tab too — and the header below
    // prints the id so the caller can actually supply it.
    const tabIdHint = tabId ? `, tabId=${tabId}` : '';
    let result = '';
    for (const segment of segments) {
      const hasMeta = withFormatting && hasFormattingInfo(segment);
      const meta = hasMeta ? buildMetaLine(segment) : null;
      // Paragraph style annotations: one line over the paragraph's real span.
      if (segment.paraMeta) {
        result += `[${segment.startIndex}-${segment.endIndex}] ${segment.text}\n`;
        continue;
      }
      // Atomic inline replacements occupy exactly [startIndex, endIndex) in the
      // doc regardless of how long their placeholder text is. Emit that real
      // range on one line so index-based edits target the object, not the text
      // that follows it.
      if (segment.atomic) {
        const line = segment.text.replace(/\n/g, ' ').trim();
        if (line) {
          result += meta
            ? `[${segment.startIndex}-${segment.endIndex}] ${meta}\n  ${line}\n`
            : `[${segment.startIndex}-${segment.endIndex}] ${line}\n`;
        }
        continue;
      }
      // Tables: the markdown below is a rendering, not index-mapped content —
      // one real span for the element, no fabricated per-row ranges. Cell
      // indices live in documents.get; targeting a cell needs the table's real
      // startIndex (this one) with editTableCell's row/column addressing.
      if (segment.spanOnly) {
        const body = segment.text.replace(/\n+$/, '');
        if (body.trim()) {
          result += `[${segment.startIndex}-${segment.endIndex}] <table — span is the real index range; rows below are a rendering, use editTableCell (tableStartIndex=${segment.startIndex}${tabIdHint}) for cell edits>\n${body}\n`;
          // Real per-cell ranges, so text inside a cell can still be targeted
          // by the range-taking tools. editTableCell styles a whole cell.
          if (segment.cellSpans?.length) {
            const map = segment.cellSpans
              .map(c => `r${c.row}c${c.column} [${c.startIndex}-${c.endIndex}]`)
              .join(', ');
            result += `cells: ${map}\n`;
          }
        }
        continue;
      }
      const lines = segment.text.split('\n');
      let offset = segment.startIndex;
      for (const line of lines) {
        if (line.trim()) {
          if (meta) {
            result += `[${offset}-${offset + line.length}] ${meta}\n  ${line}\n`;
          } else {
            result += `[${offset}-${offset + line.length}] ${line}\n`;
          }
        }
        offset += line.length + 1;
      }
    }
    return result;
  }

  function hasFormattingInfo(seg: Segment): boolean {
    return !!(seg.fontFamily || seg.fontSize || seg.bold || seg.italic || seg.underline || seg.strikethrough || seg.foregroundColor || seg.backgroundColor || seg.baselineOffset);
  }

  function buildMetaLine(seg: Segment): string {
    const parts: string[] = [];
    if (seg.fontFamily) parts.push(`font="${seg.fontFamily}"`);
    if (seg.fontSize) parts.push(`size=${seg.fontSize}pt`);
    const styles: string[] = [];
    if (seg.bold) styles.push('bold');
    if (seg.italic) styles.push('italic');
    if (seg.underline) styles.push('underline');
    if (seg.strikethrough) styles.push('strikethrough');
    if (styles.length > 0) parts.push(`style=${styles.join(',')}`);
    if (seg.baselineOffset) parts.push(`baseline=${seg.baselineOffset.toLowerCase()}`);
    if (seg.foregroundColor) parts.push(`color=${seg.foregroundColor}`);
    if (seg.backgroundColor) parts.push(`bg=${seg.backgroundColor}`);
    return parts.join(', ');
  }

  interface FontInfo { sizes: Set<number>; styles: Set<string>; charCount: number }

  const fontUsage: Map<string, FontInfo> = new Map();
  function trackFonts(segments: Segment[]) {
    if (!withFormatting) return;
    for (const seg of segments) {
      if (seg.fontFamily) {
        let info = fontUsage.get(seg.fontFamily);
        if (!info) {
          info = { sizes: new Set(), styles: new Set(), charCount: 0 };
          fontUsage.set(seg.fontFamily, info);
        }
        if (seg.fontSize) info.sizes.add(seg.fontSize);
        if (seg.bold) info.styles.add('bold');
        if (seg.italic) info.styles.add('italic');
        if (seg.underline) info.styles.add('underline');
        if (seg.strikethrough) info.styles.add('strikethrough');
        info.charCount += seg.endIndex - seg.startIndex;
      }
    }
  }

  const tabs = docData.tabs as any[] | undefined;
  let formattedContent = 'Document content with indices:\n\n';
  let totalLength = 0;

  if (tabs && tabs.length > 0) {
    const allTabs = collectAllTabsWithLevel(tabs);
    const isMultiTab = allTabs.length > 1;
    for (const { tab, level } of allTabs) {
      const bodyContent = tab.documentTab?.body?.content;
      if (isMultiTab) {
        const title = tab.tabProperties?.title || 'Untitled';
        const indent = '  '.repeat(level);
        const tabId = tab.tabProperties?.tabId;
        formattedContent += `${indent}=== Tab: ${title}${tabId ? ` (tabId=${tabId})` : ''} ===\n`;
      }
      if (bodyContent) {
        const tabInlineObjects = tab.documentTab?.inlineObjects;
        const segments = extractSegments(bodyContent, tabInlineObjects);
        trackFonts(segments);
        formattedContent += formatSegments(segments, tab.tabProperties?.tabId);
        if (segments.length > 0) {
          totalLength += segments[segments.length - 1].endIndex;
        }
      }
      if (isMultiTab) {
        formattedContent += '\n';
      }
    }
  } else {
    const bodyContent = docData.body?.content;
    if (bodyContent) {
      const legacyInlineObjects = docData.inlineObjects;
      const segments = extractSegments(bodyContent, legacyInlineObjects);
      trackFonts(segments);
      formattedContent += formatSegments(segments);
      totalLength = segments.length > 0 ? segments[segments.length - 1].endIndex : 0;
    }
  }

  if (withFormatting && fontUsage.size > 0) {
    formattedContent += '\n--- Fonts summary ---\n';
    const sorted = [...fontUsage.entries()].sort((a, b) => b[1].charCount - a[1].charCount);
    for (const [font, info] of sorted) {
      const sizesStr = info.sizes.size > 0 ? [...info.sizes].sort((a, b) => a - b).join(', ') + ' pt' : 'default size';
      const stylesStr = info.styles.size > 0 ? [...info.styles].sort().join(', ') : 'normal';
      formattedContent += `${font}: sizes [${sizesStr}], styles [${stylesStr}], ~${info.charCount} chars\n`;
    }
  }

  return { formattedContent, totalLength };
}

// Slice [offset, offset+limit) but snap the end back to a line boundary so a
// structural line prefix (e.g. "[start-end] ...") is never cut mid-line.
// When a single line is longer than `limit` no newline exists in the window;
// keep the hard cut so pagination always makes forward progress.
function sliceByLine(content: string, offset: number, limit: number): { slice: string; end: number } {
  let end = Math.min(offset + limit, content.length);
  if (end < content.length) {
    const nl = content.lastIndexOf('\n', end);
    if (nl > offset) end = nl + 1;
  }
  return { slice: content.slice(offset, end), end };
}

// Line-boundary slicing is enough for text output, where every line stands on
// its own, but a markdown table spans several lines that only render together:
// cut between them and the first page ends in a headerless fragment while the
// next begins with orphaned `| a | b |` rows. Snap the cut back to the start of
// the table instead — including the `[a-b] <table ...>` header the indexed reads
// print above it, which would otherwise be stranded at the bottom of the
// previous page. A table longer than `limit` keeps the hard cut so pagination
// still makes forward progress, but even then the cut never falls between the
// header and its first row.
function sliceMarkdownByBlock(content: string, offset: number, limit: number): { slice: string; end: number } {
  const { end } = sliceByLine(content, offset, limit);
  if (end >= content.length) return { slice: content.slice(offset, end), end };

  // Lines that only make sense next to their table: the pipe rows themselves,
  // and the per-cell index map the indexed reads print underneath them.
  const isTableBodyLine = (lineStart: number) =>
    content.startsWith('|', lineStart) || /^cells: r\d+c\d+ \[/.test(content.slice(lineStart, lineStart + 24));
  // The indexed reads prefix a table with a header line naming its real span.
  // Walking back over body lines alone would strand that header at the bottom
  // of the previous page, which is the thing being prevented.
  const isTableHeaderLine = (lineStart: number) =>
    /^\[\d+-\d+\] <table/.test(content.slice(lineStart, lineStart + 40));
  const lineStartBefore = (pos: number) => {
    const prevBreak = content.lastIndexOf('\n', pos - 2);
    return prevBreak === -1 ? 0 : prevBreak + 1;
  };

  // `end` sits at the start of the next page's first line.
  if (!isTableBodyLine(end)) return { slice: content.slice(offset, end), end };

  let tableStart = end;
  while (tableStart > offset) {
    const lineStart = lineStartBefore(tableStart);
    if (lineStart < offset || !isTableBodyLine(lineStart)) break;
    tableStart = lineStart;
  }
  // Take the header with the rows it describes.
  const headerStart = lineStartBefore(tableStart);
  if (tableStart > offset && headerStart >= offset && isTableHeaderLine(headerStart)) {
    tableStart = headerStart;
  }
  // The table starts at or before this page's own start, so it cannot be moved
  // wholly onto the next page — the block is longer than `limit`. Keep the hard
  // cut so pagination still advances, but never cut directly after the header:
  // that is the split that opens the next page with bare `| a | b |` rows and
  // nothing naming the table. Carry the first body line across with it.
  if (tableStart <= offset) {
    const lastLineStart = lineStartBefore(end);
    if (isTableHeaderLine(lastLineStart)) {
      const nextBreak = content.indexOf('\n', end);
      const extended = nextBreak === -1 ? content.length : nextBreak + 1;
      return { slice: content.slice(offset, extended), end: extended };
    }
    return { slice: content.slice(offset, end), end };
  }
  return { slice: content.slice(offset, tableStart), end: tableStart };
}

/**
 * Parse a DOCX export to extract comment positions and surrounding context.
 * Returns DOCX comment metadata and context maps keyed by DOCX comment ID.
 */
export async function resolveContextFromDocx(docxData: ArrayBuffer): Promise<DocxContextResult | null> {
  const zip = await JSZip.loadAsync(docxData);
  const commentsXml = await zip.file('word/comments.xml')?.async('string');
  const documentXml = await zip.file('word/document.xml')?.async('string');

  if (!commentsXml || !documentXml) return null;

  // ── Parse word/comments.xml ──
  const docxComments = new Map<number, DocxComment>();
  const commentTagRegex = /<w:comment\s+[^>]*?w:id="(\d+)"[^>]*>/g;
  let cMatch: RegExpExecArray | null;
  while ((cMatch = commentTagRegex.exec(commentsXml)) !== null) {
    const id = parseInt(cMatch[1]);
    const tagStr = cMatch[0];
    const authorMatch = tagStr.match(/w:author="([^"]*)"/);
    const dateMatch = tagStr.match(/w:date="([^"]*)"/);
    const author = authorMatch ? authorMatch[1] : '';
    const date = dateMatch ? dateMatch[1] : '';

    const endPos = commentsXml.indexOf('</w:comment>', cMatch.index);
    if (endPos !== -1) {
      const body = commentsXml.substring(cMatch.index, endPos);
      const texts: string[] = [];
      const tRegex = /<w:t[^>]*>([^<]*)<\/w:t>/g;
      let tMatch: RegExpExecArray | null;
      while ((tMatch = tRegex.exec(body)) !== null) {
        texts.push(tMatch[1]);
      }
      docxComments.set(id, { author, date, content: texts.join('') });
    }
  }

  // ── Parse document.xml for comment range context ──
  const contextsBefore = new Map<number, string>();
  const contextsAfter = new Map<number, string>();
  const rowCells = new Map<number, string[]>();

  const rangeStartRegex = /<w:commentRangeStart\s+w:id="(\d+)"\/>/g;
  let rMatch: RegExpExecArray | null;
  while ((rMatch = rangeStartRegex.exec(documentXml)) !== null) {
    const docxId = parseInt(rMatch[1]);
    const startPos = rMatch.index;

    // Try table row context first (most comments in table-based docs)
    const trStart = documentXml.lastIndexOf('<w:tr>', startPos);
    const trEnd = documentXml.indexOf('</w:tr>', startPos);
    if (trStart !== -1 && trEnd !== -1 && (startPos - trStart) < MAX_ROW_XML_DISTANCE) {
      const rowXml = documentXml.substring(trStart, trEnd);

      const cellTexts = extractRowCells(rowXml);
      // Find which cell contains the comment marker
      const commentMarker = `commentRangeStart w:id="${docxId}"`;
      let commentCellIdx = -1;
      let cellSearchFrom = 0;
      for (let ci = 0; ci < cellTexts.length; ci++) {
        // Walk through <w:tc> tags in order to match cell index with extractRowCells output
        const tcStart1 = rowXml.indexOf('<w:tc>', cellSearchFrom);
        const tcStart2 = rowXml.indexOf('<w:tc ', cellSearchFrom);
        const tcStart = (tcStart1 === -1 && tcStart2 === -1) ? -1 :
          (tcStart1 === -1) ? tcStart2 : (tcStart2 === -1) ? tcStart1 : Math.min(tcStart1, tcStart2);
        if (tcStart === -1) break;
        const tcEnd = rowXml.indexOf('</w:tc>', tcStart);
        if (tcEnd === -1) break;
        const cellXml = rowXml.substring(tcStart, tcEnd);
        if (cellXml.includes(commentMarker)) {
          commentCellIdx = ci;
        }
        cellSearchFrom = tcEnd + 7;
      }

      if (cellTexts.length > 0) {
        const allTexts = cellTexts;
        rowCells.set(docxId, allTexts);

        if (commentCellIdx !== -1) {
          const before = cellTexts.slice(0, commentCellIdx);
          let after = cellTexts.slice(commentCellIdx + 1);

          // If comment is in the last cell, grab the NEXT row for "after" context
          if (commentCellIdx === cellTexts.length - 1) {
            const nextTrStart = documentXml.indexOf('<w:tr>', trEnd);
            const nextTrEnd = nextTrStart !== -1 ? documentXml.indexOf('</w:tr>', nextTrStart) : -1;
            if (nextTrStart !== -1 && nextTrEnd !== -1) {
              const nextRowXml = documentXml.substring(nextTrStart, nextTrEnd);
              after = extractRowCells(nextRowXml);
            }
          }

          const commentText = cellTexts[commentCellIdx];
          contextsBefore.set(docxId, [...before, commentText].join(' | '));
          contextsAfter.set(docxId, [commentText, ...after].join(' | '));
        } else {
          contextsBefore.set(docxId, allTexts.join(' | '));
          contextsAfter.set(docxId, '');
        }
        continue;
      }
    }

    // Paragraph fallback for non-table docs
    const pStart = documentXml.lastIndexOf('<w:p ', startPos);
    const pEnd = documentXml.indexOf('</w:p>', startPos);
    if (pStart !== -1 && pEnd !== -1 && (startPos - pStart) < MAX_PARAGRAPH_XML_DISTANCE) {
      const pXml = documentXml.substring(pStart, pEnd);
      const pTexts: string[] = [];
      const tRegex = /<w:t[^>]*>([^<]*)<\/w:t>/g;
      let t: RegExpExecArray | null;
      while ((t = tRegex.exec(pXml)) !== null) pTexts.push(t[1]);
      const pText = pTexts.join('').trim();
      if (pText) {
        contextsBefore.set(docxId, pText.length > MAX_PARAGRAPH_CONTEXT_LENGTH
          ? pText.substring(0, MAX_PARAGRAPH_CONTEXT_LENGTH) + '...' : pText);
        contextsAfter.set(docxId, '');
      }
    }
  }

  return { docxComments, contextsBefore, contextsAfter, rowCells };
}

/**
 * Match Drive API comments to DOCX comments by (author, createdTime).
 * DOCX timestamps omit milliseconds, so we strip them from the API date.
 * Populates the contextMap with matched context. Also resolves Docs API
 * character offsets when flatText/offsetMap are available.
 */
export function matchDocxToDriveComments(
  driveComments: any[],
  docxResult: DocxContextResult,
  contextMap: Map<string, CommentContext>,
  flatText: string,
  offsetMap: number[],
): void {
  const { docxComments, contextsBefore, contextsAfter } = docxResult;

  for (const comment of driveComments) {
    if (contextMap.has(comment.id)) continue; // already has Tier 1 context
    if (comment.resolved) continue; // resolved comments not in DOCX

    const apiAuthor = comment.author?.displayName || '';
    const apiDate = (comment.createdTime || '').replace(/\.\d+Z$/, 'Z');

    // Find matching DOCX comment
    let matchedDocxId: number | null = null;
    for (const [docxId, docxComment] of docxComments) {
      if (docxComment.author === apiAuthor && docxComment.date === apiDate) {
        matchedDocxId = docxId;
        break;
      }
    }

    if (matchedDocxId !== null) {
      const ctxBefore = contextsBefore.get(matchedDocxId) || '';
      const ctxAfter = contextsAfter.get(matchedDocxId) || '';
      if (ctxBefore || ctxAfter) {
        const entry: CommentContext = {
          contextBefore: ctxBefore,
          contextAfter: ctxAfter,
        };

        // Find Docs API character index using row context in flatText
        const quoted = comment.quotedFileContent?.value;
        if (quoted && flatText && offsetMap.length > 0 && ctxBefore) {
          const beforePattern = ctxBefore.split(' | ').join('\n');

          const findAll = (pattern: string): number[] => {
            const results: number[] = [];
            let from = 0;
            while (true) {
              const idx = flatText.indexOf(pattern, from);
              if (idx === -1) break;
              results.push(idx);
              from = idx + 1;
            }
            return results;
          };

          let matches = findAll(beforePattern);

          if (matches.length !== 1 && ctxAfter) {
            const afterCells = ctxAfter.split(' | ');
            const afterWithoutAnchor = afterCells.slice(1).join('\n');
            if (afterWithoutAnchor) {
              const fullPattern = beforePattern + '\n' + afterWithoutAnchor;
              matches = findAll(fullPattern);
            }
          }

          if (matches.length === 1) {
            const patternStart = matches[0];
            const qIdx = patternStart + beforePattern.length - quoted.length;
            const endIdx = qIdx + quoted.length - 1;
            if (endIdx < offsetMap.length && flatText.substring(qIdx, qIdx + quoted.length) === quoted) {
              entry.startIndex = offsetMap[qIdx];
              entry.endIndex = offsetMap[endIdx] + 1;
            }
          }
        }

        contextMap.set(comment.id, entry);
      }
      // Remove from map so duplicate timestamps (rare) don't double-match
      docxComments.delete(matchedDocxId);
    }
  }
}

// ---------------------------------------------------------------------------
// Zod schemas
// ---------------------------------------------------------------------------

const CreateGoogleDocSchema = z.object({
  name: z.string().min(1, "Document name is required"),
  content: z.string(),
  parentFolderId: z.string().optional()
});

const CreateDocFromHTMLSchema = z.object({
  html: z.string().min(1, "HTML content is required"),
  name: z.string().min(1, "Document name is required"),
  parentFolderId: z.string().optional(),
});

const UpdateGoogleDocSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  content: z.string(),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
});

const GetGoogleDocContentSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  includeFormatting: z.boolean().optional(),
});

const GetGoogleDocImageSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  inlineObjectId: z.string().min(1, "Inline object ID is required"),
  outputFormat: z.enum(['image', 'base64']).optional().default('image'),
});

const InsertTextSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  text: z.string().min(1, "Text to insert is required"),
  // Google Docs use 1-based indexes; text files use 0-based offsets — accept both, validate at runtime.
  index: z.number().int().min(0, "Index must be non-negative").optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).optional().default(1),
  position: z.enum(['before', 'after']).optional().default('after'),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
}).refine(data => (data.index !== undefined) !== (data.textToFind !== undefined), {
  message: "Provide exactly one of index or textToFind"
});

const DeleteRangeSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  startIndex: z.number().int().min(0, "Start index must be non-negative").optional(),
  endIndex: z.number().int().min(0, "End index must be non-negative").optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).optional().default(1),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
}).refine(data => {
  // Exactly one targeting mode: both indices and no textToFind, or textToFind
  // and no index at all. A stray single index next to textToFind is refused
  // rather than silently ignored (#183 follow-up).
  const anyIndex = data.startIndex !== undefined || data.endIndex !== undefined;
  const bothIndices = data.startIndex !== undefined && data.endIndex !== undefined;
  return data.textToFind !== undefined ? !anyIndex : bothIndices;
}, {
  message: "Provide either startIndex+endIndex or textToFind, not both"
}).refine(data => data.startIndex === undefined || data.endIndex === undefined || data.endIndex > data.startIndex, {
  message: "End index must be greater than start index",
  path: ["endIndex"]
});

const ReadGoogleDocSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  format: z.enum(['text', 'json', 'markdown']).optional().default('text'),
  maxLength: z.number().int().min(1).optional(),
  tabId: z.string().optional()
});

const ReadGoogleDocPaginatedSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  format: z.enum(['text', 'markdown']).optional().default('text'),
  offset: z.number().int().min(0).optional().default(0),
  limit: z.number().int().min(1).max(80000).optional().default(50000),
  tabId: z.string().optional()
});

const GetGoogleDocContentPaginatedSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  includeFormatting: z.boolean().optional(),
  offset: z.number().int().min(0).optional().default(0),
  limit: z.number().int().min(1).max(80000).optional().default(50000),
});

const ListDocumentTabsSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  includeContent: z.boolean().optional().default(false)
});

const ApplyTextStyleSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  startIndex: z.number().int().min(1).optional(),
  endIndex: z.number().int().min(1).optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).optional().default(1),
  bold: z.boolean().optional(),
  italic: z.boolean().optional(),
  underline: z.boolean().optional(),
  strikethrough: z.boolean().optional(),
  fontSize: z.number().min(1).optional(),
  fontFamily: z.string().optional(),
  foregroundColor: z.string().optional(),
  backgroundColor: z.string().optional(),
  linkUrl: z.string().url().optional(),
  baselineOffset: z.enum(['SUPERSCRIPT', 'SUBSCRIPT', 'NONE']).optional(),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
});

const ParagraphBorderParamSchema = z.object({
  color: z.string().optional(),
  width: z.number().min(0).optional(),
  padding: z.number().min(0).optional(),
  dashStyle: z.enum(['SOLID', 'DOT', 'DASH']).optional(),
});

const ApplyParagraphStyleSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  startIndex: z.number().int().min(1).optional(),
  endIndex: z.number().int().min(1).optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).optional().default(1),
  indexWithinParagraph: z.number().int().min(1).optional(),
  alignment: z.enum(['START', 'END', 'CENTER', 'JUSTIFIED']).optional(),
  indentStart: z.number().min(0).optional(),
  indentEnd: z.number().min(0).optional(),
  indentFirstLine: z.number().min(0).optional(),
  spaceAbove: z.number().min(0).optional(),
  spaceBelow: z.number().min(0).optional(),
  namedStyleType: z.enum(['NORMAL_TEXT', 'TITLE', 'SUBTITLE', 'HEADING_1', 'HEADING_2', 'HEADING_3', 'HEADING_4', 'HEADING_5', 'HEADING_6']).optional(),
  keepWithNext: z.boolean().optional(),
  keepLinesTogether: z.boolean().optional(),
  avoidWidowAndOrphan: z.boolean().optional(),
  pageBreakBefore: z.boolean().optional(),
  borderTop: ParagraphBorderParamSchema.optional(),
  borderBottom: ParagraphBorderParamSchema.optional(),
  borderLeft: ParagraphBorderParamSchema.optional(),
  borderRight: ParagraphBorderParamSchema.optional(),
  borderBetween: ParagraphBorderParamSchema.optional(),
  removeBorders: z.array(z.enum(['top', 'bottom', 'left', 'right', 'between', 'all'])).optional(),
  shading: z.string().optional(),
  removeShading: z.boolean().optional(),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
});

const CreateParagraphBulletsSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  startIndex: z.number().int().min(1).optional(),
  endIndex: z.number().int().min(1).optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).optional().default(1),
  bulletPreset: z.enum([
    'BULLET_DISC_CIRCLE_SQUARE',
    'BULLET_DIAMONDX_ARROW3D_SQUARE',
    'BULLET_CHECKBOX',
    'BULLET_ARROW_DIAMOND_DISC',
    'BULLET_STAR_CIRCLE_SQUARE',
    'BULLET_ARROW3D_CIRCLE_SQUARE',
    'BULLET_LEFTTRIANGLE_DIAMOND_DISC',
    'NUMBERED_DECIMAL_ALPHA_ROMAN',
    'NUMBERED_DECIMAL_ALPHA_ROMAN_PARENS',
    'NUMBERED_DECIMAL_NESTED',
    'NUMBERED_UPPERALPHA_ALPHA_ROMAN',
    'NUMBERED_UPPERROMAN_UPPERALPHA_DECIMAL',
    'NUMBERED_ZERODECIMAL_ALPHA_ROMAN',
    'NONE'
  ]).default('BULLET_DISC_CIRCLE_SQUARE'),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
});

const ListCommentsSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  includeDeleted: z.boolean().optional(),
  pageSize: z.number().int().min(1).max(100).optional(),
  pageToken: z.string().optional(),
});

const GetCommentSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  commentId: z.string().min(1, "Comment ID is required")
});

const AddCommentSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  startIndex: z.number().int().min(1, "Start index must be at least 1").optional(),
  endIndex: z.number().int().min(1, "End index must be at least 1").optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).default(1),
  tabId: z.string().optional(),
  commentText: z.string().min(1, "Comment text is required")
}).refine(
  (d) => (d.startIndex != null && d.endIndex != null) || d.textToFind != null,
  { message: "Provide either startIndex+endIndex or textToFind" },
);

const ReplyToCommentSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  commentId: z.string().min(1, "Comment ID is required"),
  replyText: z.string().min(1, "Reply text is required"),
  resolve: z.boolean().optional().describe("Set to true to resolve the comment thread after replying")
});

const DeleteCommentSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  commentId: z.string().min(1, "Comment ID is required")
});

const InsertTableSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  rows: z.number().int().min(1, "Must have at least 1 row"),
  columns: z.number().int().min(1, "Must have at least 1 column"),
  index: z.number().int().min(1, "Index must be at least 1 (1-based)"),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
});

const EditTableCellSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  tableStartIndex: z.number().int().min(1, "Table start index is required"),
  rowIndex: z.number().int().min(0, "Row index must be at least 0 (0-based)"),
  columnIndex: z.number().int().min(0, "Column index must be at least 0 (0-based)"),
  textContent: z.string().optional().describe("New text content for the cell"),
  bold: z.boolean().optional(),
  italic: z.boolean().optional(),
  fontSize: z.number().optional(),
  alignment: z.enum(["START", "CENTER", "END", "JUSTIFIED"]).optional(),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional()
});

const TableCellBorderParamSchema = z.object({
  color: z.string().optional(),
  width: z.number().min(0).optional(),
  dashStyle: z.enum(['SOLID', 'DOT', 'DASH']).optional(),
});

const StyleDocTableSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  tableStartIndex: z.number().int().min(1, "Table start index is required"),
  rowIndex: z.number().int().min(0).optional(),
  columnIndex: z.number().int().min(0).optional(),
  rowSpan: z.number().int().min(1).optional(),
  columnSpan: z.number().int().min(1).optional(),
  backgroundColor: z.string().optional(),
  removeBackground: z.boolean().optional(),
  contentAlignment: z.enum(['TOP', 'MIDDLE', 'BOTTOM']).optional(),
  padding: z.number().min(0).optional(),
  borderTop: TableCellBorderParamSchema.optional(),
  borderBottom: TableCellBorderParamSchema.optional(),
  borderLeft: TableCellBorderParamSchema.optional(),
  borderRight: TableCellBorderParamSchema.optional(),
  removeBorders: z.array(z.enum(['top', 'bottom', 'left', 'right', 'all'])).optional(),
  columnIndices: z.array(z.number().int().min(0)).optional(),
  columnWidth: z.number().optional(),
  rowIndices: z.array(z.number().int().min(0)).optional(),
  minRowHeight: z.number().min(0).optional(),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional(),
});

const InsertImageFromUrlSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  imageUrl: z.string().url("Must be a valid URL"),
  index: z.number().int().min(1, "Index must be at least 1 (1-based)"),
  width: z.number().optional().describe("Width in points"),
  height: z.number().optional().describe("Height in points")
});

const InsertLocalImageSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  localImagePath: z.string().min(1, "Local image path is required"),
  index: z.number().int().min(1, "Index must be at least 1 (1-based)"),
  width: z.number().optional().describe("Width in points"),
  height: z.number().optional().describe("Height in points"),
  uploadToSameFolder: z.boolean().optional().default(true).describe("Upload to same folder as document"),
  makePublic: z.boolean().optional().default(false).describe("Make uploaded image publicly accessible. Required if the document is not shared with the service account.")
});

const ListGoogleDocsSchema = z.object({
  maxResults: z.number().int().min(1).max(100).optional().default(20).describe("Maximum number of documents to return (1-100)."),
  query: z.string().optional().describe("Search query to filter documents by name or content."),
  orderBy: z.enum(DRIVE_ORDER_BY_VALUES).optional().default("modifiedTime desc").describe("Sort order for results.")
});

const GetDocumentInfoSchema = z.object({
  documentId: z.string().min(1, "Document ID is required")
});

const GetGoogleDocStyleSummarySchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  tabId: z.string().optional(),
});

const DescribeGoogleDocRangeSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  startIndex: z.number().int().min(1).optional(),
  endIndex: z.number().int().min(1).optional(),
  textToFind: z.string().min(1).optional(),
  matchInstance: z.number().int().min(1).optional().default(1),
  tabId: z.string().optional(),
}).refine((d) => d.startIndex != null || d.textToFind != null, {
  message: "Provide startIndex (optionally endIndex) or textToFind",
}).refine((d) => !(d.textToFind != null && (d.startIndex != null || d.endIndex != null)), {
  // insertText and deleteRange refuse mixed targeting rather than silently
  // preferring one mode (#183); a silently ignored endIndex reads as a probe
  // of a range the caller never got.
  message: "Provide either startIndex/endIndex or textToFind, not both",
}).refine((d) => !(d.startIndex != null && d.endIndex != null && d.endIndex <= d.startIndex), {
  message: "endIndex must be greater than startIndex",
});

const FindAndReplaceInDocSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  findText: z.string().min(1, "findText is required"),
  replaceText: z.string(),
  matchCase: z.boolean().optional().default(false),
  dryRun: z.boolean().optional().default(false),
  tabId: z.string().optional(),
  expectedCount: z.number().int().min(1).optional(),
  ifRevisionId: z.string().optional(),
});

const AddDocumentTabSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  title: z.string().min(1, "Tab title is required"),
});

const RenameDocumentTabSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  tabId: z.string().min(1, "Tab ID is required"),
  title: z.string().min(1, "Tab title is required"),
});

const InsertSmartChipSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  index: z.number().int().min(1, "Index must be at least 1"),
  chipType: z.enum(["person"]),
  personEmail: z.string().email("Valid email is required for person chip"),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional(),
});

const ReadSmartChipsSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
});

const CreateFootnoteSchema = z.object({
  documentId: z.string().min(1, "Document ID is required"),
  index: z.number().int().min(1, "Index must be at least 1").optional(),
  endOfSegment: z.boolean().optional(),
  content: z.string().optional(),
  tabId: z.string().optional(),
  ifRevisionId: z.string().optional(),
}).refine(data => data.index !== undefined || data.endOfSegment === true, {
  message: "Either 'index' or 'endOfSegment: true' must be provided",
});

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

// Shared by applyParagraphStyle and its formatGoogleDocParagraph alias so the
// two schemas cannot drift.
function paragraphBorderProperty(edgeDescription: string) {
  return {
    type: "object",
    description: `${edgeDescription} Omitted subfields default to a rendering border: color #000000, width 1pt, padding 1pt, SOLID.`,
    properties: {
      color: { type: "string", description: "Hex color (e.g., #FF0000; default #000000)" },
      width: { type: "number", description: "Line width in points (default: 1)" },
      padding: { type: "number", description: "Space between the border and the paragraph text, in points (default: 1)" },
      dashStyle: { type: "string", enum: ["SOLID", "DOT", "DASH"], description: "Line style (default: SOLID)" }
    }
  };
}

// Table cell borders have no per-border padding (cell padding is separate).
function tableCellBorderProperty(edgeDescription: string) {
  return {
    type: "object",
    description: `${edgeDescription} Omitted subfields default to a rendering border: color #000000, width 1pt, SOLID.`,
    properties: {
      color: { type: "string", description: "Hex color (e.g., #FF0000; default #000000)" },
      width: { type: "number", description: "Line width in points (default: 1; 0 = invisible)" },
      dashStyle: { type: "string", enum: ["SOLID", "DOT", "DASH"], description: "Line style (default: SOLID)" }
    }
  };
}

const applyParagraphStyleInputSchema = {
  type: "object",
  properties: {
    documentId: { type: "string", description: "The document ID" },
    startIndex: { type: "number", description: "Start index (1-based) - use with endIndex" },
    endIndex: { type: "number", description: "End index (exclusive) - use with startIndex" },
    textToFind: { type: "string", description: "Text within the target paragraph" },
    matchInstance: { type: "number", description: "Which instance of textToFind (default: 1)" },
    indexWithinParagraph: { type: "number", description: "Any index within the target paragraph" },
    alignment: { type: "string", enum: ["START", "END", "CENTER", "JUSTIFIED"], description: "Text alignment" },
    indentStart: { type: "number", description: "Left indent in points" },
    indentEnd: { type: "number", description: "Right indent in points" },
    indentFirstLine: { type: "number", description: "First-line indent in points" },
    spaceAbove: { type: "number", description: "Space above in points" },
    spaceBelow: { type: "number", description: "Space below in points" },
    namedStyleType: { type: "string", enum: ["NORMAL_TEXT", "TITLE", "SUBTITLE", "HEADING_1", "HEADING_2", "HEADING_3", "HEADING_4", "HEADING_5", "HEADING_6"], description: "Named paragraph style" },
    keepWithNext: { type: "boolean", description: "Keep with next paragraph" },
    keepLinesTogether: { type: "boolean", description: "Keep all lines of the paragraph on one page" },
    avoidWidowAndOrphan: { type: "boolean", description: "Avoid widow and orphan lines" },
    pageBreakBefore: { type: "boolean", description: "Start the paragraph on a new page" },
    borderTop: paragraphBorderProperty("Border above the paragraph."),
    borderBottom: paragraphBorderProperty("Border below the paragraph (a horizontal rule)."),
    borderLeft: paragraphBorderProperty("Border left of the paragraph."),
    borderRight: paragraphBorderProperty("Border right of the paragraph."),
    borderBetween: paragraphBorderProperty("Border between consecutive same-styled paragraphs."),
    removeBorders: { type: "array", items: { type: "string", enum: ["top", "bottom", "left", "right", "between", "all"] }, description: "Border edges to clear at the paragraph level, so each falls back to the border its named style defines (usually none). Does not touch horizontalRule elements. The update applies to EVERY paragraph overlapping the target range, so removeBorders: [\"all\"] over the whole body strips every paragraph border in one call." },
    shading: { type: "string", description: "Paragraph background color as hex (e.g., #F1F3F4)" },
    removeShading: { type: "boolean", description: "Clear the paragraph background shading" },
    tabId: { type: "string", description: "Optional. Tab ID to format within (from listDocumentTabs). If omitted, operates on the first/default tab." },
    ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
  },
  required: ["documentId"]
};

export const toolDefinitions: ToolDefinition[] = [
  {
    name: "createGoogleDoc",
    description: "Create a new Google Doc",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Doc name" },
        content: { type: "string", description: "Doc content" },
        parentFolderId: { type: "string", description: "Parent folder ID" }
      },
      required: ["name", "content"]
    }
  },
  {
    name: "createDocFromHTML",
    description: "Create a Google Doc from HTML content in a single Drive API call. HTML tags are converted to native Google Doc styles: <h1> → Heading 1, <h2> → Heading 2, <p> → Normal Text, <b> → bold, <i> → italic, <ul>/<ol> → lists, <table> → tables. Useful when you want native heading/list/table styling applied in one request instead of a createGoogleDoc call followed by per-paragraph formatGoogleDocParagraph requests.",
    inputSchema: {
      type: "object",
      properties: {
        html: { type: "string", description: "HTML content. Use standard tags: <h1>, <h2>, <h3> for headings, <p> for body text, <b>/<i> for inline styles, <ul>/<ol> for lists, <table> for tables. Inline CSS styles are also supported (e.g., font-family, color, font-size)." },
        name: { type: "string", description: "Document name in Google Drive" },
        parentFolderId: { type: "string", description: "Parent folder ID or path. Defaults to root." }
      },
      required: ["html", "name"]
    }
  },
  {
    name: "updateGoogleDoc",
    description: "Update an existing Google Doc (replaces all content). The replacement is one atomic batchUpdate, so a failure leaves the document unchanged rather than wiped. For multi-tab docs, specify tabId to replace a single tab's content; leaves other tabs untouched.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Doc ID" },
        content: { type: "string", description: "New content" },
        tabId: { type: "string", description: "Optional. Tab ID to replace (from listDocumentTabs). If set, delete+insert run in a single atomic batchUpdate scoped to that tab." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId", "content"]
    }
  },
  {
    name: "insertText",
    description: "Insert text at a specific index (surgical edit, doesn't replace whole file). Works on Google Docs and text/* files (e.g. text/plain, text/markdown). Index semantics differ by file type: Google Docs use the Docs API's 1-based structural position (includes structural elements); text files use a 0-based Unicode code point (character) offset into the file content. For multi-tab Google Docs, specify tabId to target a specific tab — tabId is not supported on text files.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The Google Doc or text file ID" },
        text: { type: "string", description: "Text to insert" },
        index: { type: "number", description: "Position to insert at. Google Docs: 1-based Docs API structural index. Text files: 0-based Unicode code point (character) offset. Provide exactly one of index or textToFind." },
        textToFind: { type: "string", description: "Google Docs only: locate the insertion point by text instead of index (exact, case-sensitive match). Combine with position (before/after) and matchInstance. If the match ends at the document's final paragraph break, 'after' inserts just before that break." },
        matchInstance: { type: "number", description: "Which instance of textToFind (default: 1)" },
        position: { type: "string", enum: ["before", "after"], description: "Insert before or after the matched text (default: after). Only used with textToFind." },
        tabId: { type: "string", description: "Optional. Google Docs only — Tab ID to insert into (from listDocumentTabs). If omitted, inserts into the first/default tab. Not supported on text files." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId", "text"]
    }
  },
  {
    name: "deleteRange",
    description: "Delete content between start and end indices. Works on Google Docs and text/* files (e.g. text/plain, text/markdown). Index semantics differ by file type: Google Docs use the Docs API's 1-based structural position (includes structural elements); text files use 0-based Unicode code point (character) offsets into the file content. For multi-tab Google Docs, specify tabId to target a specific tab — tabId is not supported on text files.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The Google Doc or text file ID" },
        startIndex: { type: "number", description: "Start index (inclusive). Google Docs: 1-based Docs API structural index. Text files: 0-based Unicode code point (character) offset. Provide startIndex+endIndex or textToFind." },
        endIndex: { type: "number", description: "End index (exclusive). Google Docs: 1-based Docs API structural index. Text files: 0-based Unicode code point (character) offset." },
        textToFind: { type: "string", description: "Google Docs only: delete the matched text instead of an index range (exact, case-sensitive match). Combine with matchInstance. A match that ends with the document's final paragraph break is trimmed to keep that break, since Docs does not allow deleting it." },
        matchInstance: { type: "number", description: "Which instance of textToFind (default: 1)" },
        tabId: { type: "string", description: "Optional. Google Docs only — Tab ID to delete from (from listDocumentTabs). If omitted, deletes from the first/default tab. Not supported on text files." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "readGoogleDoc",
    description: "Read content of a Google Doc with format options. Supports multi-tab documents. Text/markdown output leads with the document's revisionId — pass it as ifRevisionId on write tools for an optimistic lock.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        format: { type: "string", enum: ["text", "json", "markdown"], description: "Output format (default: text)" },
        maxLength: { type: "number", description: "Maximum characters to return" },
        tabId: { type: "string", description: "Read a specific tab by ID (from listDocumentTabs). If omitted, all tabs are returned." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "readGoogleDocPaginated",
    description: "Read a portion of a Google Doc with pagination support. Use offset and limit to read large documents in chunks, avoiding output size limits.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        format: { type: "string", enum: ["text", "markdown"], description: "Output format (default: text)" },
        offset: { type: "number", description: "Character offset into the output text, not raw doc characters (with format=markdown the title prefix counts). 0-based, default: 0. Pass the previous response's nextOffset to get the following page." },
        limit: { type: "number", description: "Maximum characters to return per page (default: 50000, max: 80000; kept below the host's ~100K output cap to leave room for the JSON envelope and newline escaping)" },
        tabId: { type: "string", description: "Read a specific tab by ID (from listDocumentTabs). If omitted, all tabs are returned." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "listDocumentTabs",
    description: "List all tabs in a Google Doc with their IDs and hierarchy",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        includeContent: { type: "boolean", description: "Include content summary (character count) for each tab" }
      },
      required: ["documentId"]
    }
  },
  {
    name: "applyTextStyle",
    description: "Apply text formatting (bold, italic, color, etc.) to a range or found text. Use EITHER startIndex+endIndex OR textToFind for targeting.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        startIndex: { type: "number", description: "Start index (1-based) - use with endIndex" },
        endIndex: { type: "number", description: "End index (exclusive) - use with startIndex" },
        textToFind: { type: "string", description: "Text to find and format (alternative to indices)" },
        matchInstance: { type: "number", description: "Which instance of textToFind (default: 1)" },
        bold: { type: "boolean", description: "Make text bold" },
        italic: { type: "boolean", description: "Make text italic" },
        underline: { type: "boolean", description: "Underline text" },
        strikethrough: { type: "boolean", description: "Strikethrough text" },
        fontSize: { type: "number", description: "Font size in points" },
        fontFamily: { type: "string", description: "Font family name" },
        foregroundColor: { type: "string", description: "Hex color (e.g., #FF0000)" },
        backgroundColor: { type: "string", description: "Hex background color" },
        linkUrl: { type: "string", description: "URL for hyperlink" },
        baselineOffset: { type: "string", enum: ["SUPERSCRIPT", "SUBSCRIPT", "NONE"], description: "Vertical text offset: SUPERSCRIPT or SUBSCRIPT. Use NONE to reset text to the normal baseline." },
        tabId: { type: "string", description: "Optional. Tab ID to format within (from listDocumentTabs). If omitted, operates on the first/default tab." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "applyParagraphStyle",
    description: "Apply paragraph formatting — alignment, indents, spacing, named styles, borders (top/bottom/left/right/between), shading, and pagination controls. Use EITHER startIndex+endIndex OR textToFind OR indexWithinParagraph for targeting. The update styles EVERY paragraph overlapping the range, so one call over the whole body with removeBorders: [\"all\"] strips every paragraph border from a document.",
    inputSchema: applyParagraphStyleInputSchema
  },
  {
    name: "formatGoogleDocText",
    description: "Apply text formatting (bold, italic, font, color, links, superscript/subscript) to a range or found text in a Google Doc. Alias for applyTextStyle.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        startIndex: { type: "number", description: "Start index (1-based) - use with endIndex" },
        endIndex: { type: "number", description: "End index (exclusive) - use with startIndex" },
        textToFind: { type: "string", description: "Text to find and format (alternative to indices)" },
        matchInstance: { type: "number", description: "Which instance of textToFind (default: 1)" },
        bold: { type: "boolean", description: "Make text bold" },
        italic: { type: "boolean", description: "Make text italic" },
        underline: { type: "boolean", description: "Underline text" },
        strikethrough: { type: "boolean", description: "Strikethrough text" },
        fontSize: { type: "number", description: "Font size in points" },
        fontFamily: { type: "string", description: "Font family name" },
        foregroundColor: { type: "string", description: "Hex color (e.g., #FF0000)" },
        backgroundColor: { type: "string", description: "Hex background color" },
        linkUrl: { type: "string", description: "URL for hyperlink" },
        baselineOffset: { type: "string", enum: ["SUPERSCRIPT", "SUBSCRIPT", "NONE"], description: "Vertical text offset: SUPERSCRIPT or SUBSCRIPT. Use NONE to reset text to the normal baseline." },
        tabId: { type: "string", description: "Optional. Tab ID to format within (from listDocumentTabs). If omitted, operates on the first/default tab." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "formatGoogleDocParagraph",
    description: "Apply paragraph formatting (alignment, indentation, spacing, heading style, borders, shading) in a Google Doc. Alias for applyParagraphStyle.",
    inputSchema: applyParagraphStyleInputSchema
  },
  {
    name: "createParagraphBullets",
    description: "Add or remove bullet points / numbered lists on paragraphs in a Google Doc. Target paragraphs by startIndex+endIndex or textToFind. Use bulletPreset='NONE' to remove bullets.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        startIndex: { type: "number", description: "Start index (1-based) - use with endIndex" },
        endIndex: { type: "number", description: "End index (exclusive) - use with startIndex" },
        textToFind: { type: "string", description: "Text within the target paragraph(s) to bulletize" },
        matchInstance: { type: "number", description: "Which instance of textToFind (default: 1)" },
        bulletPreset: { type: "string", enum: ["BULLET_DISC_CIRCLE_SQUARE", "BULLET_DIAMONDX_ARROW3D_SQUARE", "BULLET_CHECKBOX", "BULLET_ARROW_DIAMOND_DISC", "BULLET_STAR_CIRCLE_SQUARE", "BULLET_ARROW3D_CIRCLE_SQUARE", "BULLET_LEFTTRIANGLE_DIAMOND_DISC", "NUMBERED_DECIMAL_ALPHA_ROMAN", "NUMBERED_DECIMAL_ALPHA_ROMAN_PARENS", "NUMBERED_DECIMAL_NESTED", "NUMBERED_UPPERALPHA_ALPHA_ROMAN", "NUMBERED_UPPERROMAN_UPPERALPHA_DECIMAL", "NUMBERED_ZERODECIMAL_ALPHA_ROMAN", "NONE"], description: "Bullet style preset. Use NONE to remove bullets. Default: BULLET_DISC_CIRCLE_SQUARE" },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." },
        tabId: { type: "string", description: "Optional. Tab ID to operate within (from listDocumentTabs). If omitted, operates on the first/default tab." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "findAndReplaceInDoc",
    description: "Find and replace text across a Google Document. Dry-run mode counts matches exactly across body, tables, headers, footers, and footnotes. On zero matches the response explains the likeliest lookalike cause (non-breaking spaces, curly quotes, HTML entities, case). For multi-tab docs, specify tabId to scope replacements to a single tab. When replaceText contains a newline, the write is always locked to the revision its matches were located in — even without an explicit ifRevisionId — since the compiled batch targets indices from that read; a concurrent edit fails the call instead of landing at the wrong offsets.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        findText: { type: "string", description: "Text to find" },
        replaceText: { type: "string", description: "Replacement text" },
        matchCase: { type: "boolean", description: "Case-sensitive match (default: false)" },
        dryRun: { type: "boolean", description: "Only count matches, do not modify the document (default: false). Respects tabId scoping." },
        tabId: { type: "string", description: "Optional. Tab ID to scope replacements to (from listDocumentTabs). If omitted, replaces across all tabs." },
        expectedCount: { type: "number", description: "Optional safety guard: the exact number of occurrences you expect to replace. Counted before writing; on a mismatch the call aborts without modifying the document (catches substring collisions and stale assumptions). Costs one extra document read per call, and a zero-match result costs one more to diagnose the likeliest cause — this tool is no longer a single API request when the guard is set." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId", "findText", "replaceText"]
    }
  },
  {
    name: "listComments",
    description: "List all comments in a Google Document",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        includeDeleted: { type: "boolean", description: "Whether to include deleted comments (default: false)" },
        pageSize: { type: "number", description: "Max comments to return (1-100, default: 100)" },
        pageToken: { type: "string", description: "Token for next page of results" },
      },
      required: ["documentId"]
    }
  },
  {
    name: "getComment",
    description: "Get a specific comment with its full thread of replies",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        commentId: { type: "string", description: "The comment ID" }
      },
      required: ["documentId", "commentId"]
    }
  },
  {
    name: "addComment",
    description: "Add a comment quoting a text range, targeted by startIndex+endIndex or by textToFind. The comment appears in the document's comment panel and records the quoted passage. Google Docs does not support margin-anchored comments through any public API, so the comment is unanchored by design; this is a platform limit, not a failure.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        startIndex: { type: "number", description: "Start index (1-based; or use textToFind)" },
        endIndex: { type: "number", description: "End index (exclusive; or use textToFind)" },
        textToFind: { type: "string", description: "Quote this exact text instead of using indices (case-sensitive)" },
        matchInstance: { type: "number", description: "Which occurrence of textToFind to quote (1-based, default 1)" },
        tabId: { type: "string", description: "Tab ID for multi-tab documents (defaults to the document body)" },
        commentText: { type: "string", description: "The comment content" }
      },
      required: ["documentId", "commentText"]
    }
  },
  {
    name: "replyToComment",
    description: "Add a reply to an existing comment",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        commentId: { type: "string", description: "The comment ID to reply to" },
        replyText: { type: "string", description: "The reply content" },
        resolve: { type: "boolean", description: "Set to true to resolve the comment thread after replying (default: false)" }
      },
      required: ["documentId", "commentId", "replyText"]
    }
  },
  {
    name: "deleteComment",
    description: "Delete a comment from the document",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        commentId: { type: "string", description: "The comment ID to delete" }
      },
      required: ["documentId", "commentId"]
    }
  },
  {
    name: "getGoogleDocContent",
    description: "Get content of a Google Doc with text indices for formatting",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        includeFormatting: { type: "boolean", description: "Include font, style, color, and baseline info for each text span, plus one ¶ line per paragraph with non-default paragraph styles (named style, alignment, visible borders, shading) over its real index span (default: false)" },
      },
      required: ["documentId"]
    }
  },
  {
    name: "getGoogleDocContentPaginated",
    description: "Get a portion of a Google Doc with text indices for formatting, with pagination support. Use offset and limit to read large documents in chunks.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        includeFormatting: { type: "boolean", description: "Include font, style, color, and baseline info for each text span, plus one ¶ line per paragraph with non-default paragraph styles (named style, alignment, visible borders, shading) over its real index span (default: false)" },
        offset: { type: "number", description: "Character offset into the formatted indexed output (not raw doc characters). 0-based, default: 0. Pass the previous response's nextOffset to get the following page." },
        limit: { type: "number", description: "Maximum characters to return per page (default: 50000, max: 80000; kept below the host's ~100K output cap to leave room for the JSON envelope and newline escaping). The page end is snapped back to a line boundary where possible; a single line longer than limit is hard-cut to guarantee forward progress." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "getGoogleDocImage",
    description: "Fetch the bytes of an inline image embedded in a Google Doc, keyed by its inline object ID. Returns a native image the model can see (or base64). The doc is re-fetched so the underlying image URL is always fresh — pass the objectId, never a raw contentUri (those expire). Inline images only; floating/anchored images are not supported.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        inlineObjectId: { type: "string", description: "The inline object ID shown in readGoogleDoc / getGoogleDocContent image placeholders (the objectId=... value, e.g. \"kix.abc123\")" },
        outputFormat: { type: "string", enum: ["image", "base64"], description: "\"image\" (default) returns a native MCP image block the model can view; \"base64\" returns a JSON envelope { inlineObjectId, mimeType, byteLength, dataBase64 } for programmatic use (forwarding, save-to-disk)" }
      },
      required: ["documentId", "inlineObjectId"]
    }
  },
  {
    name: "insertTable",
    description: "Insert a new table with the specified dimensions at a given index. For multi-tab docs, specify tabId to target a specific tab.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        rows: { type: "number", description: "Number of rows for the new table" },
        columns: { type: "number", description: "Number of columns for the new table" },
        index: { type: "number", description: "The index (1-based) where the table should be inserted" },
        tabId: { type: "string", description: "Optional. Tab ID to insert the table into (from listDocumentTabs). If omitted, inserts into the first/default tab." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId", "rows", "columns", "index"]
    }
  },
  {
    name: "editTableCell",
    description: "Edit the content and/or style of a specific table cell. Requires knowing the table start index. For multi-tab docs, specify tabId to target a table in a specific tab.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        tableStartIndex: { type: "number", description: "The starting index of the TABLE element" },
        rowIndex: { type: "number", description: "Row index (0-based)" },
        columnIndex: { type: "number", description: "Column index (0-based)" },
        textContent: { type: "string", description: "New text content for the cell (replaces existing)" },
        bold: { type: "boolean", description: "Make text bold" },
        italic: { type: "boolean", description: "Make text italic" },
        fontSize: { type: "number", description: "Font size in points" },
        alignment: { type: "string", enum: ["START", "CENTER", "END", "JUSTIFIED"], description: "Text alignment" },
        tabId: { type: "string", description: "Optional. Tab ID containing the table (from listDocumentTabs). If omitted, operates on the first/default tab." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId", "tableStartIndex", "rowIndex", "columnIndex"]
    }
  },
  {
    name: "styleDocTable",
    description: "Style a Docs table in one atomic call: cell borders/background/padding/content alignment (whole table via tableStartLocation, or a cell range subset), fixed column widths, and minimum row heights. Get the real tableStartIndex from the table span line readGoogleDoc or getGoogleDocContent prints (tableStartIndex=N) — never compute it from text offsets. removeBorders: [\"all\"] unboxes the table (width-0 borders). The API resolves shared edges by applying border updates right-to-left then bottom-to-top.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        tableStartIndex: { type: "number", description: "Start index of the TABLE element, as printed by readGoogleDoc / getGoogleDocContent on the table's span line (tableStartIndex=N)" },
        rowIndex: { type: "number", description: "Optional cell-subset targeting: first row (0-based); needs columnIndex too" },
        columnIndex: { type: "number", description: "Optional cell-subset targeting: first column (0-based); needs rowIndex too" },
        rowSpan: { type: "number", description: "Rows in the targeted cell range (default: 1)" },
        columnSpan: { type: "number", description: "Columns in the targeted cell range (default: 1)" },
        backgroundColor: { type: "string", description: "Cell background color as hex (e.g., #F1F3F4)" },
        removeBackground: { type: "boolean", description: "Clear the cell background color" },
        contentAlignment: { type: "string", enum: ["TOP", "MIDDLE", "BOTTOM"], description: "Vertical content alignment within cells" },
        padding: { type: "number", description: "Cell padding on all four edges, in points" },
        borderTop: tableCellBorderProperty("Cell top border."),
        borderBottom: tableCellBorderProperty("Cell bottom border."),
        borderLeft: tableCellBorderProperty("Cell left border."),
        borderRight: tableCellBorderProperty("Cell right border."),
        removeBorders: { type: "array", items: { type: "string", enum: ["top", "bottom", "left", "right", "all"] }, description: "Border edges to clear via width-0 borders. [\"all\"] over the whole table = unbox it in one call" },
        columnIndices: { type: "array", items: { type: "number" }, description: "Columns (0-based) for columnWidth; omit for all columns" },
        columnWidth: { type: "number", description: "Fixed column width in points (minimum 5)" },
        rowIndices: { type: "array", items: { type: "number" }, description: "Rows (0-based) for minRowHeight; omit for all rows" },
        minRowHeight: { type: "number", description: "Minimum row height in points" },
        tabId: { type: "string", description: "Optional. Tab ID containing the table (from listDocumentTabs). If omitted, operates on the first/default tab." },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." }
      },
      required: ["documentId", "tableStartIndex"]
    }
  },
  {
    name: "insertImageFromUrl",
    description: "Insert an inline image into a Google Document from a publicly accessible URL",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        imageUrl: { type: "string", description: "Publicly accessible URL to the image" },
        index: { type: "number", description: "The index (1-based) where the image should be inserted" },
        width: { type: "number", description: "Width of the image in points" },
        height: { type: "number", description: "Height of the image in points" }
      },
      required: ["documentId", "imageUrl", "index"]
    }
  },
  {
    name: "insertLocalImage",
    description: "Upload a local image file to Google Drive and insert it into a Google Document",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        localImagePath: { type: "string", description: "Absolute path to the local image file" },
        index: { type: "number", description: "The index (1-based) where the image should be inserted" },
        width: { type: "number", description: "Width of the image in points" },
        height: { type: "number", description: "Height of the image in points" },
        uploadToSameFolder: { type: "boolean", description: "Upload to same folder as document (default: true)" },
        makePublic: { type: "boolean", description: "Make uploaded image publicly accessible (anyone with the link can view). Set to true if the Docs API cannot access the image through the authenticated user's permissions. Default: false" }
      },
      required: ["documentId", "localImagePath", "index"]
    }
  },
  {
    name: "listGoogleDocs",
    description: "Lists Google Documents from your Google Drive with optional filtering.",
    inputSchema: {
      type: "object",
      properties: {
        maxResults: { type: "integer", description: "Maximum number of documents to return (1-100)." },
        query: { type: "string", description: "Search query to filter documents by name or content." },
        orderBy: { type: "string", enum: [...DRIVE_ORDER_BY_VALUES], description: "Sort order for results. Keys without 'desc' sort ascending, so 'modifiedTime' is oldest-first.", default: "modifiedTime desc" }
      },
      required: []
    }
  },
  {
    name: "getDocumentInfo",
    description: "Gets detailed information about a specific Google Document.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The ID of the Google Document (from the URL)." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "getGoogleDocStyleSummary",
    description: "Compact style inventory of a whole document in a few hundred tokens: fonts with their size ladders, named-style counts, text colors, bordered and shaded paragraph locations, a table inventory with real index ranges, and the heading outline. The cheap first read before any formatting work, in place of pulling the full document JSON to answer style questions. Counts cover every tab; locations are prefixed with the tab title on a multi-tab document, since index spaces restart per tab. Reports the document's revisionId, so the same read that gives you the indices gives you the lock to pass as ifRevisionId on the write.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        tabId: { type: "string", description: "Optional. Summarize only this tab (from listDocumentTabs). If omitted, covers every tab." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "describeGoogleDocRange",
    description: "Style probe for one spot: the paragraph styles (named style, alignment, borders, shading, indents, spacing, bullets) and text runs (font, size, bold/italic/underline, colors, links) overlapping a range, with the real document index of every paragraph and run so the result feeds straight into applyParagraphStyle or applyTextStyle. Target it with startIndex (optionally endIndex) or with textToFind, not both. Describes one tab's body; headers, footers, and footnotes are separate index spaces and are not included. Reports the document's revisionId, so the same read that gives you the indices gives you the lock to pass as ifRevisionId on the write.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "The document ID" },
        startIndex: { type: "number", description: "Start index (1-based). Use this or textToFind." },
        endIndex: { type: "number", description: "End index (exclusive). Defaults to startIndex + 1. Only with startIndex." },
        textToFind: { type: "string", description: "Describe the styling at this exact text (case-sensitive). Use this or startIndex." },
        matchInstance: { type: "number", description: "Which occurrence of textToFind (1-based, default 1)" },
        tabId: { type: "string", description: "Optional. Tab to probe (from listDocumentTabs). If omitted, uses the first/default tab." }
      },
      required: ["documentId"]
    }
  },
  {
    name: "addDocumentTab",
    description: "Add a new tab in a Google Doc",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        title: { type: "string", description: "Tab title" }
      },
      required: ["documentId", "title"]
    }
  },
  {
    name: "renameDocumentTab",
    description: "Rename an existing Google Doc tab",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        tabId: { type: "string", description: "Tab ID" },
        title: { type: "string", description: "New tab title" }
      },
      required: ["documentId", "tabId", "title"]
    }
  },
  {
    name: "insertSmartChip",
    description: "Insert a person smart chip (mention) at a document index. Only person chips are supported by the Docs API; date and file chips are read-only.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        index: { type: "number", description: "Insertion index (1-based)" },
        chipType: { type: "string", enum: ["person"], description: "Smart chip type (only 'person' is supported)" },
        personEmail: { type: "string", description: "Email address for the person mention" },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." },
        tabId: { type: "string", description: "Optional. Tab ID to insert into (from listDocumentTabs). If omitted, inserts into the first/default tab." }
      },
      required: ["documentId", "index", "chipType", "personEmail"]
    }
  },
  {
    name: "readSmartChips",
    description: "Read smart chip-like elements (person mentions, rich links, date chips) from the default tab of a document",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" }
      },
      required: ["documentId"]
    }
  },
  {
    name: "createFootnote",
    description: "Create a footnote in a Google Doc. Footnotes cannot be inserted inside equations, headers, footers, or other footnotes. For multi-tab docs, specify tabId to target a specific tab.",
    inputSchema: {
      type: "object",
      properties: {
        documentId: { type: "string", description: "Document ID" },
        index: { type: "number", description: "1-based character index where the footnote reference should be inserted" },
        endOfSegment: { type: "boolean", description: "If true, insert footnote at the end of the document body (use instead of index)" },
        content: { type: "string", description: "Optional text content for the footnote body" },
        ifRevisionId: { type: "string", description: "Optional optimistic lock: a revisionId from readGoogleDoc, getGoogleDocContent, getGoogleDocContentPaginated, or getDocumentInfo. The write fails cleanly if the document changed since that read. Valid for 24 hours, and only for the account that read it. Google Docs only." },
        tabId: { type: "string", description: "Optional. Tab ID to insert the footnote into (from listDocumentTabs). If omitted, inserts into the first/default tab." },
      },
      required: ["documentId"]
    }
  },
];

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handleTool(toolName: string, args: Record<string, unknown>, ctx: ToolContext): Promise<ToolResult | null> {
  switch (toolName) {

    // =========================================================================
    // CREATE / UPDATE GOOGLE DOC
    // =========================================================================

    case "createGoogleDoc": {
      const validation = CreateGoogleDocSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const parentFolderId = await ctx.resolveFolderId(a.parentFolderId);

      // Check if document already exists
      const existingFileId = await ctx.checkFileExists(a.name, parentFolderId);
      if (existingFileId) {
        return errorResponse(
          `A document named "${a.name}" already exists in this location. ` +
          `To update it, use updateGoogleDoc with documentId: ${existingFileId}`
        );
      }

      // Create empty doc
      let docResponse;
      try {
        docResponse = await ctx.getDrive().files.create({
          requestBody: {
            name: a.name,
            mimeType: 'application/vnd.google-apps.document',
            parents: [parentFolderId]
          },
          fields: 'id, name, webViewLink',
          supportsAllDrives: true
        });
      } catch (createError: any) {
        ctx.log('Drive files.create error details:', {
          message: createError.message,
          code: createError.code,
          errors: createError.errors,
          status: createError.status
        });
        throw createError;
      }
      const doc = docResponse.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      try {
        await withRetry(
          (signal) => docs.documents.batchUpdate(
            {
              documentId: doc.id!,
              requestBody: {
                requests: [
                  {
                    insertText: { location: { index: 1 }, text: a.content }
                  },
                  // Ensure the text is formatted as normal text, not as a header
                  {
                    updateParagraphStyle: {
                      range: {
                        startIndex: 1,
                        endIndex: a.content.length + 1
                      },
                      paragraphStyle: {
                        namedStyleType: 'NORMAL_TEXT'
                      },
                      fields: 'namedStyleType'
                    }
                  }
                ]
              }
            },
            { signal }
          ),
          ctx.runtimeConfig,
          `createGoogleDoc.batchUpdate(${a.name})`,
          ctx.log
        );
      } catch (batchErr: any) {
        ctx.log('batchUpdate failed after retries; doc created without content:', batchErr);
        const reason = String(batchErr?.message ?? 'unknown error').split('\n')[0].slice(0, 200);
        return {
          content: [{
            type: "text",
            text: `Created Google Doc but content insertion failed: ${doc.name}\nID: ${doc.id}\nLink: ${doc.webViewLink}\nReason: ${reason}\nRetry content insertion with updateGoogleDoc (documentId: ${doc.id}).`
          }],
          isError: true
        };
      }

      return {
        content: [{ type: "text", text: `Created Google Doc: ${doc.name}\nID: ${doc.id}\nLink: ${doc.webViewLink}` }],
        isError: false
      };
    }

    case "createDocFromHTML": {
      const validation = CreateDocFromHTMLSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const parentFolderId = await ctx.resolveFolderId(a.parentFolderId);

      // Check if document already exists
      const existingFileId = await ctx.checkFileExists(a.name, parentFolderId);
      if (existingFileId) {
        return errorResponse(
          `A document named "${a.name}" already exists in this location. ` +
          `Use a different name or delete the existing doc (ID: ${existingFileId}).`
        );
      }

      // Upload HTML content as a Google Doc via Drive API MIME conversion.
      // The Drive API automatically converts HTML tags to native Google Doc styles:
      // <h1> → HEADING_1, <h2> → HEADING_2, <p> → NORMAL_TEXT, etc.
      ctx.log('Creating Google Doc from HTML', { name: a.name, htmlLength: a.html.length });

      const htmlBuffer = Buffer.from(a.html, 'utf-8');

      let fileResponse;
      try {
        fileResponse = await ctx.getDrive().files.create({
          requestBody: {
            name: a.name,
            mimeType: 'application/vnd.google-apps.document',
            parents: [parentFolderId]
          },
          media: {
            mimeType: 'text/html',
            body: Readable.from(htmlBuffer)
          },
          fields: 'id, name, webViewLink',
          supportsAllDrives: true
        });
      } catch (createError: any) {
        ctx.log('Drive files.create (HTML) error details:', {
          message: createError.message,
          code: createError.code,
          errors: createError.errors,
          status: createError.status
        });
        throw createError;
      }

      const newDoc = fileResponse.data;
      return {
        content: [{ type: "text", text: `Created Google Doc from HTML: ${newDoc.name}\nID: ${newDoc.id}\nLink: ${newDoc.webViewLink}` }],
        isError: false
      };
    }

    case "updateGoogleDoc": {
      const validation = UpdateGoogleDocSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      if (a.tabId) {
        // Tab-scoped path: single atomic batchUpdate so a failed insert can't leave the tab wiped.
        const document = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });
        const tabs = (document.data as any).tabs as any[] | undefined;
        const tab = tabs ? findTabById(tabs, a.tabId) : null;
        if (!tab) {
          return errorResponse(`Tab with ID "${a.tabId}" not found. Use listDocumentTabs to see available tabs.`);
        }

        const bodyContent = tab.documentTab?.body?.content;
        const lastEndIndex = bodyContent?.[bodyContent.length - 1]?.endIndex ?? 1;
        const deleteEndIndex = Math.max(1, lastEndIndex - 1);

        const requests: any[] = [];
        if (deleteEndIndex > 1) {
          requests.push({
            deleteContentRange: {
              range: { startIndex: 1, endIndex: deleteEndIndex, tabId: a.tabId }
            }
          });
        }
        requests.push({
          insertText: { location: { index: 1, tabId: a.tabId }, text: a.content }
        });
        requests.push({
          updateParagraphStyle: {
            range: { startIndex: 1, endIndex: a.content.length + 1, tabId: a.tabId },
            paragraphStyle: { namedStyleType: 'NORMAL_TEXT' },
            fields: 'namedStyleType'
          }
        });

        await docs.documents.batchUpdate({
          documentId: a.documentId,
          requestBody: { requests, ...writeControlFor(a.ifRevisionId) }
        });

        return {
          content: [{ type: "text", text: `Updated Google Doc: ${document.data.title} (tab: ${a.tabId})` }],
          isError: false
        };
      }

      const document = await docs.documents.get({ documentId: a.documentId });

      // Delete all content
      const endIndex = document.data.body?.content?.[document.data.body.content.length - 1]?.endIndex || 1;
      const deleteEndIndex = Math.max(1, endIndex - 1);

      // One atomic batchUpdate, same as the tabId path above. Splitting the
      // delete from the insert meant a failed insert left the document wiped,
      // and it left the lock covering only the first of the two calls — on an
      // empty document the delete is skipped entirely, so ifRevisionId guarded
      // nothing at all.
      const requests: any[] = [];
      if (deleteEndIndex > 1) {
        requests.push({
          deleteContentRange: {
            range: { startIndex: 1, endIndex: deleteEndIndex }
          }
        });
      }
      requests.push({
        insertText: { location: { index: 1 }, text: a.content }
      });
      requests.push({
        updateParagraphStyle: {
          range: { startIndex: 1, endIndex: a.content.length + 1 },
          paragraphStyle: { namedStyleType: 'NORMAL_TEXT' },
          fields: 'namedStyleType'
        }
      });

      await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: { requests, ...writeControlFor(a.ifRevisionId) }
      });

      return {
        content: [{ type: "text", text: `Updated Google Doc: ${document.data.title}` }],
        isError: false
      };
    }

    // =========================================================================
    // DOC CONTENT
    // =========================================================================

    case "getGoogleDocContent": {
      const validation = GetGoogleDocContentSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;
      const withFormatting = a.includeFormatting === true;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const document = await docs.documents.get({
        documentId: a.documentId,
        includeTabsContent: true,
      });

      const { formattedContent, totalLength } = buildDocFormattedContent(document.data, withFormatting);

      // This is the read a caller makes immediately before an index-based
      // write, so the lock has to be obtainable from it — otherwise the indices
      // come from one fetch and the lock from a later one.
      return {
        content: [{
          type: "text",
          text: formattedContent
            + `\nTotal length: ${totalLength} characters`
            + `\nrevisionId: ${revisionIdOf(document.data)}`
        }],
        isError: false
      };
    }

    case "getGoogleDocContentPaginated": {
      const validation = GetGoogleDocContentPaginatedSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;
      const withFormatting = a.includeFormatting === true;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const document = await docs.documents.get({
        documentId: a.documentId,
        includeTabsContent: true,
      });

      const { formattedContent, totalLength } = buildDocFormattedContent(document.data, withFormatting);

      const offset = a.offset;
      const limit = a.limit;
      // Block-aware: table rows and the cell map carry no index of their own, so
      // a line-wise split can open a page with bare `| ... |` lines.
      const { slice: slicedContent, end } = sliceMarkdownByBlock(formattedContent, offset, limit);
      const hasMore = end < formattedContent.length;

      const result = {
        outputLength: formattedContent.length,
        documentLength: totalLength,
        offset,
        limit,
        nextOffset: end,
        hasMore,
        revisionId: revisionIdOf(document.data),
        content: slicedContent,
      };

      return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        isError: false
      };
    }

    case "getGoogleDocImage": {
      const validation = GetGoogleDocImageSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      // Re-fetch the doc so the resolved contentUri is fresh (they expire ~30 min).
      const docResponse = await docs.documents.get({
        documentId: a.documentId,
        includeTabsContent: true,
      });

      const embedded = findInlineObjectById(docResponse.data, a.inlineObjectId);
      if (!embedded) {
        return errorResponse(
          `Inline object "${a.inlineObjectId}" not found in document. Use readGoogleDoc or getGoogleDocContent to list valid inline image objectIds.`
        );
      }
      const contentUri = embedded.imageProperties?.contentUri;
      if (!contentUri) {
        // An image inserted from a URL exposes only sourceUri (an external URL).
        // We deliberately don't fetch it server-side (that would send Google
        // credentials to a third-party host), so surface the URL instead of the
        // misleading "embedded chart or drawing" message the readers contradict.
        const sourceUri = embedded.imageProperties?.sourceUri;
        if (sourceUri) {
          return errorResponse(
            `Inline object "${a.inlineObjectId}" has no Google-hosted image to fetch; it references an external source URL that this tool does not retrieve: ${sourceUri} — fetch it directly.`
          );
        }
        return errorResponse(
          `Inline object "${a.inlineObjectId}" has no fetchable image content (e.g. an embedded chart or drawing rather than a raster image).`
        );
      }

      // Authenticated fetch of the (fresh) contentUri — same primitive used for
      // Drive export links elsewhere in this codebase. maxContentLength aborts an
      // oversized body mid-download rather than buffering it all first; the
      // byteLength check below is a backstop for responses lacking Content-Length.
      const resp = await ctx.authClient.request({
        url: contentUri,
        responseType: 'arraybuffer',
        maxContentLength: MAX_IMAGE_BYTES,
      });
      const buffer = Buffer.from(resp.data as ArrayBuffer);

      if (buffer.byteLength > MAX_IMAGE_BYTES) {
        return errorResponse(
          `Image is too large to return inline (${(buffer.byteLength / (1024 * 1024)).toFixed(1)} MB, limit 40 MB).`
        );
      }

      const rawContentType = getResponseHeader(resp.headers, 'content-type') || 'application/octet-stream';
      const mimeType = rawContentType.split(';')[0].trim();
      const dataBase64 = buffer.toString('base64');

      if (a.outputFormat === 'base64') {
        const envelope = {
          inlineObjectId: a.inlineObjectId,
          mimeType,
          byteLength: buffer.byteLength,
          dataBase64,
        };
        return {
          content: [{ type: "text", text: JSON.stringify(envelope, null, 2) }],
          isError: false
        };
      }

      return {
        content: [{ type: "image", data: dataBase64, mimeType }],
        isError: false
      };
    }

    // =========================================================================
    // DOC EDITING TOOLS
    // =========================================================================

    case "insertText": {
      const validation = InsertTextSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      let mimeType = '';
      let fileName = 'unknown';
      try {
        const fileMeta = await ctx.getDrive().files.get({
          fileId: a.documentId,
          fields: 'mimeType, name',
          supportsAllDrives: true
        });
        mimeType = fileMeta.data.mimeType || '';
        fileName = fileMeta.data.name || 'unknown';
      } catch {
        // Under the drive.file scope, metadata reads fail for Google Docs the app
        // did not create. Fall back to the Docs API path (documents scope), which
        // is unaffected by drive.file — restoring pre-text-file-support behavior.
        mimeType = 'application/vnd.google-apps.document';
      }

      if (mimeType === 'application/vnd.google-apps.document') {
        let index: number;
        let targetNote = '';
        if (a.textToFind !== undefined) {
          const range = await findTextRange(ctx, a.documentId, a.textToFind, a.matchInstance, a.tabId);
          if (range && 'error' in range) {
            return errorResponse(range.error);
          }
          if (!range) {
            return errorResponse(`Text "${a.textToFind}" not found in document`);
          }
          index = a.position === 'before' ? range.startIndex : range.endIndex;
          // Docs refuses an insert at the segment end (just past the final
          // paragraph break). A match that runs to the end of the segment
          // therefore gets its "after" point moved before that break, which is
          // still visually after the matched text.
          if (range.segmentEndIndex !== undefined && index >= range.segmentEndIndex) {
            index = range.segmentEndIndex - 1;
          }
          targetNote = ` (${a.position} "${a.textToFind}"${a.matchInstance > 1 ? ` match ${a.matchInstance}` : ''})`;
        } else {
          index = a.index!;
        }
        if (index < 1) {
          return errorResponse('For Google Docs, index must be at least 1 (1-based).');
        }
        const location: { index: number; tabId?: string } = { index };
        if (a.tabId) location.tabId = a.tabId;

        const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
        await docs.documents.batchUpdate({
          documentId: a.documentId,
          requestBody: {
            requests: [{
              insertText: {
                location,
                text: a.text
              }
            }],
            ...writeControlFor(a.ifRevisionId)
          }
        });

        return {
          content: [{ type: "text", text: `Successfully inserted ${a.text.length} characters at index ${index}${targetNote}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
          isError: false
        };
      }

      if (isTextMime(mimeType)) {
        if (a.tabId) {
          return errorResponse('tabId is not supported for text files');
        }
        if (a.textToFind !== undefined) {
          return errorResponse('textToFind targeting is only supported on Google Docs — pass a 0-based index for text files');
        }
        if (a.ifRevisionId) {
          // Silently ignoring the lock is worse than refusing it: the caller
          // believes the write was guarded when nothing checked it.
          return errorResponse('ifRevisionId is only supported for Google Docs, not text files');
        }

        const content = await downloadTextContent(ctx.getDrive(), a.documentId);
        // Index offsets are 0-based Unicode code points, so edit on the code-point
        // array — slicing the raw JS string (UTF-16 units) would split a surrogate
        // pair and corrupt astral characters (emoji) on write-back.
        const codePoints = Array.from(content);

        if (a.index! > codePoints.length) {
          return errorResponse(`index ${a.index} is beyond end of file (length ${codePoints.length})`);
        }

        const updated = codePoints.slice(0, a.index).join('') + a.text + codePoints.slice(a.index).join('');
        await writeTextContent(ctx.getDrive(), a.documentId, mimeType, updated);

        return {
          content: [{ type: "text", text: `Successfully inserted ${a.text.length} characters at offset ${a.index} in ${fileName}` }],
          isError: false
        };
      }

      return errorResponse(
        `File "${fileName}" has MIME type "${mimeType}". insertText only supports Google Docs (application/vnd.google-apps.document) and text/* files.`
      );
    }

    case "deleteRange": {
      const validation = DeleteRangeSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      let mimeType = '';
      let fileName = 'unknown';
      try {
        const fileMeta = await ctx.getDrive().files.get({
          fileId: a.documentId,
          fields: 'mimeType, name',
          supportsAllDrives: true
        });
        mimeType = fileMeta.data.mimeType || '';
        fileName = fileMeta.data.name || 'unknown';
      } catch {
        // Under the drive.file scope, metadata reads fail for Google Docs the app
        // did not create. Fall back to the Docs API path (documents scope), which
        // is unaffected by drive.file — restoring pre-text-file-support behavior.
        mimeType = 'application/vnd.google-apps.document';
      }

      if (mimeType === 'application/vnd.google-apps.document') {
        let startIndex: number;
        let endIndex: number;
        let targetNote = '';
        if (a.textToFind !== undefined) {
          const found = await findTextRange(ctx, a.documentId, a.textToFind, a.matchInstance, a.tabId);
          if (found && 'error' in found) {
            return errorResponse(found.error);
          }
          if (!found) {
            return errorResponse(`Text "${a.textToFind}" not found in document`);
          }
          startIndex = found.startIndex;
          endIndex = found.endIndex;
          targetNote = ` ("${a.textToFind}"${a.matchInstance > 1 ? ` match ${a.matchInstance}` : ''})`;
          // Docs refuses to delete a segment's final paragraph break. A match
          // that runs to the segment end is trimmed by one so the break
          // survives; a match that is only that break has nothing left to delete.
          if (found.segmentEndIndex !== undefined && endIndex >= found.segmentEndIndex) {
            endIndex = found.segmentEndIndex - 1;
            if (endIndex <= startIndex) {
              return errorResponse(`Text "${a.textToFind}" matches only the document's final paragraph break, which Google Docs does not allow deleting`);
            }
            targetNote += ' (trailing paragraph break kept)';
          }
        } else {
          startIndex = a.startIndex!;
          endIndex = a.endIndex!;
        }
        if (startIndex < 1) {
          return errorResponse('For Google Docs, startIndex must be at least 1 (1-based).');
        }
        const range: { startIndex: number; endIndex: number; tabId?: string } = {
          startIndex,
          endIndex
        };
        if (a.tabId) range.tabId = a.tabId;

        const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
        await docs.documents.batchUpdate({
          documentId: a.documentId,
          requestBody: {
            requests: [{
              deleteContentRange: { range }
            }],
            ...writeControlFor(a.ifRevisionId)
          }
        });

        return {
          content: [{ type: "text", text: `Successfully deleted content from index ${startIndex} to ${endIndex}${targetNote}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
          isError: false
        };
      }

      if (isTextMime(mimeType)) {
        if (a.tabId) {
          return errorResponse('tabId is not supported for text files');
        }
        if (a.textToFind !== undefined) {
          return errorResponse('textToFind targeting is only supported on Google Docs — pass 0-based startIndex/endIndex for text files');
        }
        if (a.ifRevisionId) {
          // Silently ignoring the lock is worse than refusing it: the caller
          // believes the write was guarded when nothing checked it.
          return errorResponse('ifRevisionId is only supported for Google Docs, not text files');
        }

        const startIndex = a.startIndex!;
        const endIndex = a.endIndex!;
        const content = await downloadTextContent(ctx.getDrive(), a.documentId);
        // Offsets are 0-based Unicode code points, so edit on the code-point array
        // — slicing the raw JS string (UTF-16 units) would split a surrogate pair
        // and corrupt astral characters (emoji) on write-back.
        const codePoints = Array.from(content);

        if (endIndex > codePoints.length) {
          return errorResponse(`endIndex ${endIndex} is beyond end of file (length ${codePoints.length})`);
        }

        const updated = codePoints.slice(0, startIndex).join('') + codePoints.slice(endIndex).join('');
        await writeTextContent(ctx.getDrive(), a.documentId, mimeType, updated);

        return {
          content: [{ type: "text", text: `Successfully deleted ${endIndex - startIndex} characters from offset ${startIndex} to ${endIndex} in ${fileName}` }],
          isError: false
        };
      }

      return errorResponse(
        `File "${fileName}" has MIME type "${mimeType}". deleteRange only supports Google Docs (application/vnd.google-apps.document) and text/* files.`
      );
    }

    case "readGoogleDoc": {
      const validation = ReadGoogleDocSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const docResponse = await docs.documents.get({
        documentId: a.documentId,
        includeTabsContent: true,
      });

      const doc = docResponse.data;
      const format = a.format || 'text';

      if (format === 'json') {
        let result = JSON.stringify(doc, null, 2);
        if (a.maxLength && result.length > a.maxLength) {
          result = result.substring(0, a.maxLength) + '\n... (truncated)';
        }
        return {
          content: [{ type: "text", text: result }],
          isError: false
        };
      }

      const { text, error } = extractDocText(doc, a.tabId, format === 'markdown' ? 'markdown' : 'text');
      if (error) {
        return errorResponse(error);
      }

      let resultText = text;
      if (format === 'markdown') {
        resultText = withMarkdownTitle(doc.title, resultText);
      }

      // Surface the revision so callers can optimistically lock their writes
      // (pass it as ifRevisionId on any Docs write tool). The line is counted
      // against maxLength and kept whole, so a truncated read is still lockable
      // and the response still honors the cap the caller asked for.
      const revisionLine = `revisionId: ${revisionIdOf(doc)}\n`;
      if (a.maxLength) {
        const room = Math.max(0, a.maxLength - revisionLine.length);
        if (resultText.length > room) {
          resultText = resultText.substring(0, room) + '\n... (truncated)';
        }
      }
      resultText = revisionLine + resultText;

      return {
        content: [{ type: "text", text: resultText }],
        isError: false
      };
    }

    case "readGoogleDocPaginated": {
      const validation = ReadGoogleDocPaginatedSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const docResponse = await docs.documents.get({
        documentId: a.documentId,
        includeTabsContent: true,
      });

      const doc = docResponse.data;
      const format = a.format || 'text';

      const { text, error } = extractDocText(doc, a.tabId, format === 'markdown' ? 'markdown' : 'text');
      if (error) {
        return errorResponse(error);
      }

      let fullText = text;
      if (format === 'markdown') {
        fullText = withMarkdownTitle(doc.title, fullText);
      }

      const offset = a.offset;
      const limit = a.limit;
      const { slice: slicedText, end } = format === 'markdown'
        ? sliceMarkdownByBlock(fullText, offset, limit)
        : sliceByLine(fullText, offset, limit);
      const hasMore = end < fullText.length;

      const result = {
        outputLength: fullText.length,
        documentLength: text.length,
        offset,
        limit,
        nextOffset: end,
        hasMore,
        content: slicedText,
      };

      return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        isError: false
      };
    }

    case "listDocumentTabs": {
      const validation = ListDocumentTabsSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      // Use includeTabsContent to get the new tabs structure
      const docResponse = await docs.documents.get({
        documentId: a.documentId,
        includeTabsContent: true
      });

      const doc = docResponse.data;

      // Check if document has tabs (newer API feature)
      const tabs = (doc as any).tabs;
      if (!tabs || tabs.length === 0) {
        // Single-tab document or legacy format - check for body content
        let contentInfo = '';
        if (a.includeContent) {
          let charCount = 0;
          const body = doc.body;
          if (body?.content) {
            for (const element of body.content) {
              if (element.paragraph?.elements) {
                for (const elem of element.paragraph.elements) {
                  if (elem.textRun?.content) {
                    charCount += elem.textRun.content.length;
                  }
                }
              }
            }
          }
          contentInfo = ` (${charCount} characters)`;
        }
        return {
          content: [{ type: "text", text: `Document "${doc.title}" has a single tab (standard format).${contentInfo}` }],
          isError: false
        };
      }

      // Process tabs
      const processTab = (tab: any, depth: number = 0): string => {
        const indent = '  '.repeat(depth);
        let result = `${indent}- Tab: "${tab.tabProperties?.title || 'Untitled'}" (ID: ${tab.tabProperties?.tabId})`;

        if (a.includeContent && tab.documentTab?.body?.content) {
          let charCount = 0;
          for (const element of tab.documentTab.body.content) {
            if (element.paragraph?.elements) {
              for (const elem of element.paragraph.elements) {
                if (elem.textRun?.content) {
                  charCount += elem.textRun.content.length;
                }
              }
            }
          }
          result += ` (${charCount} characters)`;
        }

        if (tab.childTabs) {
          for (const childTab of tab.childTabs) {
            result += '\n' + processTab(childTab, depth + 1);
          }
        }

        return result;
      };

      let tabList = `Document "${doc.title}" tabs:\n`;
      for (const tab of tabs) {
        tabList += processTab(tab) + '\n';
      }

      return {
        content: [{ type: "text", text: tabList }],
        isError: false
      };
    }

    // =========================================================================
    // TEXT & PARAGRAPH STYLE
    // =========================================================================

    case "applyTextStyle":
    case "formatGoogleDocText": {
      const validation = ApplyTextStyleSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      let startIndex: number;
      let endIndex: number;

      // Determine target range (flat parameters)
      if (a.startIndex !== undefined && a.endIndex !== undefined) {
        startIndex = a.startIndex;
        endIndex = a.endIndex;
      } else if (a.textToFind !== undefined) {
        const range = await findTextRange(
          ctx,
          a.documentId,
          a.textToFind,
          a.matchInstance || 1,
          a.tabId
        );
        if (range && 'error' in range) {
          return errorResponse(range.error);
        }
        if (!range) {
          return errorResponse(`Text "${a.textToFind}" not found in document`);
        }
        startIndex = range.startIndex;
        endIndex = range.endIndex;
      } else {
        return errorResponse("Must provide either startIndex+endIndex or textToFind");
      }

      // Build style object from flat parameters
      const style = {
        bold: a.bold,
        italic: a.italic,
        underline: a.underline,
        strikethrough: a.strikethrough,
        fontSize: a.fontSize,
        fontFamily: a.fontFamily,
        foregroundColor: a.foregroundColor,
        backgroundColor: a.backgroundColor,
        linkUrl: a.linkUrl,
        baselineOffset: a.baselineOffset
      };

      // Build the update request
      const styleResult = buildUpdateTextStyleRequest(startIndex, endIndex, style, a.tabId);
      if (!styleResult) {
        return errorResponse("No valid style options provided");
      }

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: {
          requests: [styleResult.request],
          ...writeControlFor(a.ifRevisionId)
        }
      });

      return {
        content: [{ type: "text", text: `Successfully applied text style to range ${startIndex}-${endIndex}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
        isError: false
      };
    }

    case "applyParagraphStyle":
    case "formatGoogleDocParagraph": {
      const validation = ApplyParagraphStyleSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      let startIndex: number;
      let endIndex: number;

      // Determine target range (flat parameters)
      if (a.startIndex !== undefined && a.endIndex !== undefined) {
        startIndex = a.startIndex;
        endIndex = a.endIndex;
      } else if (a.textToFind !== undefined) {
        // Tab-scoped resolution needs an unmasked includeTabsContent fetch;
        // resolve it once here and reuse it for both the text match and the
        // enclosing-paragraph lookup instead of letting each helper fetch the
        // whole document independently (#114 follow-up). The non-tab path
        // keeps its two cheap narrow-field GETs unchanged.
        let tabContent: any[] | undefined;
        if (a.tabId) {
          const resolved = await getTabBodyContent(ctx, a.documentId, a.tabId);
          if (resolved.error) {
            return errorResponse(resolved.error);
          }
          tabContent = resolved.content;
        }

        const range = await findTextRange(
          ctx,
          a.documentId,
          a.textToFind,
          a.matchInstance || 1,
          a.tabId,
          tabContent
        );
        if (range && 'error' in range) {
          return errorResponse(range.error);
        }
        if (!range) {
          return errorResponse(`Text "${a.textToFind}" not found in document`);
        }
        // For paragraph style, get the full paragraph range
        const paraRange = await getParagraphRange(ctx, a.documentId, range.startIndex, a.tabId, tabContent);
        if (paraRange && 'error' in paraRange) {
          return errorResponse(paraRange.error);
        }
        if (!paraRange) {
          return errorResponse("Could not determine paragraph boundaries");
        }
        startIndex = paraRange.startIndex;
        endIndex = paraRange.endIndex;
      } else if (a.indexWithinParagraph !== undefined) {
        const paraRange = await getParagraphRange(ctx, a.documentId, a.indexWithinParagraph, a.tabId);
        if (paraRange && 'error' in paraRange) {
          return errorResponse(paraRange.error);
        }
        if (!paraRange) {
          return errorResponse("Could not determine paragraph boundaries");
        }
        startIndex = paraRange.startIndex;
        endIndex = paraRange.endIndex;
      } else {
        return errorResponse("Must provide either startIndex+endIndex, textToFind, or indexWithinParagraph");
      }

      // Build style object from flat parameters
      const style = {
        alignment: a.alignment,
        indentStart: a.indentStart,
        indentEnd: a.indentEnd,
        indentFirstLine: a.indentFirstLine,
        spaceAbove: a.spaceAbove,
        spaceBelow: a.spaceBelow,
        namedStyleType: a.namedStyleType,
        keepWithNext: a.keepWithNext,
        keepLinesTogether: a.keepLinesTogether,
        avoidWidowAndOrphan: a.avoidWidowAndOrphan,
        pageBreakBefore: a.pageBreakBefore,
        borderTop: a.borderTop,
        borderBottom: a.borderBottom,
        borderLeft: a.borderLeft,
        borderRight: a.borderRight,
        borderBetween: a.borderBetween,
        removeBorders: a.removeBorders,
        shading: a.shading,
        removeShading: a.removeShading
      };

      // Build the update request
      const styleResult = buildUpdateParagraphStyleRequest(startIndex, endIndex, style, a.tabId);
      if (!styleResult) {
        return errorResponse("No valid style options provided");
      }

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: {
          requests: [styleResult.request],
          ...writeControlFor(a.ifRevisionId)
        }
      });

      return {
        content: [{ type: "text", text: `Successfully applied paragraph style to range ${startIndex}-${endIndex}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
        isError: false
      };
    }


    case "createParagraphBullets": {
      const validation = CreateParagraphBulletsSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      let startIndex: number;
      let endIndex: number;

      if (a.startIndex !== undefined && a.endIndex !== undefined) {
        startIndex = a.startIndex;
        endIndex = a.endIndex;
      } else if (a.textToFind !== undefined) {
        const range = await findTextRange(
          ctx,
          a.documentId,
          a.textToFind,
          a.matchInstance || 1,
          a.tabId
        );
        if (range && 'error' in range) {
          return errorResponse(range.error);
        }
        if (!range) {
          return errorResponse(`Text "${a.textToFind}" not found in document`);
        }
        startIndex = range.startIndex;
        endIndex = range.endIndex;
      } else {
        return errorResponse("Must provide either startIndex+endIndex or textToFind");
      }

      const range = withTab({ startIndex, endIndex }, a.tabId);

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      if (a.bulletPreset === 'NONE') {
        await docs.documents.batchUpdate({
          documentId: a.documentId,
          requestBody: {
            requests: [{
              deleteParagraphBullets: { range }
            }],
            ...writeControlFor(a.ifRevisionId)
          }
        });
        return {
          content: [{ type: "text", text: `Removed bullets from range ${startIndex}-${endIndex}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
          isError: false
        };
      }

      await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: {
          requests: [{
            createParagraphBullets: {
              range,
              bulletPreset: a.bulletPreset
            }
          }],
          ...writeControlFor(a.ifRevisionId)
        }
      });

      return {
        content: [{ type: "text", text: `Applied ${a.bulletPreset} bullets to range ${startIndex}-${endIndex}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
        isError: false
      };
    }

    case "findAndReplaceInDoc": {
      const validation = FindAndReplaceInDocSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      if (a.dryRun) {
        const doc = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });
        const text = collectDocPlainText(doc.data, a.tabId);
        const count = countOccurrences(text, a.findText, a.matchCase);
        let message = `Dry run: found ${count} occurrence(s) of "${a.findText}"${a.tabId ? ` in tab ${a.tabId}` : ''} (counted across body, tables, headers, footers, and footnotes).`;
        if (count === 0) {
          const hint = diagnoseZeroMatch(text, a.findText, a.matchCase);
          if (hint) message += `\nLikely cause: ${hint}.`;
        }
        return {
          content: [{ type: 'text', text: message }],
          isError: false,
        };
      }

      // Multi-line replacements: replaceAllText does not render an embedded
      // newline as a paragraph break, so the surrounding paragraphs flatten and
      // characters are lost. Compile to deleteContentRange + insertText instead —
      // insertText renders \n as real paragraph breaks, descending order keeps
      // every index valid, and the whole batch applies atomically.
      if (a.replaceText.includes('\n')) {
        if (a.findText.includes('\n')) {
          return errorResponse(
            'Multi-line findText is not supported: matches cannot span paragraph breaks. Use a single-line findText (the replacement may contain \\n).',
          );
        }
        const doc = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });
        const text = collectDocPlainText(doc.data, a.tabId);
        const plainCount = countOccurrences(text, a.findText, a.matchCase);
        const ranges = findOccurrenceRanges(doc.data, a.findText, a.matchCase, a.tabId);
        if (ranges.length === 0) {
          let msg = `Replaced 0 occurrence(s) of "${a.findText}".`;
          const hint = diagnoseZeroMatch(text, a.findText, a.matchCase);
          if (hint) msg += `\nLikely cause: ${hint}.`;
          return { content: [{ type: 'text', text: msg }], isError: false };
        }
        if (plainCount !== ranges.length) {
          return errorResponse(
            `Cannot safely apply a multi-line replacement here: ${plainCount} plain-text match(es) but only ${ranges.length} could be index-mapped. Re-read the document and use index-based edits instead.`,
          );
        }
        if (a.expectedCount !== undefined && ranges.length !== a.expectedCount) {
          return {
            content: [{ type: 'text', text: `Aborted without writing: expectedCount=${a.expectedCount} but found ${ranges.length} occurrence(s) of "${a.findText}".` }],
            isError: true,
          };
        }
        const requests = buildMultilineReplaceRequests(ranges, a.replaceText);
        // Lock the batch to the revision the ranges were located against: the
        // compiled indices are only valid for that revision. An explicit
        // ifRevisionId wins, so a caller's older read still guards the write.
        // Held in a variable so the catch below can tell whether this write
        // actually carried a lock.
        const lock = a.ifRevisionId ?? (doc.data.revisionId || undefined);
        try {
          await docs.documents.batchUpdate({
            documentId: a.documentId,
            requestBody: {
              requests: requests as any[],
              ...writeControlFor(lock),
            },
          });
        } catch (error: any) {
          ctx.log('Error in multi-line findAndReplaceInDoc batch:', error.message);
          // A 400 on a locked write is the lock failing: these requests were
          // compiled from the same read the lock came from, so with that
          // revision pinned the indices are valid by construction. Google
          // documents the 400 for a requiredRevisionId mismatch but not the
          // message it carries, so keying on the wording would stop firing
          // without any test noticing if Google reworded it. Whether a lock
          // was sent is knowable here and cannot drift. The API's own message
          // is passed through, so a 400 from some other cause stays
          // diagnosable rather than being replaced by this explanation.
          if (lock !== undefined && (error.status ?? error.code) === 400) {
            throw new Error(
              'The document changed between the read and the write (revision mismatch). ' +
                `Re-read the document and retry. (Google Docs API: ${error.message})`,
            );
          }
          throw error;
        }
        return {
          content: [{
            type: 'text',
            text: `Replaced ${ranges.length} occurrence(s) of "${a.findText}" with a multi-line replacement (compiled to ${requests.length} delete+insert requests in one atomic batch; \\n rendered as real paragraph breaks).`,
          }],
          isError: false,
        };
      }

      // expectedCount is a pre-write guard: count exactly first, refuse to
      // write on a mismatch (substring collisions become a refused call, not a
      // revert). The count walks the same text surface replaceAllText targets.
      if (a.expectedCount !== undefined) {
        const doc = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });
        const text = collectDocPlainText(doc.data, a.tabId);
        const preCount = countOccurrences(text, a.findText, a.matchCase);
        if (preCount !== a.expectedCount) {
          let message = `Aborted without writing: expectedCount=${a.expectedCount} but found ${preCount} occurrence(s) of "${a.findText}"${a.tabId ? ` in tab ${a.tabId}` : ''}.`;
          if (preCount === 0) {
            const hint = diagnoseZeroMatch(text, a.findText, a.matchCase);
            if (hint) message += `\nLikely cause: ${hint}.`;
          } else if (preCount > a.expectedCount) {
            message += ' The extra matches may be substring collisions elsewhere in the document — use a longer, unique findText.';
          }
          return { content: [{ type: 'text', text: message }], isError: true };
        }
      }

      const replaceAllText: {
        containsText: { text: string; matchCase: boolean };
        replaceText: string;
        tabsCriteria?: { tabIds: string[] };
      } = {
        containsText: { text: a.findText, matchCase: a.matchCase },
        replaceText: a.replaceText,
      };
      if (a.tabId) replaceAllText.tabsCriteria = { tabIds: [a.tabId] };

      const response = await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: {
          requests: [{ replaceAllText }],
          ...writeControlFor(a.ifRevisionId),
        },
      });

      const occurrences = response.data.replies?.[0]?.replaceAllText?.occurrencesChanged ?? 0;
      let message = `Replaced ${occurrences} occurrence(s) of "${a.findText}"${a.tabId ? ` in tab ${a.tabId}` : ''}.`;
      if (occurrences === 0) {
        // A zero-match silent no-op has repeatedly hidden lookalike-character
        // problems; spend one read explaining why nothing matched.
        try {
          const doc = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });
          const hint = diagnoseZeroMatch(collectDocPlainText(doc.data, a.tabId), a.findText, a.matchCase);
          if (hint) message += `\nLikely cause: ${hint}.`;
        } catch {
          // diagnosis is best-effort; the zero count already stands on its own
        }
      } else if (a.expectedCount !== undefined && occurrences !== a.expectedCount) {
        message += `\nWARNING: ${a.expectedCount} matches were verified immediately before writing but ${occurrences} were changed — the document was edited concurrently. Re-read before further edits.`;
      }
      return {
        content: [{ type: 'text', text: message }],
        isError: false,
      };
    }

    // =========================================================================
    // COMMENT TOOLS (use Drive API v3)
    // =========================================================================

    case "listComments": {
      const validation = ListCommentsSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const response = await ctx.getDrive().comments.list({
        fileId: a.documentId,
        fields: 'comments(id,content,quotedFileContent,author,createdTime,resolved,replies(id,content,author,createdTime)),nextPageToken',
        pageSize: a.pageSize || 100,
        pageToken: a.pageToken,
        includeDeleted: a.includeDeleted || false,
      });

      const comments = response.data.comments || [];
      const nextPageToken = response.data.nextPageToken;

      if (comments.length === 0) {
        return {
          content: [{ type: "text", text: "No comments found in this document." }],
          isError: false
        };
      }

      // ── Comment context extraction (two-tiered) ──
      // Tier 1 (fast): Read doc via Docs API, find each comment's quotedFileContent
      //   in the body text. If there's exactly one match, extract surrounding context.
      // Tier 2 (fallback): For ambiguous or failed Tier 1 comments, export as DOCX
      //   and parse commentRangeStart/End XML markers. Match by (author, createdTime).
      const contextMap = new Map<string, CommentContext>();
      let flatText = '';
      let offsetMap: number[] = [];

      // ── Tier 1: Docs API text matching ──
      // Check MIME type upfront — only Google Docs support the Docs API
      let needsDocxFallback = false;
      let isGoogleDoc = false;
      try {
        const fileInfo = await ctx.getDrive().files.get({
          fileId: a.documentId,
          fields: 'mimeType',
          supportsAllDrives: true,
        });
        isGoogleDoc = fileInfo.data.mimeType === 'application/vnd.google-apps.document';
      } catch (err) {
        ctx.log('Failed to check file MIME type:', err);
      }

      if (isGoogleDoc) {
        try {
          const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
          const docResponse = await docs.documents.get({
            documentId: a.documentId,
            includeTabsContent: true,
          });

          const result = buildFlatTextFromDoc(docResponse.data);
          flatText = result.flatText;
          offsetMap = result.offsetMap;

          // Get surrounding context for a match at a flatText position
          function getContext(matchStart: number, matchLen: number): { before: string; after: string } {
            const matchText = flatText.substring(matchStart, matchStart + matchLen);
            const beforeStart = Math.max(0, matchStart - 120);
            let before = flatText.substring(beforeStart, matchStart).trim();
            if (beforeStart > 0) before = '...' + before;
            before = before + matchText;
            const afterEnd = Math.min(flatText.length, matchStart + matchLen + 120);
            let after = flatText.substring(matchStart + matchLen, afterEnd).trim();
            if (afterEnd < flatText.length) after = after + '...';
            after = matchText + after;
            return { before, after };
          }

          // For each comment, find all occurrences of its quotedFileContent in the doc
          const ambiguousComments: any[] = [];
          for (const comment of comments) {
            const quoted = comment.quotedFileContent?.value;
            if (!quoted) continue;

            const positions: number[] = [];
            let searchFrom = 0;
            while (true) {
              const idx = flatText.indexOf(quoted, searchFrom);
              if (idx === -1) break;
              positions.push(idx);
              searchFrom = idx + 1;
            }

            if (positions.length === 1) {
              const surrounding = getContext(positions[0], quoted.length);
              const entry: CommentContext = {
                contextBefore: surrounding.before,
                contextAfter: surrounding.after,
              };
              // Store Docs API character offsets with bounds check
              const endIdx = positions[0] + quoted.length - 1;
              if (endIdx < offsetMap.length) {
                entry.startIndex = offsetMap[positions[0]];
                entry.endIndex = offsetMap[endIdx] + 1;
              }
              if (comment.id) contextMap.set(comment.id, entry);
            } else if (positions.length > 1) {
              // Ambiguous — need DOCX fallback to disambiguate
              ambiguousComments.push(comment);
            }
            // positions.length === 0: quoted text not found (e.g., doc was edited since comment)
          }

          needsDocxFallback = ambiguousComments.length > 0;
        } catch (err) {
          ctx.log('Tier 1 context extraction failed:', err);
          needsDocxFallback = true;
        }
      }

      // ── Tier 2: DOCX export fallback for ambiguous comments ──
      if (needsDocxFallback) {
        const unresolved = comments.filter((c: any) => !contextMap.has(c.id) && !c.resolved);

        if (unresolved.length > 0) {
          try {
            const docxResponse = await ctx.getDrive().files.export({
              fileId: a.documentId,
              mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            }, { responseType: 'arraybuffer' });

            const docxResult = await resolveContextFromDocx(docxResponse.data as ArrayBuffer);
            if (docxResult) {
              matchDocxToDriveComments(comments, docxResult, contextMap, flatText, offsetMap);
            }
          } catch (err) {
            ctx.log('Tier 2 DOCX context extraction failed:', err);
          }
        }
      }

      // ── Format comments for display ──
      const formattedComments = comments.map((comment: any, index: number) => {
        const status = comment.resolved ? ' [RESOLVED]' : '';
        const author = comment.author?.displayName || 'Unknown';
        const date = comment.createdTime ? new Date(comment.createdTime).toLocaleDateString() : 'Unknown date';
        const quotedText = comment.quotedFileContent?.value;
        const commentCtx = contextMap.get(comment.id);

        let positionInfo = '';
        const indexStr = commentCtx?.startIndex != null
          ? ` [chars ${commentCtx.startIndex}-${commentCtx.endIndex}]` : '';

        if (quotedText) {
          const snippet = quotedText.length > 100 ? quotedText.substring(0, 100) + '...' : quotedText;
          positionInfo = `\n   Anchored to: "${snippet}"${indexStr}`;
        }
        if (commentCtx) {
          if (commentCtx.contextBefore) positionInfo += `\n   Context before: "${commentCtx.contextBefore}"`;
          if (commentCtx.contextAfter) positionInfo += `\n   Context after: "${commentCtx.contextAfter}"`;
        }

        let result = `${index + 1}. ${author} (${date})${status}${positionInfo}\n   Comment: ${comment.content}`;

        if (comment.replies && comment.replies.length > 0) {
          for (const reply of comment.replies) {
            const replyAuthor = reply.author?.displayName || 'Unknown';
            const replyDate = reply.createdTime ? new Date(reply.createdTime).toLocaleDateString() : 'Unknown date';
            const replyContent = reply.content || '(empty)';
            result += `\n   └─ ${replyAuthor} (${replyDate}): ${replyContent}`;
          }
        }

        result += `\n   Comment ID: ${comment.id}`;
        return result;
      }).join('\n\n');

      let text = `Found ${comments.length} comment${comments.length === 1 ? '' : 's'}:\n\n${formattedComments}`;
      if (nextPageToken) {
        text += `\n\nMore comments available. Use pageToken: "${nextPageToken}" to fetch the next page.`;
      }

      return {
        content: [{ type: "text", text }],
        isError: false
      };
    }

    case "getComment": {
      const validation = GetCommentSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const response = await ctx.getDrive().comments.get({
        fileId: a.documentId,
        commentId: a.commentId,
        fields: 'id,content,quotedFileContent,author,createdTime,resolved,replies(id,content,author,createdTime)'
      });

      const comment = response.data;
      const author = comment.author?.displayName || 'Unknown';
      const date = comment.createdTime ? new Date(comment.createdTime).toLocaleDateString() : 'Unknown date';
      const status = comment.resolved ? ' [RESOLVED]' : '';
      const quotedText = comment.quotedFileContent?.value || 'No quoted text';
      const anchor = quotedText !== 'No quoted text' ? `\nAnchored to: "${quotedText}"` : '';

      let result = `${author} (${date})${status}${anchor}\n${comment.content}`;

      if (comment.replies && comment.replies.length > 0) {
        result += '\n\nReplies:';
        comment.replies.forEach((reply: any, index: number) => {
          const replyAuthor = reply.author?.displayName || 'Unknown';
          const replyDate = reply.createdTime ? new Date(reply.createdTime).toLocaleDateString() : 'Unknown date';
          result += `\n${index + 1}. ${replyAuthor} (${replyDate})\n   ${reply.content}`;
        });
      }

      return {
        content: [{ type: "text", text: result }],
        isError: false
      };
    }

    case "addComment": {
      const validation = AddCommentSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      // Resolve the targeted body ONCE. Index spaces are per-tab, so the range
      // lookup and the quote extraction must read the same content -- reading
      // different bodies would quote text from the wrong tab.
      let content: any[];
      if (a.tabId) {
        const resolved = await getTabBodyContent(ctx, a.documentId, a.tabId);
        if (resolved.error) {
          return errorResponse(resolved.error);
        }
        content = resolved.content!;
      } else {
        const doc = await docs.documents.get({ documentId: a.documentId });
        content = doc.data.body?.content || [];
      }

      let startIndex: number;
      let endIndex: number;
      if (a.textToFind) {
        const range = await findTextRange(
          ctx, a.documentId, a.textToFind, a.matchInstance, a.tabId, content);
        if (range && 'error' in range) {
          return errorResponse(range.error);
        }
        if (!range) {
          return errorResponse(`textToFind "${a.textToFind}" not found in document.`);
        }
        startIndex = range.startIndex;
        endIndex = range.endIndex;
      } else {
        startIndex = a.startIndex!;
        endIndex = a.endIndex!;
        if (endIndex <= startIndex) {
          return errorResponse("endIndex must be greater than startIndex");
        }
      }

      // buildFlatTextFromDoc walks paragraphs and nested tables. The previous
      // scan looked only at top-level body paragraphs, so any range inside a
      // table quoted '' and left the comment with no recoverable position.
      const { flatText, offsetMap } = buildFlatTextFromDoc({ body: { content } });
      let quotedText = '';
      for (let i = 0; i < offsetMap.length; i++) {
        if (offsetMap[i] >= startIndex && offsetMap[i] < endIndex) {
          quotedText += flatText[i];
        }
      }
      if (!quotedText) {
        return errorResponse(
          `Range [${startIndex}-${endIndex}] covers no text in the document.`);
      }

      // Deliberately NO anchor. Google Docs cannot be given an anchored comment
      // through any public API: the Docs API has no comment surface at all, and
      // Drive's `anchor` field is ignored by the Workspace editors, which
      // "treat these comments as un-anchored comments" per
      // https://developers.google.com/workspace/drive/api/guides/manage-comments
      // Sending an anchor Docs cannot resolve is worse than sending none: the
      // editor renders the thread under "Original content was deleted", so the
      // comment reads as if the passage it refers to had been removed.
      // quotedFileContent is what ties the comment to its passage, and is what
      // listComments matches on to report character positions.
      const response = await ctx.getDrive().comments.create({
        fileId: a.documentId,
        fields: 'id,content,quotedFileContent,author,createdTime',
        requestBody: {
          content: a.commentText,
          quotedFileContent: {
            value: quotedText,
            mimeType: 'text/plain'
          }
        }
      });

      const snippet = quotedText.length > 60 ? `${quotedText.slice(0, 60)}...` : quotedText;
      return {
        content: [{ type: "text", text: `Comment added (id ${response.data.id}) quoting "${snippet}" at [${startIndex}-${endIndex}]. Google Docs shows API-created comments in the comment panel without a margin anchor.` }],
        isError: false
      };
    }

    case "replyToComment": {
      const validation = ReplyToCommentSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const response = await ctx.getDrive().replies.create({
        fileId: a.documentId,
        commentId: a.commentId,
        fields: 'id,content,author,createdTime',
        requestBody: {
          content: a.replyText,
          ...(a.resolve && { action: "resolve" })
        }
      });

      const resolveNote = a.resolve ? ' Comment thread resolved.' : '';
      return {
        content: [{ type: "text", text: `Reply added successfully. Reply ID: ${response.data.id}${resolveNote}` }],
        isError: false
      };
    }

    case "deleteComment": {
      const validation = DeleteCommentSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      await ctx.getDrive().comments.delete({
        fileId: a.documentId,
        commentId: a.commentId
      });

      return {
        content: [{ type: "text", text: `Comment ${a.commentId} has been deleted.` }],
        isError: false
      };
    }

    // =========================================================================
    // TABLE & MEDIA TOOLS
    // =========================================================================

    case "insertTable": {
      const validation = InsertTableSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      // tabId passed through unvalidated by design: this pure batchUpdate op
      // does no GET, so a bad tabId surfaces the raw Google API error rather
      // than the friendly listDocumentTabs hint the format/table tools give.
      // Validating here would add a GET to the happy path — not worth it.
      const request_body = {
        insertTable: {
          location: withTab({ index: a.index }, a.tabId),
          rows: a.rows,
          columns: a.columns
        }
      };

      await executeBatchUpdate(ctx, a.documentId, [request_body], a.ifRevisionId);

      return {
        content: [{ type: "text", text: `Successfully inserted ${a.rows}x${a.columns} table at index ${a.index}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
        isError: false
      };
    }

    case "editTableCell": {
      const validation = EditTableCellSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      // Get the document to find the table structure. For a tab-scoped edit we
      // must read that tab's body content (the default narrow fetch is
      // structurally tab-blind); otherwise keep the cheap default-body fetch.
      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      let docContent: any[] | undefined;
      if (a.tabId) {
        const resolved = await getTabBodyContent(ctx, a.documentId, a.tabId);
        if (resolved.error) {
          return errorResponse(resolved.error);
        }
        docContent = resolved.content;
      } else {
        const docRes = await docs.documents.get({
          documentId: a.documentId,
          fields: 'body(content)'
        });
        docContent = docRes.data.body?.content ?? undefined;
      }

      // Find the table at the specified start index
      let table: any = null;
      const findTable = (content: any[]) => {
        for (const elem of content) {
          if (elem.table && elem.startIndex === a.tableStartIndex) {
            table = elem.table;
            return;
          }
        }
      };

      if (docContent) {
        findTable(docContent);
      }

      if (!table) {
        return errorResponse(`No table found at index ${a.tableStartIndex}`);
      }

      // Get the cell
      const row = table.tableRows?.[a.rowIndex];
      if (!row) {
        return errorResponse(`Row ${a.rowIndex} not found in table`);
      }

      const cell = row.tableCells?.[a.columnIndex];
      if (!cell) {
        return errorResponse(`Column ${a.columnIndex} not found in row ${a.rowIndex}`);
      }

      // Get cell content range
      const cellStartIndex = cell.startIndex;
      const cellEndIndex = cell.endIndex;

      const requests: any[] = [];

      // If textContent is provided, delete existing content and insert new
      if (a.textContent !== undefined) {
        // Delete existing content (keeping the paragraph structure)
        const cellContentStart = cellStartIndex + 1; // Skip the cell start marker
        const cellContentEnd = cellEndIndex - 1; // Before cell end marker

        if (cellContentEnd > cellContentStart) {
          requests.push({
            deleteContentRange: {
              range: withTab({ startIndex: cellContentStart, endIndex: cellContentEnd }, a.tabId)
            }
          });
        }

        // Insert new text
        if (a.textContent.length > 0) {
          requests.push({
            insertText: {
              location: withTab({ index: cellContentStart }, a.tabId),
              text: a.textContent
            }
          });
        }
      }

      // Apply text styling if any style options provided
      if (a.bold !== undefined || a.italic !== undefined || a.fontSize !== undefined) {
        const textStyle: any = {};
        const fields: string[] = [];

        if (a.bold !== undefined) { textStyle.bold = a.bold; fields.push('bold'); }
        if (a.italic !== undefined) { textStyle.italic = a.italic; fields.push('italic'); }
        if (a.fontSize !== undefined) { textStyle.fontSize = { magnitude: a.fontSize, unit: 'PT' }; fields.push('fontSize'); }

        if (fields.length > 0) {
          // Apply to the cell content range
          const styleStart = cellStartIndex + 1;
          const styleEnd = a.textContent !== undefined
            ? styleStart + a.textContent.length
            : cellEndIndex - 1;

          requests.push({
            updateTextStyle: {
              range: withTab({ startIndex: styleStart, endIndex: styleEnd }, a.tabId),
              textStyle,
              fields: fields.join(',')
            }
          });
        }
      }

      // Apply paragraph alignment if provided
      if (a.alignment !== undefined) {
        requests.push({
          updateParagraphStyle: {
            range: withTab({ startIndex: cellStartIndex + 1, endIndex: cellEndIndex - 1 }, a.tabId),
            paragraphStyle: { alignment: a.alignment },
            fields: 'alignment'
          }
        });
      }

      if (requests.length === 0) {
        return errorResponse("No changes specified for the table cell");
      }

      await executeBatchUpdate(ctx, a.documentId, requests, a.ifRevisionId);

      return {
        content: [{ type: "text", text: `Successfully edited cell at row ${a.rowIndex}, column ${a.columnIndex}${a.tabId ? ` in tab ${a.tabId}` : ''}` }],
        isError: false
      };
    }

    case "styleDocTable": {
      const validation = StyleDocTableSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      let built: { requests: any[]; applied: string[] };
      try {
        built = buildStyleDocTableRequests(a);
      } catch (err) {
        return errorResponse(err instanceof Error ? err.message : String(err));
      }
      if (built.requests.length === 0) {
        return errorResponse("No table style options provided");
      }

      // Same path as insertTable/editTableCell: executeBatchUpdate threads the
      // lock and maps 404/403 to the server's wording instead of leaking the
      // raw gaxios error.
      await executeBatchUpdate(ctx, a.documentId, built.requests, a.ifRevisionId);

      return {
        content: [{ type: "text", text: `Styled table at index ${a.tableStartIndex}${a.tabId ? ` in tab ${a.tabId}` : ''}: ${built.applied.join('; ')}` }],
        isError: false
      };
    }

    case "insertImageFromUrl": {
      const validation = InsertImageFromUrlSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      await insertInlineImageHelper(ctx, a.documentId, a.imageUrl, a.index, a.width, a.height);

      return {
        content: [{ type: "text", text: `Successfully inserted image from URL at index ${a.index}` }],
        isError: false
      };
    }

    case "insertLocalImage": {
      const validation = InsertLocalImageSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      // Get the document's parent folder if uploadToSameFolder is true
      let parentFolderId: string | undefined;
      if (a.uploadToSameFolder !== false) {
        const fileInfo = await ctx.getDrive().files.get({
          fileId: a.documentId,
          fields: 'parents',
          supportsAllDrives: true
        });
        parentFolderId = fileInfo.data.parents?.[0];
      }

      // Upload the image to Drive
      const { webContentLink: imageUrl } = await uploadImageToDrive(ctx, a.localImagePath, {
        parentFolderId,
        makePublic: a.makePublic,
      });

      // Insert the image into the document
      await insertInlineImageHelper(ctx, a.documentId, imageUrl, a.index, a.width, a.height);

      return {
        content: [{ type: "text", text: `Successfully uploaded and inserted local image at index ${a.index}\nImage URL: ${imageUrl}` }],
        isError: false
      };
    }

    // =========================================================================
    // GOOGLE DOCS DISCOVERY & MANAGEMENT TOOLS
    // =========================================================================

    case "listGoogleDocs": {
      const validation = ListGoogleDocsSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      // Build the query string for Google Drive API
      let queryString = "mimeType='application/vnd.google-apps.document' and trashed=false";
      if (a.query) {
        const escapedQuery = escapeDriveQuery(a.query);
        queryString += ` and (name contains '${escapedQuery}' or fullText contains '${escapedQuery}')`;
      }

      const response = await ctx.getDrive().files.list({
        q: queryString,
        pageSize: a.maxResults,
        orderBy: a.orderBy,
        fields: 'files(id,name,modifiedTime,createdTime,size,webViewLink,owners(displayName,emailAddress))',
        ...ALL_DRIVES_LIST_PARAMS
      });

      const files = response.data.files || [];

      if (files.length === 0) {
        return { content: [{ type: "text", text: "No Google Docs found matching your criteria." }], isError: false };
      }

      let result = `Found ${files.length} Google Document(s) (ordered by ${a.orderBy}):\n\n`;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const modifiedDate = file.modifiedTime ? new Date(file.modifiedTime).toLocaleDateString() : 'Unknown';
        const owner = file.owners?.[0]?.displayName || 'Unknown';
        result += `${i + 1}. **${file.name}**\n`;
        result += `   ID: ${file.id}\n`;
        result += `   Modified: ${modifiedDate}\n`;
        result += `   Owner: ${owner}\n`;
        result += `   Link: ${file.webViewLink}\n\n`;
      }

      return { content: [{ type: "text", text: result }], isError: false };
    }

    case "getDocumentInfo": {
      const validation = GetDocumentInfoSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const response = await ctx.getDrive().files.get({
        fileId: a.documentId,
        fields: 'id,name,description,mimeType,size,createdTime,modifiedTime,webViewLink,owners(displayName,emailAddress),lastModifyingUser(displayName,emailAddress),shared,parents,version',
        supportsAllDrives: true,
      });

      const file = response.data;

      if (!file) {
        return errorResponse(`Document with ID ${a.documentId} not found.`);
      }

      // The Drive `version` above is not a Docs revisionId and WriteControl
      // rejects it, so fetch the real one. Narrow projection, and non-fatal:
      // a caller without edit access (or a Drive file that is not a Doc) still
      // gets the metadata rather than an error from the lock lookup.
      let docsRevisionId = REVISION_UNAVAILABLE;
      try {
        const docMeta = await ctx.google.docs({ version: 'v1', auth: ctx.authClient })
          .documents.get({ documentId: a.documentId, fields: 'revisionId' });
        docsRevisionId = revisionIdOf(docMeta.data);
      } catch {
        // Leave it unavailable; the metadata below is still worth returning.
      }

      const createdDate = file.createdTime ? new Date(file.createdTime).toLocaleString() : 'Unknown';
      const modifiedDate = file.modifiedTime ? new Date(file.modifiedTime).toLocaleString() : 'Unknown';
      const owner = file.owners?.[0];
      const lastModifier = (file as any).lastModifyingUser;

      let result = `**Document Information:**\n\n`;
      result += `**Name:** ${file.name}\n`;
      result += `**ID:** ${file.id}\n`;
      result += `**Type:** Google Document\n`;
      result += `**Created:** ${createdDate}\n`;
      result += `**Last Modified:** ${modifiedDate}\n`;

      if (owner) {
        result += `**Owner:** ${owner.displayName} (${owner.emailAddress})\n`;
      }

      if (lastModifier) {
        result += `**Last Modified By:** ${lastModifier.displayName} (${lastModifier.emailAddress})\n`;
      }

      if (file.description) {
        result += `**Description:** ${file.description}\n`;
      }

      result += `**Shared:** ${file.shared ? 'Yes' : 'No'}\n`;
      result += `**Drive Version:** ${file.version || 'Unknown'} (Drive's change counter — not a lock)\n`;
      result += `**Docs revisionId:** ${docsRevisionId} (pass as ifRevisionId to lock a write)\n`;
      result += `**View Link:** ${file.webViewLink}\n`;

      return { content: [{ type: "text", text: result }], isError: false };
    }

    case "getGoogleDocStyleSummary": {
      const validation = GetGoogleDocStyleSummarySchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const doc = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });

      const resolved = resolveBodySegments(doc.data, a.tabId);
      if ('error' in resolved) return errorResponse(resolved.error);

      const summary = summarizeDocumentStyles(resolved.segments);
      return {
        content: [{
          type: 'text',
          text: `Style summary for "${doc.data.title}"${a.tabId ? ` (tab ${a.tabId})` : ''}\n`
            + `revisionId: ${revisionIdOf(doc.data)}\n`
            + summary
        }],
        isError: false,
      };
    }

    case "describeGoogleDocRange": {
      const validation = DescribeGoogleDocRangeSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const doc = await docs.documents.get({ documentId: a.documentId, includeTabsContent: true });

      // Default to the first tab: every other index-taking Docs tool does, and
      // describing all of them at once would report overlapping ranges.
      const resolved = resolveBodySegments(doc.data, a.tabId);
      if ('error' in resolved) return errorResponse(resolved.error);
      const content = resolved.segments[0]?.content;

      let startIndex: number;
      let endIndex: number;
      if (a.textToFind) {
        // Resolve against the body we already hold, so the indices and the
        // revisionId below come from one fetch rather than two.
        const range = await findTextRange(ctx, a.documentId, a.textToFind, a.matchInstance, a.tabId, content ?? []);
        if (range && 'error' in range) return errorResponse(range.error);
        if (!range) return errorResponse(`textToFind "${a.textToFind}" not found in document.`);
        startIndex = range.startIndex;
        endIndex = range.endIndex;
      } else {
        startIndex = a.startIndex!;
        endIndex = a.endIndex ?? startIndex + 1;
      }

      const described = describeRangeStyles(content, startIndex, endIndex);
      return {
        content: [{
          type: 'text',
          text: `revisionId: ${revisionIdOf(doc.data)}\n`
            + `target range: [${startIndex}-${endIndex})${a.tabId ? ` in tab ${a.tabId}` : ''}\n`
            + described
        }],
        isError: false,
      };
    }

    case "addDocumentTab": {
      const validation = AddDocumentTabSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      await docs.documents.batchUpdate({
        documentId: a.documentId,
        // addDocumentTab is not yet in the googleapis TypeScript types — cast required
        requestBody: { requests: [{ addDocumentTab: { tabProperties: { title: a.title } } } as any] }
      });

      return { content: [{ type: 'text', text: `Requested creation of tab "${a.title}" in document ${a.documentId}.` }], isError: false };
    }

    case "renameDocumentTab": {
      const validation = RenameDocumentTabSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      await docs.documents.batchUpdate({
        documentId: a.documentId,
        // updateDocumentTabProperties is not yet in the googleapis TypeScript types — cast required.
        // Per Google Docs API spec: tabId lives INSIDE tabProperties (it's the tab identifier),
        // and `fields` is a FieldMask for which properties to update (excludes tabId).
        requestBody: { requests: [{ updateDocumentTabProperties: { tabProperties: { tabId: a.tabId, title: a.title }, fields: 'title' } } as any] }
      });

      return { content: [{ type: 'text', text: `Renamed tab ${a.tabId} to "${a.title}".` }], isError: false };
    }

    case "insertSmartChip": {
      const validation = InsertSmartChipSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      // tabId passed through unvalidated by design (see insertTable): no GET on
      // this pure batchUpdate path, so a bad tabId yields the raw Google error.
      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: {
          requests: [{
            insertPerson: {
              personProperties: { email: a.personEmail },
              location: withTab({ index: a.index }, a.tabId),
            },
          // insertPerson is not yet in the googleapis TypeScript types — cast required
          } as any],
          ...writeControlFor(a.ifRevisionId),
        },
      });

      return { content: [{ type: 'text', text: `Inserted person smart chip for ${a.personEmail} at index ${a.index}${a.tabId ? ` in tab ${a.tabId}` : ''}.` }], isError: false };
    }

    case "readSmartChips": {
      const validation = ReadSmartChipsSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });
      const doc = await docs.documents.get({ documentId: a.documentId });
      const body = (doc.data as any).body?.content || [];
      const hits: string[] = [];
      for (const block of body) {
        for (const el of block?.paragraph?.elements || []) {
          if (el?.richLink) hits.push(`richLink: ${el.richLink.richLinkProperties?.uri || 'unknown'}`);
          if (el?.person) hits.push(`person: ${el.person.personProperties?.email || 'unknown'}`);
        }
      }
      // Date chips appear as richLink elements in the Docs API model — already covered above.
      return { content: [{ type: 'text', text: hits.length ? hits.join('\n') : 'No smart chips detected (note: only the default tab is scanned).' }], isError: false };
    }

    case "createFootnote": {
      const validation = CreateFootnoteSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const docs = ctx.google.docs({ version: 'v1', auth: ctx.authClient });

      // Build the createFootnote request. tabId passed through unvalidated by
      // design (see insertTable): no GET on this pure batchUpdate path, so a
      // bad tabId yields the raw Google error.
      const createFootnoteReq: {
        location?: { index: number; tabId?: string };
        endOfSegmentLocation?: { segmentId: string; tabId?: string };
      } = {};
      if (a.index !== undefined) {
        createFootnoteReq.location = withTab({ index: a.index }, a.tabId);
      } else {
        createFootnoteReq.endOfSegmentLocation = withTab({ segmentId: "" }, a.tabId);
      }

      const res = await docs.documents.batchUpdate({
        documentId: a.documentId,
        requestBody: {
          requests: [{ createFootnote: createFootnoteReq }],
          // Lock only this creating call: the optional content insert below
          // targets the new footnote segment, whose revision necessarily
          // postdates the caller's read.
          ...writeControlFor(a.ifRevisionId),
        },
      });

      const footnoteId = res.data.replies?.[0]?.createFootnote?.footnoteId;
      if (!footnoteId) {
        return errorResponse("Failed to create footnote — no footnoteId in response.");
      }

      const locationDesc = `${a.index !== undefined ? `at index ${a.index}` : 'at end of document'}${a.tabId ? ` in tab ${a.tabId}` : ''}`;

      // Optionally insert text content into the footnote body
      if (a.content) {
        try {
          await docs.documents.batchUpdate({
            documentId: a.documentId,
            requestBody: {
              requests: [{
                insertText: {
                  location: withTab({ segmentId: footnoteId, index: 0 }, a.tabId),
                  text: a.content,
                },
              }],
            },
          });
        } catch (err: any) {
          return { content: [{ type: 'text', text: `Created footnote ${footnoteId} ${locationDesc}, but failed to insert content: ${err.message}` }], isError: true };
        }
      }

      return { content: [{ type: 'text', text: `Created footnote ${footnoteId} ${locationDesc}.${a.content ? ' Content inserted.' : ''}` }], isError: false };
    }

    default:
      return null;
  }
}
