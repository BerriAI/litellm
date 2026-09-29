// Exact occurrence counting and zero-match diagnosis for find/replace tools.
//
// `findAndReplaceInDoc` matches literally, so lookalike text produces a silent
// "Replaced 0 occurrence(s)" that reads like the text isn't in the document at
// all. The common causes are non-breaking spaces (how Docs letterspaces text),
// curly versus straight quotes, and literal `&amp;` entities. On zero matches
// the diagnosis names the likeliest cause, with the codepoints spelled out.

import type { docs_v1 } from 'googleapis';

/** A document or one tab of one — both carry the same body/header/footer/footnote shape. */
type DocLike = docs_v1.Schema$Document | docs_v1.Schema$DocumentTab;

function walkContent(
  content: docs_v1.Schema$StructuralElement[] | undefined,
  out: string[],
): void {
  for (const el of content ?? []) {
    if (el.paragraph?.elements) {
      let para = '';
      for (const pe of el.paragraph.elements) {
        if (pe.textRun?.content) para += pe.textRun.content;
      }
      out.push(para);
    } else if (el.table?.tableRows) {
      for (const row of el.table.tableRows) {
        for (const cell of row.tableCells ?? []) walkContent(cell.content, out);
      }
    } else if (el.tableOfContents?.content) {
      walkContent(el.tableOfContents.content, out);
    }
  }
}

function collectFromDocumentTab(tabLike: DocLike | undefined): string[] {
  const out: string[] = [];
  walkContent(tabLike?.body?.content, out);
  for (const segmentMap of [tabLike?.headers, tabLike?.footers, tabLike?.footnotes]) {
    if (!segmentMap) continue;
    for (const key of Object.keys(segmentMap)) walkContent(segmentMap[key]?.content, out);
  }
  return out;
}

/**
 * Full plain text of a document — body, tables (recursive), headers, footers,
 * footnotes — matching the surface `replaceAllText` actually operates on.
 * Multi-tab documents (documents.get with includeTabsContent) are walked
 * through every tab, or just `tabId` when given.
 */
export function collectDocPlainText(docData: docs_v1.Schema$Document, tabId?: string): string {
  if (Array.isArray(docData?.tabs) && docData.tabs.length > 0) {
    const texts: string[] = [];
    const visit = (tabs: docs_v1.Schema$Tab[]): void => {
      for (const tab of tabs) {
        if (!tabId || tab.tabProperties?.tabId === tabId) {
          texts.push(...collectFromDocumentTab(tab.documentTab ?? {}));
        }
        if (Array.isArray(tab.childTabs)) visit(tab.childTabs);
      }
    };
    visit(docData.tabs);
    return texts.join('\n');
  }
  return collectFromDocumentTab(docData).join('\n');
}

/** Literal (non-regex) substring count, honoring matchCase. */
export function countOccurrences(haystack: string, needle: string, matchCase: boolean): number {
  if (!needle) return 0;
  const h = matchCase ? haystack : haystack.toLowerCase();
  const n = matchCase ? needle : needle.toLowerCase();
  let count = 0;
  let i = 0;
  while ((i = h.indexOf(n, i)) !== -1) {
    count++;
    i += n.length;
  }
  return count;
}

export interface OccurrenceRange {
  startIndex: number;
  endIndex: number;
  segmentId?: string; // set for headers/footers/footnotes
  tabId?: string; // set for multi-tab documents
}

interface ParagraphRun {
  text: string;
  chars: number[]; // doc index of each character in `text`
  segmentId?: string;
  tabId?: string;
}

/**
 * The id of a header/footer/footnote segment. The three schemas are a union
 * with no common id field, so narrow once here rather than at the call site.
 */
function segmentIdOf(
  seg: docs_v1.Schema$Header | docs_v1.Schema$Footer | docs_v1.Schema$Footnote | undefined,
  fallback: string,
): string {
  const ids = seg as
    | { headerId?: string | null; footerId?: string | null; footnoteId?: string | null }
    | undefined;
  return ids?.headerId ?? ids?.footerId ?? ids?.footnoteId ?? fallback;
}

function collectParagraphRuns(docData: docs_v1.Schema$Document, tabId?: string): ParagraphRun[] {
  const out: ParagraphRun[] = [];

  function walk(
    content: docs_v1.Schema$StructuralElement[] | undefined,
    segmentId: string | undefined,
    tab: string | undefined,
  ): void {
    for (const el of content ?? []) {
      if (el.paragraph?.elements) {
        let text = '';
        const chars: number[] = [];
        for (const pe of el.paragraph.elements) {
          if (pe.textRun?.content && pe.startIndex != null) {
            const run: string = pe.textRun.content;
            for (let i = 0; i < run.length; i++) {
              text += run[i];
              chars.push(pe.startIndex + i);
            }
          }
        }
        if (text) out.push({ text, chars, segmentId, tabId: tab });
      } else if (el.table?.tableRows) {
        for (const row of el.table.tableRows) {
          for (const cell of row.tableCells ?? []) walk(cell.content, segmentId, tab);
        }
      } else if (el.tableOfContents?.content) {
        walk(el.tableOfContents.content, segmentId, tab);
      }
    }
  }

  function walkTabLike(tabLike: DocLike | undefined, tab: string | undefined): void {
    walk(tabLike?.body?.content, undefined, tab);
    for (const segmentMap of [tabLike?.headers, tabLike?.footers, tabLike?.footnotes]) {
      if (!segmentMap) continue;
      for (const key of Object.keys(segmentMap)) {
        const seg = segmentMap[key];
        walk(seg?.content, segmentIdOf(seg, key), tab);
      }
    }
  }

  if (Array.isArray(docData?.tabs) && docData.tabs.length > 0) {
    const visit = (tabs: docs_v1.Schema$Tab[]): void => {
      for (const tab of tabs) {
        const id = tab.tabProperties?.tabId as string | undefined;
        if (!tabId || id === tabId) walkTabLike(tab.documentTab ?? {}, id);
        if (Array.isArray(tab.childTabs)) visit(tab.childTabs);
      }
    };
    visit(docData.tabs);
  } else {
    walkTabLike(docData, undefined);
  }
  return out;
}

/**
 * Lowercase `text` one code point at a time, recording for every UTF-16 unit of
 * the result which source unit it came from.
 *
 * Lowercasing is not length-preserving: U+0130 (LATIN CAPITAL LETTER I WITH DOT
 * ABOVE) lowercases to two UTF-16 units, so `'Istanbul'.toLowerCase().length`
 * is 9 for the U+0130 spelling. A match position in a naively lowercased
 * haystack is therefore not a valid offset into the original text, and using it
 * as one shifts every doc index after the first expanding character. Per-code-
 * point folding also keeps the haystack and the needle consistent, since
 * whole-string lowercasing is context-sensitive (a trailing Greek sigma folds
 * to U+03C2 in a string but U+03C3 on its own).
 */
function foldWithSourceMap(text: string, matchCase: boolean): {
  hay: string;
  srcStart: number[];
  srcEnd: number[];
} {
  if (matchCase) {
    const srcStart = Array.from(text, (_, i) => i);
    return { hay: text, srcStart, srcEnd: srcStart.map(i => i + 1) };
  }
  let hay = '';
  const srcStart: number[] = [];
  const srcEnd: number[] = [];
  for (let i = 0; i < text.length; ) {
    const source = String.fromCodePoint(text.codePointAt(i) as number);
    const folded = source.toLowerCase();
    for (let k = 0; k < folded.length; k++) {
      hay += folded[k];
      srcStart.push(i);
      srcEnd.push(i + source.length);
    }
    i += source.length;
  }
  return { hay, srcStart, srcEnd };
}

/**
 * Exact doc-index ranges of every occurrence of `findText` (which must not
 * contain a newline — matches never span paragraphs here). Walks the same
 * surface as replaceAllText: body, tables, headers, footers, footnotes, tabs.
 */
export function findOccurrenceRanges(
  docData: docs_v1.Schema$Document,
  findText: string,
  matchCase: boolean,
  tabId?: string,
): OccurrenceRange[] {
  const ranges: OccurrenceRange[] = [];
  if (!findText || findText.includes('\n')) return ranges;
  const needle = foldWithSourceMap(findText, matchCase).hay;
  for (const para of collectParagraphRuns(docData, tabId)) {
    const { hay, srcStart, srcEnd } = foldWithSourceMap(para.text, matchCase);
    let i = 0;
    while ((i = hay.indexOf(needle, i)) !== -1) {
      // Map the match back through the fold before touching `para.chars`,
      // which is indexed by SOURCE offset.
      const from = srcStart[i];
      const toExclusive = srcEnd[i + needle.length - 1];
      ranges.push({
        startIndex: para.chars[from],
        endIndex: para.chars[toExclusive - 1] + 1,
        ...(para.segmentId ? { segmentId: para.segmentId } : {}),
        ...(para.tabId ? { tabId: para.tabId } : {}),
      });
      i += needle.length;
    }
  }
  return ranges;
}

/**
 * Compile a multi-line replacement into deleteContentRange + insertText pairs,
 * ordered descending by index so no request shifts a later target (the Docs
 * API's own documented best practice). insertText renders `\n` as real
 * paragraph breaks — which is the whole point: replaceAllText mangles them.
 */
export function buildMultilineReplaceRequests(ranges: OccurrenceRange[], replaceText: string): unknown[] {
  const sorted = [...ranges].sort((a, b) => b.startIndex - a.startIndex);
  const requests: unknown[] = [];
  for (const r of sorted) {
    const loc: Record<string, unknown> = { index: r.startIndex };
    const range: Record<string, unknown> = { startIndex: r.startIndex, endIndex: r.endIndex };
    if (r.segmentId) {
      loc.segmentId = r.segmentId;
      range.segmentId = r.segmentId;
    }
    if (r.tabId) {
      loc.tabId = r.tabId;
      range.tabId = r.tabId;
    }
    requests.push({ deleteContentRange: { range } });
    if (replaceText.length > 0) {
      requests.push({ insertText: { location: loc, text: replaceText } });
    }
  }
  return requests;
}

const NBSP_RE = / /g;
const CURLY_SINGLE_RE = /[‘’]/g;
const CURLY_DOUBLE_RE = /[“”]/g;

function normalizeQuotesAndSpaces(s: string): string {
  return s.replace(NBSP_RE, ' ').replace(CURLY_SINGLE_RE, "'").replace(CURLY_DOUBLE_RE, '"');
}

/**
 * When a find produced zero matches, explain the likeliest lookalike cause.
 * Returns null when no transformation produces a match.
 */
export function diagnoseZeroMatch(docText: string, findText: string, matchCase: boolean): string | null {
  const hints: string[] = [];

  if (countOccurrences(docText.replace(NBSP_RE, ' '), findText, matchCase) > 0) {
    hints.push(
      'the text matches once non-breaking spaces (U+00A0 — how Docs letterspaces text) are treated as plain spaces; copy the exact characters from a read-back or target the run by index',
    );
  } else if (countOccurrences(normalizeQuotesAndSpaces(docText), normalizeQuotesAndSpaces(findText), matchCase) > 0) {
    hints.push(
      'a match exists once curly quotes/apostrophes (U+2018/U+2019/U+201C/U+201D) are normalized — the document and findText use different quote characters; read the text back and copy them exactly',
    );
  }

  if (findText.includes('&') && countOccurrences(docText, findText.replace(/&/g, '&amp;'), matchCase) > 0) {
    hints.push(
      "the document contains the literal HTML entity '&amp;' where findText has '&' (an earlier write sent an escaped entity); search for 'amp;' to clean it up",
    );
  }

  if (matchCase && countOccurrences(docText, findText, false) > 0) {
    hints.push('a case-insensitive match exists — check capitalization or set matchCase: false');
  }

  return hints.length ? hints.join('; ') : null;
}
