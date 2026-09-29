// Compact paragraph-style annotation for the indexed reads: the answer to
// "what styling is on this paragraph?" without a full-document JSON read.

/**
 * Docs `OptionalColor` (`{ color: { rgbColor } }`) to `#rrggbb`, or null when
 * the color carries no `rgbColor`. Borders, shading, and text runs all use this
 * one shape, so rounding stays consistent across every surface that prints a
 * color. Lives here rather than in `docs.ts` only because `docs.ts` imports
 * this module; keeping the helper on this side avoids an import cycle.
 */
export function rgbColorToHex(color: any): string | null {
  if (!color?.color?.rgbColor) return null;
  const rgb = color.color.rgbColor;
  const r = Math.round((rgb.red || 0) * 255);
  const g = Math.round((rgb.green || 0) * 255);
  const b = Math.round((rgb.blue || 0) * 255);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

const BORDER_KEYS = ['borderTop', 'borderBottom', 'borderLeft', 'borderRight', 'borderBetween'] as const;

// Docs carries empty border objects (width 0/unset) on many paragraphs; only a
// positive width renders. Counting the empties made a 9-rule doc report 97.
function borderVisible(b: any): boolean {
  return !!b && (b.width?.magnitude ?? 0) > 0;
}

/**
 * Compact non-default paragraph meta bits for formatted reads: named style,
 * alignment, visible borders, shading. Empty for a plain NORMAL_TEXT paragraph
 * so unformatted content stays annotation-free.
 */
export function paragraphMetaBits(paragraph: any): string[] {
  const style = paragraph?.paragraphStyle ?? {};
  const bits: string[] = [];
  if (style.namedStyleType && style.namedStyleType !== 'NORMAL_TEXT') bits.push(style.namedStyleType);
  if (style.alignment && style.alignment !== 'START') bits.push(style.alignment.toLowerCase());
  for (const k of BORDER_KEYS) {
    if (borderVisible(style[k])) {
      const b = style[k];
      bits.push(`${k}(${rgbColorToHex(b.color) ?? 'auto'} ${b.width?.magnitude ?? '?'}${b.width?.unit ?? ''} ${b.dashStyle ?? ''})`.replace(/\s+\)/, ')'));
    }
  }
  // Guard on the resolved hex, not the object: a backgroundColor carrying no
  // rgbColor resolves to null and printed the literal `shading(null)`.
  const shade = rgbColorToHex(style.shading?.backgroundColor);
  if (shade) bits.push(`shading(${shade})`);
  return bits;
}

// ---------------------------------------------------------------------------
// Style reads: getGoogleDocStyleSummary and describeGoogleDocRange
// ---------------------------------------------------------------------------
//
// Both take a body content array that the caller has already resolved to one
// index space, never a whole document. Headers, footers, and footnotes each
// start their own index space at 0, and index spaces restart in every tab, so
// a walk that merged them would report several paragraphs with overlapping
// ranges and no way to tell which one an index-taking write should target.
// Keeping the segment choice in docs.ts makes that leak impossible here.

export interface ParagraphVisit {
  paragraph: any;
  startIndex: number;
  endIndex: number;
  inTable: boolean;
}

function visitBodyParagraphs(
  content: any[] | undefined,
  onParagraph: (v: ParagraphVisit) => void,
  onTable?: (tableElement: any) => void,
): void {
  function walk(items: any[] | undefined, inTable: boolean): void {
    for (const el of items ?? []) {
      if (el.paragraph && el.startIndex != null && el.endIndex != null) {
        onParagraph({ paragraph: el.paragraph, startIndex: el.startIndex, endIndex: el.endIndex, inTable });
      } else if (el.table?.tableRows) {
        if (onTable) onTable(el);
        for (const row of el.table.tableRows) {
          for (const cell of row.tableCells ?? []) walk(cell.content, true);
        }
      } else if (el.tableOfContents?.content) {
        walk(el.tableOfContents.content, inTable);
      }
    }
  }
  walk(content, false);
}

// One truncation rule for every list in the summary. The old code appended `…`
// to one list, cut another silently, and stopped a third at 20 without saying
// so, which left a caller unable to tell a short document from a clipped list.
function renderList(items: string[], limit: number): string {
  if (items.length <= limit) return items.join(', ');
  return `${items.slice(0, limit).join(', ')}, … (+${items.length - limit} more)`;
}

const LIST_LIMIT = 12;
const OUTLINE_LIMIT = 20;

/** One body segment to summarize, already resolved to a single index space. */
export interface StyleSummarySegment {
  /** Tab title, set only when the document has more than one tab. */
  label?: string;
  content: any[] | undefined;
}

/**
 * Compact whole-document style inventory, a few hundred tokens instead of the
 * ~160k-character `documents.get` JSON that answering the same question costs.
 *
 * Counts and the font inventory aggregate across every segment, since "what
 * fonts does this document use?" is a whole-document question. Locations are
 * per-segment: a bare index is ambiguous once a document has more than one tab,
 * so each is prefixed with its tab title when a label is supplied.
 */
export function summarizeDocumentStyles(segments: StyleSummarySegment[]): string {
  const fonts = new Map<string, { sizes: Set<number>; chars: number }>();
  const namedStyles = new Map<string, number>();
  const fgColors = new Set<string>();
  const bgColors = new Set<string>();
  const borderedParas: string[] = [];
  const shadedParas: string[] = [];
  const headings: string[] = [];
  const tables: string[] = [];
  let paraCount = 0;

  for (const segment of segments) {
    const at = (index: number | string) => (segment.label ? `${segment.label}:${index}` : `${index}`);

    visitBodyParagraphs(
      segment.content,
      ({ paragraph, startIndex }) => {
        paraCount++;
        const ps = paragraph.paragraphStyle ?? {};
        const named = ps.namedStyleType ?? 'NORMAL_TEXT';
        namedStyles.set(named, (namedStyles.get(named) ?? 0) + 1);
        if (BORDER_KEYS.some((k) => borderVisible(ps[k]))) borderedParas.push(at(startIndex));
        // Guard on the resolved hex, not the object: Docs attaches shading
        // objects that carry no color, and those render nothing.
        if (rgbColorToHex(ps.shading?.backgroundColor)) shadedParas.push(at(startIndex));

        let text = '';
        for (const pe of paragraph.elements ?? []) {
          if (!pe.textRun) continue;
          const content = pe.textRun.content ?? '';
          text += content;
          const ts = pe.textRun.textStyle ?? {};
          const family = ts.weightedFontFamily?.fontFamily;
          if (family) {
            let f = fonts.get(family);
            if (!f) fonts.set(family, (f = { sizes: new Set(), chars: 0 }));
            if (ts.fontSize?.magnitude != null) f.sizes.add(ts.fontSize.magnitude);
            f.chars += content.length;
          }
          const fg = rgbColorToHex(ts.foregroundColor);
          const bg = rgbColorToHex(ts.backgroundColor);
          if (fg && fg !== '#000000') fgColors.add(fg);
          if (bg) bgColors.add(bg);
        }

        if (/^(HEADING_[1-6]|TITLE|SUBTITLE)$/.test(named)) {
          headings.push(`[${at(startIndex)}] ${named} "${text.trim().slice(0, 60)}"`);
        }
      },
      (tableElement) => {
        const rows = tableElement.table.tableRows?.length ?? 0;
        const cols = tableElement.table.tableRows?.[0]?.tableCells?.length ?? 0;
        tables.push(`[${at(`${tableElement.startIndex}-${tableElement.endIndex}`)}] ${rows}x${cols}`);
      },
    );
  }

  const fontLines = [...fonts.entries()]
    .sort((a, b) => b[1].chars - a[1].chars)
    .map(([family, f]) => `${family} (${[...f.sizes].sort((x, y) => x - y).join('/') || '?'}pt, ${f.chars} chars)`);
  const namedLine = [...namedStyles.entries()].map(([k, n]) => `${n}x ${k}`).join(', ');

  const lines = [
    `paragraphs: ${paraCount}${namedLine ? ` (${namedLine})` : ''}`,
    `fonts: ${fontLines.join('; ') || 'default only'}`,
    `text colors: ${[...fgColors].join(', ') || 'default'}${bgColors.size ? `; highlights: ${[...bgColors].join(', ')}` : ''}`,
    `bordered paragraphs: ${borderedParas.length}${borderedParas.length ? ` (at ${renderList(borderedParas, LIST_LIMIT)})` : ''}`,
    `shaded paragraphs: ${shadedParas.length}${shadedParas.length ? ` (at ${renderList(shadedParas, LIST_LIMIT)})` : ''}`,
    `tables: ${tables.length}${tables.length ? ` - ${renderList(tables, LIST_LIMIT)}` : ''}`,
  ];
  if (headings.length) {
    const shown = headings.slice(0, OUTLINE_LIMIT);
    const more = headings.length > OUTLINE_LIMIT ? `\n  … (+${headings.length - OUTLINE_LIMIT} more)` : '';
    lines.push(`outline:\n  ${shown.join('\n  ')}${more}`);
  }
  return lines.join('\n');
}

/**
 * Compact style description of everything overlapping [startIndex, endIndex)
 * within one already-resolved body segment.
 */
export function describeRangeStyles(content: any[] | undefined, startIndex: number, endIndex: number): string {
  const out: string[] = [];

  visitBodyParagraphs(content, ({ paragraph, startIndex: ps, endIndex: pe, inTable }) => {
    if (pe <= startIndex || ps >= endIndex) return;
    const style = paragraph.paragraphStyle ?? {};

    // Named style, alignment, borders, and shading render identically in the
    // formatted reads, so they come from the one helper those use. The probe
    // additionally always names the style: paragraphMetaBits omits NORMAL_TEXT
    // because a formatted read has nothing to report there, but "what is this
    // paragraph?" is exactly this tool's question.
    const bits = paragraphMetaBits(paragraph);
    if ((style.namedStyleType ?? 'NORMAL_TEXT') === 'NORMAL_TEXT') bits.unshift('NORMAL_TEXT');

    for (const k of ['indentFirstLine', 'indentStart', 'indentEnd', 'spaceAbove', 'spaceBelow'] as const) {
      if (style[k]?.magnitude) bits.push(`${k}=${style[k].magnitude}${style[k].unit ?? ''}`);
    }
    if (paragraph.bullet) bits.push(`bullet(list ${paragraph.bullet.listId ?? '?'})`);
    if (inTable) bits.push('in-table');
    out.push(`paragraph [${ps}-${pe}]: ${bits.join(', ')}`);

    for (const el of paragraph.elements ?? []) {
      if (!el.textRun || el.startIndex == null || el.endIndex == null) continue;
      if (el.endIndex <= startIndex || el.startIndex >= endIndex) continue;
      const ts = el.textRun.textStyle ?? {};
      const rb: string[] = [];
      if (ts.weightedFontFamily?.fontFamily) rb.push(ts.weightedFontFamily.fontFamily);
      if (ts.fontSize?.magnitude != null) rb.push(`${ts.fontSize.magnitude}pt`);
      for (const flag of ['bold', 'italic', 'underline', 'strikethrough'] as const) {
        if (ts[flag]) rb.push(flag);
      }
      const fg = rgbColorToHex(ts.foregroundColor);
      const bg = rgbColorToHex(ts.backgroundColor);
      if (fg) rb.push(fg);
      if (bg) rb.push(`bg ${bg}`);
      if (ts.baselineOffset && ts.baselineOffset !== 'NONE') rb.push(ts.baselineOffset.toLowerCase());
      if (ts.link?.url) rb.push(`link ${ts.link.url}`);
      const text = (el.textRun.content ?? '').replace(/\n/g, '⏎').slice(0, 60);
      out.push(`  run [${el.startIndex}-${el.endIndex}] "${text}"${rb.length ? ` - ${rb.join(', ')}` : ' - default style'}`);
    }
  });

  return out.length ? out.join('\n') : `no paragraphs overlap [${startIndex}-${endIndex})`;
}
