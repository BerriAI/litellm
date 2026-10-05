import type { LucideIcon } from "lucide-react";

export interface FieldSpec {
  readonly group: string;
  readonly icon: LucideIcon;
  /** The field has an enumerable value set worth offering once the key has its colon. */
  readonly suggestValues: boolean;
}

/** The `key:value` vocabulary of one search box. Where the data lives is the evaluator's concern, not the language's. */
export interface QueryLanguage<F extends string> {
  readonly fields: Readonly<Record<F, FieldSpec>>;
}

interface Span {
  readonly from: number;
  readonly to: number;
}

export type QueryClause<F extends string> =
  | (Span & { readonly kind: "text"; readonly value: string })
  | (Span & {
      readonly kind: "field";
      readonly field: F;
      readonly negated: boolean;
      readonly keyTo: number;
      readonly value: string;
    });

export type FieldClause<F extends string> = Extract<QueryClause<F>, { kind: "field" }>;

export const languageFields = <F extends string>(language: QueryLanguage<F>): F[] =>
  Object.keys(language.fields) as F[];

const unquote = (raw: string): string => raw.replace(/^"([^"]*)"?$/, "$1");

/** Whitespace-separated tokens; a double-quoted stretch keeps its spaces. */
function tokenize(text: string): (Span & { raw: string })[] {
  return Array.from(text.matchAll(/(?:"[^"]*"?|\S)+/g), (match) => ({
    raw: match[0],
    from: match.index,
    to: match.index + match[0].length,
  }));
}

function parseToken<F extends string>(
  language: QueryLanguage<F>,
  { raw, from, to }: Span & { raw: string },
): QueryClause<F> {
  const match = /^(-?)([A-Za-z_]+):(.*)$/.exec(raw);
  const key = match?.[2].toLowerCase() ?? "";
  if (!match || !Object.hasOwn(language.fields, key)) return { kind: "text", value: unquote(raw), from, to };
  return {
    kind: "field",
    field: key as F,
    negated: match[1] === "-",
    keyTo: from + match[1].length + match[2].length,
    value: unquote(match[3]),
    from,
    to,
  };
}

export const parseQuery = <F extends string>(language: QueryLanguage<F>, text: string): QueryClause<F>[] =>
  tokenize(text).map((token) => parseToken(language, token));

/** `bar` matches the whole value, `*bar*` is a glob; both ignore case. */
export function valueMatcher(pattern: string): (value: string) => boolean {
  const needle = pattern.toLowerCase();
  if (!needle.includes("*")) return (value) => value.toLowerCase() === needle;
  const segments = needle.split("*");
  const first = segments[0];
  const last = segments.at(-1);
  const middle = segments.slice(1, -1).filter(Boolean);
  return (value) => {
    const candidate = value.toLowerCase();
    if (first && !candidate.startsWith(first)) return false;
    if (last && !candidate.endsWith(last)) return false;
    const suffixStart = last ? candidate.length - last.length : candidate.length;
    let cursor = first ? first.length : 0;
    if (suffixStart < cursor) return false;
    for (const segment of middle) {
      const found = candidate.indexOf(segment, cursor);
      if (found < 0 || found + segment.length > suffixStart) return false;
      cursor = found + segment.length;
    }
    return true;
  };
}
