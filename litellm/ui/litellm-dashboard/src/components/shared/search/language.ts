import type { LucideIcon } from "lucide-react";

export interface FieldSpec {
  readonly group: string;
  readonly icon: LucideIcon;
  /** The field has an enumerable value set worth offering once the key has its colon. */
  readonly suggestValues: boolean;
}

export type FilterOp = "eq" | "neq" | "glob" | "nglob";

/** Which `key:value` forms beyond equality the surface behind the box can honor. */
export interface QueryOperators {
  /** `-key:value` excludes; otherwise the dash is ordinary text. */
  readonly negation: boolean;
  /** `*` inside a value is a wildcard; otherwise it is a literal character. */
  readonly wildcard: boolean;
}

export const ALL_OPERATORS: QueryOperators = { negation: true, wildcard: true };
export const EQUALITY_ONLY: QueryOperators = { negation: false, wildcard: false };

/** The `key:value` vocabulary of one search box. Where the data lives is the evaluator's concern, not the language's. */
export interface QueryLanguage<F extends string> {
  readonly fields: Readonly<Record<F, FieldSpec>>;
  readonly ops: QueryOperators;
}

export const languageOps = <F extends string>({ ops }: QueryLanguage<F>): readonly FilterOp[] => [
  "eq",
  ...(ops.negation ? (["neq"] as const) : []),
  ...(ops.wildcard ? (["glob"] as const) : []),
  ...(ops.negation && ops.wildcard ? (["nglob"] as const) : []),
];

export const isNegatedOp = (op: FilterOp): boolean => op === "neq" || op === "nglob";
export const isGlobOp = (op: FilterOp): boolean => op === "glob" || op === "nglob";

interface Span {
  readonly from: number;
  readonly to: number;
}

export type QueryClause<F extends string> =
  | (Span & { readonly kind: "text"; readonly value: string })
  | (Span & {
      readonly kind: "field";
      readonly field: F;
      readonly op: FilterOp;
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

const filterOp = (negated: boolean, glob: boolean): FilterOp => {
  if (glob) return negated ? "nglob" : "glob";
  return negated ? "neq" : "eq";
};

function parseToken<F extends string>(
  language: QueryLanguage<F>,
  { raw, from, to }: Span & { raw: string },
): QueryClause<F> {
  const match = /^(-?)([A-Za-z_]+):(.*)$/.exec(raw);
  const text: QueryClause<F> = { kind: "text", value: unquote(raw), from, to };
  if (!match) return text;
  const key = match[2].toLowerCase();
  const negated = match[1] === "-";
  if (!Object.hasOwn(language.fields, key)) return text;
  if (negated && !language.ops.negation) return text;
  const value = unquote(match[3]);
  return {
    kind: "field",
    field: key as F,
    op: filterOp(negated, language.ops.wildcard && value.includes("*")),
    keyTo: from + match[1].length + match[2].length,
    value,
    from,
    to,
  };
}

export const parseQuery = <F extends string>(language: QueryLanguage<F>, text: string): QueryClause<F>[] =>
  tokenize(text).map((token) => parseToken(language, token));

/** Whole-value match, ignoring case; `*` is a literal. */
export function exactMatcher(pattern: string): (value: string) => boolean {
  const needle = pattern.toLowerCase();
  return (value) => value.toLowerCase() === needle;
}

/** `bar` matches the whole value, `*bar*` is a glob; both ignore case. */
export function valueMatcher(pattern: string): (value: string) => boolean {
  const needle = pattern.toLowerCase();
  if (!needle.includes("*")) return exactMatcher(needle);
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
