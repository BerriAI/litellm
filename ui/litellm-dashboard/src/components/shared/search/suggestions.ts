import { type FieldClause, languageFields, parseQuery, type QueryLanguage } from "./language";
import type { FieldValues } from "./valueSource";

export interface Suggestion<F extends string> {
  readonly id: string;
  readonly label: string;
  readonly field: F;
  /** Replaces `text.slice(from, to)`. */
  readonly from: number;
  readonly to: number;
  readonly insert: string;
  /** A value finishes the clause and closes the menu; a field keeps it open for its values. */
  readonly completesClause: boolean;
}

export interface SuggestionGroup<F extends string> {
  readonly heading: string;
  readonly items: readonly Suggestion<F>[];
}

export interface SuggestionMenu<F extends string> {
  readonly groups: readonly SuggestionGroup<F>[];
  readonly showOperators: boolean;
  /** Values are still on their way from the source. */
  readonly loading: boolean;
}

type Target<F extends string> =
  | {
      readonly kind: "field";
      readonly prefix: string;
      readonly negated: boolean;
      readonly from: number;
      readonly to: number;
    }
  | { readonly kind: "value"; readonly clause: FieldClause<F> };

const MAX_VALUES = 50;
const isSpace = (char: string | undefined) => char === undefined || /\s/.test(char);
const quoteIfNeeded = (value: string) => (/[\s"]/.test(value) ? `"${value.replaceAll('"', "")}"` : value);

function target<F extends string>(language: QueryLanguage<F>, text: string, cursor: number): Target<F> | null {
  const clause = parseQuery(language, text).find((c) => c.from < cursor && c.to === cursor);
  if (!clause) {
    const betweenTokens = isSpace(text[cursor - 1]) && isSpace(text[cursor]);
    return betweenTokens ? { kind: "field", prefix: "", negated: false, from: cursor, to: cursor } : null;
  }
  if (clause.kind === "field") return { kind: "value", clause };
  const raw = text.slice(clause.from, clause.to);
  const negated = raw.startsWith("-");
  const prefix = negated ? raw.slice(1) : raw;
  if (/[:"]/.test(prefix)) return null;
  return { kind: "field", prefix, negated, from: clause.from, to: clause.to };
}

/** The field whose value is being typed at the cursor, so a source can be asked for its values. */
export function completingField<F extends string>(language: QueryLanguage<F>, text: string, cursor: number): F | null {
  const found = target(language, text, cursor);
  return found?.kind === "value" && language.fields[found.clause.field].suggestValues ? found.clause.field : null;
}

/** The substring typed so far for the value at the cursor, wildcards stripped. */
export function completingPrefix<F extends string>(language: QueryLanguage<F>, text: string, cursor: number): string {
  const found = target(language, text, cursor);
  return found?.kind === "value" ? found.clause.value.replaceAll("*", "") : "";
}

function fieldMenu<F extends string>(
  language: QueryLanguage<F>,
  { prefix, negated, from, to }: Extract<Target<F>, { kind: "field" }>,
): SuggestionMenu<F> | null {
  const fields = languageFields(language).filter((field) => field.startsWith(prefix.toLowerCase()));
  if (fields.length === 0) return null;
  const groups = Array.from(new Set(fields.map((field) => language.fields[field].group)), (heading) => ({
    heading,
    items: fields
      .filter((field) => language.fields[field].group === heading)
      .map((field) => ({
        id: `field:${field}`,
        label: field,
        field,
        from,
        to,
        insert: `${negated ? "-" : ""}${field}:`,
        completesClause: false,
      })),
  }));
  return { groups, showOperators: false, loading: false };
}

function valueMenu<F extends string>(
  clause: FieldClause<F>,
  text: string,
  { values, loading }: FieldValues,
): SuggestionMenu<F> | null {
  const from = clause.keyTo + 1;
  const needle = clause.value.replaceAll("*", "").toLowerCase();
  const trailing = clause.to === text.length ? " " : "";
  const suggestions = values
    .filter((value) => value.toLowerCase().includes(needle))
    .slice(0, MAX_VALUES)
    .map((value) => ({
      id: `value:${value}`,
      label: value,
      field: clause.field,
      from,
      to: clause.to,
      insert: quoteIfNeeded(value) + trailing,
      completesClause: true,
    }));
  const showOperators = clause.value === "";
  if (suggestions.length === 0 && !showOperators && !loading) return null;
  return { groups: suggestions.length ? [{ heading: clause.field, items: suggestions }] : [], showOperators, loading };
}

/** What to offer at the cursor: fields while a key is being typed, values once it has a colon. */
export function suggest<F extends string>(
  language: QueryLanguage<F>,
  text: string,
  cursor: number,
  lookup: (field: F) => FieldValues,
): SuggestionMenu<F> | null {
  const found = target(language, text, cursor);
  if (!found) return null;
  if (found.kind === "field") return fieldMenu(language, found);
  const { field } = found.clause;
  return valueMenu(
    found.clause,
    text,
    language.fields[field].suggestValues ? lookup(field) : { values: [], loading: false },
  );
}
