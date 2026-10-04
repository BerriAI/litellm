import { type FieldClause, fieldValues, languageFields, parseQuery, type QueryLanguage } from "./language";

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
}

const MAX_VALUES = 50;
const isSpace = (char: string | undefined) => char === undefined || /\s/.test(char);
const quoteIfNeeded = (value: string) => (/[\s"]/.test(value) ? `"${value.replaceAll('"', "")}"` : value);

function fieldMenu<T, F extends string>(
  language: QueryLanguage<T, F>,
  prefix: string,
  negated: boolean,
  { from, to }: { from: number; to: number },
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
  return { groups, showOperators: false };
}

function valueMenu<T, F extends string>(
  language: QueryLanguage<T, F>,
  clause: FieldClause<F>,
  text: string,
  items: readonly T[],
): SuggestionMenu<F> | null {
  const from = clause.keyTo + 1;
  const needle = clause.value.replaceAll("*", "").toLowerCase();
  const trailing = clause.to === text.length ? " " : "";
  const values = language.fields[clause.field].suggestValues ? fieldValues(language, items, clause.field) : [];
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
  if (suggestions.length === 0 && !showOperators) return null;
  return { groups: suggestions.length ? [{ heading: clause.field, items: suggestions }] : [], showOperators };
}

/** What to offer at the cursor: fields while a key is being typed, values once it has a colon. */
export function suggest<T, F extends string>(
  language: QueryLanguage<T, F>,
  text: string,
  cursor: number,
  items: readonly T[],
): SuggestionMenu<F> | null {
  const clause = parseQuery(language, text).find((c) => c.from < cursor && c.to === cursor);
  if (!clause) {
    const betweenTokens = isSpace(text[cursor - 1]) && isSpace(text[cursor]);
    return betweenTokens ? fieldMenu(language, "", false, { from: cursor, to: cursor }) : null;
  }
  if (clause.kind === "field") return valueMenu(language, clause, text, items);
  const raw = text.slice(clause.from, clause.to);
  const negated = raw.startsWith("-");
  const prefix = negated ? raw.slice(1) : raw;
  if (/[:"]/.test(prefix)) return null;
  return fieldMenu(language, prefix, negated, clause);
}
