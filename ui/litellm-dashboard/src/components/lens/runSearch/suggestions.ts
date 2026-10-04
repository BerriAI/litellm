import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";

import { fieldValues, parseRunQuery, RUN_FIELDS, type QueryClause, type RunField } from "./runQuery";

export interface Suggestion {
  id: string;
  label: string;
  field: RunField;
  /** Replaces `text.slice(from, to)`. */
  from: number;
  to: number;
  insert: string;
  /** A value finishes the clause and closes the menu; a field keeps it open for its values. */
  completesClause: boolean;
}

export interface SuggestionGroup {
  heading: string;
  items: Suggestion[];
}

export interface SuggestionMenu {
  groups: SuggestionGroup[];
  showOperators: boolean;
}

const MAX_VALUES = 50;
const FIELDS = Object.keys(RUN_FIELDS) as RunField[];
const isSpace = (char: string | undefined) => char === undefined || /\s/.test(char);
const quoteIfNeeded = (value: string) => (/[\s"]/.test(value) ? `"${value.replaceAll('"', "")}"` : value);

function fieldMenu(prefix: string, negated: boolean, from: number, to: number): SuggestionMenu | null {
  const fields = FIELDS.filter((field) => field.startsWith(prefix.toLowerCase()));
  if (fields.length === 0) return null;
  const groups = Array.from(new Set(fields.map((field) => RUN_FIELDS[field].group)), (heading) => ({
    heading,
    items: fields
      .filter((field) => RUN_FIELDS[field].group === heading)
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

function valueMenu(
  clause: Extract<QueryClause, { kind: "field" }>,
  text: string,
  runs: readonly TraceSummary[],
): SuggestionMenu | null {
  const from = clause.keyTo + 1;
  const needle = clause.value.replaceAll("*", "").toLowerCase();
  const trailing = clause.to === text.length ? " " : "";
  const values = RUN_FIELDS[clause.field].suggestValues ? fieldValues(runs, clause.field) : [];
  const items = values
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
  if (items.length === 0 && !showOperators) return null;
  return { groups: items.length ? [{ heading: clause.field, items }] : [], showOperators };
}

/** What to offer at the cursor: fields while a key is being typed, values once it has a colon. */
export function suggest(text: string, cursor: number, runs: readonly TraceSummary[]): SuggestionMenu | null {
  const clause = parseRunQuery(text).find((c) => c.from < cursor && c.to === cursor);
  if (!clause) {
    const betweenTokens = isSpace(text[cursor - 1]) && isSpace(text[cursor]);
    return betweenTokens ? fieldMenu("", false, cursor, cursor) : null;
  }
  if (clause.kind === "field") return valueMenu(clause, text, runs);
  const raw = text.slice(clause.from, clause.to);
  const negated = raw.startsWith("-");
  const prefix = negated ? raw.slice(1) : raw;
  if (/[:"]/.test(prefix)) return null;
  return fieldMenu(prefix, negated, clause.from, clause.to);
}
