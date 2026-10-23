import { exactMatcher, isGlobOp, isNegatedOp, parseQuery, type QueryLanguage, valueMatcher } from "./language";
import { type SearchFilter, type SearchQuery, toSearchQuery } from "./searchQuery";

/** How to read a language's fields off items already in memory. */
export interface ClientIndex<T, F extends string> {
  readonly read: Readonly<Record<F, (item: T) => readonly string[]>>;
  /** What a bare word searches. */
  readonly freeText: (item: T) => readonly string[];
}

function matchesFilter<T, F extends string>(index: ClientIndex<T, F>, item: T, filter: SearchFilter<F>): boolean {
  const matcher = isGlobOp(filter.op) ? valueMatcher(filter.value) : exactMatcher(filter.value);
  const matches = index.read[filter.field](item).some(matcher);
  return isNegatedOp(filter.op) ? !matches : matches;
}

function matchesText<T, F extends string>(index: ClientIndex<T, F>, item: T, term: string): boolean {
  const needle = term.toLowerCase();
  return index.freeText(item).some((text) => text.toLowerCase().includes(needle));
}

/** The in-memory counterpart of a server search: every term and filter of `query` must hold. */
export function evaluate<T, F extends string>(index: ClientIndex<T, F>, items: T[], query: SearchQuery<F>): T[] {
  if (query.text.length === 0 && query.filters.length === 0) return items;
  return items.filter(
    (item) =>
      query.text.every((term) => matchesText(index, item, term)) &&
      query.filters.every((filter) => matchesFilter(index, item, filter)),
  );
}

export const filterItems = <T, F extends string>(
  language: QueryLanguage<F>,
  index: ClientIndex<T, F>,
  items: T[],
  text: string,
): T[] => evaluate(index, items, toSearchQuery(parseQuery(language, text)));

export function fieldValues<T, F extends string>(index: ClientIndex<T, F>, items: readonly T[], field: F): string[] {
  return Array.from(new Set(items.flatMap((item) => index.read[field](item))))
    .filter(Boolean)
    .sort();
}
