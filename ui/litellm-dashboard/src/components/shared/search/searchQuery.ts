import type { QueryClause } from "./language";

export type FilterOp = "eq" | "neq" | "glob" | "nglob";

export interface SearchFilter<F extends string> {
  readonly field: F;
  readonly op: FilterOp;
  readonly value: string;
}

/**
 * The wire shape of a query: what a server or the client evaluator consumes.
 * `eq` is a case-insensitive whole-value match; `glob` treats `*` as a wildcard; the `n` forms negate.
 * Every text term and every filter must hold.
 */
export interface SearchQuery<F extends string> {
  readonly text: readonly string[];
  readonly filters: readonly SearchFilter<F>[];
}

const filterOp = (negated: boolean, value: string): FilterOp => {
  if (value.includes("*")) return negated ? "nglob" : "glob";
  return negated ? "neq" : "eq";
};

/** A key typed without a value yet narrows nothing, so the list does not blank out mid-typing. */
export function toSearchQuery<F extends string>(clauses: readonly QueryClause<F>[]): SearchQuery<F> {
  const text = clauses.flatMap((clause) => (clause.kind === "text" && clause.value ? [clause.value] : []));
  const filters = clauses.flatMap((clause) =>
    clause.kind === "field" && clause.value
      ? [{ field: clause.field, op: filterOp(clause.negated, clause.value), value: clause.value }]
      : [],
  );
  return { text, filters };
}
