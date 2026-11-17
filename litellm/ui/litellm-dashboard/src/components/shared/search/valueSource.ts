import { type ClientIndex, fieldValues } from "./evaluate";

export interface FieldValues {
  readonly values: readonly string[];
  readonly loading: boolean;
}

export const NO_VALUES: FieldValues = { values: [], loading: false };

/** Where a search box gets the values it offers for a field; a server facet lookup or the loaded items. */
export interface ValueSource<F extends string> {
  /** A hook: `field` is the key being completed at the cursor, or null when none is. */
  useValues(field: F | null, prefix: string): FieldValues;
}

/** Values seen in the items already in memory. */
export function itemValues<T, F extends string>(index: ClientIndex<T, F>, items: readonly T[]): ValueSource<F> {
  return {
    useValues: (field) => (field === null ? NO_VALUES : { values: fieldValues(index, items, field), loading: false }),
  };
}
