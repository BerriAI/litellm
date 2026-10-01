import type { ColumnFiltersState } from "@tanstack/react-table";

export const MY_KEYS_FILTER_ID = "my_keys";
export const MY_KEYS_URL_KEY = "my_keys";
export const MY_KEYS_TOOLTIP =
  "Show only keys assigned to your user account. Team keys and keys you created for others are excluded.";

export function isMyKeysEnabled(myKeysParam: boolean | null, userId: string | null | undefined): boolean {
  return Boolean(userId) && myKeysParam === true;
}

export function myKeysQueryParam(enabled: boolean): boolean | null {
  return enabled ? true : null;
}

export function withMyKeysFilter(filters: ColumnFiltersState, enabled: boolean): ColumnFiltersState {
  if (!enabled) {
    return filters;
  }
  return [...filters.filter((filter) => filter.id !== "user_id"), { id: MY_KEYS_FILTER_ID, value: true }];
}

export function splitMyKeysFilter(filters: ColumnFiltersState): { myKeys: boolean; rest: ColumnFiltersState } {
  const entry = filters.find((filter) => filter.id === MY_KEYS_FILTER_ID);
  return { myKeys: entry?.value === true, rest: filters.filter((filter) => filter.id !== MY_KEYS_FILTER_ID) };
}
