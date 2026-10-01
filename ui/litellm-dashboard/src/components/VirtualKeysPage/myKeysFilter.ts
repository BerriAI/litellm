import type { ColumnFiltersState } from "@tanstack/react-table";

import { isProxyAdminTierRole } from "@/utils/roles";

export const MY_KEYS_FILTER_ID = "my_keys";
export const MY_KEYS_URL_KEY = "my_keys";
export const MY_KEYS_TOOLTIP =
  "Show only keys assigned to your user account. Team keys and keys you created for others are excluded.";

export function isMyKeysEnabled(
  myKeysParam: boolean | null,
  userRole: string | null | undefined,
  userId: string | null | undefined,
): boolean {
  if (!userId) {
    return false;
  }
  return myKeysParam ?? !isProxyAdminTierRole(userRole ?? "");
}

export function myKeysQueryParam(
  enabled: boolean,
  userRole: string | null | undefined,
  userId: string | null | undefined,
): boolean | null {
  if (enabled) {
    return true;
  }
  return isMyKeysEnabled(null, userRole, userId) ? false : null;
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
