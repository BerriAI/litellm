import { getSpendString } from "@/utils/dataUtils";

import type { DailyActivityUserPageResponse, UserActivityRow } from "../dailyActivityApi";

export type FetchUserPage = (offset: number, limit: number) => Promise<DailyActivityUserPageResponse>;

export const userRowKey = (row: UserActivityRow): string =>
  row.user_id === null || row.user_id === undefined ? "none" : `user:${row.user_id}`;

export const userPrimaryLabel = (row: UserActivityRow): string => {
  const friendlyName = row.user_email || row.user_alias;
  return friendlyName || row.user_id || "(no user)";
};

export const userSecondaryLabel = (row: UserActivityRow): string | null => {
  const hasFriendlyName = Boolean(row.user_email || row.user_alias);
  return hasFriendlyName && row.user_id ? row.user_id : null;
};

export const mergeUserActivityPages = (
  current: readonly UserActivityRow[],
  next: readonly UserActivityRow[],
  total: number,
  offset: number,
): { rows: UserActivityRow[]; nextOffset: number; hasMore: boolean } => {
  const rows: UserActivityRow[] = Array.from(
    new Map([...current, ...next].map((row) => [userRowKey(row), row])).values(),
  );
  const nextOffset = offset + next.length;
  return { rows, nextOffset, hasMore: next.length > 0 && nextOffset < total };
};

export const formatUserSpend = (value: number): string => {
  if (value === 0) return "$0.00";
  return getSpendString(value, 2);
};
