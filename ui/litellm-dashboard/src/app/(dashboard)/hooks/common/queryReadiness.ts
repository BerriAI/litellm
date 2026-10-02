import type { UseQueryResult } from "@tanstack/react-query";

export const isQueryPending = (query: Pick<UseQueryResult, "isPending" | "isEnabled">): boolean =>
  query.isEnabled && query.isPending;
