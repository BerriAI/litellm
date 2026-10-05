import { functionalUpdate, type SortingState, type Updater } from "@tanstack/react-table";

import type { TraceSummary } from "../types";

export const RUN_SORT_KEYS = ["start_ms", "duration_ms", "span_count", "error_count"] as const;
export type RunSortKey = (typeof RUN_SORT_KEYS)[number];

export const SORT_DIRS = ["asc", "desc"] as const;
export type SortDir = (typeof SORT_DIRS)[number];

/** The sequence a runs page is cut from: one server-sorted key, with the trace reference as the tie-break. */
export interface RunOrder {
  readonly key: RunSortKey;
  readonly descending: boolean;
}

export const NEWEST: RunOrder = { key: "start_ms", descending: true };

export const sortDir = (order: RunOrder): SortDir => (order.descending ? "desc" : "asc");

const SORT_VALUE: Record<RunSortKey, (run: TraceSummary) => number> = {
  start_ms: (run) => Date.parse(run.start_time),
  duration_ms: (run) => run.duration_ms,
  span_count: (run) => run.span_count,
  error_count: (run) => run.error_count,
};

const compareText = (left: string, right: string): number => {
  if (left === right) return 0;
  return left < right ? -1 : 1;
};

const reference = (run: TraceSummary): string => run.id;

/** Runs in `order`, as the server pages them: by the key, then by trace reference the same way. */
export function orderRuns(runs: readonly TraceSummary[], order: RunOrder): TraceSummary[] {
  const value = SORT_VALUE[order.key];
  const sign = order.descending ? -1 : 1;
  return [...runs].sort((left, right) => {
    const byKey = value(left) - value(right);
    return sign * (byKey !== 0 ? byKey : compareText(reference(left), reference(right)));
  });
}

export const toSorting = (order: RunOrder): SortingState => [{ id: order.key, desc: order.descending }];

const isSortKey = (id: string): id is RunSortKey => RUN_SORT_KEYS.includes(id as RunSortKey);

/** The order TanStack's sorting state asks for; an empty or unknown state keeps `current`. */
export function fromSorting(updater: Updater<SortingState>, current: RunOrder): RunOrder {
  const [first] = functionalUpdate(updater, toSorting(current));
  return first !== undefined && isSortKey(first.id) ? { key: first.id, descending: first.desc } : current;
}
