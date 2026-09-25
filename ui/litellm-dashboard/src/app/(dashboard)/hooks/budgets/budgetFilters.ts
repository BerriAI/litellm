import type { ColumnFilter, ColumnFiltersState } from "@tanstack/react-table";

export const BUDGET_DURATION_UNSET = "__unset__";

export const BUDGET_DURATION_FILTER_OPTIONS: readonly { value: string; label: string }[] = [
  { value: "1h", label: "hourly" },
  { value: "24h", label: "daily" },
  { value: "7d", label: "weekly" },
  { value: "30d", label: "monthly" },
  { value: BUDGET_DURATION_UNSET, label: "Not set" },
];

export interface MaxBudgetFilterValue {
  min?: string;
  max?: string;
  unlimitedOnly?: boolean;
}

export interface CreatedAtFilterValue {
  from?: string;
  to?: string;
}

type QueryEntry = readonly [string, string];

const entries = (key: string, value: string): QueryEntry[] => (value === "" ? [] : [[key, value]]);

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};

const asTrimmed = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

/** The date inputs give a calendar day; the route wants an instant, so widen to the viewer's whole local day. */
const isoAt = (day: string, time: string): string => {
  if (day === "") {
    return "";
  }
  const parsed = new Date(`${day}T${time}`);
  return Number.isNaN(parsed.getTime()) ? "" : parsed.toISOString();
};

/**
 * "Not set" is exclusive with the concrete durations. The route's contract does not say how it
 * combines `in` with `is_null` on one field, and under AND semantics that pair can only match
 * nothing, so we never send both.
 */
const durationParams = (value: unknown): QueryEntry[] => {
  const selected = asStringArray(value);
  if (selected.includes(BUDGET_DURATION_UNSET)) {
    return [["filter[budget_duration][is_null]", "true"]];
  }
  return entries("filter[budget_duration][in]", selected.join(","));
};

const maxBudgetParams = (value: unknown): QueryEntry[] => {
  const draft = asRecord(value);
  if (draft.unlimitedOnly === true) {
    return [["filter[max_budget][is_null]", "true"]];
  }
  return [
    ...entries("filter[max_budget][gte]", asTrimmed(draft.min)),
    ...entries("filter[max_budget][lte]", asTrimmed(draft.max)),
  ];
};

const createdAtParams = (value: unknown): QueryEntry[] => {
  const draft = asRecord(value);
  return [
    ...entries("filter[created_at][gte]", isoAt(asTrimmed(draft.from), "00:00:00.000")),
    ...entries("filter[created_at][lte]", isoAt(asTrimmed(draft.to), "23:59:59.999")),
  ];
};

const filterParams = (filter: ColumnFilter): QueryEntry[] => {
  switch (filter.id) {
    case "budget_duration":
      return durationParams(filter.value);
    case "max_budget":
      return maxBudgetParams(filter.value);
    case "created_at":
      return createdAtParams(filter.value);
    default:
      return [];
  }
};

export const serializeBudgetFilters = (filters: ColumnFiltersState): Readonly<Record<string, string>> =>
  Object.fromEntries(filters.flatMap(filterParams));
