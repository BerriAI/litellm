import type { ColumnFilter, ColumnFiltersState } from "@tanstack/react-table";

import type { ResourceListUrlState } from "../common/useResourceList";

export const BUDGET_DURATION_UNSET = "__unset__";

export const BUDGET_DURATION_FILTER_OPTIONS: readonly { value: string; label: string }[] = [
  { value: "1h", label: "hourly" },
  { value: "24h", label: "daily" },
  { value: "7d", label: "weekly" },
  { value: "30d", label: "monthly" },
  { value: BUDGET_DURATION_UNSET, label: "Not set" },
];

export const BUDGET_SORTABLE_FIELDS: readonly string[] = [
  "budget_id",
  "max_budget",
  "tpm_limit",
  "rpm_limit",
  "tpd_limit",
  "created_at",
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

const DURATION_FILTER_ID = "budget_duration";
const MAX_BUDGET_FILTER_ID = "max_budget";
const CREATED_AT_FILTER_ID = "created_at";

const URL_DURATION = "duration";
const URL_MAX_MIN = "max_min";
const URL_MAX_MAX = "max_max";
const URL_UNLIMITED = "unlimited";
const URL_CREATED_FROM = "created_from";
const URL_CREATED_TO = "created_to";
const URL_TRUE = "true";

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

/** The drawer keeps any non-empty object as an active filter, so collapse a blank draft to nothing. */
export const normalizeMaxBudget = (draft: MaxBudgetFilterValue): MaxBudgetFilterValue | undefined => {
  if (draft.unlimitedOnly === true) {
    return { unlimitedOnly: true };
  }
  const min = draft.min?.trim() ?? "";
  const max = draft.max?.trim() ?? "";
  if (min === "" && max === "") {
    return undefined;
  }
  return { ...(min === "" ? {} : { min }), ...(max === "" ? {} : { max }) };
};

export const normalizeCreatedAt = (draft: CreatedAtFilterValue): CreatedAtFilterValue | undefined => {
  const from = draft.from ?? "";
  const to = draft.to ?? "";
  if (from === "" && to === "") {
    return undefined;
  }
  return { ...(from === "" ? {} : { from }), ...(to === "" ? {} : { to }) };
};

const exclusiveDurations = (selected: string[]): string[] =>
  selected.includes(BUDGET_DURATION_UNSET) ? [BUDGET_DURATION_UNSET] : selected;

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
    case DURATION_FILTER_ID:
      return durationParams(filter.value);
    case MAX_BUDGET_FILTER_ID:
      return maxBudgetParams(filter.value);
    case CREATED_AT_FILTER_ID:
      return createdAtParams(filter.value);
    default:
      return [];
  }
};

export const serializeBudgetFilters = (filters: ColumnFiltersState): Readonly<Record<string, string>> =>
  Object.fromEntries(filters.flatMap(filterParams));

const valueOf = (filters: ColumnFiltersState, id: string): unknown => filters.find((filter) => filter.id === id)?.value;

const DURATION_VALUES = BUDGET_DURATION_FILTER_OPTIONS.map((option) => option.value);
const DAY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

const amountOrBlank = (value: string): string => (value !== "" && Number.isFinite(Number(value)) ? value : "");

const dayOrBlank = (value: string): string =>
  DAY_PATTERN.test(value) && isoAt(value, "00:00:00.000") !== "" ? value : "";

export const budgetFiltersToUrl = (filters: ColumnFiltersState): ColumnFiltersState => {
  const maxBudget = asRecord(valueOf(filters, MAX_BUDGET_FILTER_ID));
  const created = asRecord(valueOf(filters, CREATED_AT_FILTER_ID));
  return [
    { id: URL_DURATION, value: asStringArray(valueOf(filters, DURATION_FILTER_ID)) },
    { id: URL_MAX_MIN, value: asTrimmed(maxBudget.min) },
    { id: URL_MAX_MAX, value: asTrimmed(maxBudget.max) },
    { id: URL_UNLIMITED, value: maxBudget.unlimitedOnly === true ? URL_TRUE : "" },
    { id: URL_CREATED_FROM, value: asTrimmed(created.from) },
    { id: URL_CREATED_TO, value: asTrimmed(created.to) },
  ];
};

export const budgetFiltersFromUrl = (filters: ColumnFiltersState): ColumnFiltersState => {
  const text = (id: string): string => asTrimmed(valueOf(filters, id));
  const durations = asStringArray(valueOf(filters, URL_DURATION)).filter((value) => DURATION_VALUES.includes(value));
  const maxBudget = normalizeMaxBudget({
    min: amountOrBlank(text(URL_MAX_MIN)),
    max: amountOrBlank(text(URL_MAX_MAX)),
    unlimitedOnly: text(URL_UNLIMITED) === URL_TRUE,
  });
  const created = normalizeCreatedAt({
    from: dayOrBlank(text(URL_CREATED_FROM)),
    to: dayOrBlank(text(URL_CREATED_TO)),
  });
  return [
    ...(durations.length === 0 ? [] : [{ id: DURATION_FILTER_ID, value: exclusiveDurations(durations) }]),
    ...(maxBudget === undefined ? [] : [{ id: MAX_BUDGET_FILTER_ID, value: maxBudget }]),
    ...(created === undefined ? [] : [{ id: CREATED_AT_FILTER_ID, value: created }]),
  ];
};

export const BUDGET_LIST_URL_STATE: ResourceListUrlState = {
  sortFields: BUDGET_SORTABLE_FIELDS,
  filterColumns: [URL_DURATION, URL_MAX_MIN, URL_MAX_MAX, URL_UNLIMITED, URL_CREATED_FROM, URL_CREATED_TO],
  arrayFilterColumns: [URL_DURATION],
  toUrlFilters: budgetFiltersToUrl,
  fromUrlFilters: budgetFiltersFromUrl,
};
