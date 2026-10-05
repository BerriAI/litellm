import type { ColumnFilter, ColumnFiltersState } from "@tanstack/react-table";

export const MODE_FILTER_ID = "mode";
export const PROVIDER_FILTER_ID = "providers";
export const FEATURE_FILTER_ID = "features";

export const PUBLIC_MODEL_HUB_SORTABLE_FIELDS: readonly string[] = [
  "model_group",
  "mode",
  "providers",
  "max_input_tokens",
  "max_output_tokens",
  "input_cost_per_token",
  "output_cost_per_token",
  "rpm",
  "tpm",
];

type QueryEntry = readonly [string, string];

type FilterValue = string | string[];

const entries = (key: string, value: string): QueryEntry[] => (value === "" ? [] : [[key, value]]);

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

const inFilter = (field: string, value: unknown): QueryEntry[] =>
  entries(`filter[${field}][in]`, asStringArray(value).join(","));

const filterParams = (filter: ColumnFilter): QueryEntry[] => {
  switch (filter.id) {
    case MODE_FILTER_ID:
    case PROVIDER_FILTER_ID:
    case FEATURE_FILTER_ID:
      return inFilter(filter.id, filter.value);
    default:
      return [];
  }
};

export const serializePublicModelHubFilters = (filters: ColumnFiltersState): Readonly<Record<string, string>> =>
  Object.fromEntries(filters.flatMap(filterParams));

export const readFilterValues = (filters: ColumnFiltersState, id: string): string[] =>
  asStringArray(filters.find((filter) => filter.id === id)?.value);

const isEmpty = (value: FilterValue): boolean => (Array.isArray(value) ? value.length === 0 : value.trim() === "");

export const withFilterValue = (filters: ColumnFiltersState, id: string, value: FilterValue): ColumnFiltersState => {
  const others = filters.filter((filter) => filter.id !== id);
  return isEmpty(value) ? others : [...others, { id, value }];
};

/** `supports_vision` reaches the route as `vision`; the hub has always shown it as "Vision". */
export const featureLabel = (feature: string): string =>
  feature
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
