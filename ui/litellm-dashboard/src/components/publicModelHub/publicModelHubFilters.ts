import type { ColumnFilter, ColumnFiltersState } from "@tanstack/react-table";

export const MODE_FILTER_ID = "mode";
export const PROVIDER_FILTER_ID = "providers";

export const PUBLIC_MODEL_HUB_SORTABLE_FIELDS: readonly string[] = [
  "model_group",
  "mode",
  "max_input_tokens",
  "max_output_tokens",
  "input_cost_per_token",
  "output_cost_per_token",
];

export const MODE_FILTER_OPTIONS: readonly { value: string; label: string }[] = [
  "audio_speech",
  "audio_transcription",
  "chat",
  "completion",
  "embedding",
  "guardrail",
  "image_edit",
  "image_generation",
  "moderation",
  "ocr",
  "realtime",
  "rerank",
  "responses",
  "search",
  "vector_store",
  "video_generation",
].map((mode) => ({ value: mode, label: mode }));

type QueryEntry = readonly [string, string];

type FilterValue = string | string[];

const entries = (key: string, value: string): QueryEntry[] => (value === "" ? [] : [[key, value]]);

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

const asTrimmed = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

const filterParams = (filter: ColumnFilter): QueryEntry[] => {
  switch (filter.id) {
    case MODE_FILTER_ID:
      return entries("filter[mode][in]", asStringArray(filter.value).join(","));
    case PROVIDER_FILTER_ID:
      return entries("filter[providers][contains]", asTrimmed(filter.value));
    default:
      return [];
  }
};

export const serializePublicModelHubFilters = (filters: ColumnFiltersState): Readonly<Record<string, string>> =>
  Object.fromEntries(filters.flatMap(filterParams));

export const readModeFilter = (filters: ColumnFiltersState): string[] =>
  asStringArray(filters.find((filter) => filter.id === MODE_FILTER_ID)?.value);

const isEmpty = (value: FilterValue): boolean => (Array.isArray(value) ? value.length === 0 : value.trim() === "");

export const withFilterValue = (filters: ColumnFiltersState, id: string, value: FilterValue): ColumnFiltersState => {
  const others = filters.filter((filter) => filter.id !== id);
  return isEmpty(value) ? others : [...others, { id, value }];
};
