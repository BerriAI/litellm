"use client";

import { parseAsString, useQueryStates } from "nuqs";
import { useCallback, useMemo, useState } from "react";

import { matchesSearchTerm } from "@/utils/searchUtils";

export interface ModelFilterValues {
  search: string;
  provider: string;
  mode: string;
  feature: string;
}

export interface ModelFiltersState {
  values: ModelFilterValues;
  update: (patch: Partial<ModelFilterValues>) => void;
  reset: () => void;
}

interface FilterableModel {
  model_group: string;
  providers: string[];
  mode?: string;
}

const NO_FILTERS: ModelFilterValues = { search: "", provider: "", mode: "", feature: "" };

const MODEL_FILTER_PARSERS = {
  search: parseAsString.withDefault(""),
  provider: parseAsString.withDefault(""),
  mode: parseAsString.withDefault(""),
  feature: parseAsString.withDefault(""),
};

const MODEL_FILTER_URL_KEYS = { search: "q" };

const toTitleCase = (word: string): string => word.charAt(0).toUpperCase() + word.slice(1);

export const modelFeatureNames = (model: object): string[] =>
  Object.entries(model)
    .filter(([key, value]) => key.startsWith("supports_") && value === true)
    .map(([key]) =>
      key
        .replace(/^supports_/, "")
        .split("_")
        .map(toTitleCase)
        .join(" "),
    );

const MODEL_FILTER_PREDICATES: ReadonlyArray<(model: FilterableModel, filters: ModelFilterValues) => boolean> = [
  (model, { search }) => matchesSearchTerm(search, [model.model_group]),
  (model, { provider }) => provider === "" || model.providers.includes(provider),
  (model, { mode }) => mode === "" || model.mode === mode,
  (model, { feature }) => feature === "" || modelFeatureNames(model).includes(feature),
];

export const filterModelHubData = <T extends FilterableModel>(models: readonly T[], filters: ModelFilterValues): T[] =>
  models.filter((model) => MODEL_FILTER_PREDICATES.every((matches) => matches(model, filters)));

export const useLocalModelFiltersState = (): ModelFiltersState => {
  const [values, setValues] = useState<ModelFilterValues>(NO_FILTERS);
  const update = useCallback(
    (patch: Partial<ModelFilterValues>) => setValues((previous) => ({ ...previous, ...patch })),
    [],
  );
  const reset = useCallback(() => setValues(NO_FILTERS), []);
  return useMemo(() => ({ values, update, reset }), [values, update, reset]);
};

export const useModelFiltersUrlState = (): ModelFiltersState => {
  const [values, setValues] = useQueryStates(MODEL_FILTER_PARSERS, { urlKeys: MODEL_FILTER_URL_KEYS });
  const update = useCallback((patch: Partial<ModelFilterValues>) => void setValues(patch), [setValues]);
  const reset = useCallback(() => void setValues(null), [setValues]);
  return useMemo(() => ({ values, update, reset }), [values, update, reset]);
};
