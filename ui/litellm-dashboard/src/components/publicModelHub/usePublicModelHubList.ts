"use client";

import type { SortingState } from "@tanstack/react-table";
import { useCallback } from "react";

import {
  useResourceList,
  type ResourceListPage,
  type ResourceListQuery,
  type ResourceListResult,
} from "@/app/(dashboard)/hooks/common/useResourceList";
import { apiClient } from "@/components/networking";
import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";

import {
  FEATURE_FILTER_ID,
  MODE_FILTER_ID,
  PROVIDER_FILTER_ID,
  readFilterValues,
  serializePublicModelHubFilters,
  withFilterValue,
} from "./publicModelHubFilters";

export const PUBLIC_MODEL_HUB_PATH = "/public/v1/model_hub";
export const PUBLIC_MODEL_HUB_PAGE_SIZE = 50;

const QUERY_KEY = ["publicModelHub", "list"] as const;
const DEFAULT_SORTING: SortingState = [{ id: "model_group", desc: false }];

export interface PublicModelHubListResult extends ResourceListResult<ModelGroupInfo> {
  providerValues: string[];
  onProvidersChange: (values: string[]) => void;
  modeValues: string[];
  onModesChange: (values: string[]) => void;
  featureValues: string[];
  onFeaturesChange: (values: string[]) => void;
  hasActiveQuery: boolean;
}

const fetchPage = async (query: ResourceListQuery, signal: AbortSignal): Promise<ResourceListPage<ModelGroupInfo>> => {
  try {
    return await apiClient.get<ResourceListPage<ModelGroupInfo>>(PUBLIC_MODEL_HUB_PATH, { query, signal });
  } catch (error) {
    if (!signal.aborted) {
      console.error("There was an error fetching the public model data", error);
    }
    throw error;
  }
};

export const usePublicModelHubList = (enabled: boolean): PublicModelHubListResult => {
  const listOptions = {
    queryKey: QUERY_KEY,
    fetchPage,
    serializeFilters: serializePublicModelHubFilters,
    defaultSorting: DEFAULT_SORTING,
    defaultPageSize: PUBLIC_MODEL_HUB_PAGE_SIZE,
    enabled,
  };
  const list = useResourceList<ModelGroupInfo>(listOptions);

  const { onColumnFiltersChange } = list;

  const setFilter = useCallback(
    (id: string, values: string[]) => onColumnFiltersChange((previous) => withFilterValue(previous, id, values)),
    [onColumnFiltersChange],
  );

  const onProvidersChange = useCallback((values: string[]) => setFilter(PROVIDER_FILTER_ID, values), [setFilter]);
  const onModesChange = useCallback((values: string[]) => setFilter(MODE_FILTER_ID, values), [setFilter]);
  const onFeaturesChange = useCallback((values: string[]) => setFilter(FEATURE_FILTER_ID, values), [setFilter]);

  return {
    ...list,
    providerValues: readFilterValues(list.columnFilters, PROVIDER_FILTER_ID),
    onProvidersChange,
    modeValues: readFilterValues(list.columnFilters, MODE_FILTER_ID),
    onModesChange,
    featureValues: readFilterValues(list.columnFilters, FEATURE_FILTER_ID),
    onFeaturesChange,
    hasActiveQuery: list.searchValue.trim() !== "" || list.columnFilters.length > 0,
  };
};
