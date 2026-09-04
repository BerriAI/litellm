"use client";

import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import type { SortingState } from "@tanstack/react-table";
import { useCallback, useState } from "react";

import {
  useResourceList,
  type ResourceListPage,
  type ResourceListQuery,
  type ResourceListResult,
} from "@/app/(dashboard)/hooks/common/useResourceList";
import { apiClient } from "@/components/networking";
import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

import {
  MODE_FILTER_ID,
  PROVIDER_FILTER_ID,
  readModeFilter,
  serializePublicModelHubFilters,
  withFilterValue,
} from "./publicModelHubFilters";

export const PUBLIC_MODEL_HUB_PATH = "/public/v1/model_hub";
export const PUBLIC_MODEL_HUB_PAGE_SIZE = 50;

const QUERY_KEY = ["publicModelHub", "list"] as const;
const DEFAULT_SORTING: SortingState = [{ id: "model_group", desc: false }];

export interface PublicModelHubListResult extends ResourceListResult<ModelGroupInfo> {
  providerValue: string;
  onProviderChange: (value: string) => void;
  modeValues: string[];
  onModesChange: (values: string[]) => void;
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
  const [providerValue, setProviderValue] = useState("");

  const applyProvider = useDebouncedCallback(
    (value: string) => onColumnFiltersChange((previous) => withFilterValue(previous, PROVIDER_FILTER_ID, value)),
    { wait: DEBOUNCE_WAIT_MS },
  );

  const onProviderChange = useCallback(
    (value: string) => {
      setProviderValue(value);
      applyProvider(value);
    },
    [applyProvider],
  );

  const onModesChange = useCallback(
    (values: string[]) => onColumnFiltersChange((previous) => withFilterValue(previous, MODE_FILTER_ID, values)),
    [onColumnFiltersChange],
  );

  return {
    ...list,
    providerValue,
    onProviderChange,
    modeValues: readModeFilter(list.columnFilters),
    onModesChange,
    hasActiveQuery: list.searchValue.trim() !== "" || list.columnFilters.length > 0,
  };
};
