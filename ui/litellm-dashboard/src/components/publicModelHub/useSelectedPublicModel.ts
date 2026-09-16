"use client";

import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import type { ResourceListPage } from "@/app/(dashboard)/hooks/common/useResourceList";
import { apiClient } from "@/components/networking";
import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";

import { PUBLIC_MODEL_HUB_PATH } from "./usePublicModelHubList";

const LOOKUP_PAGE_SIZE = 100;

type ModelLookupPage = ResourceListPage<ModelGroupInfo>;

const findModel = (rows: readonly ModelGroupInfo[] | undefined, modelGroup: string | null) =>
  rows?.find((row) => row.model_group === modelGroup);

export interface SelectedPublicModelOptions {
  modelGroup: string | null;
  pageRows: readonly ModelGroupInfo[];
  pageSettled: boolean;
}

export const useSelectedPublicModel = ({
  modelGroup,
  pageRows,
  pageSettled,
}: SelectedPublicModelOptions): ModelGroupInfo | null => {
  const onPage = findModel(pageRows, modelGroup);
  const lookupOptions: UseQueryOptions<ModelLookupPage, Error, ModelLookupPage, readonly unknown[]> = {
    queryKey: ["publicModelHub", "model", modelGroup],
    queryFn: ({ signal }) =>
      apiClient.get<ModelLookupPage>(PUBLIC_MODEL_HUB_PATH, {
        query: { q: modelGroup, page_size: LOOKUP_PAGE_SIZE },
        signal,
      }),
    enabled: modelGroup !== null && pageSettled && onPage === undefined,
    staleTime: Infinity,
  };
  const lookup = useQuery(lookupOptions);
  return onPage ?? findModel(lookup.data?.data, modelGroup) ?? null;
};
