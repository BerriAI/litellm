"use client";

import { queryOptions, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { ApiError } from "@/lib/http/client";

import { lensKeys } from "../data/queries";
import { useLensApi } from "../data/LensServices";
import type { DatasetsApi } from "./client";
import type { BuildRequest, Dataset, DatasetCreate, RevisionSave } from "./types";

export const datasetKeys = {
  all: () => [...lensKeys.all, "datasets"] as const,
  lists: () => [...datasetKeys.all(), "list"] as const,
  list: (scope: string) => [...datasetKeys.lists(), { scope }] as const,
  details: () => [...datasetKeys.all(), "detail"] as const,
  detail: (scope: string, datasetId: string, revision: number | null) =>
    [...datasetKeys.details(), { scope, datasetId, revision }] as const,
  build: (scope: string, request: BuildRequest) => [...datasetKeys.all(), "build", { scope, request }] as const,
};

export const datasetQueries = {
  list(api: DatasetsApi, scope: string) {
    return queryOptions({ queryKey: datasetKeys.list(scope), queryFn: () => api.list() });
  },
  detail(api: DatasetsApi, scope: string, datasetId: string | null, revision: number | null = null) {
    return queryOptions({
      queryKey: datasetKeys.detail(scope, datasetId ?? "", revision),
      queryFn: () => api.get(datasetId ?? "", revision ?? undefined),
      enabled: !!datasetId,
    });
  },
  build(api: DatasetsApi, scope: string, request: BuildRequest, enabled = true) {
    const options = {
      queryKey: datasetKeys.build(scope, request),
      queryFn: () => api.build(request),
      enabled,
      staleTime: Infinity,
      gcTime: 0,
    };
    return queryOptions(options);
  },
};

export const isRevisionConflict = (error: unknown): boolean => error instanceof ApiError && error.status === 409;

export function useDatasets() {
  const api = useLensApi();
  return useQuery(datasetQueries.list(api.datasets, api.scope));
}

export function useDataset(datasetId: string | null, revision: number | null = null) {
  const api = useLensApi();
  return useQuery(datasetQueries.detail(api.datasets, api.scope, datasetId, revision));
}

export function useBuildCases(request: BuildRequest, enabled = true) {
  const api = useLensApi();
  return useQuery(datasetQueries.build(api.datasets, api.scope, request, enabled));
}

export function useInvalidateDatasets() {
  const client = useQueryClient();
  return () =>
    Promise.all([
      client.invalidateQueries({ queryKey: datasetKeys.lists() }),
      client.invalidateQueries({ queryKey: datasetKeys.details() }),
    ]);
}

export function useCreateDataset() {
  const api = useLensApi();
  const invalidate = useInvalidateDatasets();
  return useMutation({
    retry: false,
    mutationFn: (body: DatasetCreate) => api.datasets.create(body),
    onSettled: invalidate,
  });
}

export function useSaveRevision() {
  const api = useLensApi();
  const invalidate = useInvalidateDatasets();
  return useMutation({
    retry: false,
    mutationFn: ({ datasetId, body }: { datasetId: string; body: RevisionSave }) =>
      api.datasets.saveRevision(datasetId, body),
    onSettled: invalidate,
  });
}

export async function downloadDataset(api: DatasetsApi, dataset: Pick<Dataset, "id" | "name" | "revision">) {
  const blob = await api.exportJsonl(dataset.id, dataset.revision);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${dataset.name || dataset.id}-r${dataset.revision}.jsonl`;
  link.click();
  URL.revokeObjectURL(url);
}

export function useExportDataset() {
  const api = useLensApi();
  return useMutation({
    retry: false,
    mutationFn: (dataset: Pick<Dataset, "id" | "name" | "revision">) => downloadDataset(api.datasets, dataset),
  });
}
