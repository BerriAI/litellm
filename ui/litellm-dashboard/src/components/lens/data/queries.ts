import { infiniteQueryOptions, keepPreviousData, queryOptions } from "@tanstack/react-query";
import { hasActiveJob } from "../model/status";
import type { Sample, ActivitySelection, Job } from "../model/types";
import type { KeyPage, LensApi } from "./service";

export type { Key } from "./service";

type Activity = Awaited<ReturnType<LensApi["activity"]>>;

export const lensKeys = {
  all: ["lens"] as const,
  lists: () => [...lensKeys.all, "list"] as const,
  list: (scope: string) => [...lensKeys.lists(), { scope }] as const,
  histories: () => [...lensKeys.all, "history"] as const,
  history: (scope: string, lensId: string | undefined, offset: number) =>
    [...lensKeys.histories(), { scope, lensId, offset }] as const,
  runs: () => [...lensKeys.all, "run"] as const,
  run: (scope: string, lensId: string | undefined, batchId: string) =>
    [...lensKeys.runs(), { scope, lensId, batchId }] as const,
  models: (scope: string) => [...lensKeys.all, "models", { scope }] as const,
  modelDetails: (scope: string) => [...lensKeys.all, "model-details", { scope }] as const,
  activity: (scope: string) => [...lensKeys.all, "activity-available", { scope }] as const,
  preview: (scope: string, selection: ActivitySelection, asOf: string) =>
    [...lensKeys.all, "preview", { scope, selection, asOf }] as const,
  analysisKeys: (scope: string, query: string) => [...lensKeys.all, "analysis-keys", { scope, query }] as const,
  analysisKeyInfo: (scope: string, keyId: string | undefined) =>
    [...lensKeys.all, "analysis-key-info", { scope, keyId }] as const,
};

export const lensQueries = {
  list(api: LensApi) {
    return queryOptions({ queryKey: lensKeys.list(api.scope), queryFn: () => api.lenses(), staleTime: 5000 });
  },
  models(api: LensApi) {
    return queryOptions({ queryKey: lensKeys.models(api.scope), queryFn: () => api.models() });
  },
  modelDetails(api: LensApi) {
    return queryOptions({ queryKey: lensKeys.modelDetails(api.scope), queryFn: () => api.modelDetails() });
  },
  activity(api: LensApi, loaded: boolean) {
    const options = {
      queryKey: lensKeys.activity(api.scope),
      queryFn: () => api.activity(),
      enabled: loaded,
      refetchInterval: ({ state }: { state: { data?: Activity } }) => (state.data?.traces ? false : 5000),
    };
    return queryOptions(options);
  },
  history(api: LensApi, { lensId, historyOffset }: { lensId: string | undefined; historyOffset: number }) {
    const options = {
      queryKey: lensKeys.history(api.scope, lensId, historyOffset),
      enabled: !!lensId,
      queryFn: () => api.runs(lensId as string, historyOffset),
      refetchInterval: ({ state }: { state: { data?: Job[] } }) =>
        state.data && hasActiveJob(state.data) ? 10000 : false,
    };
    return queryOptions(options);
  },
  run(api: LensApi, lensId: string | undefined, batchId: string) {
    const options = {
      queryKey: lensKeys.run(api.scope, lensId, batchId),
      enabled: !!lensId && !["latest", "all"].includes(batchId),
      queryFn: () => api.run(lensId as string, batchId),
    };
    return queryOptions(options);
  },
  preview(api: LensApi, { scope, asOf, enabled }: { scope: ActivitySelection; asOf: string; enabled: boolean }) {
    const options = {
      queryKey: lensKeys.preview(api.scope, scope, asOf),
      initialPageParam: "",
      queryFn: ({ pageParam }: { pageParam: string }) => api.sample(scope, pageParam, asOf),
      getNextPageParam: (lastPage: Sample) => lastPage.next_cursor ?? undefined,
      enabled,
      staleTime: 30000,
      gcTime: 60_000,
      placeholderData: keepPreviousData,
    };
    return infiniteQueryOptions(options);
  },
};

export function analysisKeysQuery(api: LensApi, query: string) {
  const options = {
    queryKey: lensKeys.analysisKeys(api.scope, query),
    initialPageParam: 1,
    queryFn: ({ pageParam, signal }: { pageParam: number; signal: AbortSignal }) => api.keys(query, pageParam, signal),
    getNextPageParam: (lastPage: KeyPage, pages: KeyPage[]) =>
      pages.length < lastPage.total_pages ? pages.length + 1 : undefined,
  };
  return infiniteQueryOptions(options);
}

export function analysisKeyInfoQuery(api: LensApi, keyId?: string) {
  const options = {
    queryKey: lensKeys.analysisKeyInfo(api.scope, keyId),
    enabled: !!keyId,
    queryFn: () => api.keyInfo(keyId as string),
  };
  return queryOptions(options);
}
