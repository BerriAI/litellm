import { infiniteQueryOptions, queryOptions, type Query } from "@tanstack/react-query";
import type { LensList, Settings, ActivitySelection } from "../model/types";
import type { KeyPage, LensApi } from "./service";

export type { Key } from "./service";

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
  evidence: (scope: string, lensId: string | undefined, evidenceId: string | undefined, offset: number) =>
    [...lensKeys.all, "evidence", { scope, lensId, evidenceId, offset }] as const,
  models: (scope: string) => [...lensKeys.all, "models", { scope }] as const,
  modelDetails: (scope: string) => [...lensKeys.all, "model-details", { scope }] as const,
  activity: (scope: string) => [...lensKeys.all, "activity-available", { scope }] as const,
  discovery: (scope: string, source: Settings["source"], hours: number | undefined, asOf: string) =>
    [...lensKeys.all, "discovery", { scope, source, hours, asOf }] as const,
  preview: (scope: string, selection: ActivitySelection, offset: number, asOf: string) =>
    [...lensKeys.all, "preview", { scope, selection, offset, asOf }] as const,
  agents: (scope: string, asOf: string) => [...lensKeys.all, "agents", { scope, asOf }] as const,
  analysisKeys: (scope: string, query: string) => [...lensKeys.all, "analysis-keys", { scope, query }] as const,
  analysisKeyInfo: (scope: string, keyId: string | undefined) =>
    [...lensKeys.all, "analysis-key-info", { scope, keyId }] as const,
};

export const lensQueries = {
  list(api: LensApi, demo: boolean, workerSetup: boolean) {
    const options = {
      queryKey: lensKeys.list(api.scope),
      queryFn: () => api.lenses(),
      refetchInterval: (current: Query<LensList>): number | false => {
        if (demo) return false;
        const running = current.state.data?.lenses.some((item) =>
          item.jobs.some((job) => ["queued", "running"].includes(job.status)),
        );
        return workerSetup || running ? 2000 : 10000;
      },
    };
    return queryOptions(options);
  },
  models(api: LensApi) {
    return queryOptions({ queryKey: lensKeys.models(api.scope), queryFn: () => api.models() });
  },
  modelDetails(api: LensApi) {
    return queryOptions({ queryKey: lensKeys.modelDetails(api.scope), queryFn: () => api.modelDetails() });
  },
  activity(api: LensApi, loaded: boolean, demo: boolean) {
    const options = {
      queryKey: lensKeys.activity(api.scope),
      queryFn: () => api.activity(),
      enabled: loaded,
      refetchInterval: demo ? (false as const) : 5000,
    };
    return queryOptions(options);
  },
  history(
    api: LensApi,
    { lensId, historyOffset, demo }: { lensId: string | undefined; historyOffset: number; demo: boolean },
  ) {
    const options = {
      queryKey: lensKeys.history(api.scope, lensId, historyOffset),
      enabled: !!lensId,
      queryFn: () => api.runs(lensId as string, historyOffset),
      refetchInterval: demo ? (false as const) : 10000,
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
  evidence(
    api: LensApi,
    {
      lensId,
      evidenceId,
      requestOffset,
      source,
    }: {
      lensId: string | undefined;
      evidenceId: string | undefined;
      requestOffset: number;
      source: string | undefined;
    },
  ) {
    const options = {
      queryKey: lensKeys.evidence(api.scope, lensId, evidenceId, requestOffset),
      enabled: !!lensId && source === "requests",
      queryFn: () => api.execution(lensId as string, evidenceId ?? "", requestOffset),
    };
    return queryOptions(options);
  },
  discovery(api: LensApi, { value, asOf, enabled }: { value: ActivitySelection; asOf: string; enabled: boolean }) {
    const unfiltered = { source: value.source, service: "", filters: [], lookback_hours: value.lookback_hours };
    const options = {
      queryKey: lensKeys.discovery(api.scope, value.source, value.lookback_hours, asOf),
      queryFn: () => api.sample(unfiltered, 0, asOf),
      staleTime: 60000,
      enabled,
    };
    return queryOptions(options);
  },
  preview(
    api: LensApi,
    { scope, offset, asOf, enabled }: { scope: ActivitySelection; offset: number; asOf: string; enabled: boolean },
  ) {
    const options = {
      queryKey: lensKeys.preview(api.scope, scope, offset, asOf),
      queryFn: () => api.sample(scope, offset, asOf),
      enabled,
      staleTime: 30000,
    };
    return queryOptions(options);
  },
  agents(api: LensApi, asOf: string, source: Settings["source"]) {
    const options = {
      queryKey: lensKeys.agents(api.scope, asOf),
      queryFn: () => api.agents(),
      enabled: source !== "requests",
      staleTime: 60000,
    };
    return queryOptions(options);
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
