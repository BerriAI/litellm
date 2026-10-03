import { infiniteQueryOptions, queryOptions, type Query } from "@tanstack/react-query";
import { z } from "zod";
import type { ApiClient } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import { type LensList, type Job, type Sample, type Settings, type ActivitySelection } from "../model/types";
import type { AnalysisModelInfo } from "../setup/fields/analysisModels";

export const lensKeys = {
  all: ["lens"] as const,
  lists: () => [...lensKeys.all, "list"] as const,
  list: (accessToken: string) => [...lensKeys.lists(), { accessToken }] as const,
  histories: () => [...lensKeys.all, "history"] as const,
  history: (accessToken: string, lensId: string | undefined, offset: number) =>
    [...lensKeys.histories(), { accessToken, lensId, offset }] as const,
  runs: () => [...lensKeys.all, "run"] as const,
  run: (accessToken: string, lensId: string | undefined, batchId: string) =>
    [...lensKeys.runs(), { accessToken, lensId, batchId }] as const,
  evidence: (accessToken: string, lensId: string | undefined, evidenceId: string | undefined, offset: number) =>
    [...lensKeys.all, "evidence", { accessToken, lensId, evidenceId, offset }] as const,
  models: (accessToken: string) => [...lensKeys.all, "models", { accessToken }] as const,
  modelDetails: (accessToken: string) => [...lensKeys.all, "model-details", { accessToken }] as const,
  activity: (accessToken: string) => [...lensKeys.all, "activity-available", { accessToken }] as const,
  discovery: (accessToken: string, source: Settings["source"], hours: number | undefined, asOf: string) =>
    [...lensKeys.all, "discovery", { accessToken, source, hours, asOf }] as const,
  preview: (accessToken: string, scope: ActivitySelection, offset: number, asOf: string) =>
    [...lensKeys.all, "preview", { accessToken, scope, offset, asOf }] as const,
  agents: (accessToken: string, asOf: string) => [...lensKeys.all, "agents", { accessToken, asOf }] as const,
  analysisKeys: (accessToken: string, query: string) =>
    [...lensKeys.all, "analysis-keys", { accessToken, query }] as const,
  analysisKeyInfo: (accessToken: string, keyId: string | undefined) =>
    [...lensKeys.all, "analysis-key-info", { accessToken, keyId }] as const,
};

export const lensQueries = {
  list(apiClient: ApiClient, accessToken: string, demo: boolean, workerSetup: boolean) {
    const options = {
      queryKey: lensKeys.list(accessToken),
      queryFn: () => apiClient.get<LensList>("/lens", { accessToken }),
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
  models(apiClient: ApiClient, accessToken: string) {
    const options = {
      queryKey: lensKeys.models(accessToken),
      queryFn: () => apiClient.get<{ data: { id: string }[] }>("/models", { accessToken }),
    };
    return queryOptions(options);
  },
  modelDetails(apiClient: ApiClient, accessToken: string) {
    const options = {
      queryKey: lensKeys.modelDetails(accessToken),
      queryFn: () => apiClient.get<{ data: AnalysisModelInfo[] }>("/model_group/info", { accessToken }),
    };
    return queryOptions(options);
  },
  activity(apiClient: ApiClient, accessToken: string, loaded: boolean, demo: boolean) {
    const options = {
      queryKey: lensKeys.activity(accessToken),
      queryFn: () => apiClient.get<{ traces: boolean; requests: boolean }>("/lens/activity/available", { accessToken }),
      enabled: loaded,
      refetchInterval: demo ? (false as const) : 5000,
    };
    return queryOptions(options);
  },
  history(
    apiClient: ApiClient,
    accessToken: string,
    { lensId, historyOffset, demo }: { lensId: string | undefined; historyOffset: number; demo: boolean },
  ) {
    const options = {
      queryKey: lensKeys.history(accessToken, lensId, historyOffset),
      enabled: !!lensId,
      queryFn: () => apiClient.get<Job[]>(`/lens/${lensId}/runs`, { accessToken, query: { offset: historyOffset } }),
      refetchInterval: demo ? (false as const) : 10000,
    };
    return queryOptions(options);
  },
  run(apiClient: ApiClient, accessToken: string, lensId: string | undefined, batchId: string) {
    const options = {
      queryKey: lensKeys.run(accessToken, lensId, batchId),
      enabled: !!lensId && !["latest", "all"].includes(batchId),
      queryFn: () => apiClient.get<Job>(`/lens/${lensId}/runs/${batchId}`, { accessToken }),
    };
    return queryOptions(options);
  },
  evidence(
    apiClient: ApiClient,
    accessToken: string,
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
      queryKey: lensKeys.evidence(accessToken, lensId, evidenceId, requestOffset),
      enabled: !!lensId && source === "requests",
      queryFn: () =>
        apiClient.get<components["schemas"]["ExecutionContent"]>(
          `/lens/${lensId}/executions/${encodeURIComponent(evidenceId ?? "")}`,
          { accessToken, query: { offset: requestOffset } },
        ),
    };
    return queryOptions(options);
  },
  sample(
    apiClient: ApiClient,
    accessToken: string,
    { selection, pageOffset, asOf }: { selection: ActivitySelection; pageOffset: number; asOf: string },
  ) {
    const { lookback_hours, ...selectionSettings } = selection;
    return apiClient.post<Sample>("/lens/preview/sample", {
      accessToken,
      body: {
        offset: pageOffset,
        as_of: asOf,
        settings: {
          ...selectionSettings,
          execution_ids: [],
          name: "Preview",
          model: "preview",
          checks: [{ id: "preview", instruction: "Preview recorded activity" }],
        },
        lookback_hours: lookback_hours ?? 24,
      },
    });
  },
  discovery(
    apiClient: ApiClient,
    accessToken: string,
    { value, asOf, enabled }: { value: ActivitySelection; asOf: string; enabled: boolean },
  ) {
    const options = {
      queryKey: lensKeys.discovery(accessToken, value.source, value.lookback_hours, asOf),
      queryFn: () =>
        lensQueries.sample(apiClient, accessToken, {
          selection: {
            source: value.source,
            service: "",
            filters: [],
            lookback_hours: value.lookback_hours,
          },
          pageOffset: 0,
          asOf,
        }),
      staleTime: 60000,
      enabled,
    };
    return queryOptions(options);
  },
  preview(
    apiClient: ApiClient,
    accessToken: string,
    { scope, offset, asOf, enabled }: { scope: ActivitySelection; offset: number; asOf: string; enabled: boolean },
  ) {
    const options = {
      queryKey: lensKeys.preview(accessToken, scope, offset, asOf),
      queryFn: () => lensQueries.sample(apiClient, accessToken, { selection: scope, pageOffset: offset, asOf }),
      enabled,
      staleTime: 30000,
    };
    return queryOptions(options);
  },
  agents(apiClient: ApiClient, accessToken: string, asOf: string, source: Settings["source"]) {
    const options = {
      queryKey: lensKeys.agents(accessToken, asOf),
      queryFn: () => apiClient.get<string[]>("/lens/agents", { accessToken }),
      enabled: source !== "requests",
      staleTime: 60000,
    };
    return queryOptions(options);
  },
};

const keySchema = z.object({ token: z.string(), key_alias: z.string().nullable().optional() });
const pageSchema = z.object({ keys: z.array(keySchema), total_pages: z.number() });
export type Key = z.infer<typeof keySchema>;

export function analysisKeysQuery(apiClient: ApiClient, accessToken: string, query: string) {
  const options = {
    queryKey: lensKeys.analysisKeys(accessToken, query),
    initialPageParam: 1,
    queryFn: async ({ pageParam, signal }: { pageParam: number; signal: AbortSignal }) =>
      pageSchema.parse(
        await apiClient.get("/key/list", {
          accessToken,
          signal,
          query: {
            page: String(pageParam),
            size: "25",
            return_full_object: "true",
            key_alias: query || undefined,
            substring_matching: "true",
            include_team_keys: "true",
            include_created_by_keys: "true",
            status: "active",
          },
        }),
      ),
    getNextPageParam: (lastPage: z.infer<typeof pageSchema>, pages: z.infer<typeof pageSchema>[]) =>
      pages.length < lastPage.total_pages ? pages.length + 1 : undefined,
  };
  return infiniteQueryOptions(options);
}

const keyInfoFields = {
  key_alias: z.string().nullable().optional(),
  models: z.array(z.string()),
  max_budget: z.number().nullable(),
  budget_duration: z.string().nullable().optional(),
  rpm_limit: z.number().nullable().optional(),
  tpm_limit: z.number().nullable().optional(),
  expires: z.string().nullable().optional(),
  status: z.string().optional(),
};
const keyInfoSchema = z.object({ info: z.object(keyInfoFields) });

export function analysisKeyInfoQuery(apiClient: ApiClient, accessToken: string, keyId?: string) {
  const options = {
    queryKey: lensKeys.analysisKeyInfo(accessToken, keyId),
    enabled: !!keyId,
    queryFn: async () =>
      keyInfoSchema.parse(await apiClient.get("/key/info", { accessToken, query: { key: keyId } })).info,
  };
  return queryOptions(options);
}
