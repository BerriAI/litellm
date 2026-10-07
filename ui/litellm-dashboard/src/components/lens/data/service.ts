import { z } from "zod";
import type { ApiClient } from "@/lib/http/client";
import { getAuthHeaderName } from "@/lib/http/runtime";
import type { Client } from "openapi-fetch";
import type { components, paths } from "@/lib/http/schema";
import { liveDatasetsApi, type DatasetsApi } from "../datasets/client";
import type {
  ActivitySelection,
  AnalysisModelInfo,
  Job,
  Lens,
  LensList,
  RunWindow,
  Sample,
  Settings,
  SignalConfig,
  WorkerCreated,
} from "../model/types";

export type ExecutionContent = components["schemas"]["ExecutionContent"];
export type FindingStatus = components["schemas"]["FindingUpdate"]["status"];

const keySchema = z.object({ token: z.string(), key_alias: z.string().nullable().optional() });
const keyPageSchema = z.object({ keys: z.array(keySchema), total_pages: z.number() });
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
export type Key = z.infer<typeof keySchema>;
export type KeyPage = z.infer<typeof keyPageSchema>;
export type KeyInfo = z.infer<typeof keyInfoSchema>["info"];

export type ReviewPage = components["schemas"]["ReviewPage"];

export interface AnalysisKeyRequest {
  readonly model: string;
  readonly budget: number;
}

export interface LensApi {
  /** Partitions query caches between backends (one token, or the demo). */
  readonly scope: string;
  readonly datasets: DatasetsApi;
  lenses(): Promise<LensList>;
  activity(): Promise<components["schemas"]["ActivityAvailability"]>;
  runs(lensId: string, offset: number): Promise<Job[]>;
  run(lensId: string, jobId: string): Promise<Job>;
  reviews(lensId: string, jobId: string, after: number): Promise<ReviewPage>;
  execution(lensId: string, executionId: string, offset: number): Promise<ExecutionContent>;
  sample(selection: ActivitySelection, offset: number, asOf: string): Promise<Sample>;
  agents(): Promise<string[]>;
  models(): Promise<{ data: { id: string }[] }>;
  modelDetails(): Promise<{ data: AnalysisModelInfo[] }>;
  keys(alias: string, page: number, signal: AbortSignal): Promise<KeyPage>;
  keyInfo(keyId: string): Promise<KeyInfo>;
  saveLens(id: string | undefined, settings: Settings): Promise<Lens>;
  startRun(lensId: string, request?: RunWindow): Promise<void>;
  watchAll(): Promise<components["schemas"]["WatchAllResult"]>;
  signalConfig(): Promise<SignalConfig>;
  saveSignalConfig(config: SignalConfig): Promise<SignalConfig>;
  cancelRun(lensId: string): Promise<void>;
  reviewFinding(lensId: string, findingId: string, status: FindingStatus, reason: string): Promise<void>;
  registerWorker(analysisKeyId: string): Promise<WorkerCreated>;
  setWorkerBillingKey(workerId: string, analysisKeyId: string): Promise<void>;
  revokeWorker(workerId: string): Promise<void>;
  generateAnalysisKey(request: AnalysisKeyRequest): Promise<{ token_id?: string }>;
  deleteKeys(keys: readonly string[]): Promise<void>;
}

type LensClient = Client<paths>;

async function required<T>(request: Promise<{ data?: T }>): Promise<T> {
  const { data } = await request;
  if (data === undefined) throw new Error("The proxy returned an empty response");
  return data;
}

async function sent(request: Promise<unknown>): Promise<void> {
  await request;
}

export function liveLensApi(client: LensClient, apiClient: ApiClient, accessToken: string): LensApi {
  const headers = { [getAuthHeaderName()]: `Bearer ${accessToken}` };
  const lens = (lens_id: string) => ({ headers, params: { path: { lens_id } } });
  const worker = (worker_id: string) => ({ headers, params: { path: { worker_id } } });
  return {
    scope: accessToken,
    datasets: liveDatasetsApi(client, apiClient, accessToken),
    lenses: () => required(client.GET("/lens", { headers })),
    activity: () => required(client.GET("/lens/activity/available", { headers })),
    runs: (lensId, offset) =>
      required(
        client.GET("/lens/{lens_id}/runs", { headers, params: { path: { lens_id: lensId }, query: { offset } } }),
      ),
    run: (lensId, jobId) =>
      required(
        client.GET("/lens/{lens_id}/runs/{job_id}", {
          headers,
          params: { path: { lens_id: lensId, job_id: jobId } },
        }),
      ),
    reviews: (lensId, jobId, after) =>
      required(
        client.GET("/lens/{lens_id}/runs/{job_id}/reviews", {
          headers,
          params: { path: { lens_id: lensId, job_id: jobId }, query: { after } },
        }),
      ),
    execution: (lensId, executionId, offset) =>
      required(
        client.GET("/lens/{lens_id}/executions/{execution_id}", {
          headers,
          params: { path: { lens_id: lensId, execution_id: executionId }, query: { offset } },
        }),
      ),
    sample: (selection, offset, asOf) =>
      required(
        client.POST("/lens/preview/sample", {
          headers,
          body: {
            offset,
            as_of: asOf,
            selection: {
              source: selection.source,
              service: selection.service ?? "",
              agent_name: selection.agent_name ?? "",
              filters: selection.filters ?? [],
              sample_size: selection.sample_size,
              sample_percent: selection.sample_percent ?? 100,
              team_id: selection.team_id ?? "",
              execution_ids: [],
            },
            lookback_hours: selection.lookback_hours ?? 24,
          },
        }),
      ),
    agents: () => required(client.GET("/lens/agents", { headers })),
    models: () => apiClient.get("/models", { accessToken }),
    modelDetails: () => apiClient.get("/model_group/info", { accessToken }),
    keys: async (alias, page, signal) =>
      keyPageSchema.parse(
        await apiClient.get("/key/list", {
          accessToken,
          signal,
          query: {
            page: String(page),
            size: "25",
            return_full_object: "true",
            key_alias: alias || undefined,
            substring_matching: "true",
            include_team_keys: "true",
            include_created_by_keys: "true",
            status: "active",
          },
        }),
      ),
    keyInfo: async (keyId) =>
      keyInfoSchema.parse(await apiClient.get("/key/info", { accessToken, query: { key: keyId } })).info,
    saveLens: (id, settings) =>
      required(
        id
          ? client.PUT("/lens/{lens_id}", { ...lens(id), body: settings })
          : client.POST("/lens", { headers, body: settings }),
      ),
    startRun: (lensId, request = {}) => sent(client.POST("/lens/{lens_id}/runs", { ...lens(lensId), body: request })),
    watchAll: () => required(client.POST("/lens/watch-all", { headers })),
    signalConfig: () => required(client.GET("/lens/signals", { headers })),
    saveSignalConfig: (config) => required(client.PUT("/lens/signals", { headers, body: config })),
    cancelRun: (lensId) => sent(client.POST("/lens/{lens_id}/cancel", lens(lensId))),
    reviewFinding: (lensId, findingId, status, reason) =>
      sent(
        client.PATCH("/lens/{lens_id}/findings/{finding_id}", {
          headers,
          params: { path: { lens_id: lensId, finding_id: findingId } },
          body: { status, reason },
        }),
      ),
    registerWorker: (analysisKeyId) =>
      required(
        client.POST("/lens/workers/register", {
          headers,
          body: { name: "Lens worker", analysis_key_id: analysisKeyId },
        }),
      ),
    setWorkerBillingKey: (workerId, analysisKeyId) =>
      sent(
        client.PUT("/lens/workers/{worker_id}/billing-key", {
          ...worker(workerId),
          body: { analysis_key_id: analysisKeyId },
        }),
      ),
    revokeWorker: (workerId) => sent(client.DELETE("/lens/workers/{worker_id}", worker(workerId))),
    generateAnalysisKey: (request) =>
      apiClient.post<{ token_id?: string }>("/key/generate", {
        accessToken,
        body: {
          key_alias: "Lens analysis",
          models: [request.model],
          max_budget: request.budget,
          budget_duration: "1mo",
          metadata: { purpose: "lens" },
        },
      }),
    deleteKeys: (keys) => apiClient.post("/key/delete", { accessToken, body: { keys } }),
  };
}
