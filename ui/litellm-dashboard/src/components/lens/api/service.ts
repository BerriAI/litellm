import { z } from "zod";
import type { ApiClient } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import type { ActivitySelection, Job, Lens, LensList, Sample, Settings, WorkerCreated } from "../model/types";
import type { AnalysisModelInfo } from "../setup/fields/analysisModels";

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

export interface AnalysisKeyRequest {
  readonly model: string;
  readonly budget: number;
}

export interface LensApi {
  /** Partitions query caches between backends (one token, or the demo). */
  readonly scope: string;
  lenses(): Promise<LensList>;
  activity(): Promise<{ traces: boolean; requests: boolean }>;
  runs(lensId: string, offset: number): Promise<Job[]>;
  run(lensId: string, jobId: string): Promise<Job>;
  execution(lensId: string, executionId: string, offset: number): Promise<ExecutionContent>;
  sample(selection: ActivitySelection, offset: number, asOf: string): Promise<Sample>;
  agents(): Promise<string[]>;
  models(): Promise<{ data: { id: string }[] }>;
  modelDetails(): Promise<{ data: AnalysisModelInfo[] }>;
  keys(alias: string, page: number, signal: AbortSignal): Promise<KeyPage>;
  keyInfo(keyId: string): Promise<KeyInfo>;
  saveLens(id: string | undefined, settings: Settings): Promise<Lens>;
  startRun(lensId: string): Promise<void>;
  cancelRun(lensId: string): Promise<void>;
  reviewFinding(lensId: string, findingId: string, status: FindingStatus, reason: string): Promise<void>;
  registerWorker(analysisKeyId: string | null): Promise<WorkerCreated>;
  setWorkerBillingKey(workerId: string, analysisKeyId: string | null): Promise<void>;
  revokeWorker(workerId: string): Promise<void>;
  generateAnalysisKey(request: AnalysisKeyRequest): Promise<{ token_id?: string }>;
  deleteKeys(keys: readonly string[]): Promise<void>;
}

export function liveLensApi(apiClient: ApiClient, accessToken: string): LensApi {
  const encode = encodeURIComponent;
  return {
    scope: accessToken,
    lenses: () => apiClient.get<LensList>("/lens", { accessToken }),
    activity: () => apiClient.get("/lens/activity/available", { accessToken }),
    runs: (lensId, offset) => apiClient.get<Job[]>(`/lens/${lensId}/runs`, { accessToken, query: { offset } }),
    run: (lensId, jobId) => apiClient.get<Job>(`/lens/${lensId}/runs/${jobId}`, { accessToken }),
    execution: (lensId, executionId, offset) =>
      apiClient.get<ExecutionContent>(`/lens/${lensId}/executions/${encode(executionId)}`, {
        accessToken,
        query: { offset },
      }),
    sample: (selection, offset, asOf) => {
      const { lookback_hours, ...selectionSettings } = selection;
      return apiClient.post<Sample>("/lens/preview/sample", {
        accessToken,
        body: {
          offset,
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
    agents: () => apiClient.get<string[]>("/lens/agents", { accessToken }),
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
      apiClient.request<Lens>(id ? "PUT" : "POST", id ? `/lens/${id}` : "/lens", { accessToken, body: settings }),
    startRun: (lensId) => apiClient.post(`/lens/${lensId}/runs`, { accessToken, body: {} }),
    cancelRun: (lensId) => apiClient.post(`/lens/${lensId}/cancel`, { accessToken, body: {} }),
    reviewFinding: (lensId, findingId, status, reason) =>
      apiClient.patch(`/lens/${lensId}/findings/${findingId}`, { accessToken, body: { status, reason } }),
    registerWorker: (analysisKeyId) =>
      apiClient.post<WorkerCreated>("/lens/workers/register", {
        accessToken,
        body: { name: "Lens worker", analysis_key_id: analysisKeyId },
      }),
    setWorkerBillingKey: (workerId, analysisKeyId) =>
      apiClient.put(`/lens/workers/${workerId}/billing-key`, { accessToken, body: { analysis_key_id: analysisKeyId } }),
    revokeWorker: (workerId) => apiClient.delete(`/lens/workers/${workerId}`, { accessToken }),
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
