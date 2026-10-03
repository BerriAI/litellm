"use client";

import { validateWorkerAddress, analysisAccessSchema, type AnalysisAccess } from "../setup/worker/workerSchema";

import { useMutation } from "@tanstack/react-query";
import type { ApiClient } from "@/lib/http/client";
import type { Lens, Settings, WorkerCreated, LensList } from "../model/types";

import { useLensApi } from "./useLensApi";

export function useLensUpdate(accessToken: string) {
  const apiClient = useLensApi();
  return useMutation({
    retry: false,
    mutationFn: ({ path, body, method = "post" }: { path: string; body: unknown; method?: "post" | "put" | "patch" }) =>
      apiClient[method]<unknown>(path, { accessToken, body }),
  });
}

export function useSaveLens(accessToken: string) {
  const apiClient = useLensApi();
  return useMutation({
    retry: false,
    mutationFn: ({ id, settings }: { id?: string; settings: Settings }) =>
      apiClient.request<Lens>(id ? "PUT" : "POST", id ? `/lens/${id}` : "/lens", { accessToken, body: settings }),
  });
}

export function useRevokeWorker(accessToken: string) {
  const apiClient = useLensApi();
  return useMutation({
    retry: false,
    mutationFn: (id: string) => apiClient.delete<unknown>(`/lens/workers/${id}`, { accessToken }),
  });
}

export async function createAnalysisKey(
  apiClient: ApiClient,
  accessToken: string,
  access: AnalysisAccess,
): Promise<string> {
  const parsed = analysisAccessSchema.safeParse(access);
  if (!parsed.success) throw new Error(parsed.error.issues[0].message);
  const result = await apiClient.post<{ token_id?: string }>("/key/generate", {
    accessToken,
    body: {
      key_alias: "Lens analysis",
      models: [parsed.data.model],
      max_budget: parsed.data.budget,
      budget_duration: "1mo",
      metadata: { purpose: "lens" },
    },
  });
  if (!result.token_id) throw new Error("The proxy did not return the new key's ID");
  return result.token_id;
}

export function usePrepareWorker(
  accessToken: string,
  { onChanged, onPrepared }: { onChanged: () => void; onPrepared: (created: WorkerCreated | null) => void },
) {
  const apiClient = useLensApi();
  return useMutation({
    retry: false,
    mutationFn: async ({
      address,
      useExisting,
      analysisKey,
      access,
      editingWorker,
    }: {
      address: string;
      useExisting: boolean;
      analysisKey: string | null;
      access: AnalysisAccess;
      editingWorker: string | null;
    }) => {
      let newKey: string | null = null;
      try {
        validateWorkerAddress(address);
        const keyId = useExisting ? analysisKey : await createAnalysisKey(apiClient, accessToken, access);
        if (!useExisting) newKey = keyId;
        if (editingWorker) {
          await apiClient.put<unknown>(`/lens/workers/${editingWorker}/billing-key`, {
            accessToken,
            body: { analysis_key_id: keyId },
          });
          onPrepared(null);
          onChanged();
          return null;
        }
        const created = await apiClient.post<WorkerCreated>("/lens/workers/register", {
          accessToken,
          body: { name: "Lens worker", analysis_key_id: keyId },
        });
        onPrepared(created);
        onChanged();
        return created;
      } catch (e) {
        const message = e instanceof Error ? e.message : "Could not create credential";
        if (newKey) {
          try {
            const current = await apiClient.get<LensList>("/lens", { accessToken });
            if (!current.workers.some((worker) => !worker.revoked && worker.analysis_key_id === newKey)) {
              await apiClient.post<unknown>("/key/delete", { accessToken, body: { keys: [newKey] } });
            }
            onChanged();
          } catch {
            throw new Error(
              `${message}. Could not confirm cleanup. Check the Lens analysis key in Virtual Keys before retrying.`,
            );
          }
        }
        throw new Error(message);
      }
    },
  });
}
