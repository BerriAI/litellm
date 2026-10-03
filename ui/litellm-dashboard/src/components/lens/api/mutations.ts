"use client";

import { validateWorkerAddress, analysisAccessSchema, type AnalysisAccess } from "../setup/worker/workerSchema";

import { useMutation } from "@tanstack/react-query";
import type { Lens, Settings, WorkerCreated } from "../model/types";

import type { LensApi } from "./service";
import { useLensApi } from "../services";

export type LensWrite = (api: LensApi) => Promise<unknown>;

export function useLensUpdate(accessToken: string) {
  const api = useLensApi(accessToken);
  return useMutation({ retry: false, mutationFn: (write: LensWrite) => write(api) });
}

export function useSaveLens(accessToken: string) {
  const api = useLensApi(accessToken);
  return useMutation({
    retry: false,
    mutationFn: ({ id, settings }: { id?: string; settings: Settings }): Promise<Lens> => api.saveLens(id, settings),
  });
}

export function useRevokeWorker(accessToken: string) {
  const api = useLensApi(accessToken);
  return useMutation({ retry: false, mutationFn: (id: string) => api.revokeWorker(id) });
}

export async function createAnalysisKey(api: LensApi, access: AnalysisAccess): Promise<string> {
  const parsed = analysisAccessSchema.safeParse(access);
  if (!parsed.success) throw new Error(parsed.error.issues[0].message);
  const result = await api.generateAnalysisKey(parsed.data);
  if (!result.token_id) throw new Error("The proxy did not return the new key's ID");
  return result.token_id;
}

export function usePrepareWorker(
  accessToken: string,
  { onChanged, onPrepared }: { onChanged: () => void; onPrepared: (created: WorkerCreated | null) => void },
) {
  const api = useLensApi(accessToken);
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
        const keyId = useExisting ? analysisKey : await createAnalysisKey(api, access);
        if (!useExisting) newKey = keyId;
        if (editingWorker) {
          await api.setWorkerBillingKey(editingWorker, keyId);
          onPrepared(null);
          onChanged();
          return null;
        }
        const created = await api.registerWorker(keyId);
        onPrepared(created);
        onChanged();
        return created;
      } catch (e) {
        const message = e instanceof Error ? e.message : "Could not create credential";
        if (newKey) {
          try {
            const current = await api.lenses();
            if (!current.workers.some((worker) => !worker.revoked && worker.analysis_key_id === newKey)) {
              await api.deleteKeys([newKey]);
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
