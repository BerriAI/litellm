"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { lensKeys } from "../../data/queries";
import type { LensApi } from "../../data/service";
import { useLensApi } from "../../data/LensServices";
import type { WorkerCreated } from "../../model/types";
import { validateWorkerAddress, analysisAccessSchema, type AnalysisAccess } from "./workerSchema";

export interface WorkerRegistration {
  readonly address: string;
  readonly useExisting: boolean;
  readonly analysisKey: string | null;
  readonly access: AnalysisAccess;
  readonly editingWorker: string | null;
}

export async function createAnalysisKey(api: LensApi, access: AnalysisAccess): Promise<string> {
  const parsed = analysisAccessSchema.safeParse(access);
  if (!parsed.success) throw new Error(parsed.error.issues[0].message);
  const result = await api.generateAnalysisKey(parsed.data);
  if (!result.token_id) throw new Error("The proxy did not return the new key's ID");
  return result.token_id;
}

async function releaseUnusedKey(api: LensApi, keyId: string): Promise<void> {
  const current = await api.lenses();
  if (current.workers.some((worker) => !worker.revoked && worker.analysis_key_id === keyId)) return;
  await api.deleteKeys([keyId]);
}

async function prepareWorker(api: LensApi, registration: WorkerRegistration): Promise<WorkerCreated | null> {
  const { address, useExisting, analysisKey, access, editingWorker } = registration;
  validateWorkerAddress(address);
  const keyId = useExisting ? analysisKey : await createAnalysisKey(api, access);
  const newKey = useExisting ? null : keyId;
  try {
    if (editingWorker) {
      await api.setWorkerBillingKey(editingWorker, keyId);
      return null;
    }
    return await api.registerWorker(keyId);
  } catch (e) {
    const message = e instanceof Error ? e.message : "Could not create credential";
    if (!newKey) throw new Error(message);
    try {
      await releaseUnusedKey(api, newKey);
    } catch {
      throw new Error(
        `${message}. Could not confirm cleanup. Check the Lens analysis key in Virtual Keys before retrying.`,
      );
    }
    throw new Error(message);
  }
}

/** Registers a worker (or re-points its billing key), releasing a freshly minted key if registration fails. */
export function usePrepareWorker() {
  const api = useLensApi();
  const client = useQueryClient();
  return useMutation({
    retry: false,
    mutationFn: (registration: WorkerRegistration) => prepareWorker(api, registration),
    onSettled: () => client.invalidateQueries({ queryKey: lensKeys.list(api.scope) }),
  });
}
