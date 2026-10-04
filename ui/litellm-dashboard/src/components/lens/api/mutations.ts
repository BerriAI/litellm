"use client";

import { useMutation } from "@tanstack/react-query";
import type { Lens, Settings } from "../model/types";

import type { LensApi } from "./service";
import { useLensApi } from "../LensServices";

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
