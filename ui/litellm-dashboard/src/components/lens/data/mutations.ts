"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { Lens, LensList, Settings } from "../model/types";

import type { LensApi } from "./service";
import { lensKeys } from "./queries";
import { useLensApi } from "./LensServices";

export type LensWrite = (api: LensApi) => Promise<unknown>;

export function useInvalidateLenses() {
  const api = useLensApi();
  const client = useQueryClient();
  return () =>
    Promise.all([
      client.invalidateQueries({ queryKey: lensKeys.list(api.scope) }),
      client.invalidateQueries({ queryKey: lensKeys.histories() }),
    ]);
}

export function useLensUpdate() {
  const api = useLensApi();
  const invalidate = useInvalidateLenses();
  return useMutation({ retry: false, mutationFn: (write: LensWrite) => write(api), onSettled: invalidate });
}

function upsertLens(list: LensList | undefined, saved: Lens): LensList | undefined {
  if (!list) return list;
  const known = list.lenses.some((lens) => lens.id === saved.id);
  const lenses = known ? list.lenses.map((lens) => (lens.id === saved.id ? saved : lens)) : [...list.lenses, saved];
  return { ...list, lenses };
}

export function useSaveLens() {
  const api = useLensApi();
  const client = useQueryClient();
  return useMutation({
    retry: false,
    mutationFn: ({ id, settings }: { id?: string; settings: Settings }): Promise<Lens> => api.saveLens(id, settings),
    onSuccess: (saved) => {
      client.setQueryData<LensList>(lensKeys.list(api.scope), (current) => upsertLens(current, saved));
      return client.invalidateQueries({ queryKey: lensKeys.list(api.scope) });
    },
  });
}

export function useRevokeWorker() {
  const api = useLensApi();
  const client = useQueryClient();
  return useMutation({
    retry: false,
    mutationFn: (id: string) => api.revokeWorker(id),
    onSettled: () => client.invalidateQueries({ queryKey: lensKeys.list(api.scope) }),
  });
}
