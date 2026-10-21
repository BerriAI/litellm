"use client";

import { useNow } from "@/hooks/useNow";

import type { LensApi } from "../data/service";
import { queueReason, type QueueReason } from "../model/status";
import type { Job, Lens, LensList } from "../model/types";

export interface QueueContext {
  api: LensApi;
  lenses: readonly Lens[];
  workers: readonly LensList["workers"][number][];
  onConnect: () => void;
  onOpenLens: (id: string) => void;
}

export function useQueueReason(job: Job, queue: QueueContext | undefined): QueueReason | null {
  const now = useNow(1000);
  if (job.status !== "queued" || !queue) return null;
  return queueReason(job, queue.lenses, queue.workers, now);
}
