"use client";

import { useNow } from "@/hooks/useNow";
import { workerConnected } from "../model/status";
import type { LensList } from "../model/types";

const HEARTBEAT_TICK_MS = 5000;

export function useWorkerConnected(workers: LensList["workers"] | null | undefined): boolean {
  const now = useNow(HEARTBEAT_TICK_MS);
  return workers?.some((worker) => workerConnected(worker, now)) ?? false;
}
