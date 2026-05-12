"use client";

import { useState, useCallback, useEffect } from "react";
import { switchToWorkerUrl, WorkerInfo } from "@/components/networking";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";

const SELECTED_WORKER_KEY = "litellm_selected_worker_id";

interface UseWorkerReturn {
  isControlPlane: boolean;
  workers: WorkerInfo[];
  selectedWorkerId: string | null;
  selectedWorker: WorkerInfo | null;
  selectWorker: (workerId: string) => void;
  disconnectFromWorker: () => void;
}

export const useWorker = (): UseWorkerReturn => {
  const { data: uiConfig } = useUIConfig();
  const isControlPlane = uiConfig?.is_control_plane ?? false;
  const workers: WorkerInfo[] = uiConfig?.workers ?? [];

  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem(SELECTED_WORKER_KEY);
  });

  // Once workers are loaded, restore proxyBaseUrl from the persisted selection
  useEffect(() => {
    if (!selectedWorkerId || workers.length === 0) return;
    const worker = workers.find((w) => w.worker_id === selectedWorkerId);
    if (worker) {
      switchToWorkerUrl(worker.url);
    }
  }, [selectedWorkerId, workers]);

  const selectedWorker =
    workers.find((w) => w.worker_id === selectedWorkerId) ?? null;

  const selectWorker = useCallback(
    (workerId: string) => {
      const worker = workers.find((w) => w.worker_id === workerId);
      if (!worker) return;
      setSelectedWorkerId(workerId);
      localStorage.setItem(SELECTED_WORKER_KEY, workerId);
      switchToWorkerUrl(worker.url);
    },
    [workers],
  );

  const disconnectFromWorker = useCallback(() => {
    setSelectedWorkerId(null);
    localStorage.removeItem(SELECTED_WORKER_KEY);
    switchToWorkerUrl(null);
  }, []);

  return {
    isControlPlane,
    workers,
    selectedWorkerId,
    selectedWorker,
    selectWorker,
    disconnectFromWorker,
  };
};
