import { useCallback, useEffect, useState } from "react";
import { apiClient } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import {
  observedSettingsSchema,
  observedReportResponseSchema,
  observedStatusSchema,
  type ObservedSettings,
  type ObservedSnapshot,
  type ObservedStatus,
} from "./observedData";

export type ObservedViewData = { settings: ObservedSettings; report: ObservedSnapshot | null; status: ObservedStatus };

export function useObservedReport(accessToken: string) {
  const [data, setData] = useState<ObservedViewData | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const options = { accessToken, signal: controller.signal };
    async function poll(previous?: ObservedViewData) {
      try {
        const status = observedStatusSchema.parse(
          await apiClient.get<unknown>("/roi-calculator/observed/sync", options),
        );
        const finished = previous?.status.running && !status.running;
        const changed = !previous || previous.status.finished_at !== status.finished_at || finished;
        const updated = changed
          ? await Promise.all([
              apiClient
                .get<unknown>("/roi-calculator/observed/settings", options)
                .then((value) => observedSettingsSchema.parse(value)),
              apiClient
                .get<unknown>("/roi-calculator/observed/report", options)
                .then((value) => observedReportResponseSchema.parse(value)),
            ])
          : null;
        if (controller.signal.aborted) return;
        const existing = previous ? { ...previous, status } : null;
        const next = updated ? { settings: updated[0], report: updated[1].report, status } : existing;
        setData(next);
        setError("");
        timer = setTimeout(() => void poll(next ?? undefined), status.running ? 2000 : 30000);
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(extractProxyErrorMessage(reason));
        timer = setTimeout(() => void poll(previous), 10000);
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [accessToken, revision]);
  return { data, error, refresh };
}
