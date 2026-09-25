import { parseAsNumberLiteral, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useMemo } from "react";
import { LOG_FILTERS, LOG_SAMPLE_SIZES, type LogViewerState } from "@/components/GuardrailsMonitor/useLogViewerState";
import { useUrlTab } from "@/hooks/useUrlTab";

const DETAIL_TABS = ["overview", "logs"] as const;

const logParsers = {
  log_filter: parseAsStringLiteral(LOG_FILTERS).withDefault("all"),
  log_sample: parseAsNumberLiteral(LOG_SAMPLE_SIZES).withDefault(10),
  request_id: parseAsString.withOptions({ history: "push" }),
};

export function useGuardrailDetailUrlState() {
  const [tab, setTab] = useUrlTab(DETAIL_TABS, "overview");
  const [{ log_filter, log_sample, request_id }, setLogParams] = useQueryStates(logParsers);

  const logViewerState = useMemo<LogViewerState>(
    () => ({
      filter: log_filter,
      setFilter: (filter) => void setLogParams({ log_filter: filter }),
      sampleSize: log_sample,
      setSampleSize: (size) => void setLogParams({ log_sample: size }),
      requestId: request_id,
      setRequestId: (requestId) =>
        void setLogParams({ request_id: requestId }, requestId === null ? { history: "replace" } : {}),
    }),
    [log_filter, log_sample, request_id, setLogParams],
  );

  const clear = useCallback(() => {
    setTab("overview");
    void setLogParams(null, { history: "replace" });
  }, [setTab, setLogParams]);

  return { tab, setTab, logViewerState, clear };
}
