import { useQuery } from "@tanstack/react-query";
import { useNow } from "@/hooks/useNow";
import { isTracingNotEnabled, useTraceAvailability } from "@/components/view_logs/TraceView/useAgentTraces";
import { lensQueries } from "../api/queries";
import { workerConnected } from "../model/status";
import { useLensApi } from "../services";

function traceSetupState(traces: ReturnType<typeof useTraceAvailability>, configured: boolean) {
  const disabled = isTracingNotEnabled(traces.error);
  return {
    tracesReady: traces.data === true && !traces.error,
    tracingEnabled: !disabled && (traces.isSuccess || configured),
    missingTraces: disabled || traces.data === false,
    error: disabled ? null : traces.error,
  };
}

export function useLensSetup(accessToken: string, enabled: boolean, canInvestigate: boolean, settingUp: boolean) {
  const api = useLensApi(accessToken);
  const now = useNow(2000);
  const traces = useTraceAvailability(accessToken, enabled);
  const list = useQuery({ ...lensQueries.list(api, settingUp), enabled: enabled && canInvestigate });
  const activity = useQuery(lensQueries.activity(api, enabled && canInvestigate && list.isSuccess));
  const data = list.data ?? { lenses: [], workers: [], tracing_enabled: false };
  const traceState = traceSetupState(traces, data.tracing_enabled);
  const hasRequests = activity.data?.requests === true;
  const requestsReady = hasRequests && !activity.error;
  const connected = data.workers.some((worker) => workerConnected(worker, now));
  const hasInvestigations = data.lenses.length > 0;
  const activityReady = traceState.tracesReady || requestsReady;
  const activityError = activityReady ? null : traceState.error || activity.error;
  const error = list.error || activityError;
  const loadingActivity = !activityReady && list.isSuccess && activity.isPending;
  const loadingInvestigations = canInvestigate && (list.isPending || loadingActivity);
  const refresh = () => {
    void traces.refetch();
    if (canInvestigate) {
      void list.refetch();
      void activity.refetch();
    }
  };
  return {
    tracingEnabled: traceState.tracingEnabled,
    tracesReady: traceState.tracesReady,
    requestsReady,
    hasRequests,
    connected,
    hasInvestigations,
    missingTraces: traceState.missingTraces,
    loading: (!activityReady && traces.isPending) || loadingInvestigations,
    checking: traces.isFetching || list.isFetching || activity.isFetching,
    ready: activityReady && connected && !error,
    workers: data.workers,
    error: error?.message,
    refresh,
  };
}

export type LensSetupState = ReturnType<typeof useLensSetup>;
