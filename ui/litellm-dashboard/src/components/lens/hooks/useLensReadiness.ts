"use client";

import { useQuery } from "@tanstack/react-query";
import { isTracingNotEnabled, useTraceAvailability } from "@/components/lens/traces/list/useAgentTraces";
import { useLensAccessToken, useLensApi } from "../data/LensServices";
import { lensQueries } from "../data/queries";
import { readiness, type Readiness, type ReadinessInput } from "../model/readiness";
import { useWorkerConnected } from "./useWorkerConnected";

export interface LensReadiness extends Readiness {
  readonly loading: boolean;
  readonly checking: boolean;
  readonly error: string | undefined;
  refresh(): void;
}

export function useLensReadiness(canInvestigate: boolean): LensReadiness {
  const api = useLensApi();
  const accessToken = useLensAccessToken();
  const traces = useTraceAvailability(accessToken, true);
  const list = useQuery({ ...lensQueries.list(api), enabled: canInvestigate });
  const activity = useQuery(lensQueries.activity(api, canInvestigate && list.isSuccess));
  const connected = useWorkerConnected(list.data?.workers);
  const tracesDisabled = isTracingNotEnabled(traces.error);
  const input: ReadinessInput = {
    traces: {
      recorded: traces.data,
      failed: traces.isError,
      disabled: tracesDisabled,
      checked: traces.isSuccess,
    },
    tracingConfigured: list.data?.tracing_enabled ?? false,
    activity: activity.data,
    activityError: activity.error,
    connected,
    listError: list.error,
    investigations: list.data?.lenses.length ?? 0,
  };
  const state = readiness(input);
  const activityError = state.activityReady ? null : (tracesDisabled ? null : traces.error) || activity.error;
  const error = list.error || activityError;
  const loadingTraces = !state.activityReady && traces.isPending;
  const loadingActivity = !state.activityReady && list.isSuccess && activity.isPending;
  const loadingInvestigations = canInvestigate && (list.isPending || loadingActivity);
  const refresh = () => {
    void traces.refetch();
    if (!canInvestigate) return;
    void list.refetch();
    void activity.refetch();
  };
  return {
    ...state,
    loading: loadingTraces || loadingInvestigations,
    checking: traces.isFetching || list.isFetching || activity.isFetching,
    error: error?.message,
    refresh,
  };
}
