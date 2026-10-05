export interface Readiness {
  readonly tracingEnabled: boolean;
  readonly tracesReady: boolean;
  readonly requestsReady: boolean;
  readonly activityReady: boolean;
  readonly connected: boolean;
  readonly hasInvestigations: boolean;
  readonly ready: boolean;
}

export interface ReadinessInput {
  readonly traces: {
    readonly recorded: boolean | undefined;
    readonly failed: boolean;
    readonly disabled: boolean;
    readonly checked: boolean;
  };
  readonly tracingConfigured: boolean;
  readonly activity: { readonly traces: boolean; readonly requests: boolean } | undefined;
  readonly activityError: unknown;
  readonly connected: boolean;
  readonly listError: unknown;
  readonly investigations: number;
}

export function readiness(input: ReadinessInput): Readiness {
  const { traces, activity, activityError, connected, listError } = input;
  /** Traces confirmed straight from trace storage count as recorded activity even if the activity check fails. */
  const tracesSeen = traces.recorded === true && !traces.failed;
  const tracesReady = tracesSeen || (activity?.traces === true && !activityError);
  const requestsReady = activity?.requests === true && !activityError;
  const activityReady = tracesReady || requestsReady;
  return {
    tracingEnabled: !traces.disabled && (traces.checked || input.tracingConfigured),
    tracesReady,
    requestsReady,
    activityReady,
    connected,
    hasInvestigations: input.investigations > 0,
    ready: activityReady && connected && !listError,
  };
}

export function initialSetupStep(state: Readiness): number {
  if (state.activityReady) return state.connected ? 3 : 2;
  return state.tracingEnabled ? 1 : 0;
}
