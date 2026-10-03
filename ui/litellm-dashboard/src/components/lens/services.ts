"use client";

import { createContext, useContext, useMemo } from "react";
import { apiClient } from "@/components/networking";
import { liveTracesApi, type TracesApi } from "@/components/view_logs/TraceView/tracesApi";
import { liveLensApi, type LensApi } from "./api/service";

/** Without a provider, hooks fall back to the live HTTP implementations for the caller's token. */
export interface LensServices {
  readonly lens: LensApi;
  readonly traces: TracesApi;
}

export const LensServicesContext = createContext<LensServices | null>(null);

export function useLensApi(accessToken: string): LensApi {
  const provided = useContext(LensServicesContext)?.lens;
  const live = useMemo(() => liveLensApi(apiClient, accessToken), [accessToken]);
  return provided ?? live;
}

export function useTracesApi(accessToken: string): TracesApi {
  const provided = useContext(LensServicesContext)?.traces;
  const live = useMemo(() => liveTracesApi(accessToken), [accessToken]);
  return provided ?? live;
}
