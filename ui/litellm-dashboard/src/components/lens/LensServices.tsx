"use client";

import { createContext, useContext, useMemo, type ReactNode } from "react";
import { apiClient } from "@/components/networking";
import { liveTracesApi, TracesApiContext, type TracesApi } from "@/components/view_logs/TraceView/tracesApi";
import { liveLensApi, type LensApi } from "./api/service";

export interface LensServices {
  readonly lens: LensApi;
  readonly traces: TracesApi;
}

const LensApiContext = createContext<LensApi | null>(null);

export function liveLensServices(accessToken: string): LensServices {
  return { lens: liveLensApi(apiClient, accessToken), traces: liveTracesApi(accessToken) };
}

export function useLensApi(): LensApi {
  const provided = useContext(LensApiContext);
  if (!provided) throw new Error("useLensApi needs a LensServicesProvider above it");
  return provided;
}

export function useLiveLensServices(accessToken: string): LensServices {
  return useMemo(() => liveLensServices(accessToken), [accessToken]);
}

export function LensServicesProvider({ services, children }: { services: LensServices; children: ReactNode }) {
  return (
    <LensApiContext.Provider value={services.lens}>
      <TracesApiContext.Provider value={services.traces}>{children}</TracesApiContext.Provider>
    </LensApiContext.Provider>
  );
}
