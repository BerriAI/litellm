"use client";

import { createContext, useContext, useMemo, type ReactNode } from "react";
import { apiClient } from "@/components/networking";
import { liveTracesApi, TracesApiContext, type TracesApi } from "@/components/lens/traces/api";
import { liveLensApi, type LensApi } from "./service";

export interface LensServices {
  readonly accessToken: string;
  readonly lens: LensApi;
  readonly traces: TracesApi;
}

const LensServicesContext = createContext<LensServices | null>(null);

export function liveLensServices(accessToken: string): LensServices {
  return { accessToken, lens: liveLensApi(apiClient, accessToken), traces: liveTracesApi(accessToken) };
}

function useLensServices(): LensServices {
  const provided = useContext(LensServicesContext);
  if (!provided) throw new Error("Lens services need a LensServicesProvider above them");
  return provided;
}

export const useLensApi = (): LensApi => useLensServices().lens;

export const useLensAccessToken = (): string => useLensServices().accessToken;

export function useLiveLensServices(accessToken: string): LensServices {
  return useMemo(() => liveLensServices(accessToken), [accessToken]);
}

export function LensServicesProvider({ services, children }: { services: LensServices; children: ReactNode }) {
  return (
    <LensServicesContext.Provider value={services}>
      <TracesApiContext.Provider value={services.traces}>{children}</TracesApiContext.Provider>
    </LensServicesContext.Provider>
  );
}
