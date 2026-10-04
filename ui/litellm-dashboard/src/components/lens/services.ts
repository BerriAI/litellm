"use client";

import { createContext, useContext, useMemo } from "react";
import { apiClient } from "@/components/networking";
import type { TracesApi } from "@/components/view_logs/TraceView/tracesApi";
import { liveLensApi, type LensApi } from "./api/service";

export interface LensServices {
  readonly lens: LensApi;
  readonly traces: TracesApi;
}

export const LensApiContext = createContext<LensApi | null>(null);

/** Without a provider, falls back to the live HTTP implementation for the caller's token. */
export function useLensApi(accessToken: string): LensApi {
  const provided = useContext(LensApiContext);
  const live = useMemo(() => liveLensApi(apiClient, accessToken), [accessToken]);
  return provided ?? live;
}
