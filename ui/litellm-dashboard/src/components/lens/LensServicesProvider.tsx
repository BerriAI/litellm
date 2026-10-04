"use client";

import type { ReactNode } from "react";
import { TracesApiContext } from "@/components/view_logs/TraceView/tracesApi";
import { LensApiContext, type LensServices } from "./services";

export function LensServicesProvider({ services, children }: { services: LensServices; children: ReactNode }) {
  return (
    <LensApiContext.Provider value={services.lens}>
      <TracesApiContext.Provider value={services.traces}>{children}</TracesApiContext.Provider>
    </LensApiContext.Provider>
  );
}
