"use client";

import { createContext, useContext } from "react";
import type { ApiClient } from "@/lib/http/client";

export interface LensDemo {
  client: ApiClient;
  copyTrace: (traceId: string, spanId?: string) => string;
}

export const LensDemoContext = createContext<LensDemo | null>(null);
export const useLensDemo = () => useContext(LensDemoContext);
