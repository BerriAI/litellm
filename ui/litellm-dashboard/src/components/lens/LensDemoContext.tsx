"use client";

import { createContext, useContext } from "react";

export interface LensDemo {
  copyTrace: (traceId: string, spanId?: string) => string;
}

export const LensDemoContext = createContext<LensDemo | null>(null);
export const useLensDemo = () => useContext(LensDemoContext);
