"use client";

import { createContext, useContext } from "react";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { keepTerminology, toSiTerminology, type Term } from "@/lib/terminology";

const TerminologyContext = createContext<Term>(keepTerminology);

export function TerminologyProvider({ children }: { children: React.ReactNode }) {
  const { data: uiConfig } = useUIConfig();
  return (
    <TerminologyContext.Provider value={uiConfig?.use_si_terminology ? toSiTerminology : keepTerminology}>
      {children}
    </TerminologyContext.Provider>
  );
}

export const useTerminology = (): Term => useContext(TerminologyContext);
