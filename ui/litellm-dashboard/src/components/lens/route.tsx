"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";

export const LENS_TABS = { traces: "Traces", findings: "Findings", investigations: "Investigations" } as const;
export type LensTab = keyof typeof LENS_TABS;
const lensTabs = Object.keys(LENS_TABS) as LensTab[];

export interface LensRoute {
  readonly tab: LensTab | null;
  readonly lensId: string | null;
  setTab(tab: LensTab): void;
  setLensId(lensId: string | null): void;
}

const LensRouteContext = createContext<LensRoute | null>(null);

/** Keeps navigation out of the URL, e.g. for a sample session that must not leave links behind. */
export function MemoryLensRoute({ initialTab, children }: { initialTab: LensTab; children: ReactNode }) {
  const [tab, setTab] = useState<LensTab | null>(initialTab);
  const [lensId, setLensId] = useState<string | null>(null);
  return <LensRouteContext.Provider value={{ tab, lensId, setTab, setLensId }}>{children}</LensRouteContext.Provider>;
}

/** Without a provider, navigation lives in the URL. */
export function useLensRoute(): LensRoute {
  const provided = useContext(LensRouteContext);
  const [tab, setTab] = useQueryState("tab", parseAsStringLiteral(lensTabs).withOptions({ history: "push" }));
  const [lensId, setLensId] = useQueryState("lens", parseAsString.withOptions({ history: "push" }));
  return (
    provided ?? {
      tab,
      lensId,
      setTab: (next) => void setTab(next),
      setLensId: (next) => void setLensId(next),
    }
  );
}
