"use client";

import { parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback } from "react";
import { OPEN_TRACE_PARSERS, RUN_FILTER_PARSERS } from "@/components/view_logs/TraceView/traceRouting";

export const LENS_TABS = { traces: "Traces", findings: "Findings", investigations: "Investigations" } as const;
export type LensTab = keyof typeof LENS_TABS;
const lensTabs = Object.keys(LENS_TABS) as LensTab[];

const LENS_PARSERS = {
  tab: parseAsStringLiteral(lensTabs),
  lens: parseAsString,
  demo: parseAsBoolean.withDefault(false),
};

/** Every Lens-owned key. Leaving the sample session clears them so sample ids never point at live data. */
const SESSION_PARSERS = { ...LENS_PARSERS, ...OPEN_TRACE_PARSERS, ...RUN_FILTER_PARSERS };
const CLEARED_SESSION = {
  lens: null,
  demo: null,
  trace: null,
  trace_ref: null,
  span: null,
  view: null,
  span_tab: null,
  q: null,
  agent: null,
  status: null,
  hours: null,
} satisfies Record<Exclude<keyof typeof SESSION_PARSERS, "tab">, null>;

export interface LensRoute {
  readonly tab: LensTab | null;
  readonly lensId: string | null;
  readonly demo: boolean;
  setTab(tab: LensTab): void;
  setLensId(lensId: string | null): void;
  setDemo(demo: boolean): void;
}

/** Lens navigation lives in the URL, sample session included, so any view is a shareable link. */
export function useLensRoute(): LensRoute {
  const [{ tab, lens, demo }, setParams] = useQueryStates(SESSION_PARSERS, { history: "push" });
  const setTab = useCallback((next: LensTab) => void setParams({ tab: next }), [setParams]);
  const setLensId = useCallback((next: string | null) => void setParams({ lens: next }), [setParams]);
  const setDemo = useCallback((next: boolean) => void setParams(next ? { demo: true } : CLEARED_SESSION), [setParams]);
  return { tab, lensId: lens, demo, setTab, setLensId, setDemo };
}
