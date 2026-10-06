import { parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useState } from "react";

import { TIME_RANGE_PARSERS } from "@/components/shared/timeRange/routing";
import type { TraceSummary } from "./types";

export const TRACE_VIEWS = ["steps", "thread"] as const;
export type TraceView = (typeof TRACE_VIEWS)[number];

export const SPAN_TABS = ["content", "request", "attributes"] as const;
export type SpanTab = (typeof SPAN_TABS)[number];

export interface TraceRef {
  traceId: string;
  traceRef?: string;
}

export const traceRefOf = (run: TraceSummary): TraceRef => ({ traceId: run.trace_id, traceRef: run.trace_ref });
export const traceKey = (ref: TraceRef): string => ref.traceRef || ref.traceId;

/** A shareable link that opens just this run on the current page, dropping list filters and step selection. */
export const traceShareUrl = (ref: TraceRef, location: Pick<Location, "origin" | "pathname" | "search">): string => {
  const demo = new URLSearchParams(location.search).get("demo") === "true";
  const params = new URLSearchParams({
    ...(demo ? { demo: "true" } : {}),
    trace: ref.traceId,
    ...(ref.traceRef ? { trace_ref: ref.traceRef } : {}),
  });
  return `${location.origin}${location.pathname}?${params}`;
};

/** Which step, view and detail section of an open run are showing. Owned by the URL in the drawer, locally in sheets. */
export interface RunSelection {
  spanId: string | null;
  view: TraceView;
  spanTab: SpanTab;
  stepQuery: string;
  errorsOnly: boolean;
  selectSpan: (id: string) => void;
  setView: (view: TraceView) => void;
  setSpanTab: (tab: SpanTab) => void;
  setStepQuery: (query: string) => void;
  setErrorsOnly: (errorsOnly: boolean) => void;
}

export const OPEN_TRACE_PARSERS = {
  trace: parseAsString,
  trace_ref: parseAsString,
  span: parseAsString,
  view: parseAsStringLiteral(TRACE_VIEWS).withDefault("steps"),
  span_tab: parseAsStringLiteral(SPAN_TABS).withDefault("content"),
  steps_q: parseAsString.withDefault(""),
  errors: parseAsBoolean.withDefault(false),
  fullscreen: parseAsBoolean.withDefault(false),
};

export const RUN_FILTER_PARSERS = {
  q: parseAsString.withDefault(""),
  agent: parseAsString.withDefault(""),
  status: parseAsStringLiteral(["all", "ok", "error"]).withDefault("all"),
  ...TIME_RANGE_PARSERS,
};

export interface OpenTraceRouting {
  trace: TraceRef | null;
  openTrace: (ref: TraceRef | null) => void;
  selection: RunSelection;
  fullScreen: boolean;
  setFullScreen: (fullScreen: boolean) => void;
}

const FRESH_RUN = { span: null, view: null, span_tab: null, steps_q: null, errors: null };

/** The open run is a history entry; moving within it (step, view, section) replaces the current one. */
export function useOpenTraceRouting(): OpenTraceRouting {
  const [params, setParams] = useQueryStates(OPEN_TRACE_PARSERS, { history: "push" });
  const openTrace = useCallback(
    (ref: TraceRef | null) => {
      const run = { trace: ref?.traceId ?? null, trace_ref: ref?.traceRef || null };
      void setParams(ref === null ? { ...FRESH_RUN, ...run, fullscreen: null } : { ...FRESH_RUN, ...run });
    },
    [setParams],
  );
  const selectSpan = useCallback(
    (id: string) => void setParams({ span: id, span_tab: null }, { history: "replace" }),
    [setParams],
  );
  const setView = useCallback((view: TraceView) => void setParams({ view }, { history: "replace" }), [setParams]);
  const setSpanTab = useCallback(
    (span_tab: SpanTab) => void setParams({ span_tab }, { history: "replace" }),
    [setParams],
  );
  const setStepQuery = useCallback(
    (steps_q: string) => void setParams({ steps_q }, { history: "replace" }),
    [setParams],
  );
  const setErrorsOnly = useCallback(
    (errors: boolean) => void setParams({ errors }, { history: "replace" }),
    [setParams],
  );
  const setFullScreen = useCallback(
    (fullscreen: boolean) => void setParams({ fullscreen }, { history: "replace" }),
    [setParams],
  );
  return {
    trace: params.trace === null ? null : { traceId: params.trace, traceRef: params.trace_ref ?? undefined },
    openTrace,
    selection: {
      spanId: params.span,
      view: params.view,
      spanTab: params.span_tab,
      stepQuery: params.steps_q,
      errorsOnly: params.errors,
      selectSpan,
      setView,
      setSpanTab,
      setStepQuery,
      setErrorsOnly,
    },
    fullScreen: params.fullscreen,
    setFullScreen,
  };
}

/** Same shape as the routed selection, for run views opened inside another surface (a finding's evidence sheet). */
export function useLocalRunSelection(initialSpanId: string | null): RunSelection {
  const [spanId, setSpanId] = useState(initialSpanId);
  const [view, setView] = useState<TraceView>("steps");
  const [spanTab, setSpanTab] = useState<SpanTab>("content");
  const [stepQuery, setStepQuery] = useState("");
  const [errorsOnly, setErrorsOnly] = useState(false);
  const selectSpan = useCallback((id: string) => {
    setSpanId(id);
    setSpanTab("content");
  }, []);
  return { spanId, view, spanTab, stepQuery, errorsOnly, selectSpan, setView, setSpanTab, setStepQuery, setErrorsOnly };
}

export function useRunFilterRouting() {
  const [{ q, agent, status }, setParams] = useQueryStates(RUN_FILTER_PARSERS);
  return {
    query: q,
    agent,
    status,
    setQuery: (query: string) => void setParams({ q: query }),
    setAgent: (agent: string) => void setParams({ agent }),
    setStatus: (status: "all" | "ok" | "error") => void setParams({ status }),
  };
}
