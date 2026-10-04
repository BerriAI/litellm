"use client";

import { createContext, useContext, useMemo } from "react";
import {
  agentTraceCall,
  agentTraceListCall,
  agentTraceSpanCall,
  agentTraceSpanErrorCall,
  apiClient,
  getProxyBaseUrl,
} from "../../networking";
import type { SpanDetail, SpanErrorPage, Trace, TracePage } from "./traceTypes";

export interface TraceWindow {
  readonly startMs: number;
  readonly endMs: number;
  readonly cursor?: string | null;
}

export interface TraceHandoff {
  readonly text: string;
  readonly copied: string;
}

export interface TracesApi {
  /** False for a fixed snapshot: nothing new arrives, so live tail and tracing setup don't apply. */
  readonly live: boolean;
  handoff(traceId: string, spanId?: string | null, traceRef?: string): TraceHandoff;
  list(window: TraceWindow): Promise<TracePage>;
  anyRecorded(): Promise<boolean>;
  trace(traceId: string, traceRef?: string, cursor?: string | null): Promise<Trace>;
  span(traceId: string, spanId: string, traceRef?: string): Promise<SpanDetail>;
  spanError(
    traceId: string,
    spanId: string,
    options: { readonly traceRef?: string; readonly cursor?: string | null },
  ): Promise<SpanErrorPage>;
}

/** A one-liner Claude Code / Codex can run to read the trace. */
export const agentHandoffText = (traceId: string, spanId?: string | null, traceRef?: string): string => {
  const url = `${getProxyBaseUrl().replace(/\/$/, "")}/v1/traces/${traceId}?format=md${spanId ? `&span_id=${spanId}` : ""}${traceRef ? `&trace_ref=${traceRef}` : ""}`;
  const what = spanId ? "this step of a LiteLLM agent trace" : "this LiteLLM agent trace";
  return `Read ${what} and explain what happened and why it failed:\ncurl -s -H "Authorization: Bearer $LITELLM_API_KEY" "${url}"`;
};

export function liveTracesApi(accessToken: string): TracesApi {
  return {
    live: true,
    handoff: (traceId, spanId, traceRef) => ({
      text: agentHandoffText(traceId, spanId, traceRef),
      copied: "Command copied",
    }),
    list: (window) => agentTraceListCall({ accessToken, ...window }),
    anyRecorded: async () => {
      const page = await apiClient.get<TracePage>("/v1/traces", { accessToken, query: { start_ms: 0 } });
      return page.data.length > 0;
    },
    trace: (traceId, traceRef, cursor) => agentTraceCall(accessToken, traceId, traceRef, cursor),
    span: (traceId, spanId, traceRef) => agentTraceSpanCall(accessToken, traceId, spanId, traceRef),
    spanError: (traceId, spanId, options) => agentTraceSpanErrorCall(accessToken, traceId, spanId, options),
  };
}

export const TracesApiContext = createContext<TracesApi | null>(null);

/** Without a provider, falls back to the live HTTP implementation for the caller's token. */
export function useTracesApi(accessToken: string): TracesApi {
  const provided = useContext(TracesApiContext);
  const live = useMemo(() => liveTracesApi(accessToken), [accessToken]);
  return provided ?? live;
}

export const useTracesLive = (): boolean => useContext(TracesApiContext)?.live ?? true;
