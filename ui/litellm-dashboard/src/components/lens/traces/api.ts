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
import type { AgentSummary } from "../agents/agentRollup";
import type {
  SpanDetail,
  SpanQuery,
  TraceDetailQuery,
  SpanErrorPage,
  Trace,
  TraceListQuery,
  TracePage,
  TraceFindingCount,
  TraceFindingsRequest,
  TraceSignals,
  TraceAgentList,
  TraceAgentsQuery,
} from "./types";

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
  agents(window: TraceWindow): Promise<AgentSummary[]>;
  findings(traces: TraceFindingsRequest["traces"]): Promise<TraceFindingCount[]>;
  signals(traces: TraceFindingsRequest["traces"]): Promise<TraceSignals[]>;
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
  const base = `${getProxyBaseUrl().replace(/\/$/, "")}/v1/traces/${encodeURIComponent(traceId)}`;
  const path = spanId ? `${base}/spans/${encodeURIComponent(spanId)}` : base;
  const query = spanId
    ? ({ trace_ref: traceRef } satisfies SpanQuery)
    : ({ trace_ref: traceRef, page_size: 200 } satisfies TraceDetailQuery);
  const params = new URLSearchParams(
    Object.entries(query)
      .filter(([, value]) => value !== undefined)
      .map(([key, value]) => [key, String(value)]),
  ).toString();
  const url = params ? `${path}?${params}` : path;
  const quotedUrl = "'" + url.replaceAll("'", "'\"'\"'") + "'";
  const what = spanId ? "this step of a LiteLLM agent trace" : "this LiteLLM agent trace";
  const guidance = spanId
    ? "The JSON response contains this step's captured input, output, and attributes. Report missing content and capture warnings explicitly."
    : `The JSON response contains span summaries. Follow next_cursor by adding cursor to this URL until it is null, preserving trace_ref and page_size. Fetch captured content at ${base}/spans/{span_id}, using the same trace_ref. Report missing content and capture warnings explicitly.`;
  return `Read ${what}, explain what happened, and investigate any issues:\ncurl --fail-with-body -sS -H "Authorization: Bearer $LITELLM_API_KEY" ${quotedUrl}\n${guidance}`;
};

export function liveTracesApi(accessToken: string): TracesApi {
  return {
    live: true,
    handoff: (traceId, spanId, traceRef) => ({
      text: agentHandoffText(traceId, spanId, traceRef),
      copied: "Command copied",
    }),
    list: (window) => agentTraceListCall({ accessToken, ...window }),
    agents: async ({ startMs, endMs }) => {
      const page = await apiClient.get<TraceAgentList>("/v1/traces/agents", {
        accessToken,
        query: { start_ms: startMs, end_ms: endMs } satisfies TraceAgentsQuery,
      });
      return page.agents ?? [];
    },
    findings: (traces) =>
      apiClient.post<TraceFindingCount[]>("/lens/traces/findings", {
        accessToken,
        body: { traces } satisfies TraceFindingsRequest,
      }),
    signals: (traces) =>
      apiClient.post<TraceSignals[]>("/lens/traces/signals", {
        accessToken,
        body: { traces } satisfies TraceFindingsRequest,
      }),
    anyRecorded: async () => {
      const page = await apiClient.get<TracePage>("/v1/traces", {
        accessToken,
        query: { start_ms: 0 } satisfies TraceListQuery,
      });
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
