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
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";
import type { RunOrder } from "./list/runOrder";
import type { RunField, RunValues, SpanDetail, SpanErrorPage, Trace, TraceHistogram, TracePage } from "./types";

/** Which runs: the window and the `q` search the server applies before counting or paging. */
export interface RunSelection {
  readonly window: TimeWindow;
  readonly q: string;
  readonly asOfMs?: number;
}

/** Where in the ordered sequence a page starts; `null` is the first page. */
export interface RunPage {
  readonly cursor: string | null;
}

/** One page of the runs list, as the server's independent read axes. */
export interface RunListRequest {
  readonly selection: RunSelection;
  readonly order: RunOrder;
  readonly page: RunPage;
}

export interface TraceHandoff {
  readonly text: string;
  readonly copied: string;
}

export interface TracesApi {
  /** Partitions query caches between backends (one token, or the demo). */
  readonly scope: string;
  /** False for a fixed snapshot: nothing new arrives, so live tail and tracing setup don't apply. */
  readonly live: boolean;
  handoff(id: string, spanId?: string | null): TraceHandoff;
  list(request: RunListRequest): Promise<TracePage>;
  histogram(selection: RunSelection, buckets: number): Promise<TraceHistogram>;
  values(field: RunField, contains: string, range: TimeWindow): Promise<readonly string[]>;
  anyRecorded(): Promise<boolean>;
  trace(id: string, cursor?: string | null): Promise<Trace>;
  span(id: string, spanId: string): Promise<SpanDetail>;
  spanError(traceId: string, spanId: string, options: { readonly cursor?: string | null }): Promise<SpanErrorPage>;
}

export const agentHandoffText = (id: string, spanId?: string | null): string => {
  const base = `${getProxyBaseUrl().replace(/\/$/, "")}/v1/traces/${encodeURIComponent(id)}`;
  const urls = spanId ? [`${base}/spans/${encodeURIComponent(spanId)}`] : [base, `${base}/spans`];
  const commands = urls.map((url) => `curl -s -H "Authorization: Bearer $LITELLM_API_KEY" "${url}"`).join("\n");
  return `Read this LiteLLM agent trace and explain what happened. Follow next_cursor on the spans endpoint to load remaining steps:\n${commands}`;
};

export function liveTracesApi(accessToken: string): TracesApi {
  return {
    scope: accessToken,
    live: true,
    handoff: (traceId, spanId) => ({
      text: agentHandoffText(traceId, spanId),
      copied: "Command copied",
    }),
    list: (request) => agentTraceListCall(accessToken, request),
    histogram: ({ window, q, asOfMs }, buckets) =>
      apiClient.get<TraceHistogram>("/v1/traces/histogram", {
        accessToken,
        query: { start_ms: window.startMs, end_ms: window.endMs, as_of_ms: asOfMs, q: q || undefined, buckets },
      }),
    values: async (field, contains, range) => {
      const found = await apiClient.get<RunValues>(`/v1/traces/values/${field}`, {
        accessToken,
        query: { start_ms: range.startMs, end_ms: range.endMs, contains: contains || undefined },
      });
      return found.values;
    },
    anyRecorded: async () => {
      const page = await apiClient.get<TracePage>("/v1/traces", { accessToken, query: { start_ms: 0 } });
      return page.data.length > 0;
    },
    trace: (id, cursor) => agentTraceCall(accessToken, id, cursor),
    span: (id, spanId) => agentTraceSpanCall(accessToken, id, spanId),
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
