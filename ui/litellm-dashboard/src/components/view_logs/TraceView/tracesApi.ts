import {
  agentTraceCall,
  agentTraceListCall,
  agentTraceSpanCall,
  agentTraceSpanErrorCall,
  apiClient,
} from "../../networking";
import type { SpanDetail, SpanErrorPage, Trace, TracePage } from "./traceTypes";

export interface TraceWindow {
  readonly startMs: number;
  readonly endMs: number;
  readonly cursor?: string | null;
}

export interface TracesApi {
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

export function liveTracesApi(accessToken: string): TracesApi {
  return {
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
