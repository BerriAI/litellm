import type { components, paths } from "@/lib/http/schema";

export type TraceMetadata = paths["/v1/traces/{id}"]["get"]["responses"][200]["content"]["application/json"];
export type TraceSpansPage = paths["/v1/traces/{id}/spans"]["get"]["responses"][200]["content"]["application/json"];
export type TracePage = paths["/v1/traces"]["get"]["responses"][200]["content"]["application/json"];
export type TraceHistogram = paths["/v1/traces/histogram"]["get"]["responses"][200]["content"]["application/json"];
export type RunValues = paths["/v1/traces/values/{field}"]["get"]["responses"][200]["content"]["application/json"];
export type RunField = paths["/v1/traces/values/{field}"]["get"]["parameters"]["path"]["field"];
export type SpanErrorPage =
  paths["/v1/traces/{id}/spans/{span_id}/error"]["get"]["responses"][200]["content"]["application/json"];
export type SpanDetail =
  paths["/v1/traces/{id}/spans/{span_id}"]["get"]["responses"][200]["content"]["application/json"];
export type Span = components["schemas"]["Span"];
export type SpanType = Span["type"];
export type SpanStatus = Span["status"];
export type AgentNode = TraceMetadata["agents"][number];
export type TraceSummary = TraceMetadata["summary"];
export type Trace = TraceMetadata & {
  spans: Span[];
  next_cursor?: string | null;
  spans_complete?: boolean;
};

export interface TraceToolCall {
  name: string;
  args: unknown;
}

export interface TraceMessage {
  role: string;
  content: string;
  name?: string;
  tool_calls?: TraceToolCall[];
}
