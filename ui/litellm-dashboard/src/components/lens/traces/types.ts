import type { components, paths } from "@/lib/http/schema";

export type Trace = paths["/v1/traces/{trace_id}"]["get"]["responses"][200]["content"]["application/json"];
export type TracePage = paths["/v1/traces"]["get"]["responses"][200]["content"]["application/json"];
export type SpanErrorPage =
  paths["/v1/traces/{trace_id}/spans/{span_id}/error"]["get"]["responses"][200]["content"]["application/json"];
export type TraceListQuery = NonNullable<paths["/v1/traces"]["get"]["parameters"]["query"]>;
export type TraceDetailQuery = NonNullable<paths["/v1/traces/{trace_id}"]["get"]["parameters"]["query"]>;
export type SpanQuery = NonNullable<paths["/v1/traces/{trace_id}/spans/{span_id}"]["get"]["parameters"]["query"]>;
export type SpanErrorQuery = NonNullable<
  paths["/v1/traces/{trace_id}/spans/{span_id}/error"]["get"]["parameters"]["query"]
>;
export type TraceQueryBody = components["schemas"]["TraceQueryRequest"];
type ApiSpanDetail =
  paths["/v1/traces/{trace_id}/spans/{span_id}"]["get"]["responses"][200]["content"]["application/json"];
export type Span = Trace["spans"][number];
export type SpanType = Span["type"];
export type SpanStatus = Span["status"];
export type AgentNode = Trace["agents"][number];
export type TraceSummary = Trace["summary"];
export type SpanDetail = Omit<ApiSpanDetail, "input_ui" | "output_ui"> &
  Partial<Pick<ApiSpanDetail, "input_ui" | "output_ui">>;
export type UIToolCall = components["schemas"]["UIToolCall"];
export type UIMessage = components["schemas"]["UIMessage"];
export type UIField = components["schemas"]["UIField"];
export type UIContent = ApiSpanDetail["input_ui"];

export interface TraceToolCall {
  name: string;
  args: unknown;
}

export type TraceMessage = Omit<UIMessage, "role" | "tool_calls"> & {
  role: string;
  tool_calls?: TraceToolCall[];
};
