/**
 * Agent tracing types. Mirrors `litellm/tracing/types.py` exactly.
 *
 * A trace is one agent run made of spans (agent / llm / tool / chain / framework).
 */

export type SpanType = "agent" | "llm" | "tool" | "chain" | "framework";
export type SpanStatus = "ok" | "error" | "unset";

export interface Span {
  span_id: string;
  parent_span_id: string | null;
  name: string;
  type: SpanType;
  /** The agent this span runs inside, e.g. "researcher". */
  agent: string;
  /** Relative to trace start. */
  start_offset_ms: number;
  duration_ms: number;
  status: SpanStatus;
  /** Exception message when status is "error". */
  error?: string | null;
  input_preview: string;
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  litellm_request_id: string | null;
}

/** One distinct agent in a trace. 200 invocations of `researcher` = one node. */
export interface AgentNode {
  name: string;
  parent_agent: string | null;
  invocations: number;
  llm_calls: number;
  tool_calls: number;
  duration_ms: number;
}

export interface TraceSummary {
  trace_id: string;
  trace_ref?: string;
  name: string;
  service: string;
  input_preview: string;
  /** ISO 8601 */
  start_time: string;
  duration_ms: number;
  status: SpanStatus;
  span_count: number;
  agent_count: number;
  llm_calls: number;
  tool_calls: number;
  /** Spans with an error status; > 0 means the run shows as failed. */
  error_count: number;
  input_tokens: number;
  output_tokens: number;
  models: string[];
}

export interface Trace {
  summary: TraceSummary;
  agents: AgentNode[];
  spans: Span[];
}

export interface TracePage {
  data: TraceSummary[];
  next_cursor: string | null;
}

/** `input` / `output` are JSON strings (messages for llm spans, raw args / result for tools). */
export interface SpanDetail {
  span_id: string;
  input: string;
  output: string;
  attributes: Record<string, string>;
}

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
