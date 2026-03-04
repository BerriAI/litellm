/**
 * Types for agent trace visualization in GuardrailsMonitor LogViewer.
 * Supports orchestrator → sub-agents → LLM + MCP hierarchy.
 */

export type SpanType =
  | "orchestrator"
  | "agent"
  | "llm"
  | "mcp"
  | "function";

export type SpanStatus = "success" | "error" | "running" | "retry";

export interface Span {
  id: string;
  name: string;
  type: SpanType;
  status: SpanStatus;
  startMs: number;
  durationMs: number;
  model?: string;
  tokens?: { prompt: number; completion: number };
  cost?: number;
  mcpServer?: string;
  mcpTool?: string;
  input?: string;
  output?: string;
  children?: Span[];
}

export interface AgentTraceSession {
  id: string;
  shortId: string;
  rootAgentName: string;
  timestamp: string;
  relativeTime: string;
  totalSpans: number;
  totalCost: number;
  totalDurationMs: number;
  status: SpanStatus;
  spans: Span[];
}
