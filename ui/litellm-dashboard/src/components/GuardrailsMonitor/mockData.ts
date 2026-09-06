/**
 * Types for Guardrails Monitor dashboard (data from usage API).
 */

export interface LogEntry {
  id: string;
  timestamp: string;
  input?: string;
  output?: string;
  input_snippet?: string;
  output_snippet?: string;
  score?: number;
  action: "blocked" | "passed" | "flagged";
  model?: string;
  reason?: string;
  latency_ms?: number;
}
