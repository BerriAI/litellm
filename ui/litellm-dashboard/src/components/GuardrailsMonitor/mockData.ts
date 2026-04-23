/**
 * Types for Guardrails Monitor dashboard (data from usage API).
 */

export interface PerformanceRow {
  id: string;
  name: string;
  type: string;
  provider: string;
  requestsEvaluated: number;
  failRate: number;
  avgScore?: number;
  avgLatency?: number;
  p95Latency?: number;
  falsePositiveRate?: number;
  falseNegativeRate?: number;
  status: "healthy" | "warning" | "critical";
  trend: "up" | "down" | "stable";
}

export interface GuardrailDetailRecord {
  name: string;
  type: string;
  provider: string;
  requestsEvaluated: number;
  failRate: number;
  avgScore?: number;
  avgLatency?: number;
  p95Latency?: number;
  falsePositiveRate?: number;
  falsePositiveCount?: number;
  falseNegativeRate?: number;
  falseNegativeCount?: number;
  status: string;
  description: string;
}

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
