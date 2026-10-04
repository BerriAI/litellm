import type { TraceSummary } from "../../traceTypes";

export const run = (overrides: Partial<TraceSummary>): TraceSummary => ({
  agent_count: 1,
  agent_invocations: 1,
  duration_ms: 10,
  error_count: 0,
  input_preview: "",
  input_tokens: 0,
  llm_calls: 1,
  models: [],
  name: "run",
  output_tokens: 0,
  service: "svc",
  span_count: 1,
  spend: null,
  start_time: "2026-10-01T00:00:00Z",
  status: "ok",
  tool_calls: 0,
  trace_id: "trace",
  ...overrides,
});

const refundOverrides = {
  trace_id: "aaa111",
  name: "support",
  input_preview: "Where is my refund?",
  agent_names: ["billing-agent", "triage"],
  models: ["gpt-5"],
};
const refund = run(refundOverrides);
const researchOverrides = {
  trace_id: "bbb222",
  name: "research_lead",
  input_preview: "Compare vector stores",
  agent_names: ["researcher"],
  models: ["claude-opus"],
  error_count: 2,
};
const research = run(researchOverrides);
const plain = run({ trace_id: "ccc333", name: "health", service: "cron" });
export const runs = [refund, research, plain];
