import type { TraceAgent, TraceSummary } from "../traces/types";
import { traceAgentNames } from "../traces/utils";

export type AgentSummary = TraceAgent;

const failed = (trace: TraceSummary): boolean => trace.status === "error" || trace.error_count > 0;

const latest = (times: readonly string[]): string => times.reduce((a, b) => (Date.parse(a) >= Date.parse(b) ? a : b));

/** One row per agent across the given runs, newest activity first; mirrors what `/v1/traces/agents` returns. */
export function rollUpAgents(traces: readonly TraceSummary[]): AgentSummary[] {
  const names = [...new Set(traces.flatMap(traceAgentNames))];
  return names
    .map((name) => {
      const runs = traces.filter((trace) => traceAgentNames(trace).includes(name));
      return {
        name,
        runs: runs.length,
        failed_runs: runs.filter(failed).length,
        last_seen: latest(runs.map((trace) => trace.start_time)),
        frameworks: [...new Set(runs.flatMap((trace) => trace.frameworks ?? []))].sort(),
      };
    })
    .sort((a, b) => Date.parse(b.last_seen) - Date.parse(a.last_seen) || a.name.localeCompare(b.name));
}
