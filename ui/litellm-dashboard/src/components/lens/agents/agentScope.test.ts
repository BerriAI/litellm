import { describe, expect, it } from "vitest";

import type { TraceSummary } from "../traces/types";
import { rollUpAgents } from "./agentRollup";
import { matchesAgent } from "./AgentPicker";
import { resolveAgent } from "./useAgentSelection";

const summary = (overrides: Partial<TraceSummary>): TraceSummary =>
  ({
    trace_id: "t",
    name: "run",
    service: "svc",
    agent_names: [],
    frameworks: [],
    input_preview: "",
    start_time: "2026-10-07T12:00:00+00:00",
    duration_ms: 1,
    status: "ok",
    span_count: 1,
    agent_count: 1,
    agent_invocations: 1,
    llm_calls: 0,
    tool_calls: 0,
    error_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    models: [],
    spend: null,
    priced_calls: 0,
    ...overrides,
  }) as TraceSummary;

describe("resolveAgent", () => {
  const available = ["moyai", "researcher", "writer"];

  it("lets a shared link pick the agent", () => {
    expect(resolveAgent("writer", "moyai", available)).toBe("writer");
  });

  it("reopens the agent this browser picked last", () => {
    expect(resolveAgent("", "researcher", available)).toBe("researcher");
  });

  it("falls back to the most recently active agent when the remembered one is gone", () => {
    expect(resolveAgent("", "retired", available)).toBe("moyai");
  });

  it("opens the most recently active agent on a first visit", () => {
    expect(resolveAgent("", "", available)).toBe("moyai");
  });

  it("has no agent to scope to before any traces arrive", () => {
    expect(resolveAgent("", "moyai", [])).toBeNull();
  });
});

describe("rollUpAgents", () => {
  it("counts runs and failures per agent and orders by latest activity", () => {
    const agents = rollUpAgents([
      summary({ agent_names: ["moyai"], start_time: "2026-10-07T10:00:00+00:00" }),
      summary({ agent_names: ["moyai", "researcher"], status: "error", start_time: "2026-10-07T12:00:00+00:00" }),
      summary({ agent_names: ["researcher"], error_count: 2, start_time: "2026-10-07T13:00:00+00:00" }),
      summary({ agent_names: ["writer"], frameworks: ["langgraph"], start_time: "2026-10-07T09:00:00+00:00" }),
    ]);
    expect(agents).toEqual([
      { name: "researcher", runs: 2, failed_runs: 2, last_seen: "2026-10-07T13:00:00+00:00", frameworks: [] },
      { name: "moyai", runs: 2, failed_runs: 1, last_seen: "2026-10-07T12:00:00+00:00", frameworks: [] },
      { name: "writer", runs: 1, failed_runs: 0, last_seen: "2026-10-07T09:00:00+00:00", frameworks: ["langgraph"] },
    ]);
  });
});

describe("matchesAgent", () => {
  it("finds agents by a case-insensitive part of the name", () => {
    expect(matchesAgent({ name: "Support-Bot" }, " support")).toBe(true);
    expect(matchesAgent({ name: "moyai" }, "research")).toBe(false);
  });
});
