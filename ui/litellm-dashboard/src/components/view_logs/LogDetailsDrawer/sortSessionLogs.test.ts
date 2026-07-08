import { describe, it, expect } from "vitest";
import { sortSessionLogs, getRowDurationMs } from "./utils";
import { LogEntry } from "../columns";

function makeLog(overrides: Partial<LogEntry>): LogEntry {
  return {
    request_id: "req",
    startTime: "2026-07-08T00:00:00.000Z",
    endTime: "2026-07-08T00:00:01.000Z",
    call_type: "completion",
    ...overrides,
  } as LogEntry;
}

describe("getRowDurationMs", () => {
  it("prefers request_duration_ms when present", () => {
    const row = makeLog({
      request_duration_ms: 4200,
      startTime: "2026-07-08T00:00:00.000Z",
      endTime: "2026-07-08T00:00:01.000Z",
    });
    expect(getRowDurationMs(row)).toBe(4200);
  });

  it("falls back to endTime minus startTime when duration missing", () => {
    const row = makeLog({
      request_duration_ms: undefined,
      startTime: "2026-07-08T00:00:00.000Z",
      endTime: "2026-07-08T00:00:02.500Z",
    });
    expect(getRowDurationMs(row)).toBe(2500);
  });

  it("returns 0 when timestamps are unparseable and no duration", () => {
    const row = makeLog({ request_duration_ms: undefined, startTime: "nope", endTime: "nope" });
    expect(getRowDurationMs(row)).toBe(0);
  });
});

describe("sortSessionLogs", () => {
  const llmEarlyShort = makeLog({
    request_id: "llm-early-short",
    call_type: "completion",
    startTime: "2026-07-08T00:00:00.000Z",
    request_duration_ms: 500,
  });
  const mcpMiddleLong = makeLog({
    request_id: "mcp-middle-long",
    call_type: "call_mcp_tool",
    startTime: "2026-07-08T00:00:05.000Z",
    request_duration_ms: 9000,
  });
  const llmLateMedium = makeLog({
    request_id: "llm-late-medium",
    call_type: "completion",
    startTime: "2026-07-08T00:00:10.000Z",
    request_duration_ms: 3000,
  });

  const logs = [llmLateMedium, llmEarlyShort, mcpMiddleLong];

  it("sorts by duration descending (longest first), interleaving MCP and LLM", () => {
    const result = sortSessionLogs(logs, "duration").map((r) => r.request_id);
    expect(result).toEqual(["mcp-middle-long", "llm-late-medium", "llm-early-short"]);
  });

  it("sorts by start time ascending (chronological), interleaving MCP and LLM", () => {
    const result = sortSessionLogs(logs, "start_time").map((r) => r.request_id);
    expect(result).toEqual(["llm-early-short", "mcp-middle-long", "llm-late-medium"]);
  });

  it("does not mutate the input array", () => {
    const input = [llmLateMedium, llmEarlyShort, mcpMiddleLong];
    const snapshot = [...input];
    sortSessionLogs(input, "duration");
    expect(input).toEqual(snapshot);
  });
});
