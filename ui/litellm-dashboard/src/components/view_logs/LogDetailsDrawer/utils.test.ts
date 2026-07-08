import { describe, expect, it } from "vitest";
import { sortSessionLogs } from "./utils";

const llm = (id: string, startTime: string) => ({ request_id: id, call_type: "acompletion", startTime });
const mcp = (id: string, startTime: string) => ({ request_id: id, call_type: "call_mcp_tool", startTime });

const ids = (rows: { request_id: string }[]) => rows.map((row) => row.request_id);

describe("sortSessionLogs", () => {
  const rows = [
    mcp("mcp-early", "2026-07-08T10:00:01.000Z"),
    llm("llm-late", "2026-07-08T10:00:02.000Z"),
    mcp("mcp-late", "2026-07-08T10:00:03.000Z"),
    llm("llm-early", "2026-07-08T10:00:00.000Z"),
  ];

  it("grouped mode keeps MCP calls last, newest first within each group", () => {
    expect(ids(sortSessionLogs(rows, "grouped"))).toEqual(["llm-late", "llm-early", "mcp-late", "mcp-early"]);
  });

  it("chronological mode interleaves all calls by start time, oldest first", () => {
    expect(ids(sortSessionLogs(rows, "chronological"))).toEqual(["llm-early", "mcp-early", "llm-late", "mcp-late"]);
  });

  it("does not mutate the input array", () => {
    const input = [...rows];
    sortSessionLogs(input, "chronological");
    expect(ids(input)).toEqual(ids(rows));
  });
});
