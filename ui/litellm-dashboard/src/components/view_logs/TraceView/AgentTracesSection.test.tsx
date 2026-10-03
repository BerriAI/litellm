import { describe, expect, it } from "vitest";
import { filterRuns } from "./AgentTracesSection";
import traceList from "./__fixtures__/trace_list.json";
import type { TracePage, TraceSummary } from "./traceTypes";

const runs = (traceList as TracePage).data as TraceSummary[];

describe("filterRuns", () => {
  it("status filter 'Failed' keeps only runs with errors", () => {
    const failed = filterRuns(runs, "", "all", "error");
    expect(failed.length).toBeGreaterThan(0);
    expect(failed.every((r) => r.error_count > 0)).toBe(true);
    const ok = filterRuns(runs, "", "all", "ok");
    expect(ok.every((r) => r.error_count === 0)).toBe(true);
    expect(failed.length + ok.length).toBe(runs.length);
  });
});
