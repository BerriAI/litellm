import { describe, expect, it, vi } from "vitest";
import { createLensDemo, createLensDemoData } from "./lensDemoData";
import type { TracePage } from "@/components/view_logs/TraceView/traceTypes";
import { evidenceTarget } from "./lensData";

describe("Lens demo data", () => {
  it("links every finding to the quoted original step and assessed run", () => {
    const data = createLensDemoData();
    for (const lens of data.lenses) {
      for (const job of lens.jobs) {
        for (const finding of job.findings ?? []) {
          for (const evidence of finding.evidence) {
            const target = evidenceTarget(evidence.execution_id);
            const run = data.runs.find(({ trace }) => trace.summary.trace_id === target?.id);
            const step = run?.details.find((span) => span.span_id === evidence.span_id);
            expect(step?.output).toContain(evidence.quote);
            expect(job.sample?.executions.map((execution) => execution.id)).toContain(evidence.execution_id);
            expect(
              job.assessments.find((assessment) => assessment.execution_id === evidence.execution_id)?.[
                finding.kind === "issue" ? "issue_checks" : "pattern_checks"
              ],
            ).toContain(finding.check_id);
          }
        }
      }
    }
  });

  it("keeps trace totals, timestamps, and agent names consistent", () => {
    const data = createLensDemoData();
    for (const { trace } of data.runs) {
      expect(trace.summary.span_count).toBe(trace.spans.length);
      expect(trace.summary.agent_names).toContain(trace.agents[0].name);
      expect(trace.summary.error_count).toBe(trace.spans.filter((span) => span.status === "error").length);
      for (const span of trace.spans) {
        expect(span.start_offset_ms + span.duration_ms).toBeLessThanOrEqual(trace.summary.duration_ms);
      }
    }
  });

  it("filters time windows locally and rejects writes or unknown reads without network access", async () => {
    const network = vi.spyOn(globalThis, "fetch");
    const now = Date.now();
    const { client } = createLensDemo(now);
    const all = await client.get<TracePage>("/v1/traces");
    const recent = await client.get<TracePage>("/v1/traces", { query: { start_ms: now - 3600_000, end_ms: now } });
    expect(recent.data.length).toBeGreaterThan(0);
    expect(recent.data.length).toBeLessThan(all.data.length);
    expect(recent.data.every((trace) => Date.parse(trace.start_time) >= now - 3600_000)).toBe(true);
    await expect(client.post("/lens", { body: {} })).rejects.toMatchObject({ status: 403 });
    await expect(client.get("/lens/real-investigation")).rejects.toMatchObject({ status: 404 });
    expect(network).not.toHaveBeenCalled();
    network.mockRestore();
  });
});
