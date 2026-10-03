import { describe, expect, it, vi } from "vitest";
import { createLensDemo, createLensDemoData } from "./createLensDemo";
import { evidenceTarget } from "../model/findings";

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

  it("includes a long release review with unique steps, complete details and three failed checks", () => {
    const run = createLensDemoData().runs[6];
    const ids = new Set(run.trace.spans.map((span) => span.span_id));
    expect(run.trace.spans).toHaveLength(362);
    expect(ids.size).toBe(362);
    expect(run.trace.summary.error_count).toBe(3);
    expect(run.trace.summary.status).toBe("ok");
    for (const span of run.trace.spans) {
      if (span.parent_span_id) expect(ids.has(span.parent_span_id)).toBe(true);
      expect(run.details.find((detail) => detail.span_id === span.span_id)).toBeDefined();
    }
    expect(JSON.parse(run.details.at(-1)!.input)).toHaveLength(120);
    expect(run.details.at(-1)!.output).toContain("Hold the release");
  });

  it("filters time windows locally and rejects writes or unknown reads without network access", async () => {
    const network = vi.spyOn(globalThis, "fetch");
    const now = Date.now();
    const { services } = createLensDemo(now);
    const all = await services.traces.list({ startMs: 0, endMs: now });
    const recent = await services.traces.list({ startMs: now - 3600_000, endMs: now });
    expect(recent.data.length).toBeGreaterThan(0);
    expect(recent.data.length).toBeLessThan(all.data.length);
    expect(recent.data.every((trace) => Date.parse(trace.start_time) >= now - 3600_000)).toBe(true);
    const settings = services.lens.lenses().then((list) => list.lenses[0].settings);
    await expect(services.lens.saveLens(undefined, await settings)).rejects.toMatchObject({ status: 403 });
    await expect(services.lens.run("real-investigation", "job")).rejects.toMatchObject({ status: 404 });
    await expect(services.traces.trace("missing")).rejects.toMatchObject({ status: 404 });
    expect(network).not.toHaveBeenCalled();
    network.mockRestore();
  });
});
