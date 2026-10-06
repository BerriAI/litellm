import { describe, expect, it } from "vitest";

import { SAMPLE_TRACE_SERVICE, sampleTraceExport } from "./sampleTrace";

interface ExportedSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: { key: string; value: { stringValue: string } }[];
}

const spansOf = (body: object): ExportedSpan[] =>
  (body as { resourceSpans: { scopeSpans: { spans: ExportedSpan[] }[] }[] }).resourceSpans[0].scopeSpans[0].spans;

const operation = (span: ExportedSpan) =>
  span.attributes.find((a) => a.key === "gen_ai.operation.name")?.value.stringValue;

describe("sampleTraceExport", () => {
  it("uses OTLP/JSON hex ids and returns the trace id the spans carry", () => {
    const { traceId, body } = sampleTraceExport(Date.now());
    const spans = spansOf(body);
    expect(traceId).toMatch(/^[0-9a-f]{32}$/);
    for (const span of spans) {
      expect(span.traceId).toBe(traceId);
      expect(span.spanId).toMatch(/^[0-9a-f]{16}$/);
    }
  });

  it("nests an LLM call and a tool call under one root agent", () => {
    const spans = spansOf(sampleTraceExport(Date.now()).body);
    const roots = spans.filter((s) => !s.parentSpanId);
    expect(roots).toHaveLength(1);
    expect(operation(roots[0])).toBe("invoke_agent");
    const children = spans.filter((s) => s.parentSpanId === roots[0].spanId);
    expect(children.map(operation).sort()).toEqual(["chat", "execute_tool"]);
  });

  it("ends at the given time and tags the sample service", () => {
    const now = 1_790_000_000_000;
    const { body } = sampleTraceExport(now);
    const root = spansOf(body).find((s) => !s.parentSpanId)!;
    expect(BigInt(root.endTimeUnixNano)).toBe(BigInt(now) * BigInt(1_000_000));
    expect(BigInt(root.startTimeUnixNano)).toBeLessThan(BigInt(root.endTimeUnixNano));
    expect(JSON.stringify(body)).toContain(SAMPLE_TRACE_SERVICE);
  });
});
