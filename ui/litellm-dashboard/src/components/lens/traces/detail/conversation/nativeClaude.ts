import type { Span, SpanDetail } from "../../types";

export function isClaudeSupplement(span: Span): boolean {
  return (
    span.framework === "claude-code" && ["claude_code.tool_result", "claude_code.api_request_body"].includes(span.name)
  );
}

function object(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function payload(detail: SpanDetail): Record<string, unknown> | undefined {
  try {
    return object(JSON.parse(detail.output));
  } catch {
    return undefined;
  }
}

export function claudeToolDetails(
  spans: readonly Span[],
  details: ReadonlyMap<string, SpanDetail>,
): ReadonlyMap<string, SpanDetail> {
  const inputs = new Map<string, SpanDetail>();
  const outputs = new Map<string, string>();
  for (const span of spans.filter(isClaudeSupplement)) {
    const detail = details.get(span.span_id);
    if (!detail) continue;
    const id = detail.attributes["tool_use_id"];
    if (span.name === "claude_code.tool_result" && id && detail.input) inputs.set(id, detail);
    const results = payload(detail)?.tool_results;
    if (!Array.isArray(results)) continue;
    for (const value of results) {
      const result = object(value);
      if (typeof result?.id === "string" && typeof result.content === "string") {
        outputs.set(result.id, result.content);
      }
    }
  }
  return new Map(
    [...details].map(([id, detail]) => {
      const callId = detail.attributes["gen_ai.tool.call.id"] || detail.attributes["tool_use_id"];
      const input = inputs.get(callId);
      const output = outputs.get(callId);
      return [
        id,
        {
          ...detail,
          ...(input ? { input: input.input, input_ui: input.input_ui } : {}),
          ...(!detail.output && output !== undefined
            ? { output, output_ui: { kind: "text" as const, text: output } }
            : {}),
        },
      ];
    }),
  );
}

export function claudeCaptureWarnings(details: ReadonlyMap<string, SpanDetail>): string[] {
  return [...details.values()].flatMap((detail) => {
    if (detail.attributes["event.name"] !== "api_request_body") return [];
    const warning = payload(detail)?.warning;
    return typeof warning === "string" ? [warning] : [];
  });
}
