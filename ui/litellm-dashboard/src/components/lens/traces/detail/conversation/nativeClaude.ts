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

export function claudeCaptureWarnings(details: ReadonlyMap<string, SpanDetail>): string[] {
  return [...details.values()].flatMap((detail) => {
    if (detail.attributes["event.name"] !== "api_request_body") return [];
    const warning = payload(detail)?.warning;
    return typeof warning === "string" ? [warning] : [];
  });
}

export function unexecutedToolResults(detail: SpanDetail): { id: string; content: string; isError: boolean }[] {
  try {
    const results: unknown = JSON.parse(detail.attributes["lens.content.unexecuted_tool_results"] ?? "[]");
    if (!Array.isArray(results)) return [];
    return results.flatMap((value) => {
      const result = object(value);
      return typeof result?.id === "string" && typeof result.content === "string"
        ? [{ id: result.id, content: result.content, isError: result.is_error === true }]
        : [];
    });
  } catch {
    return [];
  }
}
