import { parseJson } from "@/components/view_logs/TraceView/traceUtils";

import type { Review } from "./types";

type Span = Review["spans"][number];

export type TimelineItem =
  | { kind: "ask"; text: string; span: number }
  | { kind: "tool"; name: string; args: string; result: string; error: boolean; span: number }
  | { kind: "reply"; text: string; span: number }
  | { kind: "failure"; text: string; span: number }
  | { kind: "note"; label: string; text: string; span: number };

interface Message {
  role: string;
  content: string;
}

const OMITTED = /\n?\[\.\.\. preview omitted; read this span for evidence \.\.\.\]\n?/;
const CUT_OUTPUT_HEADER = /\nO(?:u(?:t(?:p(?:u(?:t:?)?)?)?)?)? ?$/;
const SECTION_START =/(?:^|\n)(Input|Output|Status): ?/g;
const OK_STATUS = /^STATUS_CODE_(UNSET|OK)\b/;
const MESSAGE = /"role":\s*"(\w+)",\s*"content":\s*"((?:[^"\\]|\\.)*)(")?/g;
const TOOL_PROBLEM = /"error"|\bdenied\b|\bforbidden\b|\bunauthori[sz]ed\b|\bnot (?:allowed|permitted)\b/i;
const SNIPPET = 160;

function decode(escaped: string): string {
  const parsed = parseJson(`"${escaped.replace(/\\u[0-9a-fA-F]{0,3}$|\\$/, "")}"`);
  return typeof parsed === "string" ? parsed : escaped;
}

function tidy(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function sections(text: string): Readonly<Partial<Record<"Input" | "Output" | "Status", string>>> {
  const starts = [...text.matchAll(SECTION_START)];
  return Object.fromEntries(
    starts.map((match, n) => {
      const from = (match.index ?? 0) + match[0].length;
      const to = starts[n + 1]?.index ?? text.length;
      return [match[1], text.slice(from, to)];
    }),
  );
}

function parts(preview: string) {
  const [rawHead, tail = ""] = preview.split(OMITTED);
  const head = rawHead.replace(CUT_OUTPUT_HEADER, "\nOutput: ");
  const whole = sections(preview.replace(OMITTED, "\n"));
  const ending = sections(tail);
  const before = sections(head);
  const cutOutput = before.Output !== undefined ? `…${tail.split(/\nStatus: /)[0] ?? ""}` : "";
  return {
    input: before.Input ?? "",
    output: tail ? ending.Output ?? cutOutput : whole.Output ?? "",
    status: (whole.Status ?? "").trim(),
    tail: tail.split(/\nStatus: /)[0] ?? "",
    truncated: !!tail,
  };
}

function messages(text: string): Message[] {
  return [...text.matchAll(MESSAGE)].map(([, role, content, closed]) => ({
    role,
    content: tidy(decode(content)) + (closed ? "" : "…"),
  }));
}

function compact(text: string): string {
  const trimmed = text.trim();
  const parsed = parseJson(trimmed);
  const flat = parsed !== null && typeof parsed === "object" ? JSON.stringify(parsed) : trimmed;
  return tidy(flat).slice(0, SNIPPET);
}

export function stepFailure(preview: string): string | null {
  const { status } = parts(preview);
  if (!status || OK_STATUS.test(status)) return null;
  return status.replace(/^STATUS_CODE_ERROR\s*/, "") || "failed";
}

export function userAsk(preview: string): string | null {
  const asked = messages(parts(preview).input).filter((m) => m.role === "user" || m.role === "human");
  return asked.at(-1)?.content || null;
}

function replyFromTail(tail: string): string | null {
  const end = tail.trimEnd();
  if (!end.endsWith('"}]') || end.includes('\\"') || /"function"|\{"|":\s/.test(end)) return null;
  const text = tidy(end.slice(0, -3));
  return text ? `…${text}` : null;
}

export function assistantReply(preview: string): string | null {
  const { output, tail, truncated } = parts(preview);
  const said = messages(output).filter((m) => m.role === "assistant" && m.content && m.content !== "…");
  if (said.length) return said.at(-1)?.content ?? null;
  return truncated && !output ? replyFromTail(tail) : null;
}

export function toolCall(span: Pick<Span, "name" | "preview">): Omit<Extract<TimelineItem, { kind: "tool" }>, "span"> {
  const { input, output, status } = parts(span.preview);
  const result = compact(output);
  const failed = !!status && !OK_STATUS.test(status);
  return {
    kind: "tool",
    name: span.name.replace(/^execute_tool\s+/, ""),
    args: compact(input),
    result: result || (failed ? stepFailure(span.preview) ?? "" : ""),
    error: failed || TOOL_PROBLEM.test(output),
  };
}

function overlaps(a: string, b: string): boolean {
  const strip = (s: string) => s.replace(/…/g, "").trim();
  const [x, y] = [strip(a), strip(b)];
  return !!x && !!y && (x.includes(y) || y.includes(x));
}

function itemFor(span: Span, index: number): TimelineItem | null {
  if (span.kind === "tool") return { ...toolCall(span), span: index };
  if (span.kind === "llm") {
    const failure = stepFailure(span.preview);
    if (failure) return { kind: "failure", text: `Model call failed: ${failure}`, span: index };
    const reply = assistantReply(span.preview);
    return reply ? { kind: "reply", text: reply, span: index } : null;
  }
  return null;
}

function fallback(spans: readonly Span[]): TimelineItem[] {
  return spans.map((span, index) => {
    const { input, output } = parts(span.preview);
    const said = messages(input).filter((m) => m.role !== "system").at(-1)?.content;
    return { kind: "note", label: span.name, text: said ?? compact(output || input), span: index };
  });
}

export function timeline(spans: readonly Span[]): TimelineItem[] {
  const askAt = spans.findIndex((span) => userAsk(span.preview));
  const ask = askAt >= 0 ? userAsk(spans[askAt].preview) : null;
  const agentAt = spans.findIndex((span) => span.kind === "agent" && assistantReply(span.preview));
  const final = agentAt >= 0 ? assistantReply(spans[agentAt].preview) : null;
  const steps = spans.flatMap((span, index) => itemFor(span, index) ?? []);
  const middle = final ? steps.filter((item) => item.kind !== "reply" || !overlaps(item.text, final)) : steps;
  const items: TimelineItem[] = [
    ...(ask ? [{ kind: "ask" as const, text: ask, span: askAt }] : []),
    ...middle,
    ...(final ? [{ kind: "reply" as const, text: final, span: agentAt }] : []),
  ];
  return items.length ? items : fallback(spans);
}
