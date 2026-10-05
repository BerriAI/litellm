import type { Span, SpanDetail, TraceMessage, TraceToolCall, UIContent } from "../../types";
import { isFrameworkSpan, parseJson, parseMessages, prettyPayload } from "../../utils";
import { toTraceMessage } from "../content/payload";

export const CONVERSATION_PAGE_SIZE = 20;

export function conversationSteps(spans: readonly Span[]): Span[] {
  const parents = new Set(spans.map((span) => span.parent_span_id));
  return spans
    .filter((span) => {
      const isEvent = ["agent", "llm", "tool"].includes(span.type) || !parents.has(span.span_id);
      return !isFrameworkSpan(span) && (span.parent_span_id === null || isEvent);
    })
    .sort((a, b) => a.start_offset_ms - b.start_offset_ms);
}

function contentText(value: string, content?: UIContent): string {
  if (content?.kind === "text") return content.text;
  if (content?.kind === "fields")
    return JSON.stringify(Object.fromEntries(content.fields.map((field) => [field.key, field.value])), null, 2);
  if (content?.kind === "messages") return content.messages.map((message) => message.content).join("\n");
  return prettyPayload(value);
}

function messages(value: string, content: UIContent | undefined, role: string): TraceMessage[] {
  if (content?.kind === "messages") return content.messages.map(toTraceMessage);
  const parsed = !content ? parseMessages(value) : null;
  if (parsed) return parsed;
  const text = contentText(value, content);
  return text ? [{ role, content: text }] : [];
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value !== null && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, stableValue(item)]),
    );
  return value;
}

const messageKey = (message: TraceMessage): string =>
  JSON.stringify(
    stableValue({
      role: message.role,
      content: message.content,
      tool_calls: message.tool_calls?.length ? message.tool_calls : undefined,
    }),
  );

export function newConversationMessages(
  previous: readonly TraceMessage[],
  current: readonly TraceMessage[],
): TraceMessage[] {
  const oldKeys = previous.map(messageKey);
  const newKeys = current.map(messageKey);
  if (newKeys.every((key, index) => key === oldKeys[index])) return [];
  for (let overlap = Math.min(oldKeys.length, newKeys.length); overlap > 0; overlap--) {
    if (oldKeys.slice(-overlap).every((key, index) => key === newKeys[index])) return current.slice(overlap);
  }
  return [...current];
}

export interface ConversationItem {
  id: string;
  span: Span;
  messages: TraceMessage[];
  toolCall?: TraceToolCall;
  toolResult?: string;
  showError?: boolean;
}

function toolItem(
  span: Span,
  detail: SpanDetail,
  pending: TraceToolCall[],
  items: ConversationItem[],
): ConversationItem {
  const args =
    parseJson(detail.input) ??
    (detail.input_ui?.kind === "fields"
      ? Object.fromEntries(detail.input_ui.fields.map((field) => [field.key, field.value]))
      : detail.input);
  const call = { name: span.name, args };
  const match = pending.findIndex(
    (candidate) =>
      candidate.name === call.name &&
      JSON.stringify(stableValue(candidate.args)) === JSON.stringify(stableValue(call.args)),
  );
  if (match >= 0) {
    const [matched] = pending.splice(match, 1);
    for (const item of items)
      for (const message of item.messages) {
        if (message.tool_calls?.includes(matched))
          message.tool_calls = message.tool_calls.filter((call) => call !== matched);
      }
  }
  const result = contentText(detail.output, detail.output_ui);
  return { id: span.span_id, span, messages: [], toolCall: call, toolResult: result };
}

interface ConversationEvent {
  span: Span;
  time: number;
  output: boolean;
}

function conversationEvents(
  steps: readonly Span[],
  byId: ReadonlyMap<string, Span>,
  details: ReadonlyMap<string, SpanDetail>,
  complete: boolean,
): ConversationEvent[] {
  const missingIndex = steps.findIndex((span) => !details.has(span.span_id));
  const loaded = missingIndex < 0 ? steps : steps.slice(0, missingIndex);
  const boundary = missingIndex < 0 ? -Infinity : steps[missingIndex].start_offset_ms;
  const ends = new Map<string, number>();
  const depths = new Map<string, number>();
  for (const span of steps) {
    const end = span.start_offset_ms + span.duration_ms;
    const visited = new Set<string>();
    let ancestor: Span | undefined = span;
    while (ancestor && !visited.has(ancestor.span_id)) {
      visited.add(ancestor.span_id);
      ends.set(ancestor.span_id, Math.max(ends.get(ancestor.span_id) ?? -Infinity, end));
      ancestor = ancestor.parent_span_id ? byId.get(ancestor.parent_span_id) : undefined;
    }
    depths.set(span.span_id, visited.size);
  }
  return loaded
    .flatMap((span) => {
      const start = { span, time: span.start_offset_ms, output: false };
      if (span.type === "tool" || (span.type !== "agent" && span.parent_span_id !== null)) return [start];
      const end = ends.get(span.span_id)!;
      return (complete && missingIndex < 0) || end < boundary ? [start, { span, time: end, output: true }] : [start];
    })
    .sort(
      (a, b) =>
        a.time - b.time ||
        Number(a.output) - Number(b.output) ||
        (a.output ? -1 : 1) * (depths.get(a.span.span_id)! - depths.get(b.span.span_id)!),
    );
}

function isDescendant(id: string, ancestorId: string, byId: ReadonlyMap<string, Span>): boolean {
  let parentId = byId.get(id)?.parent_span_id;
  const visited = new Set<string>();
  while (parentId && !visited.has(parentId)) {
    if (parentId === ancestorId) return true;
    visited.add(parentId);
    parentId = byId.get(parentId)?.parent_span_id;
  }
  return false;
}

function withoutForwardedAnswers(
  spanId: string,
  output: TraceMessage[],
  completedOutputs: ReadonlyMap<string, TraceMessage[]>,
  byId: ReadonlyMap<string, Span>,
): TraceMessage[] {
  return [...completedOutputs].reduce(
    (fresh, [childId, childOutput]) =>
      isDescendant(childId, spanId, byId) ? newConversationMessages(childOutput, fresh) : fresh,
    output,
  );
}

export function buildConversation(
  spans: readonly Span[],
  details: ReadonlyMap<string, SpanDetail>,
  complete: boolean,
): ConversationItem[] {
  const byId = new Map(spans.map((span) => [span.span_id, span]));
  const histories = new Map<string, TraceMessage[]>();
  const completedOutputs = new Map<string, TraceMessage[]>();
  const pendingCalls = new Map<string, TraceToolCall[]>();
  const items: ConversationItem[] = [];
  const events = conversationEvents(conversationSteps(spans), byId, details, complete);
  const branch = (span: Span): string => {
    if (span.type === "agent" || span.parent_span_id === null) return span.span_id;
    let parent = span.parent_span_id ? byId.get(span.parent_span_id) : undefined;
    const visited = new Set<string>();
    while (parent && !visited.has(parent.span_id)) {
      visited.add(parent.span_id);
      if (parent.type === "agent") return parent.span_id;
      parent = parent.parent_span_id ? byId.get(parent.parent_span_id) : undefined;
    }
    return span.parent_span_id ?? span.span_id;
  };
  for (const event of events) {
    const { span } = event;
    const detail = details.get(span.span_id)!;
    const key = branch(span);
    const history = histories.get(key) ?? [];
    if (event.output) {
      const output = messages(detail.output, detail.output_ui, "assistant");
      const fresh = withoutForwardedAnswers(
        span.span_id,
        newConversationMessages(history, output),
        completedOutputs,
        byId,
      );
      const item = { id: `${span.span_id}-output`, span, messages: fresh, showError: span.status === "error" };
      if (fresh.length || item.showError) items.push(item);
      completedOutputs.set(span.span_id, output);
      histories.set(key, [...history, ...fresh]);
      continue;
    }
    if (span.type === "tool") {
      const item = toolItem(span, detail, pendingCalls.get(key) ?? [], items);
      items.push(item);
      histories.set(key, [...history, { role: "tool", name: span.name, content: item.toolResult ?? "" }]);
      continue;
    }
    const input = messages(detail.input, detail.input_ui, "user");
    const output = messages(detail.output, detail.output_ui, "assistant");
    const fresh = newConversationMessages(history, input);
    if (span.type === "agent" || span.parent_span_id === null) {
      histories.set(key, input);
      if (fresh.length) items.push({ id: span.span_id, span, messages: fresh });
      continue;
    }
    const combined = [...fresh, ...output];
    const retained = input.length && (fresh.length || input.length >= history.length) ? input : history;
    histories.set(key, [...retained, ...output]);
    pendingCalls.set(
      key,
      output.flatMap((message) => message.tool_calls ?? []),
    );
    const item = {
      id: span.span_id,
      span,
      showError: span.status === "error",
      messages: combined.map((message) => ({
        ...message,
        tool_calls: message.tool_calls ? [...message.tool_calls] : undefined,
      })),
    };
    if (combined.length || item.showError) items.push(item);
  }
  return items
    .map((item) => ({
      ...item,
      messages: item.messages.filter((message) => Boolean(message.content) || Boolean(message.tool_calls?.length)),
    }))
    .filter((item) => item.messages.length || item.toolResult !== undefined || item.showError);
}
