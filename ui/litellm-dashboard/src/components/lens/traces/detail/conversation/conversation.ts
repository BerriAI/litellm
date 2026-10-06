import { groupBy, orderBy } from "es-toolkit";
import type { Span, SpanDetail, TraceMessage, TraceToolCall, UIContent } from "../../types";
import { isFrameworkSpan, parseAssistantSummary, parseMessages, prettyPayload } from "../../utils";
import { toTraceMessage, toolInput } from "../content/payload";
import { claudeCaptureWarnings, claudeToolDetails, isClaudeSupplement } from "./nativeClaude";

export const CONVERSATION_PAGE_SIZE = 20;

export function conversationSteps(spans: readonly Span[]): Span[] {
  const parents = new Set(spans.map((span) => span.parent_span_id));
  return spans
    .filter((span) => {
      const isEvent = ["agent", "llm", "tool"].includes(span.type) || !parents.has(span.span_id);
      const visible = !isFrameworkSpan(span) && (span.parent_span_id === null || isEvent);
      return isClaudeSupplement(span) || visible;
    })
    .sort((a, b) => a.start_offset_ms - b.start_offset_ms);
}

export function pendingConversationBranches(
  spans: readonly Span[],
  details: ReadonlyMap<string, SpanDetail>,
  hasMoreSpans: boolean,
): ReadonlySet<string> {
  const byId = new Map(spans.map((span) => [span.span_id, span]));
  const steps = new Set(conversationSteps(spans).map((span) => span.span_id));
  const pageBoundary = spans.reduce((latest, span) => Math.max(latest, span.start_offset_ms), -Infinity);
  const pending = spans.filter((span) => {
    const missingDetail = steps.has(span.span_id) && !details.has(span.span_id);
    const mayHaveLaterChildren = hasMoreSpans && span.start_offset_ms + span.duration_ms >= pageBoundary;
    return missingDetail || mayHaveLaterChildren;
  });
  const initial = {
    pending: new Set(pending.map((span) => span.span_id)),
    ancestors: new Map(
      spans.flatMap((span) =>
        span.parent_span_id && byId.has(span.parent_span_id) ? [[span.span_id, span.parent_span_id] as const] : [],
      ),
    ),
  };
  const doublingPasses = Math.ceil(Math.log2(Math.max(1, spans.length)));
  return Array.from({ length: doublingPasses }).reduce<typeof initial>((state) => {
    if (!state.pending.size || !state.ancestors.size) return state;
    return {
      pending: new Set([
        ...state.pending,
        ...[...state.pending].flatMap((id) => {
          const ancestor = state.ancestors.get(id);
          return ancestor ? [ancestor] : [];
        }),
      ]),
      ancestors: new Map(
        [...state.ancestors].flatMap(([id, ancestor]) => {
          const next = state.ancestors.get(ancestor);
          return next ? [[id, next] as const] : [];
        }),
      ),
    };
  }, initial).pending;
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
  const source = content?.kind === "text" ? content.text : value;
  const parsed = parseMessages(source) ?? (role === "assistant" ? parseAssistantSummary(source) : null);
  if (parsed) return parsed;
  const text = contentText(value, content);
  return text ? [{ role, content: text }] : [];
}

function inputMessages(detail: SpanDetail): TraceMessage[] {
  return detail.attributes["lens.capture.messages_separate"] === "true"
    ? []
    : messages(detail.input, detail.input_ui, "user");
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
  agentId?: string;
  agentName?: string;
  branchId?: string;
  parentBranchId?: string;
  model?: string;
  time?: number;
  showError?: boolean;
}

function toolItem(
  span: Span,
  detail: SpanDetail,
  pending: TraceToolCall[],
  items: ConversationItem[],
): ConversationItem {
  const args = toolInput(detail.input, detail.input_ui);
  const call = { name: span.name, args };
  const match = pending.findIndex(
    (candidate) =>
      candidate.name === call.name &&
      (candidate.args === undefined ||
        JSON.stringify(stableValue(candidate.args)) === JSON.stringify(stableValue(call.args))),
  );
  if (match >= 0) {
    const [matched] = pending.splice(match, 1);
    for (const item of items)
      for (const message of item.messages) {
        if (message.tool_calls?.includes(matched))
          message.tool_calls = message.tool_calls.filter((call) => call !== matched);
      }
  }
  const result =
    detail.output_ui?.kind === "text"
      ? detail.output_ui.text
      : prettyPayload(detail.output) || contentText(detail.output, detail.output_ui);
  return { id: span.span_id, span, messages: [], toolCall: call, toolResult: result };
}

interface ConversationEvent {
  span: Span;
  time: number;
  output: boolean;
}

function messageTime(span: Span, steps: readonly Span[], details: ReadonlyMap<string, SpanDetail>): number {
  const attributes = details.get(span.span_id)?.attributes;
  if (attributes?.["event.name"] !== "assistant_response") return span.start_offset_ms;
  const source = attributes["query_source"];
  const matches = steps.filter((candidate) => {
    const recorded = details.get(candidate.span_id)?.attributes["query_source_safe"];
    const unnamedAgent = recorded === "agent" && source?.startsWith("agent:");
    const sameSource = recorded === source || recorded === source?.replace(/^agent:/, "agent.") || unnamedAgent;
    const sameRequest = candidate.parent_span_id === span.parent_span_id && candidate.model === span.model;
    const matchingCall = candidate.type === "llm" && sameRequest && sameSource;
    return matchingCall && Math.abs(candidate.start_offset_ms + candidate.duration_ms - span.start_offset_ms) < 2;
  });
  if (matches.length !== 1) return span.start_offset_ms;
  const request = matches[0];
  const firstContent = Number(details.get(request.span_id)?.attributes["first_content_ms"]);
  return Number.isFinite(firstContent) && firstContent >= 0 && firstContent <= request.duration_ms
    ? request.start_offset_ms + firstContent
    : span.start_offset_ms;
}

function conversationEvents(
  steps: readonly Span[],
  byId: ReadonlyMap<string, Span>,
  details: ReadonlyMap<string, SpanDetail>,
  { complete, pendingBranches }: { complete: boolean; pendingBranches?: ReadonlySet<string> },
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
      const start = { span, time: messageTime(span, loaded, details), output: false };
      if (span.type === "tool" || (span.type !== "agent" && span.parent_span_id !== null)) return [start];
      const end = ends.get(span.span_id)!;
      const branchComplete = (complete && missingIndex < 0) || pendingBranches?.has(span.span_id) === false;
      return branchComplete || end < boundary ? [start, { span, time: end, output: true }] : [start];
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

function isNativeAgent(span: Span): boolean {
  return span.framework === "claude-code" && span.type === "tool" && ["Agent", "Task"].includes(span.name);
}

function conversationBranch(span: Span, byId: ReadonlyMap<string, Span>): string {
  if (span.type === "agent" || span.parent_span_id === null) return span.span_id;
  let parent = span.parent_span_id ? byId.get(span.parent_span_id) : undefined;
  const visited = new Set<string>();
  while (parent && !visited.has(parent.span_id)) {
    visited.add(parent.span_id);
    if (parent.type === "agent" || isNativeAgent(parent)) return parent.span_id;
    parent = parent.parent_span_id ? byId.get(parent.parent_span_id) : undefined;
  }
  return span.parent_span_id ?? span.span_id;
}

function agentIdentity(span: Span, details: ReadonlyMap<string, SpanDetail>): string {
  return isNativeAgent(span) ? span.span_id : details.get(span.span_id)?.attributes["gen_ai.agent.id"] || span.span_id;
}

function agentLabels(agents: readonly Span[], details: ReadonlyMap<string, SpanDetail>): ReadonlyMap<string, string> {
  const identity = (agent: Span): string => agentIdentity(agent, details);
  const unique = agents.filter(
    (agent, index) => agents.findIndex((other) => identity(other) === identity(agent)) === index,
  );
  const named = unique.map((agent) => {
    const detail = details.get(agent.span_id);
    const args = detail && isNativeAgent(agent) ? toolInput(detail.input, detail.input_ui) : undefined;
    const description = args && typeof args === "object" && "description" in args ? args.description : undefined;
    const name =
      agent.framework === "claude-code" && !isNativeAgent(agent)
        ? agent.agent || agent.name || "Agent"
        : agent.name || agent.agent || "Agent";
    return { agent, name: typeof description === "string" && description ? description : name };
  });
  return new Map(
    Object.values(groupBy(named, ({ name }) => JSON.stringify(name))).flatMap((group) =>
      orderBy(group, [({ agent }) => agent.start_offset_ms, ({ agent }) => agent.span_id], ["asc", "asc"]).map(
        ({ agent, name }, index) => [identity(agent), group.length > 1 ? `${name} (${index + 1})` : name] as const,
      ),
    ),
  );
}

export function buildConversation(
  spans: readonly Span[],
  recordedDetails: ReadonlyMap<string, SpanDetail>,
  complete: boolean,
  pendingBranches?: ReadonlySet<string>,
): ConversationItem[] {
  const details = claudeToolDetails(spans, recordedDetails);
  const byId = new Map(spans.map((span) => [span.span_id, span]));
  const histories = new Map<string, TraceMessage[]>();
  const completedOutputs = new Map<string, TraceMessage[]>();
  const pendingCalls = new Map<string, TraceToolCall[]>();
  const items: ConversationItem[] = [];
  const events = conversationEvents(conversationSteps(spans), byId, details, { complete, pendingBranches });
  const branch = (span: Span): string => conversationBranch(span, byId);
  for (const event of events) {
    const { span } = event;
    if (isClaudeSupplement(span)) continue;
    const detail = details.get(span.span_id)!;
    if (["generate_session_title", "prompt_suggestion"].includes(detail.attributes["query_source_safe"])) continue;
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
      const item = {
        id: `${span.span_id}-output`,
        span,
        time: event.time,
        messages: fresh,
        showError: span.status === "error",
      };
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
    const input = inputMessages(detail);
    const output = messages(detail.output, detail.output_ui, "assistant");
    const fresh =
      detail.attributes["lens.capture.source"] === "session_transcript"
        ? input
        : newConversationMessages(history, input);
    if (span.type === "agent" || span.parent_span_id === null) {
      histories.set(key, input);
      const item = { id: span.span_id, span, time: event.time, messages: fresh };
      if (fresh.length) items.push(item);
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
      time: event.time,
      showError: span.status === "error",
      messages: combined.map((message) => ({
        ...message,
        tool_calls: message.tool_calls ? [...message.tool_calls] : undefined,
      })),
    };
    if (combined.length || item.showError) items.push(item);
  }
  const agents = [...new Set(conversationSteps(spans).map(branch))].flatMap((id) => {
    const span = byId.get(id);
    return span ? [span] : [];
  });
  const labels = agentLabels(agents, details);
  const actor = (id: string): string => {
    const span = byId.get(id);
    return span ? agentIdentity(span, details) : id;
  };
  return items
    .map((item) => ({
      ...item,
      agentId: actor(branch(item.span)),
      agentName: labels.get(actor(branch(item.span))) || item.span.agent,
      branchId: branch(item.span),
      parentBranchId: (() => {
        const parentId = byId.get(branch(item.span))?.parent_span_id;
        const parent = parentId ? byId.get(parentId) : undefined;
        return parent ? branch(parent) : undefined;
      })(),
      model: item.span.model || details.get(item.span.span_id)?.attributes["gen_ai.request.model"] || undefined,
      messages: item.messages.filter((message) => Boolean(message.content) || Boolean(message.tool_calls?.length)),
    }))
    .filter((item) => item.messages.length || item.toolResult !== undefined || item.showError);
}
export type ConversationGroup =
  | { kind: "item"; item: ConversationItem }
  | { kind: "branch"; id: string; name: string; children: ConversationGroup[] };

export function conversationBranchGroups(groups: ConversationGroup[], branchId: string | null): ConversationGroup[] {
  if (branchId === null) return groups;
  for (const group of groups) {
    if (group.kind === "item") continue;
    if (group.id === branchId) return group.children;
    const nested = conversationBranchGroups(group.children, branchId);
    if (nested.length) return nested;
  }
  return [];
}

export function groupConversation(items: readonly ConversationItem[], spans: readonly Span[]): ConversationGroup[] {
  const byId = new Map(spans.map((span) => [span.span_id, span]));
  const parentById = new Map(
    conversationSteps(spans).flatMap((span) => {
      const branch = byId.get(conversationBranch(span, byId));
      if (!branch) return [];
      const parent = branch.parent_span_id ? byId.get(branch.parent_span_id) : undefined;
      return [[branch.span_id, parent ? conversationBranch(parent, byId) : undefined] as const];
    }),
  );
  const roots = new Set(
    [...parentById].flatMap(([id, parent]) => {
      if (!parent) return [id];
      return parentById.has(parent) ? [] : [parent];
    }),
  );
  const directBranch = (item: ConversationItem, parent?: string): string | undefined => {
    let id = item.branchId;
    const visited = new Set<string>();
    while (id && !visited.has(id)) {
      visited.add(id);
      const ancestor = parentById.get(id);
      if (parent ? ancestor === parent : ancestor !== undefined && roots.has(ancestor)) return id;
      id = ancestor;
    }
    return undefined;
  };
  const build = (parent?: string, ancestors = new Set<string>()): ConversationGroup[] => {
    const seen = new Set<string>();
    return items.flatMap((item): ConversationGroup[] => {
      if (parent ? item.branchId === parent : !item.parentBranchId) return [{ kind: "item", item }];
      const id = directBranch(item, parent);
      if (!id || seen.has(id) || ancestors.has(id)) return [];
      seen.add(id);
      const first = items.find((candidate) => candidate.branchId === id);
      const name = first?.agentName || byId.get(id)?.name || "Subagent";
      return [{ kind: "branch", id, name, children: build(id, new Set([...ancestors, id])) }];
    });
  };
  return build();
}

export function conversationWarnings(details: ReadonlyMap<string, SpanDetail>, complete: boolean): string[] {
  const warnings = [
    ...claudeCaptureWarnings(details),
    ...[...details.values()].flatMap((detail) =>
      detail.attributes["lens.capture.warning"] ? [detail.attributes["lens.capture.warning"]] : [],
    ),
  ];
  if (
    complete &&
    ![...details.values()].some((detail) => detail.attributes["event.name"] === "assistant_response") &&
    [...details.values()].some(
      (detail) =>
        detail.attributes["span.type"] === "llm_request" &&
        !detail.output &&
        !messages(detail.output, detail.output_ui, "assistant").length,
    )
  ) {
    warnings.push(
      "This Claude Code trace has no recorded assistant replies. Enable assistant response logs for future sessions.",
    );
  }
  return [...new Set(warnings)];
}
