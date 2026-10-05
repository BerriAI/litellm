/**
 * Pure helpers for the agent trace views. No React in here: everything the span tree
 * computes lives here so it can be unit-tested directly.
 */
import { type ErrorSource, type SpanTreeState, type TreeRow } from "./tree";
import type { Span, TraceMessage, TraceSummary, TraceToolCall } from "./types";

/* ------------------------------------------------------------------ */
/*  Formatting                                                         */
/* ------------------------------------------------------------------ */

export const traceAgentNames = (trace: TraceSummary): readonly string[] =>
  trace.agent_names ?? (trace.service ? [trace.service] : []);

export const fmtMs = (ms: number): string => {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.max(ms, 0).toFixed(ms < 10 ? 1 : 0)}ms`;
};

export const fmtTok = (n: number): string => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

export const shortId = (id: string, length = 16): string => (id.length > length ? `${id.slice(0, length)}…` : id);

/* ------------------------------------------------------------------ */
/*  Visible tree (framework spans hidden + children re-parented)       */
/* ------------------------------------------------------------------ */

/** Root-level key in the children map. */
export const ROOT_KEY = "__root__";

export type ChildrenMap = Map<string, Span[]>;

export interface VisibleTree {
  children: ChildrenMap;
  visibleCount: number;
}

/**
 * Framework plumbing: middleware wrappers plus LangGraph's generic "model" / "tools"
 * graph nodes. The root span is never hidden.
 */
const GRAPH_NODE_NAMES = new Set(["model", "tools"]);

export const isFrameworkSpan = (span: Span): boolean => {
  const isGraphNode = span.type === "chain" && GRAPH_NODE_NAMES.has(span.name);
  const isPlumbing = span.type === "framework" || isGraphNode;
  return span.parent_span_id !== null && isPlumbing;
};

const byStart = (a: Span, b: Span): number => a.start_offset_ms - b.start_offset_ms;

export const indexSpans = (spans: readonly Span[]): Map<string, Span> => new Map(spans.map((s) => [s.span_id, s]));

const pushChild = (children: ChildrenMap, key: string, span: Span): void => {
  const list = children.get(key);
  if (list) list.push(span);
  else children.set(key, [span]);
};

/**
 * Children map for the waterfall. With `showFramework` off, framework spans are
 * dropped and their children attach to the nearest visible ancestor. Spans whose
 * parent is missing from the trace attach to the root level.
 */
export function buildVisibleTree(spans: readonly Span[], showFramework: boolean): VisibleTree {
  const byId = indexSpans(spans);
  const hidden = (span: Span) => !showFramework && isFrameworkSpan(span);
  const visibleParentKey = (span: Span): string => {
    let parent = span.parent_span_id ? byId.get(span.parent_span_id) : undefined;
    while (parent && hidden(parent)) {
      parent = parent.parent_span_id ? byId.get(parent.parent_span_id) : undefined;
    }
    return parent ? parent.span_id : ROOT_KEY;
  };
  const children: ChildrenMap = new Map();
  let visibleCount = 0;
  for (const span of spans) {
    if (hidden(span)) continue;
    visibleCount++;
    pushChild(children, visibleParentKey(span), span);
  }
  children.forEach((list) => list.sort(byStart));
  return { children, visibleCount };
}

/* ------------------------------------------------------------------ */
/*  Run view rows (Input, span tree with grouping, Output)             */
/* ------------------------------------------------------------------ */

/** Same-named siblings fold into one row at this count ... */
export const GROUP_THRESHOLD_OK = 6;
/** ... or as soon as this many of them failed. */
export const GROUP_THRESHOLD_ERROR = 3;
/** Group rows reveal this many members at a time. */
export const GROUP_PAGE_SIZE = 20;

export const median = (values: readonly number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

const LITELLM_ERROR = /rate.?limit|429|guardrail|budget|litellm/i;

/** Who failed: the tool, LiteLLM (rate limit / guardrail / budget), or the model. Null when the span is fine. */
export function errorSource(span: Span): ErrorSource | null {
  if (span.status !== "error") return null;
  if (span.type === "tool") return "tool";
  if (LITELLM_ERROR.test(span.error ?? "")) return "litellm";
  return "model";
}

type GroupOrSpan = Span | { group: Span[] };

const groupKey = (span: Pick<Span, "name" | "type" | "agent">): string => `${span.agent}|${span.name}|${span.type}`;

/** Siblings sharing agent + name + type fold into one group once there are enough of them (or enough failures). */
function groupChildren(children: readonly Span[]): GroupOrSpan[] {
  const byKey = new Map<string, Span[]>();
  for (const child of children) {
    const key = groupKey(child);
    const list = byKey.get(key);
    if (list) list.push(child);
    else byKey.set(key, [child]);
  }
  const emitted = new Set<string>();
  const out: GroupOrSpan[] = [];
  for (const child of children) {
    const key = groupKey(child);
    const group = byKey.get(key) ?? [child];
    const failed = group.filter((s) => s.status === "error").length;
    if (group.length >= GROUP_THRESHOLD_OK || failed >= GROUP_THRESHOLD_ERROR) {
      if (!emitted.has(key)) {
        emitted.add(key);
        out.push({ group });
      }
    } else {
      out.push(child);
    }
  }
  return out;
}

export const groupRowId = (parentKey: string, span: Pick<Span, "name" | "type" | "agent">): string =>
  `grp::${parentKey}::${groupKey(span)}`;

interface RowContext {
  children: ChildrenMap;
  state: SpanTreeState;
  rows: TreeRow[];
}

function pushSpan(ctx: RowContext, span: Span, depth: number): void {
  const hasChildren = (ctx.children.get(span.span_id)?.length ?? 0) > 0;
  const collapsed = ctx.state.collapsedSpanIds.has(span.span_id);
  const row: TreeRow = { kind: "span", id: span.span_id, span, depth, hasChildren, collapsed };
  ctx.rows.push(row);
  if (hasChildren && !collapsed) pushLevel(ctx, span.span_id, depth + 1);
}

function pushGroup(ctx: RowContext, parentKey: string, members: Span[], depth: number): void {
  const first = members[0];
  const id = groupRowId(parentKey, first);
  const failedCount = members.filter((s) => s.status === "error").length;
  const expanded = ctx.state.expandedGroupIds.has(id);
  const groupRow: TreeRow = {
    kind: "group",
    id,
    depth,
    name: first.name,
    type: first.type,
    agent: first.agent,
    members,
    failedCount,
    p50Duration: median(members.map((s) => s.duration_ms)),
    isFailureGroup: failedCount === members.length,
    expanded,
  };
  ctx.rows.push(groupRow);
  if (!expanded) return;
  const reveal = Math.min(ctx.state.groupRevealCounts[id] ?? GROUP_PAGE_SIZE, members.length);
  members.slice(0, reveal).forEach((member) => pushSpan(ctx, member, depth + 1));
  if (reveal < members.length) {
    const moreRow: TreeRow = {
      kind: "load-more",
      id: `${id}::more`,
      depth: depth + 1,
      groupId: id,
      remaining: members.length - reveal,
    };
    ctx.rows.push(moreRow);
  }
}

function pushLevel(ctx: RowContext, parentKey: string, depth: number): void {
  for (const item of groupChildren(ctx.children.get(parentKey) ?? [])) {
    if ("group" in item) pushGroup(ctx, parentKey, item.group, depth);
    else pushSpan(ctx, item, depth);
  }
}

/**
 * Rows for the run view: the span tree, framework spans optionally hidden with their children lifted
 * and same-named siblings folded into paged groups.
 */
export function buildTreeRows(spans: readonly Span[], state: SpanTreeState): TreeRow[] {
  const ctx: RowContext = { children: buildVisibleTree(spans, !state.hideFramework).children, state, rows: [] };
  pushLevel(ctx, ROOT_KEY, 0);
  return ctx.rows;
}

export function findTraceSteps(
  spans: readonly Span[],
  query: string,
  errorsOnly: boolean,
  hideFramework: boolean,
): Span[] {
  const search = query.trim().toLowerCase();
  return spans
    .filter((span) => {
      const visible = !hideFramework || !isFrameworkSpan(span);
      const matchesStatus = !errorsOnly || span.status === "error";
      const matchesQuery =
        !search ||
        [span.name, span.agent, span.model, span.input_preview, span.span_id].some((value) =>
          value?.toLowerCase().includes(search),
        );
      return visible && matchesStatus && matchesQuery;
    })
    .sort(byStart);
}

/** Tree state with every visible ancestor of `spanId` expanded and any group holding it paged far enough. */
export function revealSpanInState(spans: readonly Span[], state: SpanTreeState, spanId: string): SpanTreeState {
  const { children } = buildVisibleTree(spans, !state.hideFramework);
  const parentOf = new Map<string, string>();
  children.forEach((list, key) => list.forEach((s) => parentOf.set(s.span_id, key)));
  if (!parentOf.has(spanId)) return state;
  const collapsed = new Set(state.collapsedSpanIds);
  const expanded = new Set(state.expandedGroupIds);
  const reveal = { ...state.groupRevealCounts };
  let current = spanId;
  while (parentOf.has(current)) {
    const parentKey = parentOf.get(current) as string;
    collapsed.delete(parentKey);
    for (const item of groupChildren(children.get(parentKey) ?? [])) {
      if (!("group" in item)) continue;
      const index = item.group.findIndex((s) => s.span_id === current);
      if (index < 0) continue;
      const id = groupRowId(parentKey, item.group[0]);
      expanded.add(id);
      reveal[id] = Math.max(reveal[id] ?? GROUP_PAGE_SIZE, Math.ceil((index + 1) / GROUP_PAGE_SIZE) * GROUP_PAGE_SIZE);
    }
    current = parentKey;
  }
  return { ...state, collapsedSpanIds: collapsed, expandedGroupIds: expanded, groupRevealCounts: reveal };
}

/** `spanId` if it shows in the tree, else its nearest ancestor that does (framework spans can be hidden). */
export function nearestVisibleSpanId(spans: readonly Span[], spanId: string, hideFramework: boolean): string {
  if (!hideFramework) return spanId;
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  let current = byId.get(spanId);
  while (current && isFrameworkSpan(current) && current.parent_span_id !== null) {
    current = byId.get(current.parent_span_id);
  }
  return current?.span_id ?? spanId;
}

/* ------------------------------------------------------------------ */
/*  Trace-level rollups                                                */
/* ------------------------------------------------------------------ */

/** Earliest failing non-root span (the root just echoes its children), else the root. */
export function firstErrorSpan(spans: readonly Span[]): Span | null {
  const failed = spans.filter((s) => s.status === "error").sort(byStart);
  return failed.find((s) => s.parent_span_id !== null) ?? failed[0] ?? null;
}

/* ------------------------------------------------------------------ */
/*  Span detail payloads                                               */
/* ------------------------------------------------------------------ */

export const parseJson = (value: string): unknown => {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const blockText = (block: unknown): string | null => {
  if (typeof block !== "object" || block === null) return null;
  const text: unknown =
    Reflect.get(block, "text") ?? (Reflect.get(block, "type") === "text" ? Reflect.get(block, "content") : undefined);
  return typeof text === "string" ? text : null;
};

const NON_TEXT_BLOCKS: ReadonlySet<unknown> = new Set(["reasoning", "function_call", "tool_use", "tool_call"]);

const blockType = (block: unknown): unknown =>
  typeof block === "object" && block !== null ? Reflect.get(block, "type") : undefined;

const isKnownBlock = (block: unknown): boolean => blockText(block) !== null || NON_TEXT_BLOCKS.has(blockType(block));

/** Block-list content (reasoning / function_call / text blocks) as its text; unknown content unchanged. */
export function messageText(content: string): string {
  if (!content.startsWith("[")) return content;
  const blocks = parseJson(content);
  if (!Array.isArray(blocks) || blocks.length === 0 || !blocks.every(isKnownBlock)) return content;
  return blocks
    .map(blockText)
    .filter((text): text is string => text !== null)
    .join("\n\n");
}

const LANGCHAIN_ROLE: Readonly<Record<string, string>> = {
  human: "user",
  ai: "assistant",
  system: "system",
  tool: "tool",
};

const isMessageContent = (value: unknown): value is string | unknown[] =>
  typeof value === "string" || Array.isArray(value);

const langchainToolCalls = (data: object): TraceToolCall[] | undefined => {
  const calls: unknown = Reflect.get(data, "tool_calls");
  if (!Array.isArray(calls) || calls.length === 0) return undefined;
  return calls
    .filter((call): call is object => typeof call === "object" && call !== null)
    .map((call) => ({ name: String(Reflect.get(call, "name") ?? "tool"), args: Reflect.get(call, "args") ?? {} }));
};

/** LangChain's `dumpd` message: `{"type": "human", "data": {"content": ...}}`. */
const parseLangchainMessage = (value: object): TraceMessage | null => {
  const role = LANGCHAIN_ROLE[String(Reflect.get(value, "type"))];
  const data: unknown = Reflect.get(value, "data");
  if (!role || typeof data !== "object" || data === null) return null;
  const content: unknown = Reflect.get(data, "content");
  if (!isMessageContent(content)) return null;
  const name: unknown = Reflect.get(data, "name");
  const toolCalls = langchainToolCalls(data);
  return {
    role,
    content: messageText(typeof content === "string" ? content : JSON.stringify(content)),
    ...(typeof name === "string" && name ? { name } : {}),
    ...(toolCalls ? { tool_calls: toolCalls } : {}),
  };
};

const parseToolCall = (call: unknown): TraceToolCall => {
  const unknownCall = { name: "Tool call", args: call };
  if (!call || typeof call !== "object") return unknownCall;
  const fn: unknown = Reflect.get(call, "function");
  const source = fn && typeof fn === "object" ? fn : call;
  const name: unknown = Reflect.get(source, "name");
  if (typeof name !== "string" || !name) return unknownCall;
  if ("args" in source) return { name, args: source.args };
  const args: unknown = Reflect.get(source, "arguments");
  return { name, args: typeof args === "string" ? parseJson(args) ?? args : args };
};

const parseMessage = (value: unknown): TraceMessage | null => {
  if (typeof value !== "object" || value === null) return null;
  if (!("role" in value) && "data" in value) return parseLangchainMessage(value);
  const role: unknown = Reflect.get(value, "role");
  const content: unknown = Reflect.get(value, "content") ?? Reflect.get(value, "parts");
  const rawCalls: unknown = Reflect.get(value, "tool_calls");
  const callEntries = Array.isArray(rawCalls) ? rawCalls : [rawCalls];
  const hasCalls = rawCalls != null && callEntries.length > 0;
  const emptyToolMessage = content == null && hasCalls;
  const hasContent = isMessageContent(content) || emptyToolMessage;
  if (typeof role !== "string" || !hasContent) return null;
  const calls = callEntries.map(parseToolCall);
  const text = typeof content === "string" ? content : JSON.stringify(content ?? "");
  return {
    ...value,
    role,
    content: content == null ? "" : messageText(text),
    ...(rawCalls != null ? { tool_calls: calls } : {}),
  };
};

export function parseAssistantSummary(value: string): TraceMessage[] | null {
  const parsed = parseJson(value);
  if (!Array.isArray(parsed) || !parsed.length) return null;
  const isSummary = (item: unknown): item is { content: string | null; tool_names: string[] } => {
    if (!item || typeof item !== "object") return false;
    const content: unknown = Reflect.get(item, "content");
    const tools: unknown = Reflect.get(item, "tool_names");
    const knownFields = Object.keys(item).every((key) => key === "content" || key === "tool_names");
    const validContent = content === null || typeof content === "string";
    const validTools = Array.isArray(tools) && tools.every((name: unknown) => typeof name === "string");
    return knownFields && validContent && validTools;
  };
  if (!parsed.every(isSummary)) return null;
  return parsed.map((item) => ({
    role: "assistant",
    content: item.content ?? "",
    tool_calls: item.tool_names.map((name: string) => ({ name, args: undefined })),
  }));
}

/** An llm span's input (array of messages) or output (one message); null when it isn't one. */
export function parseMessages(value: string): TraceMessage[] | null {
  const parsed = parseJson(value);
  const messages = (Array.isArray(parsed) ? parsed : [parsed]).map(parseMessage);
  return messages.every((message) => message !== null) ? messages : null;
}

/** Pretty JSON when the payload is JSON, else the raw string. */
export const prettyPayload = (value: string): string => {
  const parsed = parseJson(value);
  if (parsed === null || typeof parsed === "string") return typeof parsed === "string" ? parsed : value;
  return JSON.stringify(parsed, null, 2);
};

/* ------------------------------------------------------------------ */
/*  List view helpers                                                  */
/* ------------------------------------------------------------------ */

const PREVIEW_USER_CONTENT =
  /"(?:role|type)":\s*"(?:user|human)",\s*"(?:content|data)":\s*(?:\{"content":\s*)?"((?:[^"\\]|\\.)*)/;

/**
 * Human text for an input preview. Previews are often a (possibly truncated) JSON
 * message array; show the last user/tool message's content when we can find it.
 */
const decodeJsonString = (escaped: string): string => {
  const complete = escaped.replace(/\\u[0-9a-fA-F]{0,3}$|\\$/, "");
  const parsed = parseJson(`"${complete}"`);
  return typeof parsed === "string" ? parsed : complete;
};

export function previewText(preview: string): string {
  if (!preview || /^\s*\[\s*\]\s*$/.test(preview)) return "";
  const messages = parseMessages(preview);
  if (messages) {
    const last = [...messages].reverse().find((m) => m.role === "user" || m.role === "tool") ?? messages.at(-1);
    return last?.content || "";
  }
  if (!/^\s*[[{]/.test(preview)) return preview;
  const match = PREVIEW_USER_CONTENT.exec(preview);
  return match ? decodeJsonString(match[1]).replace(/\s+/g, " ") : preview;
}

/** Trace display name; root spans without a name fall back to the service. */
export const traceDisplayName = (summary: Pick<TraceSummary, "name" | "service">): string =>
  summary.name || summary.service || "(unnamed trace)";

/* ------------------------------------------------------------------ */
/*  Tree connector guides                                              */
/* ------------------------------------------------------------------ */

export interface TreeGuide {
  /** Per ancestor level 1..depth-1: whether that ancestor has a later sibling, so its rail continues. */
  rails: readonly boolean[];
  /** Last child of its parent: the elbow ends here instead of continuing down. */
  last: boolean;
  /** The next row is this row's child, so a stem runs down from this row's tile. */
  stem: boolean;
}

/** Rail / elbow / stem flags for each row from row depths alone, in one backward pass. */
export function treeGuides(depths: readonly number[]): TreeGuide[] {
  const levels: boolean[] = [];
  return depths
    .map((depth, i) => ({ depth, i }))
    .reverse()
    .map(({ depth, i }) => {
      const guide: TreeGuide = {
        rails: Array.from({ length: Math.max(0, depth - 1) }, (_, k) => levels[k + 1] === true),
        last: levels[depth] !== true,
        stem: (depths[i + 1] ?? -1) > depth,
      };
      levels.length = depth;
      levels[depth] = true;
      return guide;
    })
    .reverse();
}
