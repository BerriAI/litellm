/**
 * Pure helpers for the agent trace views. No React in here: everything the span tree
 * computes lives here so it can be unit-tested directly.
 */
import { type ErrorSource, type SpanTreeState, type TreeRow } from "./traceTree";
import type { Span, TraceMessage, TraceSummary } from "./traceTypes";

/* ------------------------------------------------------------------ */
/*  Formatting                                                         */
/* ------------------------------------------------------------------ */

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
export const isFrameworkSpan = (span: Span): boolean =>
  span.parent_span_id !== null &&
  (span.type === "framework" || (span.type === "chain" && (span.name === "model" || span.name === "tools")));

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

/** Siblings sharing name + type fold into one group once there are enough of them (or enough failures). */
function groupChildren(children: readonly Span[]): GroupOrSpan[] {
  const byKey = new Map<string, Span[]>();
  for (const child of children) {
    const key = `${child.name}|${child.type}`;
    const list = byKey.get(key);
    if (list) list.push(child);
    else byKey.set(key, [child]);
  }
  const emitted = new Set<string>();
  const out: GroupOrSpan[] = [];
  for (const child of children) {
    const key = `${child.name}|${child.type}`;
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

export const groupRowId = (parentKey: string, span: Pick<Span, "name" | "type">): string =>
  `grp::${parentKey}::${span.name}::${span.type}`;

interface RowContext {
  children: ChildrenMap;
  state: SpanTreeState;
  rows: TreeRow[];
}

function pushSpan(ctx: RowContext, span: Span, depth: number): void {
  const hasChildren = (ctx.children.get(span.span_id)?.length ?? 0) > 0;
  const collapsed = ctx.state.collapsedSpanIds.has(span.span_id);
  ctx.rows.push({ kind: "span", id: span.span_id, span, depth, hasChildren, collapsed });
  if (hasChildren && !collapsed) pushLevel(ctx, span.span_id, depth + 1);
}

function pushGroup(ctx: RowContext, parentKey: string, members: Span[], depth: number): void {
  const first = members[0];
  const id = groupRowId(parentKey, first);
  const failedCount = members.filter((s) => s.status === "error").length;
  const expanded = ctx.state.expandedGroupIds.has(id);
  ctx.rows.push({
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
  });
  if (!expanded) return;
  const reveal = Math.min(ctx.state.groupRevealCounts[id] ?? GROUP_PAGE_SIZE, members.length);
  members.slice(0, reveal).forEach((member) => pushSpan(ctx, member, depth + 1));
  if (reveal < members.length) {
    ctx.rows.push({
      kind: "load-more",
      id: `${id}::more`,
      depth: depth + 1,
      groupId: id,
      remaining: members.length - reveal,
    });
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

const isMessage = (value: unknown): value is TraceMessage =>
  typeof value === "object" && value !== null && "role" in value && typeof (value as TraceMessage).role === "string";

/** An llm span's input (array of messages) or output (one message); null when it isn't one. */
export function parseMessages(value: string): TraceMessage[] | null {
  const parsed = parseJson(value);
  if (Array.isArray(parsed)) return parsed.every(isMessage) ? parsed : null;
  return isMessage(parsed) ? [parsed] : null;
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
export function previewText(preview: string): string {
  if (!preview) return "";
  const messages = parseMessages(preview);
  if (messages) {
    const last = [...messages].reverse().find((m) => m.role === "user" || m.role === "tool") ?? messages.at(-1);
    return last?.content || preview;
  }
  if (!/^\s*[[{]/.test(preview)) return preview;
  const match = PREVIEW_USER_CONTENT.exec(preview);
  return match ? match[1].replace(/\\n/g, " ").replace(/\\"/g, '"') : preview;
}

/** Trace display name; root spans without a name fall back to the service. */
export const traceDisplayName = (summary: Pick<TraceSummary, "name" | "service">): string =>
  summary.name || summary.service || "(unnamed trace)";
