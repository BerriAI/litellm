/**
 * Pure helpers for the agent trace views. No React in here: everything the tree,
 * steps and graph views compute lives here so it can be unit-tested directly.
 */
import type { AgentNode, Span, SpanStatus, Trace, TraceMessage, TraceSummary } from "./traceTypes";

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

const plural = (n: number, word: string): string => `${n} ${word}${n === 1 ? "" : "s"}`;

/** "◆ Agent · 7 LLM · 26 tool" or "◆ 2 agents · 7 LLM · 26 tool". */
export const agentBadgeLabel = (summary: Pick<TraceSummary, "agent_count" | "llm_calls" | "tool_calls">): string => {
  const agents = summary.agent_count > 1 ? plural(summary.agent_count, "agent") : "Agent";
  return `◆ ${agents} · ${summary.llm_calls} LLM · ${summary.tool_calls} tool`;
};

/** LLM spans are labelled by model group (what the caller asked for); everything else by span name. */
export const spanLabel = (span: Span): string => (span.type === "llm" ? span.model || span.name : span.name);

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
/*  Subtree rollups                                                    */
/* ------------------------------------------------------------------ */

export interface SubtreeStats {
  errors: number;
}

/** Error count of every span's full subtree, keyed by span id. */
export function subtreeStats(spans: readonly Span[]): Map<string, SubtreeStats> {
  const raw = buildVisibleTree(spans, true).children;
  const stats = new Map<string, SubtreeStats>();
  const visit = (span: Span): SubtreeStats => {
    const cached = stats.get(span.span_id);
    if (cached) return cached;
    const own: SubtreeStats = { errors: span.status === "error" ? 1 : 0 };
    for (const child of raw.get(span.span_id) ?? []) {
      const childStats = visit(child);
      own.errors += childStats.errors;
    }
    stats.set(span.span_id, own);
    return own;
  };
  spans.forEach(visit);
  return stats;
}

/* ------------------------------------------------------------------ */
/*  Sibling agent grouping (auto-collapse of fan-out)                  */
/* ------------------------------------------------------------------ */

/** A parent with more than this many same-named agent children renders them as one group row. */
export const GROUP_THRESHOLD = 10;
/** Group rows expand this many invocations at a time. */
export const GROUP_PAGE_SIZE = 20;

export interface SpanGroup {
  key: string;
  name: string;
  spans: Span[];
  p50Ms: number;
  errors: number;
}

export type TreeItem = { kind: "span"; span: Span } | { kind: "group"; group: SpanGroup };

export const median = (values: readonly number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export const groupKey = (parentKey: string, name: string): string => `${parentKey}::${name}`;

const buildGroup = (parentKey: string, name: string, members: Span[], stats: Map<string, SubtreeStats>): SpanGroup => ({
  key: groupKey(parentKey, name),
  name,
  spans: members,
  p50Ms: median(members.map((s) => s.duration_ms)),
  errors: members.filter((s) => (stats.get(s.span_id)?.errors ?? 0) > 0).length,
});

/**
 * Collapse runs of same-named agent siblings (more than `threshold`) into one group
 * item, placed where the first member would have been. Other children pass through.
 */
export function groupSiblingAgents(
  parentKey: string,
  children: readonly Span[],
  stats: Map<string, SubtreeStats>,
  threshold = GROUP_THRESHOLD,
): TreeItem[] {
  const agentsByName = new Map<string, Span[]>();
  for (const child of children) {
    if (child.type !== "agent") continue;
    const list = agentsByName.get(child.name);
    if (list) list.push(child);
    else agentsByName.set(child.name, [child]);
  }
  const grouped = new Set([...agentsByName].filter(([, list]) => list.length > threshold).map(([name]) => name));
  const emitted = new Set<string>();
  const items: TreeItem[] = [];
  for (const child of children) {
    if (child.type !== "agent" || !grouped.has(child.name)) {
      items.push({ kind: "span", span: child });
      continue;
    }
    if (emitted.has(child.name)) continue;
    emitted.add(child.name);
    items.push({ kind: "group", group: buildGroup(parentKey, child.name, agentsByName.get(child.name) ?? [], stats) });
  }
  return items;
}

/* ------------------------------------------------------------------ */
/*  Flattening the tree into rows                                      */
/* ------------------------------------------------------------------ */

export interface TreeUiState {
  /** Span ids whose children are hidden. */
  collapsed: ReadonlySet<string>;
  /** Group key -> how many invocations are revealed (0 = just the summary row). */
  groupShown: Readonly<Record<string, number>>;
}

export type TreeRow =
  | { kind: "span"; span: Span; depth: number; hasChildren: boolean; isCollapsed: boolean }
  | { kind: "group"; group: SpanGroup; depth: number; shown: number }
  | { kind: "more"; group: SpanGroup; depth: number; shown: number };

interface FlattenContext {
  children: ChildrenMap;
  stats: Map<string, SubtreeStats>;
  ui: TreeUiState;
  rows: TreeRow[];
}

function pushSpanRow(ctx: FlattenContext, span: Span, depth: number): void {
  const hasChildren = (ctx.children.get(span.span_id)?.length ?? 0) > 0;
  const isCollapsed = ctx.ui.collapsed.has(span.span_id);
  ctx.rows.push({ kind: "span", span, depth, hasChildren, isCollapsed });
  if (hasChildren && !isCollapsed) walkChildren(ctx, span.span_id, depth + 1);
}

function pushGroupRows(ctx: FlattenContext, group: SpanGroup, depth: number): void {
  const shown = Math.min(ctx.ui.groupShown[group.key] ?? 0, group.spans.length);
  ctx.rows.push({ kind: "group", group, depth, shown });
  if (shown === 0) return;
  group.spans.slice(0, shown).forEach((span) => pushSpanRow(ctx, span, depth + 1));
  if (shown < group.spans.length) ctx.rows.push({ kind: "more", group, depth: depth + 1, shown });
}

function walkChildren(ctx: FlattenContext, parentKey: string, depth: number): void {
  for (const item of groupSiblingAgents(parentKey, ctx.children.get(parentKey) ?? [], ctx.stats)) {
    if (item.kind === "span") pushSpanRow(ctx, item.span, depth);
    else pushGroupRows(ctx, item.group, depth);
  }
}

/** Depth-first rows for the waterfall, honouring collapsed spans and group pagination. */
export function flattenTree(children: ChildrenMap, stats: Map<string, SubtreeStats>, ui: TreeUiState): TreeRow[] {
  const ctx: FlattenContext = { children, stats, ui, rows: [] };
  walkChildren(ctx, ROOT_KEY, 0);
  return ctx.rows;
}

/** Ids of the span rows, in render order (drives J/K navigation). */
export const spanRowIds = (rows: readonly TreeRow[]): string[] =>
  rows.flatMap((row) => (row.kind === "span" ? [row.span.span_id] : []));

/**
 * Tree UI state with `spanId` guaranteed visible: its visible ancestors are
 * expanded and any group containing it (or an ancestor) reveals enough pages.
 */
export function revealSpan(
  children: ChildrenMap,
  stats: Map<string, SubtreeStats>,
  ui: TreeUiState,
  spanId: string,
): TreeUiState {
  const parentOf = new Map<string, string>();
  children.forEach((list, key) => list.forEach((s) => parentOf.set(s.span_id, key)));
  if (!parentOf.has(spanId)) return ui;
  const collapsed = new Set(ui.collapsed);
  const groupShown = { ...ui.groupShown };
  let current = spanId;
  while (parentOf.has(current)) {
    const parentKey = parentOf.get(current) as string;
    collapsed.delete(parentKey);
    const items = groupSiblingAgents(parentKey, children.get(parentKey) ?? [], stats);
    for (const item of items) {
      if (item.kind !== "group") continue;
      const index = item.group.spans.findIndex((s) => s.span_id === current);
      if (index < 0) continue;
      const needed = Math.ceil((index + 1) / GROUP_PAGE_SIZE) * GROUP_PAGE_SIZE;
      groupShown[item.group.key] = Math.max(groupShown[item.group.key] ?? 0, needed);
    }
    current = parentKey;
  }
  return { collapsed, groupShown };
}

/* ------------------------------------------------------------------ */
/*  Steps view                                                         */
/* ------------------------------------------------------------------ */

export interface TraceStep {
  span: Span;
  /** 1-based, in time order. */
  number: number;
  /** Nearest enclosing non-root agent span name, e.g. "researcher". */
  subagent: string | null;
  /** Nesting level of subagents (0 = root agent). */
  depth: number;
  /** For LLM steps: the tools the agent ran right after this decision. */
  toolNames: string[];
}

const isStepSpan = (span: Span): boolean =>
  span.type === "llm" || span.type === "tool" || (span.type === "agent" && span.parent_span_id !== null);

interface AgentContext {
  subagent: string | null;
  depth: number;
  invocationId: string | null;
}

function agentContext(span: Span, byId: Map<string, Span>): AgentContext {
  let parent = span.parent_span_id ? byId.get(span.parent_span_id) : undefined;
  let subagent: string | null = null;
  let invocationId: string | null = null;
  let depth = 0;
  while (parent) {
    if (parent.type === "agent") {
      invocationId ??= parent.span_id;
      if (parent.parent_span_id !== null) {
        subagent ??= parent.name;
        depth++;
      }
    }
    parent = parent.parent_span_id ? byId.get(parent.parent_span_id) : undefined;
  }
  return { subagent, depth, invocationId };
}

/** Every LLM decision, tool call and subagent invocation, in time order. */
export function stepsFromSpans(spans: readonly Span[]): TraceStep[] {
  const byId = indexSpans(spans);
  const ordered = spans.filter(isStepSpan).sort(byStart);
  const contexts = ordered.map((span) => agentContext(span, byId));
  return ordered.map((span, i) => {
    const toolNames: string[] = [];
    if (span.type === "llm") {
      for (let j = i + 1; j < ordered.length; j++) {
        if (contexts[j].invocationId !== contexts[i].invocationId) continue;
        if (ordered[j].type === "llm") break;
        if (ordered[j].type === "tool") toolNames.push(ordered[j].name);
      }
    }
    return { span, number: i + 1, subagent: contexts[i].subagent, depth: contexts[i].depth, toolNames };
  });
}

/* ------------------------------------------------------------------ */
/*  Trace-level rollups                                                */
/* ------------------------------------------------------------------ */

export const summaryHasErrors = (summary: Pick<TraceSummary, "status" | "error_count">): boolean =>
  summary.status === "error" || summary.error_count > 0;

export const traceHasErrors = (trace: Trace): boolean =>
  summaryHasErrors(trace.summary) || trace.spans.some((s) => s.status === "error");

/** Earliest failing non-root span (the root just echoes its children), else the root. */
export function firstErrorSpan(spans: readonly Span[]): Span | null {
  const failed = spans.filter((s) => s.status === "error").sort(byStart);
  return failed.find((s) => s.parent_span_id !== null) ?? failed[0] ?? null;
}

export const agentsWithErrors = (spans: readonly Span[]): Set<string> =>
  new Set(spans.filter((s) => s.status === "error").map((s) => s.agent));

export const invocationsOf = (spans: readonly Span[], agentName: string): Span[] =>
  spans.filter((s) => s.type === "agent" && s.name === agentName).sort(byStart);

export const llmSpans = (spans: readonly Span[]): Span[] => spans.filter((s) => s.type === "llm").sort(byStart);

export const statusTone = (status: SpanStatus): "success" | "error" | "neutral" => {
  if (status === "error") return "error";
  return status === "ok" ? "success" : "neutral";
};

/* ------------------------------------------------------------------ */
/*  Agent graph layout                                                 */
/* ------------------------------------------------------------------ */

export const GRAPH_NODE_HEIGHT = 48;
export const GRAPH_MIN_NODE_WIDTH = 132;
export const GRAPH_MAX_NODE_WIDTH = 232;
const GRAPH_COLUMN_GAP = 96;
const GRAPH_ROW_GAP = 28;
const GRAPH_PADDING = 16;

export interface GraphNode {
  agent: AgentNode;
  x: number;
  y: number;
  width: number;
}

export interface GraphEdge {
  from: GraphNode;
  to: GraphNode;
  label: string;
}

export interface GraphLayout {
  nodes: GraphNode[];
  edges: GraphEdge[];
  width: number;
  height: number;
}

function agentDepths(agents: readonly AgentNode[]): Map<string, number> {
  const byName = new Map(agents.map((a) => [a.name, a]));
  const depths = new Map<string, number>();
  const depthOf = (agent: AgentNode, seen: Set<string>): number => {
    const known = depths.get(agent.name);
    if (known !== undefined) return known;
    const parent = agent.parent_agent ? byName.get(agent.parent_agent) : undefined;
    const depth = parent && !seen.has(parent.name) ? depthOf(parent, new Set([...seen, agent.name])) + 1 : 0;
    depths.set(agent.name, depth);
    return depth;
  };
  agents.forEach((a) => depthOf(a, new Set([a.name])));
  return depths;
}

/** Left-to-right layered layout: one column per nesting level. */
export function layoutAgentGraph(agents: readonly AgentNode[]): GraphLayout {
  const depths = agentDepths(agents);
  const columnWidth = GRAPH_MAX_NODE_WIDTH + GRAPH_COLUMN_GAP;
  const rowsPerColumn = new Map<number, number>();
  const nodes: GraphNode[] = agents.map((agent) => {
    const depth = depths.get(agent.name) ?? 0;
    const row = rowsPerColumn.get(depth) ?? 0;
    rowsPerColumn.set(depth, row + 1);
    return {
      agent,
      x: GRAPH_PADDING + depth * columnWidth,
      y: GRAPH_PADDING + row * (GRAPH_NODE_HEIGHT + GRAPH_ROW_GAP),
      width: GRAPH_MIN_NODE_WIDTH,
    };
  });
  const byName = new Map(nodes.map((n) => [n.agent.name, n]));
  const edges: GraphEdge[] = nodes.flatMap((node) => {
    const parent = node.agent.parent_agent ? byName.get(node.agent.parent_agent) : undefined;
    return parent ? [{ from: parent, to: node, label: `×${node.agent.invocations}` }] : [];
  });
  const columns = Math.max(1, ...[...rowsPerColumn.keys()].map((d) => d + 1));
  const rows = Math.max(1, ...rowsPerColumn.values());
  return {
    nodes,
    edges,
    width: GRAPH_PADDING * 2 + (columns - 1) * columnWidth + GRAPH_MAX_NODE_WIDTH,
    height: GRAPH_PADDING * 2 + rows * GRAPH_NODE_HEIGHT + (rows - 1) * GRAPH_ROW_GAP,
  };
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

export const traceStartMs = (summary: TraceSummary): number => Date.parse(summary.start_time);

const PREVIEW_USER_CONTENT = /"role":\s*"(?:user|human)",\s*"content":\s*"((?:[^"\\]|\\.)*)/;

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
  if (!preview.trimStart().startsWith("[")) return preview;
  const match = PREVIEW_USER_CONTENT.exec(preview);
  return match ? match[1].replace(/\\n/g, " ").replace(/\\"/g, '"') : preview;
}

/** Trace display name; root spans without a name fall back to the service. */
export const traceDisplayName = (summary: Pick<TraceSummary, "name" | "service">): string =>
  summary.name || summary.service || "(unnamed trace)";

export interface TimedItem<T> {
  at: number;
  item: T;
}

/** Merge two newest-first lists into one newest-first list. */
export function interleaveByTime<A, B>(
  a: readonly TimedItem<A>[],
  b: readonly TimedItem<B>[],
): ({ kind: "a"; at: number; item: A } | { kind: "b"; at: number; item: B })[] {
  const merged: ({ kind: "a"; at: number; item: A } | { kind: "b"; at: number; item: B })[] = [
    ...a.map((x) => ({ kind: "a" as const, ...x })),
    ...b.map((x) => ({ kind: "b" as const, ...x })),
  ];
  return merged.sort((x, y) => y.at - x.at);
}
