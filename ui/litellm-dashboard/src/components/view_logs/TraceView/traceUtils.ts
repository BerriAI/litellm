/**
 * Pure helpers for the agent trace views. No React in here: the outline the drawer
 * renders is computed here so it can be unit-tested against real traces.
 */
import type { Span, SpanType, Trace, TraceMessage, TraceSummary } from "./traceTypes";

/* ------------------------------------------------------------------ */
/*  Formatting                                                         */
/* ------------------------------------------------------------------ */

/** Short human durations: "<1ms", "840ms", "8.3s", "1m 29s". */
export const fmtMs = (ms: number): string => {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes}m ${Math.round((ms % 60_000) / 1000)}s`;
};

/** "$1.14", "$0.128", "$0.0029", "$0". */
export const fmtCost = (cost: number | null | undefined): string => {
  if (cost == null) return "—";
  if (cost === 0) return "$0";
  if (cost >= 1) return `$${cost.toFixed(2)}`;
  if (cost >= 0.01) return `$${cost.toFixed(3)}`;
  return `$${Number(cost.toPrecision(2))}`;
};

export const fmtTok = (n: number): string => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

const plural = (n: number, word: string): string => `${n} ${word}${n === 1 ? "" : "s"}`;

/** "just now", "2m ago", "3h ago", "4d ago". */
export function fmtRelative(iso: string, now: number = Date.now()): string {
  const seconds = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m ago`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86_400)}d ago`;
}

/** "claude-sonnet-4-5" / "openai/claude-haiku-4-5-20251001" -> "sonnet-4-5" / "haiku-4-5". */
export const shortModel = (model: string): string =>
  model
    .replace(/^.*\//, "")
    .replace(/^claude-/, "")
    .replace(/-\d{8}$/, "");

/** Exception text without the glued-on Python traceback header. */
export const cleanError = (error: string | null | undefined): string =>
  (error ?? "").split("Traceback (most recent call last)")[0].trim();

export const spanSpend = (span: Span): number => span.litellm?.spend ?? 0;

/** LLM decisions + tool calls: what the list row and the header call "steps". */
export const stepCount = (summary: Pick<TraceSummary, "llm_calls" | "tool_calls">): number =>
  summary.llm_calls + summary.tool_calls;

/** `14 steps · 48.2s · $0.21 · 3 errors` for a trace list row. */
export function traceRowMeta(summary: TraceSummary): string {
  const parts = [plural(stepCount(summary), "step"), fmtMs(summary.duration_ms), fmtCost(summary.spend)];
  if (summary.error_count > 0) parts.push(plural(summary.error_count, "error"));
  return parts.join(" · ");
}

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

/** Exporters occasionally send the same span twice; keep the first copy. */
export function dedupeSpans(spans: readonly Span[]): Span[] {
  const seen = new Set<string>();
  return spans.filter((s) => !seen.has(s.span_id) && Boolean(seen.add(s.span_id)));
}

const pushChild = (children: ChildrenMap, key: string, span: Span): void => {
  const list = children.get(key);
  if (list) list.push(span);
  else children.set(key, [span]);
};

/**
 * Children map with framework spans dropped (unless `showFramework`) and their
 * children attached to the nearest visible ancestor. Orphans attach to the root level.
 */
export function buildVisibleTree(spans: readonly Span[], showFramework = false): VisibleTree {
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
  spend: number;
  errors: number;
}

/** Spend and error count of every span's full (raw) subtree, keyed by span id. */
export function subtreeStats(spans: readonly Span[]): Map<string, SubtreeStats> {
  const raw = buildVisibleTree(spans, true).children;
  const stats = new Map<string, SubtreeStats>();
  const visit = (span: Span): SubtreeStats => {
    const cached = stats.get(span.span_id);
    if (cached) return cached;
    const own: SubtreeStats = { spend: spanSpend(span), errors: span.status === "error" ? 1 : 0 };
    stats.set(span.span_id, own);
    for (const child of raw.get(span.span_id) ?? []) {
      const childStats = visit(child);
      own.spend += childStats.spend;
      own.errors += childStats.errors;
    }
    return own;
  };
  spans.forEach(visit);
  return stats;
}

export const median = (values: readonly number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/* ------------------------------------------------------------------ */
/*  Outline tree: passthrough tools, agent folds, failure groups       */
/* ------------------------------------------------------------------ */

/** More than this many same-named sibling agents fold into one `researcher ×200` row. */
export const FOLD_THRESHOLD = 5;
/** Folded invocations reveal this many at a time. */
export const FOLD_PAGE_SIZE = 20;

export interface SpanNode {
  kind: "span";
  span: Span;
  /** The tool span that launched this subagent (`task`), when it was collapsed into it. */
  via?: Span;
  children: OutlineNode[];
}

export interface AgentFoldNode {
  kind: "agents";
  key: string;
  name: string;
  invocations: SpanNode[];
  spend: number;
  /** Invocations with at least one failed span inside. */
  failed: number;
  p50Ms: number;
}

export interface FailureGroupNode {
  kind: "failures";
  key: string;
  /** The failing tool. */
  name: string;
  failures: number;
  /** The failed calls plus the LLM retries between them, in order. */
  nodes: OutlineNode[];
}

export type OutlineNode = SpanNode | AgentFoldNode | FailureGroupNode;

const isAgentNode = (node: OutlineNode): node is SpanNode => node.kind === "span" && node.span.type === "agent";

/** Replace runs of more than `threshold` same-named agent siblings with one fold node at the first one's slot. */
export function foldSiblingAgents(
  parentKey: string,
  nodes: readonly OutlineNode[],
  stats: Map<string, SubtreeStats>,
  threshold = FOLD_THRESHOLD,
): OutlineNode[] {
  const byName = new Map<string, SpanNode[]>();
  for (const node of nodes) {
    if (!isAgentNode(node)) continue;
    const list = byName.get(node.span.name);
    if (list) list.push(node);
    else byName.set(node.span.name, [node]);
  }
  const folded = new Set([...byName].filter(([, list]) => list.length > threshold).map(([name]) => name));
  if (folded.size === 0) return [...nodes];
  const emitted = new Set<string>();
  const out: OutlineNode[] = [];
  for (const node of nodes) {
    if (!isAgentNode(node) || !folded.has(node.span.name)) {
      out.push(node);
      continue;
    }
    const name = node.span.name;
    if (emitted.has(name)) continue;
    emitted.add(name);
    const invocations = byName.get(name) ?? [];
    out.push({
      kind: "agents",
      key: `${parentKey}::agents::${name}`,
      name,
      invocations,
      spend: invocations.reduce((sum, n) => sum + (stats.get(n.span.span_id)?.spend ?? 0), 0),
      failed: invocations.filter((n) => (stats.get(n.span.span_id)?.errors ?? 0) > 0).length,
      p50Ms: median(invocations.map((n) => n.span.duration_ms)),
    });
  }
  return out;
}

const isFailedLeafTool = (node: OutlineNode | undefined, name?: string): node is SpanNode =>
  node?.kind === "span" &&
  node.span.type === "tool" &&
  node.span.status === "error" &&
  node.children.length === 0 &&
  (name === undefined || node.span.name === name);

const isLlmNode = (node: OutlineNode | undefined): node is SpanNode =>
  node?.kind === "span" && node.span.type === "llm";

/** End (exclusive) of a failure run of tool `name` starting at `start`: failed calls, optionally each preceded by the LLM retry that made it. */
function failureRunEnd(nodes: readonly OutlineNode[], start: number, name: string): number {
  let i = start;
  let end = start;
  while (i < nodes.length) {
    if (isFailedLeafTool(nodes[i], name)) {
      end = ++i;
    } else if (isLlmNode(nodes[i]) && isFailedLeafTool(nodes[i + 1], name)) {
      i += 1;
    } else {
      break;
    }
  }
  return end;
}

/**
 * Fold repeated failures of the same tool into one `verify_claim ×4 failed` node, in
 * place: back-to-back failed calls, or a retry loop (LLM -> failed call, again and again).
 */
export function groupFailedToolCalls(parentKey: string, nodes: readonly OutlineNode[]): OutlineNode[] {
  const out: OutlineNode[] = [];
  let i = 0;
  while (i < nodes.length) {
    const node = nodes[i];
    const startsAt = isFailedLeafTool(node) ? node : isLlmNode(node) && isFailedLeafTool(nodes[i + 1]) ? nodes[i + 1] : null;
    if (startsAt && startsAt.kind === "span") {
      const end = failureRunEnd(nodes, i, startsAt.span.name);
      const members = nodes.slice(i, end);
      const failures = members.filter((m) => isFailedLeafTool(m)).length;
      if (failures > 1) {
        out.push({
          kind: "failures",
          key: `${parentKey}::failures::${startsAt.span.span_id}`,
          name: startsAt.span.name,
          failures,
          nodes: members,
        });
        i = end;
        continue;
      }
    }
    out.push(node);
    i++;
  }
  return out;
}

/** The outline's nodes under the root agent (framework hidden, subagent launches collapsed, folds applied). */
export function buildOutlineTree(spans: readonly Span[]): { root: Span | null; nodes: OutlineNode[] } {
  const unique = dedupeSpans(spans);
  const { children } = buildVisibleTree(unique, false);
  const stats = subtreeStats(unique);
  const toNode = (span: Span): SpanNode => {
    const kids = (children.get(span.span_id) ?? []).map(toNode);
    // A tool whose only job was to launch a subagent (Deep Agents `task`) renders as the subagent.
    if (span.type === "tool" && kids.length === 1 && kids[0].span.type === "agent" && !kids[0].via) {
      return { ...kids[0], via: span };
    }
    const shaped = groupFailedToolCalls(span.span_id, foldSiblingAgents(span.span_id, kids, stats));
    return { kind: "span", span, children: shaped };
  };
  const top = children.get(ROOT_KEY) ?? [];
  const root = top.find((s) => s.parent_span_id === null) ?? top[0] ?? null;
  if (!root) return { root: null, nodes: [] };
  const rootNode = toNode(root);
  const others = top.filter((s) => s !== root).map(toNode);
  return { root, nodes: [...rootNode.children, ...others] };
}

/* ------------------------------------------------------------------ */
/*  Human labels                                                       */
/* ------------------------------------------------------------------ */

/** What a sibling node looks like as a call made by the LLM step before it. */
const callName = (node: OutlineNode): { name: string; count: number } => {
  if (node.kind === "agents") return { name: node.name, count: node.invocations.length };
  if (node.kind === "failures") return { name: node.name, count: node.failures };
  if (node.via) return { name: `${node.via.name}(${node.span.name})`, count: 1 };
  return { name: node.span.name, count: 1 };
};

/** `task(researcher) ×4, write_file`: consecutive duplicates collapsed. */
export function summarizeCalls(names: readonly { name: string; count: number }[]): string {
  const merged: { name: string; count: number }[] = [];
  for (const entry of names) {
    const last = merged.at(-1);
    if (last && last.name === entry.name) last.count += entry.count;
    else merged.push({ ...entry });
  }
  return merged.map((m) => (m.count > 1 ? `${m.name} ×${m.count}` : m.name)).join(", ");
}

/** Calls the LLM at `index` made: the siblings after it, up to the next LLM step. */
export function callsAfter(nodes: readonly OutlineNode[], index: number): { name: string; count: number }[] {
  const calls: { name: string; count: number }[] = [];
  for (let i = index + 1; i < nodes.length; i++) {
    const node = nodes[i];
    if (node.kind === "span" && node.span.type === "llm") break;
    calls.push(callName(node));
  }
  return calls;
}

const LABEL_MAX_CHARS = 60;
const clip = (text: string, max = LABEL_MAX_CHARS): string => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

/**
 * The outline label for a span. LLM steps read as the decision they made
 * (`→ task(researcher) ×4`), or `Answer` when no call followed.
 */
export function humanLabel(span: Span, calls: readonly { name: string; count: number }[] = []): string {
  if (span.type === "llm") {
    if (calls.length > 0) return clip(`→ ${summarizeCalls(calls)}`);
    return "Answer";
  }
  return span.name;
}

/** First line of a span's input as plain text. */
export const firstLine = (text: string): string => text.split("\n").find((l) => l.trim())?.trim() ?? "";

/** `customer_id="acme-404"` from a tool's JSON args preview. */
export function argsPreview(preview: string): string {
  const parsed = parseJson(preview);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return Object.entries(parsed as Record<string, unknown>)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
      .join(" ");
  }
  return preview.replace(/^\{/, "").replace(/\}$/, "");
}

/* ------------------------------------------------------------------ */
/*  Outline rows (flattened, for rendering + keyboard nav)             */
/* ------------------------------------------------------------------ */

export type OutlineGlyph = SpanType | "input" | "output";

export interface OutlineRow {
  id: string;
  kind: "input" | "output" | "span" | "agents" | "failures" | "more";
  depth: number;
  glyph: OutlineGlyph;
  label: string;
  /** Faint inline text after the label (tool args, subagent task). */
  detail?: string;
  /** Faint right-hand text (model · tokens, steps, spend). */
  meta?: string;
  /** Right-hand text shown in red (e.g. "3 failed"). */
  metaError?: string;
  durationMs?: number;
  /** Label renders red. */
  error: boolean;
  span?: Span;
  via?: Span;
  fold?: AgentFoldNode;
  failures?: FailureGroupNode;
  hasChildren: boolean;
  /** Collapsed by default? (Folds, failure groups and folded invocations are.) */
  defaultOpen: boolean;
  /** Row ids that must all be open for this row to show. */
  ancestors: string[];
  /** For folded invocations and "Show more": which fold page it belongs to. */
  page?: { fold: string; index: number };
}

export interface OutlineUiState {
  /** Row id -> open, overriding the row's default. */
  open: Readonly<Record<string, boolean>>;
  /** Fold row id -> how many invocations are revealed. */
  shown: Readonly<Record<string, number>>;
}

export const EMPTY_OUTLINE_UI: OutlineUiState = { open: {}, shown: {} };

export const INPUT_ROW_ID = "input";
export const OUTPUT_ROW_ID = "output";

const countSteps = (nodes: readonly OutlineNode[]): number =>
  nodes.reduce((sum, node) => {
    if (node.kind === "agents") return sum + node.invocations.reduce((s, inv) => s + countSteps(inv.children), 0);
    if (node.kind === "failures") return sum + countSteps(node.nodes);
    if (node.span.type === "agent") return sum + countSteps(node.children);
    return sum + 1 + countSteps(node.children);
  }, 0);

interface FlattenContext {
  rows: OutlineRow[];
  stats: Map<string, SubtreeStats>;
}

function spanRow(
  node: SpanNode,
  depth: number,
  ancestors: string[],
  calls: { name: string; count: number }[],
  ctx: FlattenContext,
  invocationIndex?: number,
): OutlineRow {
  const { span } = node;
  const row: OutlineRow = {
    id: span.span_id,
    kind: "span",
    depth,
    glyph: span.type,
    label: humanLabel(span, calls),
    durationMs: span.duration_ms,
    error: span.status === "error",
    span,
    via: node.via,
    hasChildren: node.children.length > 0,
    defaultOpen: invocationIndex === undefined,
    ancestors,
  };
  if (span.type === "llm") {
    const model = span.litellm?.model_group || span.model || span.litellm?.model;
    const tokens = span.input_tokens + span.output_tokens;
    row.meta = [model ? shortModel(model) : null, tokens ? `${fmtTok(tokens)} tok` : null].filter(Boolean).join(" · ");
  } else if (span.type === "tool") {
    row.detail = argsPreview(span.input_preview);
  } else if (span.type === "agent") {
    if (invocationIndex !== undefined) {
      row.label = `#${invocationIndex + 1}`;
    } else {
      row.label = span.name;
    }
    row.detail = firstLine(previewText(span.input_preview));
    const steps = countSteps(node.children);
    row.meta = plural(steps, "step");
    const errors = ctx.stats.get(span.span_id)?.errors ?? 0;
    if (errors > 0 && span.status !== "error") row.metaError = `${errors} failed`;
  }
  return row;
}

function walk(nodes: readonly OutlineNode[], depth: number, ancestors: string[], ctx: FlattenContext): void {
  nodes.forEach((node, index) => {
    if (node.kind === "span") {
      const calls = node.span.type === "llm" ? callsAfter(nodes, index) : [];
      const row = spanRow(node, depth, ancestors, calls, ctx);
      ctx.rows.push(row);
      walk(node.children, depth + 1, [...ancestors, row.id], ctx);
      return;
    }
    if (node.kind === "failures") {
      const id = `failures:${node.key}`;
      ctx.rows.push({
        id,
        kind: "failures",
        depth,
        glyph: "tool",
        label: `${node.name} ×${node.failures} failed`,
        error: true,
        failures: node,
        hasChildren: true,
        defaultOpen: false,
        ancestors,
      });
      walk(node.nodes, depth + 1, [...ancestors, id], ctx);
      return;
    }
    const id = `agents:${node.key}`;
    ctx.rows.push({
      id,
      kind: "agents",
      depth,
      glyph: "agent",
      label: `${node.name} ×${node.invocations.length}`,
      meta: fmtCost(node.spend),
      metaError: node.failed > 0 ? `${node.failed} failed` : undefined,
      error: false,
      fold: node,
      hasChildren: true,
      defaultOpen: false,
      ancestors,
    });
    node.invocations.forEach((inv, i) => {
      const row = spanRow(inv, depth + 1, [...ancestors, id], [], ctx, i);
      row.page = { fold: id, index: i };
      ctx.rows.push(row);
      walk(inv.children, depth + 2, [...ancestors, id, row.id], ctx);
    });
    ctx.rows.push({
      id: `more:${node.key}`,
      kind: "more",
      depth: depth + 1,
      glyph: "agent",
      label: "Show more",
      error: false,
      hasChildren: false,
      defaultOpen: false,
      ancestors: [...ancestors, id],
      page: { fold: id, index: node.invocations.length },
    });
  });
}

/**
 * Every outline row with nothing collapsed: Input, the steps depth-first in time
 * order, Output. Visibility is applied separately by `visibleOutlineRows`.
 */
export function buildOutline(trace: Trace): OutlineRow[] {
  const spans = dedupeSpans(trace.spans);
  const { root, nodes } = buildOutlineTree(spans);
  const ctx: FlattenContext = { rows: [], stats: subtreeStats(spans) };
  const input = firstLine(previewText(trace.summary.input_preview || root?.input_preview || ""));
  ctx.rows.push({
    id: INPUT_ROW_ID,
    kind: "input",
    depth: 0,
    glyph: "input",
    label: "Input",
    detail: input,
    error: false,
    span: root ?? undefined,
    hasChildren: false,
    defaultOpen: true,
    ancestors: [],
  });
  walk(nodes, 0, [], ctx);
  ctx.rows.push({
    id: OUTPUT_ROW_ID,
    kind: "output",
    depth: 0,
    glyph: "output",
    label: "Output",
    durationMs: trace.summary.duration_ms,
    error: false,
    span: root ?? undefined,
    hasChildren: false,
    defaultOpen: true,
    ancestors: [],
  });
  return ctx.rows;
}

export const isRowOpen = (row: OutlineRow, ui: OutlineUiState): boolean => ui.open[row.id] ?? row.defaultOpen;

/** Rows currently on screen: every ancestor open, fold pages respected, "Show more" only while some are hidden. */
export function visibleOutlineRows(rows: readonly OutlineRow[], ui: OutlineUiState): OutlineRow[] {
  const byId = new Map(rows.map((r) => [r.id, r]));
  const shownFor = (fold: string) => ui.shown[fold] ?? FOLD_PAGE_SIZE;
  return rows.filter((row) => {
    if (!row.ancestors.every((id) => isRowOpen(byId.get(id) as OutlineRow, ui))) return false;
    if (row.page) {
      const shown = shownFor(row.page.fold);
      if (row.kind === "more") return shown < row.page.index;
      if (row.page.index >= shown) return false;
    }
    // Rows nested under a folded invocation are hidden when that invocation is off-page.
    return row.ancestors.every((id) => {
      const ancestor = byId.get(id) as OutlineRow;
      return !ancestor.page || ancestor.page.index < shownFor(ancestor.page.fold);
    });
  });
}

/** Visible rows for a trace: the flat list the outline renders and J/K walks. */
export const outlineRows = (trace: Trace, ui: OutlineUiState = EMPTY_OUTLINE_UI): OutlineRow[] =>
  visibleOutlineRows(buildOutline(trace), ui);

/** UI state with `rowId` visible: its ancestors opened and fold pages advanced. */
export function revealRow(rows: readonly OutlineRow[], ui: OutlineUiState, rowId: string): OutlineUiState {
  const byId = new Map(rows.map((r) => [r.id, r]));
  const target = byId.get(rowId);
  if (!target) return ui;
  const open = { ...ui.open };
  const shown = { ...ui.shown };
  for (const row of [target, ...target.ancestors.map((id) => byId.get(id) as OutlineRow)]) {
    if (row !== target) open[row.id] = true;
    if (row.page) {
      const needed = Math.ceil((row.page.index + 1) / FOLD_PAGE_SIZE) * FOLD_PAGE_SIZE;
      shown[row.page.fold] = Math.max(shown[row.page.fold] ?? FOLD_PAGE_SIZE, needed);
    }
  }
  return { open, shown };
}

/** Rows the drawer can select (everything but "Show more"). */
export const selectableIds = (rows: readonly OutlineRow[]): string[] =>
  rows.filter((r) => r.kind !== "more").map((r) => r.id);

/** Where the drawer opens: the requested span, else the first failure, else Output. */
export function initialOutlineSelection(
  rows: readonly OutlineRow[],
  initialSpanId?: string | null,
): { selectedId: string; ui: OutlineUiState } {
  const requested = initialSpanId ? rows.find((r) => r.id === initialSpanId) : undefined;
  const target = requested ?? rows.find((r) => r.error) ?? rows.find((r) => r.id === OUTPUT_ROW_ID);
  const selectedId = target?.id ?? OUTPUT_ROW_ID;
  return { selectedId, ui: revealRow(rows, EMPTY_OUTLINE_UI, selectedId) };
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
  if (Array.isArray(parsed)) return parsed.length > 0 && parsed.every(isMessage) ? parsed : null;
  return isMessage(parsed) ? [parsed] : null;
}

/** Pretty JSON when the payload is JSON, else the raw string. */
export const prettyPayload = (value: string): string => {
  const parsed = parseJson(value);
  if (parsed === null || typeof parsed === "string") return typeof parsed === "string" ? parsed : value;
  return JSON.stringify(parsed, null, 2);
};

/** Compact JSON for short args, pretty for long ones. */
export const compactPayload = (value: string, maxInline = 100): string => {
  const parsed = parseJson(value);
  if (parsed === null || typeof parsed === "string") return typeof parsed === "string" ? parsed : value;
  const compact = JSON.stringify(parsed);
  return compact.length <= maxInline ? compact : JSON.stringify(parsed, null, 2);
};

/* ------------------------------------------------------------------ */
/*  List view helpers                                                  */
/* ------------------------------------------------------------------ */

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

/** The list row / drawer title: first line of what was asked, else the agent name. */
export const traceTitle = (summary: Pick<TraceSummary, "name" | "service" | "input_preview">): string =>
  firstLine(previewText(summary.input_preview)) || traceDisplayName(summary);

/* ------------------------------------------------------------------ */
/*  Copy for agent                                                     */
/* ------------------------------------------------------------------ */

const curlLine = (url: string): string => `curl -s -H "Authorization: Bearer $LITELLM_API_KEY" "${url}"`;

/** Paste-ready instruction for Claude / Codex that fetches the whole trace as markdown. */
export const copyForAgentText = (url: string, failed: boolean): string =>
  `Read this LiteLLM agent trace and explain what happened${failed ? " and why it failed" : ""}:\n${curlLine(url)}`;

/** Same, for one step. */
export const copyStepText = (url: string): string =>
  `Read this step of a LiteLLM agent trace and explain what it did:\n${curlLine(url)}`;
