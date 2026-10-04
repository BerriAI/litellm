import { useCallback, useMemo, useState } from "react";

import type { RunSelection } from "../../routing";
import type { SpanTreeState, TreeRow } from "../../tree";
import type { Trace } from "../../types";
import {
  buildTreeRows,
  findTraceSteps,
  firstErrorSpan,
  GROUP_PAGE_SIZE,
  isFrameworkSpan,
  nearestVisibleSpanId,
  revealSpanInState,
} from "../../utils";

const INITIAL_STATE: SpanTreeState = {
  hideFramework: true,
  collapsedSpanIds: new Set(),
  expandedGroupIds: new Set(),
  groupRevealCounts: {},
};

/** First failed span if the run has errors (with its tree path opened), otherwise the root agent. */
export function initialRunSelection(
  trace: Trace,
  initialSpanId?: string,
): { selectedId: string; state: SpanTreeState } {
  if (initialSpanId && trace.spans.some((span) => span.span_id === initialSpanId)) {
    const selectedId = nearestVisibleSpanId(trace.spans, initialSpanId, false);
    const state = revealSpanInState(trace.spans, { ...INITIAL_STATE, hideFramework: false }, selectedId);
    return { selectedId, state };
  }
  const initialState = {
    ...INITIAL_STATE,
    collapsedSpanIds: new Set(
      trace.spans.filter((span) => span.type === "agent" && span.parent_span_id !== null).map((span) => span.span_id),
    ),
  };
  const failed = firstErrorSpan(trace.spans);
  if (!failed || failed.parent_span_id === null) {
    const root = trace.spans.find((s) => s.parent_span_id === null);
    return { selectedId: root?.span_id ?? "", state: initialState };
  }
  const visibleFailure = trace.spans
    .filter((s) => s.status === "error" && s.parent_span_id !== null && !isFrameworkSpan(s))
    .sort((a, b) => a.start_offset_ms - b.start_offset_ms)[0];
  const selectedId = visibleFailure?.span_id ?? nearestVisibleSpanId(trace.spans, failed.span_id, true);
  return { selectedId, state: revealSpanInState(trace.spans, initialState, selectedId) };
}

const toggle = (set: ReadonlySet<string>, id: string): Set<string> => {
  const next = new Set(set);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
};

/** Fold state, filtering and selection for one run's step tree. */
export function useRunTree(trace: Trace, selection: RunSelection) {
  const { selectSpan, stepQuery: query, errorsOnly } = selection;
  const [initial] = useState(() => initialRunSelection(trace, selection.spanId ?? undefined));
  const [state, setState] = useState<SpanTreeState>(initial.state);
  const selectedId = selection.spanId ?? initial.selectedId;
  const filtering = Boolean(query.trim()) || errorsOnly;

  const treeRows = useMemo(() => buildTreeRows(trace.spans, state), [trace, state]);
  const rows = useMemo<TreeRow[]>(
    () =>
      filtering
        ? findTraceSteps(trace.spans, query, errorsOnly, state.hideFramework).map((span) => ({
            kind: "span",
            id: span.span_id,
            span,
            depth: 0,
            hasChildren: false,
            collapsed: false,
          }))
        : treeRows,
    [trace, state.hideFramework, query, errorsOnly, filtering, treeRows],
  );
  const selectedRow: TreeRow | undefined =
    rows.find((row) => row.id === selectedId) ?? treeRows.find((row) => row.id === selectedId) ?? rows[0];

  const select = useCallback(
    (id: string) => {
      selectSpan(id);
      setState((prev) => revealSpanInState(trace.spans, prev, id));
    },
    [trace.spans, selectSpan],
  );
  const toggleSpan = useCallback(
    (id: string) => setState((prev) => ({ ...prev, collapsedSpanIds: toggle(prev.collapsedSpanIds, id) })),
    [],
  );
  const toggleGroup = useCallback(
    (id: string) => setState((prev) => ({ ...prev, expandedGroupIds: toggle(prev.expandedGroupIds, id) })),
    [],
  );
  const loadMore = useCallback(
    (groupId: string) =>
      setState((prev) => ({
        ...prev,
        groupRevealCounts: {
          ...prev.groupRevealCounts,
          [groupId]: (prev.groupRevealCounts[groupId] ?? GROUP_PAGE_SIZE) + GROUP_PAGE_SIZE,
        },
      })),
    [],
  );
  const setHideFramework = (hideFramework: boolean) => setState((prev) => ({ ...prev, hideFramework }));
  const collapseAll = () =>
    setState((prev) => ({
      ...prev,
      collapsedSpanIds: new Set(trace.spans.filter((span) => span.parent_span_id !== null).map((span) => span.span_id)),
      expandedGroupIds: new Set(),
    }));

  const setRowExpanded = (row: TreeRow, expand: boolean) => {
    if (row.kind === "span" && row.hasChildren && row.collapsed === expand) toggleSpan(row.id);
    if (row.kind === "group" && row.expanded !== expand) toggleGroup(row.id);
  };
  const moveBy = (delta: number) => {
    const index = rows.findIndex((row) => row.id === selectedRow?.id);
    const next = rows[Math.min(rows.length - 1, Math.max(0, index + delta))];
    if (next) select(next.id);
  };
  const fold = (expand: boolean) => {
    const row = rows.find((candidate) => candidate.id === selectedRow?.id);
    if (row) setRowExpanded(row, expand);
  };

  return {
    rows,
    selectedRow,
    selectedId: selectedRow?.id ?? selectedId,
    hideFramework: state.hideFramework,
    filtering,
    select,
    toggleSpan,
    toggleGroup,
    loadMore,
    setHideFramework,
    collapseAll,
    moveBy,
    fold,
  };
}
