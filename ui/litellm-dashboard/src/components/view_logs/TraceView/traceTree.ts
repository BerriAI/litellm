/**
 * Shared contract for the agent run view (span tree + detail pane).
 * Rows are produced by `buildTreeRows` in traceUtils.ts and rendered by SpanTree / DetailPane.
 */
import type { Span, SpanType } from "./traceTypes";

export type ErrorSource = "model" | "tool" | "litellm";

export interface SpanRowData {
  kind: "span";
  id: string;
  span: Span;
  depth: number;
  hasChildren: boolean;
  collapsed: boolean;
}

export interface GroupRowData {
  kind: "group";
  id: string;
  depth: number;
  name: string;
  type: SpanType;
  agent: string;
  members: Span[];
  failedCount: number;
  p50Duration: number;
  /** Every member failed (e.g. a tool that failed ×12). */
  isFailureGroup: boolean;
  expanded: boolean;
}

export interface LoadMoreRowData {
  kind: "load-more";
  id: string;
  depth: number;
  groupId: string;
  remaining: number;
}

export type TreeRow = SpanRowData | GroupRowData | LoadMoreRowData;

export interface SpanTreeState {
  hideFramework: boolean;
  collapsedSpanIds: ReadonlySet<string>;
  expandedGroupIds: ReadonlySet<string>;
  groupRevealCounts: Readonly<Record<string, number>>;
}
