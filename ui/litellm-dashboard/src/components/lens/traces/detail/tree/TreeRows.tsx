"use client";

import { MoreHorizontal } from "lucide-react";

import { cn } from "@/lib/cva.config";

import { formatCost } from "../../list/AgentTracesTable";
import { FoldChevron } from "../../ui/Collapse";
import { FAMILY_BAR, SPAN_FAMILY, SpanIcon } from "../../ui/SpanIcon";
import type { GroupRowData, LoadMoreRowData, SpanRowData } from "../../tree";
import type { SpanType } from "../../types";
import { fmtMs, previewText, type TreeGuide } from "../../utils";
import { groupFacts, SpanHoverCard, spanFacts } from "./SpanHoverCard";
import { toolSummary } from "../content/payload";
import { BAR_TRACK, barGeometry } from "./timeline";

export type TreeLayout = "tree" | "waterfall";

const INDENT = "w-4";
const ROW =
  "group/row relative flex h-8 w-full items-center pr-2 pl-2 text-left outline-none transition-colors duration-100 focus-visible:shadow-[inset_0_0_0_2px_var(--ring)] motion-reduce:transition-none";
const GUIDE = "bg-border";

const rowTone = (selected: boolean): string =>
  selected
    ? "bg-trace-chain-soft before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:bg-trace-chain"
    : "hover:bg-muted/60";

/** One indent column per ancestor: a rail where that ancestor has more children, an elbow into this row. */
function Guides({ depth, guide }: { depth: number; guide: TreeGuide }) {
  return Array.from({ length: depth }, (_, level) => {
    const parentColumn = level === depth - 1;
    const rail = parentColumn ? !guide.last : guide.rails[level] === true;
    return (
      <span key={level} aria-hidden="true" data-testid="tree-gutter" className={cn("relative h-full shrink-0", INDENT)}>
        {rail && <span data-guide="rail" className={cn("absolute inset-y-0 left-2.5 w-px", GUIDE)} />}
        {parentColumn && (
          <>
            <span data-guide="elbow" className={cn("absolute top-0 left-2.5 h-1/2 w-px", GUIDE)} />
            <span className={cn("absolute top-1/2 left-2.5 h-px w-1.5", GUIDE)} />
          </>
        )}
      </span>
    );
  });
}

function Caret({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <span
      role="button"
      tabIndex={-1}
      aria-label={open ? "Collapse" : "Expand"}
      className="grid size-5 shrink-0 place-items-center rounded-sm text-muted-foreground hover:bg-muted hover:text-foreground"
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
    >
      <FoldChevron open={open} className="size-3.5" />
    </span>
  );
}

function Bar({
  start,
  duration,
  total,
  type,
  failed,
}: {
  start: number;
  duration: number;
  total: number;
  type: SpanType;
  failed: boolean;
}) {
  const { left, width } = barGeometry(start, duration, total);
  return (
    <span
      aria-hidden="true"
      data-testid="span-waterfall"
      className={cn("relative h-full shrink-0 border-l border-border/60", BAR_TRACK)}
    >
      <span
        className={cn(
          "absolute top-1/2 h-3 -translate-y-1/2 rounded-sm",
          failed ? "bg-destructive" : FAMILY_BAR[SPAN_FAMILY[type]],
        )}
        style={{ left: `${left}%`, width: `${width}%` }}
      />
    </span>
  );
}

interface RowFrameProps {
  id: string;
  depth: number;
  guide: TreeGuide;
  selected: boolean;
  expanded: boolean | undefined;
  onClick: () => void;
  onDoubleClick?: () => void;
  children: React.ReactNode;
}

function RowFrame({ id, depth, guide, selected, expanded, onClick, onDoubleClick, children }: RowFrameProps) {
  return (
    <button
      type="button"
      role="treeitem"
      aria-expanded={expanded}
      aria-selected={selected}
      data-row-id={id}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      className={cn(ROW, rowTone(selected))}
    >
      <Guides depth={depth} guide={guide} />
      {children}
    </button>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="flex min-w-0 flex-1 items-center gap-2 pl-1.5">{children}</span>;
}

const NAME = "min-w-0 truncate text-sm";
const META = "shrink-0 text-xs text-muted-foreground tabular-nums";

export interface RowContext {
  guide: TreeGuide;
  selected: boolean;
  layout: TreeLayout;
  traceStartMs: number;
  totalMs: number;
  filtering: boolean;
  onSelect: (id: string) => void;
  onToggleSpan: (id: string) => void;
  onToggleGroup: (id: string) => void;
}

const readablePreview = (preview: string): string => {
  const text = previewText(preview);
  return /^\s*[[{]/.test(text) ? "" : text;
};

const subtitle = (row: SpanRowData, filtering: boolean): string => {
  const { span } = row;
  if (span.type === "tool") return toolSummary(span.input_preview);
  if (filtering) return readablePreview(span.input_preview) || span.agent;
  return span.type === "agent" && span.parent_span_id ? readablePreview(span.input_preview) : "";
};

export function SpanRow({ row, ctx }: { row: SpanRowData; ctx: RowContext }) {
  const { span } = row;
  const failed = span.status === "error";
  const hint = ctx.layout === "tree" ? subtitle(row, ctx.filtering) : "";
  const modelChip = span.type === "llm" && span.model !== span.name ? span.model : null;
  return (
    <SpanHoverCard facts={spanFacts(span)} traceStartMs={ctx.traceStartMs}>
      <RowFrame
        id={row.id}
        depth={row.depth}
        guide={ctx.guide}
        selected={ctx.selected}
        expanded={row.hasChildren ? !row.collapsed : undefined}
        onClick={() => ctx.onSelect(row.id)}
        onDoubleClick={() => row.hasChildren && ctx.onToggleSpan(row.id)}
      >
        <SpanIcon type={span.type} model={span.model} error={failed} size="md" />
        <Label>
          <span className={cn(NAME, failed ? "font-medium text-destructive" : "text-foreground")}>{span.name}</span>
          {modelChip && (
            <span className="min-w-0 shrink truncate rounded-sm border border-border px-1 font-mono text-xs leading-4 text-muted-foreground">
              {modelChip}
            </span>
          )}
          {ctx.layout === "tree" && <span className={META}>{fmtMs(span.duration_ms)}</span>}
          {ctx.layout === "tree" && span.spend_match != null && (
            <span className={META} data-testid="step-cost">
              {span.spend_match === "matched" && span.spend != null ? formatCost(span.spend) : "—"}
            </span>
          )}
          {hint && (
            <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground" title={hint}>
              {hint}
            </span>
          )}
        </Label>
        {ctx.layout === "waterfall" && (
          <Bar
            start={span.start_offset_ms}
            duration={span.duration_ms}
            total={ctx.totalMs}
            type={span.type}
            failed={failed}
          />
        )}
        <span className="flex w-5 shrink-0 justify-end">
          {row.hasChildren && <Caret open={!row.collapsed} onToggle={() => ctx.onToggleSpan(row.id)} />}
        </span>
      </RowFrame>
    </SpanHoverCard>
  );
}

export function GroupRow({ row, ctx }: { row: GroupRowData; ctx: RowContext }) {
  const start = Math.min(...row.members.map((m) => m.start_offset_ms));
  const end = Math.max(...row.members.map((m) => m.start_offset_ms + m.duration_ms));
  return (
    <SpanHoverCard facts={groupFacts(row)} traceStartMs={ctx.traceStartMs}>
      <RowFrame
        id={row.id}
        depth={row.depth}
        guide={ctx.guide}
        selected={ctx.selected}
        expanded={row.expanded}
        onClick={() => {
          ctx.onSelect(row.id);
          ctx.onToggleGroup(row.id);
        }}
      >
        <SpanIcon type={row.type} model={row.members[0]?.model ?? null} error={row.failedCount > 0} size="md" />
        <Label>
          <span className={cn(NAME, row.isFailureGroup ? "font-medium text-destructive" : "text-foreground")}>
            {row.name}
          </span>
          <span className="shrink-0 rounded-sm bg-muted px-1 text-xs leading-4 text-muted-foreground tabular-nums">
            ×{row.members.length}
          </span>
          {row.failedCount > 0 && <span className={cn(META, "text-destructive")}>{row.failedCount} failed</span>}
        </Label>
        {ctx.layout === "waterfall" && (
          <Bar start={start} duration={end - start} total={ctx.totalMs} type={row.type} failed={row.isFailureGroup} />
        )}
        <span className="flex w-5 shrink-0 justify-end">
          <Caret open={row.expanded} onToggle={() => ctx.onToggleGroup(row.id)} />
        </span>
      </RowFrame>
    </SpanHoverCard>
  );
}

export function LoadMoreRow({
  row,
  guide,
  onLoadMore,
}: {
  row: LoadMoreRowData;
  guide: TreeGuide;
  onLoadMore: (groupId: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onLoadMore(row.groupId)}
      className={cn(ROW, "text-xs text-muted-foreground hover:bg-trace-row-hover hover:text-foreground")}
    >
      <Guides depth={row.depth} guide={guide} />
      <span className="flex items-center gap-2 pl-1.5">
        <MoreHorizontal className="size-3.5" /> Load 20 more
        <span>({row.remaining} remaining)</span>
      </span>
    </button>
  );
}
