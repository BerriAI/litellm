"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/cva.config";

import { SPAN_BAR_CLASS, SpanTypePill } from "./TracePills";
import type { Span } from "./traceTypes";
import { fmtMs, GROUP_PAGE_SIZE, spanLabel, type SpanGroup, type TreeRow } from "./traceUtils";

interface SpanTreeProps {
  rows: TreeRow[];
  traceDurationMs: number;
  selectedId: string | null;
  onSelect: (spanId: string) => void;
  onToggleCollapse: (spanId: string) => void;
  /** Reveal `shown` invocations of a collapsed agent group (0 hides them again). */
  onGroupShow: (groupKey: string, shown: number) => void;
}

const MIN_BAR_PERCENT = 0.3;

const spanMeta = (span: Span): string => fmtMs(span.duration_ms);

export const groupSummary = (group: SpanGroup): string => {
  const errors = group.errors ? ` · ${group.errors} error${group.errors === 1 ? "" : "s"}` : "";
  return `${group.name} ×${group.spans.length} · p50 ${fmtMs(group.p50Ms)}${errors}`;
};

function Guides({ depth }: { depth: number }) {
  return (
    <>
      {Array.from({ length: depth }, (_, i) => (
        <span key={i} className="ml-1.5 w-3.5 shrink-0 self-stretch border-l border-border" />
      ))}
    </>
  );
}

function WaterfallBar({ span, total }: { span: Span; total: number }) {
  const left = total > 0 ? (span.start_offset_ms / total) * 100 : 0;
  const width = total > 0 ? Math.max((span.duration_ms / total) * 100, MIN_BAR_PERCENT) : MIN_BAR_PERCENT;
  return (
    <div className="relative h-2.5 rounded-sm bg-muted">
      <i
        className={cn(
          "absolute top-0 h-2.5 min-w-[2px] rounded-sm",
          span.status === "error" ? "bg-destructive" : SPAN_BAR_CLASS[span.type],
        )}
        style={{ left: `${left.toFixed(2)}%`, width: `${width.toFixed(2)}%` }}
      />
    </div>
  );
}

interface SpanRowProps {
  row: Extract<TreeRow, { kind: "span" }>;
  total: number;
  selected: boolean;
  onSelect: (spanId: string) => void;
  onToggleCollapse: (spanId: string) => void;
}

function SpanRow({ row, total, selected, onSelect, onToggleCollapse }: SpanRowProps) {
  const { span } = row;
  const handleToggle = (event: React.MouseEvent) => {
    event.stopPropagation();
    onToggleCollapse(span.span_id);
  };
  return (
    <div
      role="treeitem"
      aria-selected={selected}
      aria-level={row.depth + 1}
      data-span-id={span.span_id}
      onClick={() => onSelect(span.span_id)}
      className={cn(
        "grid min-h-7 cursor-pointer grid-cols-[minmax(0,1fr)_28%] items-center gap-2.5 border-l-2 border-transparent px-3 py-0.5 hover:bg-muted/60",
        selected && "border-l-primary bg-primary/5",
      )}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <Guides depth={row.depth} />
        <button
          type="button"
          aria-label={row.isCollapsed ? `Expand ${span.name}` : `Collapse ${span.name}`}
          className={cn("w-3.5 shrink-0 text-[10px] text-muted-foreground", !row.hasChildren && "invisible")}
          onClick={handleToggle}
          tabIndex={row.hasChildren ? 0 : -1}
        >
          {row.isCollapsed ? "▸" : "▾"}
        </button>
        <SpanTypePill type={span.type} />
        <span className={cn("truncate", span.status === "error" && "text-destructive")}>{spanLabel(span)}</span>
        <span className="ml-auto whitespace-nowrap pl-1.5 text-[11px] text-muted-foreground">{spanMeta(span)}</span>
      </div>
      <WaterfallBar span={span} total={total} />
    </div>
  );
}

function GroupRow({
  row,
  onGroupShow,
}: {
  row: Extract<TreeRow, { kind: "group" }>;
  onGroupShow: SpanTreeProps["onGroupShow"];
}) {
  const open = row.shown > 0;
  return (
    <div
      role="treeitem"
      aria-expanded={open}
      aria-selected={false}
      aria-level={row.depth + 1}
      onClick={() => onGroupShow(row.group.key, open ? 0 : GROUP_PAGE_SIZE)}
      className="flex min-h-7 cursor-pointer items-center gap-1.5 px-3 py-0.5 hover:bg-muted/60"
    >
      <Guides depth={row.depth} />
      <span className="w-3.5 shrink-0 text-[10px] text-muted-foreground">{open ? "▾" : "▸"}</span>
      <SpanTypePill type="agent" />
      <span className={cn("truncate font-medium", row.group.errors > 0 && "text-destructive")}>
        {groupSummary(row.group)}
      </span>
    </div>
  );
}

function MoreRow({
  row,
  onGroupShow,
}: {
  row: Extract<TreeRow, { kind: "more" }>;
  onGroupShow: SpanTreeProps["onGroupShow"];
}) {
  const remaining = row.group.spans.length - row.shown;
  return (
    <div className="flex min-h-7 items-center gap-1.5 px-3 py-0.5">
      <Guides depth={row.depth} />
      <button
        type="button"
        className="text-xs text-primary hover:underline"
        onClick={() => onGroupShow(row.group.key, row.shown + GROUP_PAGE_SIZE)}
      >
        Show {Math.min(remaining, GROUP_PAGE_SIZE)} more of {remaining} {row.group.name} invocations
      </button>
    </div>
  );
}

const rowKey = (row: TreeRow): string => (row.kind === "span" ? row.span.span_id : `${row.kind}:${row.group.key}`);

/** Waterfall tree of visible spans. Rows are computed by `flattenTree` in the drawer. */
export function SpanTree({
  rows,
  traceDurationMs,
  selectedId,
  onSelect,
  onToggleCollapse,
  onGroupShow,
}: SpanTreeProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selectedId) return;
    const node = scrollRef.current?.querySelector(`[data-span-id="${selectedId}"]`);
    if (node && "scrollIntoView" in node) (node as HTMLElement).scrollIntoView({ block: "nearest" });
  }, [selectedId, rows]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="grid grid-cols-[minmax(0,1fr)_28%] gap-2.5 border-b px-3 py-1 text-[11px] text-muted-foreground">
        <span>Span</span>
        <span className="flex justify-between">
          <span>0</span>
          <span>{fmtMs(traceDurationMs / 2)}</span>
          <span>{fmtMs(traceDurationMs)}</span>
        </span>
      </div>
      <div ref={scrollRef} role="tree" aria-label="Span tree" className="min-h-0 flex-1 overflow-auto text-[13px]">
        {rows.map((row) => {
          if (row.kind === "group") return <GroupRow key={rowKey(row)} row={row} onGroupShow={onGroupShow} />;
          if (row.kind === "more") return <MoreRow key={rowKey(row)} row={row} onGroupShow={onGroupShow} />;
          return (
            <SpanRow
              key={rowKey(row)}
              row={row}
              total={traceDurationMs}
              selected={row.span.span_id === selectedId}
              onSelect={onSelect}
              onToggleCollapse={onToggleCollapse}
            />
          );
        })}
      </div>
    </div>
  );
}
