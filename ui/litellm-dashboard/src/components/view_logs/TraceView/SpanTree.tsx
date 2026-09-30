"use client";

import {
  Bot,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  CircleDot,
  CornerDownRight,
  MoreHorizontal,
  Network,
  Wrench,
} from "lucide-react";
import { useEffect, useRef } from "react";

import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/cva.config";

import { DurationBar } from "./DurationBar";
import type { TreeRow } from "./traceTree";
import type { SpanType } from "./traceTypes";
import { fmtMs } from "./traceUtils";

interface SpanTreeProps {
  rows: TreeRow[];
  spanCount: number;
  totalMs: number;
  selectedId: string;
  hideFramework: boolean;
  onSelect: (id: string) => void;
  onToggleHideFramework: (checked: boolean) => void;
  onToggleSpan: (id: string) => void;
  onToggleGroup: (id: string) => void;
  onLoadMore: (groupId: string) => void;
}

const ROW_GRID = "grid h-8 w-full grid-cols-[minmax(275px,1fr)_minmax(94px,27%)_64px] items-center px-2";
const INDENT_PX = 15;

const rowClass = (selected: boolean): string =>
  cn(
    ROW_GRID,
    "border-b border-border/60 text-left outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset",
    selected ? "bg-accent shadow-[inset_2px_0_0_var(--foreground)]" : "hover:bg-muted/60",
  );

/** Span list with a framework toggle, a timeline column and keyboard hints. */
export function SpanTree({
  rows,
  spanCount,
  totalMs,
  selectedId,
  hideFramework,
  onSelect,
  onToggleHideFramework,
  onToggleSpan,
  onToggleGroup,
  onLoadMore,
}: SpanTreeProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current
      ?.querySelector(`[data-row-id="${CSS.escape(selectedId)}"]`)
      ?.scrollIntoView?.({ block: "nearest" });
  }, [selectedId]);

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col border-r border-border bg-card" aria-label="Run spans">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-border bg-muted/40 px-2.5">
        <span className="shrink-0 font-mono text-[9px] text-muted-foreground tabular-nums">
          {`${spanCount.toLocaleString()} spans`}
        </span>
        <label className="ml-auto flex shrink-0 cursor-pointer items-center gap-2 font-mono text-[9px] text-muted-foreground">
          hide framework
          <Switch
            size="sm"
            checked={hideFramework}
            onCheckedChange={(checked) => onToggleHideFramework(checked)}
            aria-label="Hide framework spans"
          />
        </label>
      </div>
      <div
        className={cn(
          ROW_GRID,
          "h-7 shrink-0 border-b border-border bg-muted/60 font-mono text-[8px] tracking-[0.1em] text-muted-foreground uppercase",
        )}
      >
        <span className="pl-6">Span</span>
        <span>Timeline</span>
        <span className="text-right">Time</span>
      </div>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto" role="tree" aria-label="Spans in time order">
        {rows.map((row) => (
          <TreeRowItem
            key={row.id}
            row={row}
            selected={selectedId === row.id}
            totalMs={totalMs}
            onSelect={onSelect}
            onToggleSpan={onToggleSpan}
            onToggleGroup={onToggleGroup}
            onLoadMore={onLoadMore}
          />
        ))}
      </div>
      <div className="flex h-7 shrink-0 items-center gap-3 border-t border-border bg-muted/60 px-3 font-mono text-[8px] text-muted-foreground">
        <span>
          <Kbd>J</Kbd>/<Kbd>K</Kbd> move
        </span>
        <span>
          <Kbd>←</Kbd>/<Kbd>→</Kbd> fold
        </span>
        <span>
          <Kbd>Esc</Kbd> close
        </span>
      </div>
    </section>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded-[3px] border border-border border-b-2 bg-muted px-[3px] font-mono text-muted-foreground">
      {children}
    </kbd>
  );
}

interface TreeRowItemProps {
  row: TreeRow;
  selected: boolean;
  totalMs: number;
  onSelect: (id: string) => void;
  onToggleSpan: (id: string) => void;
  onToggleGroup: (id: string) => void;
  onLoadMore: (groupId: string) => void;
}

function TreeRowItem({ row, selected, totalMs, onSelect, onToggleSpan, onToggleGroup, onLoadMore }: TreeRowItemProps) {
  if (row.kind === "load-more") {
    return (
      <button
        type="button"
        onClick={() => onLoadMore(row.groupId)}
        className="flex h-8 w-full items-center gap-2 border-b border-border/60 px-2 font-mono text-[9px] text-foreground hover:bg-muted/60"
        style={{ paddingLeft: `${18 + row.depth * INDENT_PX}px` }}
      >
        <MoreHorizontal className="size-3" /> load 20 more{" "}
        <span className="text-muted-foreground">({row.remaining} remaining)</span>
      </button>
    );
  }

  if (row.kind === "group") {
    const start = Math.min(...row.members.map((m) => m.start_offset_ms));
    const end = Math.max(...row.members.map((m) => m.start_offset_ms + m.duration_ms));
    return (
      <button
        type="button"
        role="treeitem"
        aria-expanded={row.expanded}
        aria-selected={selected}
        data-row-id={row.id}
        onClick={() => {
          onSelect(row.id);
          onToggleGroup(row.id);
        }}
        className={rowClass(selected)}
      >
        <div className="flex min-w-0 items-center" style={{ paddingLeft: `${row.depth * INDENT_PX}px` }}>
          {row.expanded ? (
            <ChevronDown className="mr-0.5 size-3 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="mr-0.5 size-3 shrink-0 text-muted-foreground" />
          )}
          <TypeIcon type={row.type} error={row.failedCount > 0} />
          <span
            className={cn(
              "ml-1.5 truncate font-mono text-[10px]",
              row.isFailureGroup ? "text-destructive" : "text-foreground",
            )}
          >
            {row.name}
          </span>
          <span className="ml-1 font-mono text-[9px] text-muted-foreground">×{row.members.length}</span>
          <span className="ml-2 hidden truncate font-mono text-[8px] text-muted-foreground xl:block">
            p50={fmtMs(row.p50Duration)}
          </span>
          {row.failedCount > 0 && (
            <span className="ml-2 shrink-0 font-mono text-[8px] text-destructive">{row.failedCount} failed</span>
          )}
        </div>
        <DurationBar startMs={start} durationMs={end - start} totalMs={totalMs} error={row.isFailureGroup} />
        <span className="text-right font-mono text-[9px] text-muted-foreground tabular-nums">
          {fmtMs(row.p50Duration)}
        </span>
      </button>
    );
  }

  const { span } = row;
  const failed = span.status === "error";
  return (
    <button
      type="button"
      role="treeitem"
      aria-expanded={row.hasChildren ? !row.collapsed : undefined}
      aria-selected={selected}
      data-row-id={row.id}
      onClick={() => onSelect(row.id)}
      onDoubleClick={() => row.hasChildren && onToggleSpan(row.id)}
      className={rowClass(selected)}
    >
      <div className="flex min-w-0 items-center" style={{ paddingLeft: `${row.depth * INDENT_PX}px` }}>
        {row.hasChildren ? (
          <span
            role="button"
            tabIndex={-1}
            aria-label={row.collapsed ? "Expand" : "Collapse"}
            className="mr-0.5 shrink-0"
            onClick={(event) => {
              event.stopPropagation();
              onToggleSpan(row.id);
            }}
          >
            {row.collapsed ? (
              <ChevronRight className="size-3 text-muted-foreground" />
            ) : (
              <ChevronDown className="size-3 text-muted-foreground" />
            )}
          </span>
        ) : (
          <CornerDownRight className="mr-0.5 size-3 shrink-0 text-muted-foreground/50" />
        )}
        <TypeIcon type={span.type} error={failed} />
        <span className={cn("ml-1.5 truncate font-mono text-[10px]", failed ? "text-destructive" : "text-foreground")}>
          {span.name}
        </span>
        {span.model && (
          <span className="ml-2 hidden truncate font-mono text-[8px] text-muted-foreground 2xl:block">
            {span.model.split("/").pop()}
          </span>
        )}
      </div>
      <DurationBar startMs={span.start_offset_ms} durationMs={span.duration_ms} totalMs={totalMs} error={failed} />
      <span
        className={cn(
          "text-right font-mono text-[9px] tabular-nums",
          failed ? "text-destructive" : "text-muted-foreground",
        )}
      >
        {fmtMs(span.duration_ms)}
      </span>
    </button>
  );
}

function TypeIcon({ type, error }: { type: SpanType; error: boolean }) {
  const className = cn("size-3 shrink-0", error ? "text-destructive" : "text-muted-foreground");
  if (type === "agent") return <Bot className={className} />;
  if (type === "llm") return <BrainCircuit className={className} />;
  if (type === "tool") return <Wrench className={className} />;
  if (type === "framework") return <Network className={className} />;
  return <CircleDot className={className} />;
}
