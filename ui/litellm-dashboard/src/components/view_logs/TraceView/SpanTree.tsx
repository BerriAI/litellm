"use client";

import { MoreHorizontal, PanelRightOpen, Search, X } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import { FoldChevron } from "./Collapse";
import { PaneBar } from "./PaneBar";
import { groupFacts, SpanHoverCard, spanFacts } from "./SpanHoverCard";
import { SpanIcon } from "./SpanIcon";
import type { GroupRowData, SpanRowData, TreeRow } from "./traceTree";
import type { TraceSummary } from "./traceTypes";
import { fmtMs, previewText, type TreeGuide, treeGuides } from "./traceUtils";

interface SpanTreeProps {
  rows: TreeRow[];
  summary: TraceSummary;
  selectedId: string;
  hideFramework: boolean;
  onSelect: (id: string) => void;
  onToggleHideFramework: (checked: boolean) => void;
  onToggleSpan: (id: string) => void;
  onToggleGroup: (id: string) => void;
  onLoadMore: (groupId: string) => void;
  onOpenDetails?: () => void;
  /** Inside the side drawer J/K switch runs, so spans move with the arrow keys. */
  embedded?: boolean;
  query: string;
  onQueryChange: (query: string) => void;
  errorsOnly: boolean;
  onErrorsOnlyChange: (enabled: boolean) => void;
  filtering: boolean;
  onClearFilters: () => void;
  onCollapseAll: () => void;
}

const rowClass = (selected: boolean): string =>
  cn(
    "relative flex w-full items-stretch border-l-2 px-3.5 text-left outline-none transition-colors duration-150 ease-[cubic-bezier(0.4,0,0.2,1)] focus-visible:shadow-[inset_0_0_0_2px_var(--ring)] motion-reduce:transition-none",
    selected ? "border-l-primary bg-accent" : "border-l-transparent hover:bg-muted/60",
  );

const LINE = "border-l border-dashed border-border";
const ELBOW = "border-b border-l border-dashed border-border";
const NAME = "line-clamp-2 break-words text-sm leading-tight font-medium";
const META = "text-xs leading-4 text-muted-foreground tabular-nums";

/** One 20px column per ancestor: pass-through rails for open branches, an elbow into this row's tile. */
function Gutters({ depth, guide }: { depth: number; guide: TreeGuide }) {
  return Array.from({ length: depth }, (_, level) => {
    const parentColumn = level === depth - 1;
    const passThrough = parentColumn ? !guide.last : guide.rails[level] === true;
    return (
      <span key={level} aria-hidden="true" data-testid="tree-gutter" className="relative w-5 shrink-0">
        {passThrough && (
          <span data-guide="rail" className={cn("absolute inset-y-0 left-[calc(50%-0.5px)] w-px", LINE)} />
        )}
        {parentColumn && (
          <span data-guide="elbow" className={cn("absolute top-0 left-[calc(50%-0.5px)] h-3.5 w-1/2", ELBOW)} />
        )}
      </span>
    );
  });
}

function TileColumn({ tile, stem }: { tile: React.ReactNode; stem: boolean }) {
  return (
    <span className="flex w-5 shrink-0 flex-col items-center pt-1">
      <span className="relative z-raised">{tile}</span>
      {stem && <span aria-hidden="true" data-guide="stem" className={cn("w-px grow", LINE)} />}
    </span>
  );
}

interface RowProps {
  guide: TreeGuide;
  selected: boolean;
  traceStartMs: number;
  totalMs: number;
  onSelect: (id: string) => void;
  onToggleSpan: (id: string) => void;
  onToggleGroup: (id: string) => void;
  filtering: boolean;
}

function Caret({ open, onToggle, label }: { open: boolean; onToggle: () => void; label: string }) {
  return (
    <span
      role="button"
      tabIndex={-1}
      aria-label={label}
      className="grid size-6 shrink-0 place-items-center self-start rounded-sm p-0.5 text-muted-foreground transition-colors duration-150 hover:bg-muted motion-reduce:transition-none"
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
    >
      <FoldChevron open={open} className="size-4" />
    </span>
  );
}

function RowBody({ children, trailing }: { children: React.ReactNode; trailing: React.ReactNode }) {
  return (
    <span className="ml-2 grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-start gap-x-2 py-1">
      <span className="flex min-w-0 flex-col gap-1">{children}</span>
      {trailing}
    </span>
  );
}

interface WaterfallProps {
  startMs: number;
  durationMs: number;
  totalMs: number;
  tone: string;
}

/** Thin bar showing where this span sits inside the run's timeline. */
function Waterfall({ startMs, durationMs, totalMs, tone }: WaterfallProps) {
  const left = totalMs > 0 ? Math.min(99, (startMs / totalMs) * 100) : 0;
  const width = totalMs > 0 ? Math.max(0.8, Math.min(100 - left, (durationMs / totalMs) * 100)) : 0.8;
  return (
    <span
      aria-hidden="true"
      data-testid="span-waterfall"
      className="relative block h-[3px] w-full overflow-hidden rounded-full bg-muted"
    >
      <span className={cn("absolute inset-y-0 rounded-full", tone)} style={{ left: `${left}%`, width: `${width}%` }} />
    </span>
  );
}

const barTone = (type: string, failed: boolean): string => {
  if (failed) return "bg-destructive";
  if (type === "tool") return "bg-muted-foreground/40";
  if (type === "llm") return "bg-muted-foreground/50";
  return "bg-muted-foreground/60";
};

function SpanRow({
  row,
  guide,
  selected,
  traceStartMs,
  totalMs,
  onSelect,
  onToggleSpan,
  filtering,
}: RowProps & { row: SpanRowData }) {
  const { span } = row;
  const failed = span.status === "error";
  const duration = <span className={cn(META, "shrink-0")}>{fmtMs(span.duration_ms)}</span>;
  const caret = row.hasChildren && (
    <Caret open={!row.collapsed} label={row.collapsed ? "Expand" : "Collapse"} onToggle={() => onToggleSpan(row.id)} />
  );
  const leafDuration = !row.hasChildren && <span className="flex min-h-5 items-center">{duration}</span>;
  return (
    <SpanHoverCard facts={spanFacts(span)} traceStartMs={traceStartMs}>
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
        <Gutters depth={row.depth} guide={guide} />
        <TileColumn tile={<SpanIcon type={span.type} model={span.model} error={failed} />} stem={guide.stem} />
        <RowBody trailing={caret || leafDuration}>
          <span className={cn("flex min-w-0 items-center gap-2", row.hasChildren ? "min-h-6" : "min-h-5")}>
            <span className={cn(NAME, failed ? "text-destructive" : "text-foreground")}>{span.name}</span>
            {row.hasChildren && duration}
          </span>
          {span.type === "agent" && span.parent_span_id && span.input_preview ? (
            <span className="truncate text-xs text-muted-foreground" title={previewText(span.input_preview)}>
              {previewText(span.input_preview)}
            </span>
          ) : (
            filtering && (
              <span className="truncate text-xs text-muted-foreground">
                {previewText(span.input_preview) || span.agent}
              </span>
            )
          )}
          <Waterfall
            startMs={span.start_offset_ms}
            durationMs={span.duration_ms}
            totalMs={totalMs}
            tone={barTone(span.type, failed)}
          />
        </RowBody>
      </button>
    </SpanHoverCard>
  );
}

function GroupRow({
  row,
  guide,
  selected,
  traceStartMs,
  totalMs,
  onSelect,
  onToggleGroup,
}: RowProps & { row: GroupRowData }) {
  const groupStart = Math.min(...row.members.map((m) => m.start_offset_ms));
  const groupEnd = Math.max(...row.members.map((m) => m.start_offset_ms + m.duration_ms));
  return (
    <SpanHoverCard facts={groupFacts(row)} traceStartMs={traceStartMs}>
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
        <Gutters depth={row.depth} guide={guide} />
        <TileColumn
          tile={<SpanIcon type={row.type} model={row.members[0]?.model ?? null} error={row.failedCount > 0} />}
          stem={guide.stem}
        />
        <RowBody
          trailing={
            <Caret
              open={row.expanded}
              label={row.expanded ? "Collapse" : "Expand"}
              onToggle={() => onToggleGroup(row.id)}
            />
          }
        >
          <span className="flex min-h-6 min-w-0 items-center gap-2">
            <span className={cn(NAME, row.isFailureGroup ? "text-destructive" : "text-foreground")}>{row.name}</span>
            <span className={cn(META, "shrink-0")}>×{row.members.length}</span>
          </span>
          {row.failedCount > 0 && <span className={cn(META, "text-destructive")}>{row.failedCount} failed</span>}
          <Waterfall
            startMs={groupStart}
            durationMs={groupEnd - groupStart}
            totalMs={totalMs}
            tone={barTone(row.type, row.isFailureGroup)}
          />
        </RowBody>
      </button>
    </SpanHoverCard>
  );
}

export function SpanTree({
  rows,
  summary,
  selectedId,
  hideFramework,
  onSelect,
  onToggleHideFramework,
  onToggleSpan,
  onToggleGroup,
  onLoadMore,
  onOpenDetails,
  embedded = false,
  query,
  onQueryChange,
  errorsOnly,
  onErrorsOnlyChange,
  filtering,
  onClearFilters,
  onCollapseAll,
}: SpanTreeProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const guides = useMemo(() => treeGuides(rows.map((row) => row.depth)), [rows]);
  const traceStartMs = Date.parse(summary.start_time);

  useEffect(() => {
    scrollRef.current
      ?.querySelector(`[data-row-id="${CSS.escape(selectedId)}"]`)
      ?.scrollIntoView?.({ block: "nearest" });
  }, [selectedId]);

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col border-r border-border bg-card" aria-label="Run spans">
      <PaneBar className="justify-between">
        <span className="text-sm font-medium">
          Steps{" "}
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            {filtering && `${rows.length.toLocaleString()} of `}
            {summary.span_count.toLocaleString()}
          </span>
        </span>
        <div className="flex items-center gap-1">
          {onOpenDetails && (
            <Button variant="ghost" size="icon-xs" onClick={onOpenDetails} aria-label="Show details">
              <PanelRightOpen className="size-4" />
            </Button>
          )}
          <DropdownMenu>
            <DropdownMenuTrigger render={<Button variant="ghost" size="icon-xs" aria-label="Step display options" />}>
              <MoreHorizontal className="size-4" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52">
              <DropdownMenuCheckboxItem checked={hideFramework} onCheckedChange={onToggleHideFramework}>
                Hide framework spans
              </DropdownMenuCheckboxItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onCollapseAll}>Collapse branches</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </PaneBar>
      <PaneBar>
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            aria-label="Search steps"
            placeholder="Search steps"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            className="h-7 pr-7 pl-7 text-xs shadow-none md:text-xs"
          />
          {query && (
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Clear step search"
              className="absolute top-1/2 right-0.5 -translate-y-1/2"
              onClick={() => onQueryChange("")}
            >
              <X className="size-3" />
            </Button>
          )}
        </div>
        <Button
          variant={!errorsOnly ? "secondary" : "ghost"}
          size="xs"
          aria-pressed={!errorsOnly}
          onClick={() => onErrorsOnlyChange(false)}
        >
          All
        </Button>
        <Button
          variant={errorsOnly ? "secondary" : "ghost"}
          size="xs"
          aria-pressed={errorsOnly}
          onClick={() => onErrorsOnlyChange(true)}
        >
          Errors <span className="ml-1 text-muted-foreground">{summary.error_count}</span>
        </Button>
        {filtering && (
          <span role="status" className="sr-only">
            {rows.length} found
          </span>
        )}
      </PaneBar>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto py-2">
        {rows.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            <p>No matching steps</p>
            <Button variant="link" size="sm" onClick={onClearFilters}>
              Clear filters
            </Button>
          </div>
        )}
        <div role="tree" aria-label="Spans in time order">
          {rows.map((row, i) => {
            if (row.kind === "load-more") {
              return (
                <button
                  key={row.id}
                  type="button"
                  onClick={() => onLoadMore(row.groupId)}
                  className={cn(rowClass(false), "h-8 text-sm text-muted-foreground")}
                >
                  <Gutters depth={row.depth} guide={guides[i]} />
                  <span className="ml-0.5 flex items-center gap-2">
                    <MoreHorizontal className="relative z-raised size-3.5" /> Load 20 more
                    <span className="text-muted-foreground">({row.remaining} remaining)</span>
                  </span>
                </button>
              );
            }
            const shared = {
              guide: guides[i],
              selected: selectedId === row.id,
              traceStartMs,
              totalMs: summary.duration_ms,
              onSelect,
              onToggleSpan,
              onToggleGroup,
              filtering,
            };
            return row.kind === "group" ? (
              <GroupRow key={row.id} row={row} {...shared} />
            ) : (
              <SpanRow key={row.id} row={row} {...shared} />
            );
          })}
        </div>
      </div>
      <div className="flex min-h-10 shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-t border-border px-3 py-1.5 text-xs text-muted-foreground">
        {embedded ? (
          <>
            <span className="whitespace-nowrap">
              <Kbd>↑</Kbd>/<Kbd>↓</Kbd> step
            </span>
            <span className="whitespace-nowrap">
              <Kbd>J</Kbd>/<Kbd>K</Kbd> trace
            </span>
          </>
        ) : (
          <span className="whitespace-nowrap">
            <Kbd>J</Kbd>/<Kbd>K</Kbd> move
          </span>
        )}
        <span className="whitespace-nowrap">
          <Kbd>←</Kbd>/<Kbd>→</Kbd> fold
        </span>
        <span className="whitespace-nowrap">
          <Kbd>Esc</Kbd> close
        </span>
      </div>
    </section>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded-sm border border-border border-b-2 bg-muted px-[3px] font-mono text-muted-foreground">
      {children}
    </kbd>
  );
}
