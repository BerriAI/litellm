"use client";

import { BarChart3, Clock, Coins, MoreHorizontal } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/cva.config";

import { formatCost } from "./AgentTracesTable";
import { FoldChevron } from "./Collapse";
import { groupFacts, SpanHoverCard, spanFacts } from "./SpanHoverCard";
import { SpanIcon } from "./SpanIcon";
import type { GroupRowData, SpanRowData, TreeRow } from "./traceTree";
import type { TraceSummary } from "./traceTypes";
import { fmtMs, fmtTok, type TreeGuide, treeGuides } from "./traceUtils";

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
}

const LEVEL_PX = 22;
const ICON_PX = 20;
const ROW_PAD_PX = 14;

const rowClass = (selected: boolean): string =>
  cn(
    "relative flex w-full items-start gap-2 py-[7px] pr-3.5 text-left text-[13px] transition-colors duration-150 outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset motion-reduce:transition-none",
    selected ? "bg-info/10 shadow-[inset_2px_0_0_var(--info)]" : "hover:bg-muted/60",
  );

function Pill({ icon: Icon, children }: { icon: typeof Clock; children: React.ReactNode }) {
  return (
    <span className="inline-flex shrink-0 items-center gap-1 rounded-[3px] border border-trace-border bg-card px-1 text-[12px] text-trace-text tabular-nums">
      <Icon className="size-[11px] text-muted-foreground" />
      {children}
    </span>
  );
}

function Summary({ summary }: { summary: TraceSummary }) {
  const tokens = summary.input_tokens + summary.output_tokens;
  return (
    <div className="px-3.5 pt-1.5 pb-2.5">
      <div className="flex items-center gap-1.5 text-[12px] font-semibold text-trace-text">
        <BarChart3 className="size-3.5 text-info" />
        Summary
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 pl-5 text-[13px] text-trace-duration tabular-nums">
        <span>{`${summary.span_count.toLocaleString()} spans`}</span>
        <span className="text-muted-foreground/50">/</span>
        <Clock className="size-3" />
        <span>{fmtMs(summary.duration_ms)}</span>
        {tokens > 0 && (
          <>
            <span className="text-muted-foreground/50">/</span>
            <Coins className="size-3" />
            <span>{fmtTok(tokens)}</span>
          </>
        )}
        {summary.spend != null && (
          <>
            <span className="text-muted-foreground/50">/</span>
            <span>{formatCost(summary.spend)}</span>
          </>
        )}
      </div>
    </div>
  );
}

/** Vertical rails for open ancestor branches plus a rounded elbow into this row's icon. */
function Connectors({ depth, guide }: { depth: number; guide: TreeGuide }) {
  if (depth === 0) return null;
  const railLeft = (level: number) => ROW_PAD_PX + (level - 1) * LEVEL_PX + ICON_PX / 2;
  return (
    <span aria-hidden="true">
      {guide.rails.map((open, k) =>
        open ? (
          <span key={k} className="absolute inset-y-0 border-l border-border" style={{ left: railLeft(k + 1) }} />
        ) : null,
      )}
      {!guide.last && <span className="absolute inset-y-0 border-l border-border" style={{ left: railLeft(depth) }} />}
      <span
        className="absolute top-0 h-[17px] rounded-bl-md border-b border-l border-border"
        style={{ left: railLeft(depth), width: LEVEL_PX - ICON_PX / 2 }}
      />
    </span>
  );
}

const indent = (depth: number) => ({ paddingLeft: ROW_PAD_PX + depth * LEVEL_PX });

interface RowProps {
  guide: TreeGuide;
  selected: boolean;
  traceStartMs: number;
  onSelect: (id: string) => void;
  onToggleSpan: (id: string) => void;
  onToggleGroup: (id: string) => void;
}

function Caret({ open, onToggle, label }: { open: boolean; onToggle: () => void; label: string }) {
  return (
    <span
      role="button"
      tabIndex={-1}
      aria-label={label}
      className="mt-[3px] ml-auto grid size-4 shrink-0 place-items-center text-muted-foreground/70 hover:text-foreground"
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
    >
      <FoldChevron open={open} className="size-3.5" />
    </span>
  );
}

function SpanRow({ row, guide, selected, traceStartMs, onSelect, onToggleSpan }: RowProps & { row: SpanRowData }) {
  const { span } = row;
  const failed = span.status === "error";
  const tokens = span.input_tokens + span.output_tokens;
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
        style={indent(row.depth)}
      >
        <Connectors depth={row.depth} guide={guide} />
        <span className="relative z-raised mt-px">
          <SpanIcon type={span.type} model={span.model} error={failed} />
        </span>
        <span className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="flex min-w-0 items-center gap-1.5">
            <span className={cn("truncate text-[14px] font-medium", failed ? "text-destructive" : "text-trace-text")}>
              {span.name}
            </span>
            {span.type === "llm" && span.model ? (
              <span className="truncate rounded-[3px] border border-trace-border bg-card px-1 text-[12px] text-trace-text">
                {span.model.split("/").pop()}
              </span>
            ) : (
              <span className="shrink-0 text-[13px] text-trace-duration tabular-nums">{fmtMs(span.duration_ms)}</span>
            )}
          </span>
          {span.type === "llm" && (
            <span className="flex items-center gap-1.5 text-[13px] text-trace-duration tabular-nums">
              {fmtMs(span.duration_ms)}
              {tokens > 0 && <Pill icon={Coins}>{fmtTok(tokens)}</Pill>}
            </span>
          )}
        </span>
        {row.hasChildren && (
          <Caret
            open={!row.collapsed}
            label={row.collapsed ? "Expand" : "Collapse"}
            onToggle={() => onToggleSpan(row.id)}
          />
        )}
      </button>
    </SpanHoverCard>
  );
}

function GroupRow({ row, guide, selected, traceStartMs, onSelect, onToggleGroup }: RowProps & { row: GroupRowData }) {
  const failedAll = row.isFailureGroup;
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
        style={indent(row.depth)}
      >
        <Connectors depth={row.depth} guide={guide} />
        <span className="relative z-raised mt-px">
          <SpanIcon type={row.type} model={row.members[0]?.model ?? null} error={row.failedCount > 0} />
        </span>
        <span className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="flex min-w-0 items-center gap-1.5">
            <span
              className={cn("truncate text-[14px] font-medium", failedAll ? "text-destructive" : "text-trace-text")}
            >
              {row.name}
            </span>
            <span className="shrink-0 text-muted-foreground">×{row.members.length}</span>
          </span>
          <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground tabular-nums">
            <span>p50 {fmtMs(row.p50Duration)}</span>
            {row.failedCount > 0 && <span className="text-destructive">{row.failedCount} failed</span>}
          </span>
        </span>
        <Caret
          open={row.expanded}
          label={row.expanded ? "Collapse" : "Expand"}
          onToggle={() => onToggleGroup(row.id)}
        />
      </button>
    </SpanHoverCard>
  );
}

/** Span tree with connector lines, a run summary on top, a framework toggle and keyboard hints. */
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
      <div className="flex h-11 shrink-0 items-center gap-2 px-3.5">
        <span className="text-[13px] font-medium text-muted-foreground">Spans</span>
        <label className="ml-auto flex shrink-0 cursor-pointer items-center gap-2 text-[12px] text-muted-foreground">
          Hide framework
          <Switch
            size="sm"
            checked={hideFramework}
            onCheckedChange={(checked) => onToggleHideFramework(checked)}
            aria-label="Hide framework spans"
          />
        </label>
      </div>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto pb-4">
        <Summary summary={summary} />
        <div role="tree" aria-label="Spans in time order" className="border-t border-border/60 pt-1">
          {rows.map((row, i) => {
            if (row.kind === "load-more") {
              return (
                <button
                  key={row.id}
                  type="button"
                  onClick={() => onLoadMore(row.groupId)}
                  className="relative flex h-8 w-full items-center gap-2 text-[12px] text-foreground hover:bg-muted/60"
                  style={indent(row.depth)}
                >
                  <Connectors depth={row.depth} guide={guides[i]} />
                  <MoreHorizontal className="relative z-raised size-3.5" /> Load 20 more
                  <span className="text-muted-foreground">({row.remaining} remaining)</span>
                </button>
              );
            }
            const shared = {
              guide: guides[i],
              selected: selectedId === row.id,
              traceStartMs,
              onSelect,
              onToggleSpan,
              onToggleGroup,
            };
            return row.kind === "group" ? (
              <GroupRow key={row.id} row={row} {...shared} />
            ) : (
              <SpanRow key={row.id} row={row} {...shared} />
            );
          })}
        </div>
      </div>
      <div className="flex h-8 shrink-0 items-center gap-3 border-t border-border px-3.5 text-[11px] text-muted-foreground">
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
