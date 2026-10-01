"use client";

import { BarChart3, Clock, Coins, ListTree, MoreHorizontal, Timer } from "lucide-react";
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
  /** Inside the side drawer J/K switch runs, so spans move with the arrow keys. */
  embedded?: boolean;
}

const rowClass = (selected: boolean): string =>
  cn(
    "relative flex w-full items-stretch border-l-2 px-3.5 text-left outline-none transition-colors duration-150 ease-[cubic-bezier(0.4,0,0.2,1)] focus-visible:shadow-[inset_0_0_0_2px_var(--trace-brand)] motion-reduce:transition-none",
    selected ? "border-l-trace-brand bg-trace-row-selected" : "border-l-transparent hover:bg-trace-row-hover",
  );

const LINE = "border-dashed border-trace-line [border-width:0] [border-left-width:1px]";
const ELBOW = "border-dashed border-trace-line [border-width:0] [border-left-width:1px] [border-bottom-width:1px]";
const NAME = "truncate text-[13px] leading-[1.2] font-medium";
const MONO_NAME = "truncate font-mono text-[12px] leading-[1.2]";
const META = "font-mono text-[11px] leading-[14px] text-trace-duration tabular-nums";

function Stat({ icon: Icon, children }: { icon: typeof Clock; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1">
      <Icon className="size-3" />
      {children}
    </span>
  );
}

const Slash = () => <span className="text-trace-key">/</span>;

function Summary({ summary }: { summary: TraceSummary }) {
  const tokens = summary.input_tokens + summary.output_tokens;
  return (
    <div className="flex flex-wrap items-center gap-2 px-4 py-2 transition-colors duration-150 hover:bg-trace-turn motion-reduce:transition-none">
      <span className="flex items-center gap-1.5 text-[12px] leading-[13.8px] font-semibold text-trace-text">
        <span className="grid size-5 place-items-center rounded-[4px] bg-trace-row-selected text-[var(--trace-summary-glyph,#0d3d77)]">
          <BarChart3 className="size-3" strokeWidth={1.5} />
        </span>
        Summary
      </span>
      <span className="flex basis-full flex-wrap items-center gap-x-2 pl-6 font-mono text-[11.5px] leading-[13.8px] text-trace-text-secondary tabular-nums">
        <Stat icon={ListTree}>{`${summary.span_count.toLocaleString()} spans`}</Stat>
        <Slash />
        <Stat icon={Timer}>{fmtMs(summary.duration_ms)}</Stat>
        {tokens > 0 && (
          <>
            <Slash />
            <Stat icon={Coins}>{fmtTok(tokens)}</Stat>
          </>
        )}
        {summary.spend != null && (
          <>
            <Slash />
            <span>{formatCost(summary.spend)}</span>
          </>
        )}
      </span>
    </div>
  );
}

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
}

function Caret({ open, onToggle, label }: { open: boolean; onToggle: () => void; label: string }) {
  return (
    <span
      role="button"
      tabIndex={-1}
      aria-label={label}
      className="grid size-6 shrink-0 place-items-center self-start rounded-[4px] p-0.5 text-trace-text-secondary transition-colors duration-150 hover:bg-trace-tab-active motion-reduce:transition-none"
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
      className="relative block h-[3px] w-full overflow-hidden rounded-full bg-trace-chip"
    >
      <span className={cn("absolute inset-y-0 rounded-full", tone)} style={{ left: `${left}%`, width: `${width}%` }} />
    </span>
  );
}

const barTone = (type: string, failed: boolean): string => {
  if (failed) return "bg-destructive";
  if (type === "tool") return "bg-trace-tool";
  if (type === "llm") return "bg-trace-llm";
  return "bg-trace-chain";
};

function SpanRow({
  row,
  guide,
  selected,
  traceStartMs,
  totalMs,
  onSelect,
  onToggleSpan,
}: RowProps & { row: SpanRowData }) {
  const { span } = row;
  const failed = span.status === "error";
  const tokens = span.input_tokens + span.output_tokens;
  const isLlm = span.type === "llm";
  const duration = <span className={cn(META, "shrink-0")}>{fmtMs(span.duration_ms)}</span>;
  const caret = row.hasChildren && (
    <Caret open={!row.collapsed} label={row.collapsed ? "Expand" : "Collapse"} onToggle={() => onToggleSpan(row.id)} />
  );
  const leafDuration = !row.hasChildren && !isLlm && <span className="flex min-h-5 items-center">{duration}</span>;
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
            <span
              className={cn(span.type === "tool" ? MONO_NAME : NAME, failed ? "text-destructive" : "text-trace-text")}
            >
              {span.name}
            </span>
            {isLlm && span.model && (
              <span className="h-4 max-w-[150px] shrink-0 truncate rounded-[3px] bg-trace-chip px-1 font-mono text-[11px] leading-4 text-trace-text-secondary">
                {span.model.split("/").pop()}
              </span>
            )}
            {(row.hasChildren || isLlm) && duration}
            {isLlm && tokens > 0 && <span className={META}>{fmtTok(tokens)} tok</span>}
          </span>
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
            <span
              className={cn(
                row.type === "tool" ? MONO_NAME : NAME,
                row.isFailureGroup ? "text-destructive" : "text-trace-text",
              )}
            >
              {row.name}
            </span>
            <span className={META}>×{row.members.length}</span>
            <span className={META}>p50 {fmtMs(row.p50Duration)}</span>
            {row.failedCount > 0 && <span className={cn(META, "text-destructive")}>{row.failedCount} failed</span>}
          </span>
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
  embedded = false,
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
      <div className="mt-3.5 mr-2 mb-2 ml-4 flex h-8 shrink-0 items-center gap-1">
        <span className="text-[13px] leading-[15.6px] font-semibold tracking-[-0.26px] text-trace-key">Spans</span>
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
        <div role="tree" aria-label="Spans in time order">
          {rows.map((row, i) => {
            if (row.kind === "load-more") {
              return (
                <button
                  key={row.id}
                  type="button"
                  onClick={() => onLoadMore(row.groupId)}
                  className={cn(rowClass(false), "h-8 text-[13px] tracking-[-0.26px] text-trace-text-secondary")}
                >
                  <Gutters depth={row.depth} guide={guides[i]} />
                  <span className="ml-0.5 flex items-center gap-2">
                    <MoreHorizontal className="relative z-raised size-3.5" /> Load 20 more
                    <span className="text-trace-duration">({row.remaining} remaining)</span>
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
        {embedded ? (
          <>
            <span>
              <Kbd>↑</Kbd>/<Kbd>↓</Kbd> step
            </span>
            <span>
              <Kbd>J</Kbd>/<Kbd>K</Kbd> trace
            </span>
          </>
        ) : (
          <span>
            <Kbd>J</Kbd>/<Kbd>K</Kbd> move
          </span>
        )}
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
