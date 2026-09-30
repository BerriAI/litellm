"use client";

import { ChevronRight } from "lucide-react";
import { useEffect, useRef } from "react";

import { cn } from "@/lib/cva.config";

import { fmtMs, type OutlineGlyph, type OutlineRow } from "./traceUtils";

interface TraceOutlineProps {
  rows: OutlineRow[];
  traceDurationMs: number;
  selectedId: string | null;
  isOpen: (row: OutlineRow) => boolean;
  onSelect: (row: OutlineRow) => void;
  onToggle: (row: OutlineRow) => void;
  onShowMore: (row: OutlineRow) => void;
}

const INDENT_PX = 16;
const BASE_PX = 16;

const GLYPH: Record<OutlineGlyph, string> = {
  input: "↳",
  output: "↲",
  agent: "◆",
  llm: "●",
  tool: "■",
  chain: "○",
  framework: "○",
};

function Glyph({ glyph }: { glyph: OutlineGlyph }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-flex w-3 shrink-0 justify-center text-muted-foreground/70",
        glyph === "tool" || glyph === "llm" ? "text-[7px]" : "text-[10px]",
        (glyph === "input" || glyph === "output") && "text-[12px]",
      )}
    >
      {GLYPH[glyph]}
    </span>
  );
}

const TIMELINE_PX = 72;

/** Where the step sat in the run: a thin grayscale bar in a fixed-width column (Datadog-style). */
function Timeline({ row, total, selected }: { row: OutlineRow; total: number; selected: boolean }) {
  const hasBar = row.span !== undefined && row.kind !== "input" && total > 0;
  const start = row.kind === "output" || !row.span ? 0 : row.span.start_offset_ms;
  const duration = row.kind === "output" ? total : (row.span?.duration_ms ?? 0);
  const left = Math.min(Math.max(start / total, 0), 1) * 100;
  const width = Math.min(Math.max((duration / total) * 100, 1.5), 100 - left);
  return (
    <span aria-hidden className="relative ml-3 hidden h-1 shrink-0 rounded-full bg-muted lg:block" style={{ width: TIMELINE_PX }}>
      {hasBar && (
        <span
          className={cn("absolute h-full rounded-full", selected ? "bg-foreground/50" : "bg-foreground/20")}
          style={{ left: `${left}%`, width: `${width}%` }}
        />
      )}
    </span>
  );
}

function IndentGuides({ depth }: { depth: number }) {
  return (
    <>
      {Array.from({ length: depth }, (_, i) => (
        <span
          key={i}
          aria-hidden
          className="absolute top-0 bottom-0 w-px bg-border/70"
          style={{ left: BASE_PX + 6 + i * INDENT_PX }}
        />
      ))}
    </>
  );
}

function RowMeta({ row }: { row: OutlineRow }) {
  return (
    <span className="ml-auto flex shrink-0 items-center gap-3 pl-3 text-xs tabular-nums text-muted-foreground">
      {row.meta && <span className="hidden sm:inline">{row.meta}</span>}
      {row.metaError && <span>{row.metaError}</span>}
      <span className="w-12 text-right">{row.durationMs != null ? fmtMs(row.durationMs) : ""}</span>
    </span>
  );
}

/** The run as one vertical list: Input, every step in time order, Output. */
export function TraceOutline({
  rows,
  traceDurationMs,
  selectedId,
  isOpen,
  onSelect,
  onToggle,
  onShowMore,
}: TraceOutlineProps) {
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!selectedId) return;
    const node = listRef.current?.querySelector(`[data-row-id="${CSS.escape(selectedId)}"]`);
    if (node instanceof HTMLElement && typeof node.scrollIntoView === "function") node.scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  return (
    <ul ref={listRef} role="tree" aria-label="Trace outline" className="min-h-0 flex-1 overflow-auto py-2 text-[13px]">
      {rows.map((row) => {
        const selected = row.id === selectedId;
        const open = isOpen(row);
        const paddingLeft = BASE_PX + row.depth * INDENT_PX;
        if (row.kind === "more") {
          return (
            <li key={row.id} data-row-id={row.id} className="relative">
              <IndentGuides depth={row.depth} />
              <button
                type="button"
                onClick={() => onShowMore(row)}
                className="h-8 w-full text-left text-xs text-muted-foreground hover:text-foreground"
                style={{ paddingLeft: paddingLeft + 16 }}
              >
                Show 20 more
              </button>
            </li>
          );
        }
        return (
          <li
            key={row.id}
            role="treeitem"
            aria-selected={selected}
            aria-expanded={row.hasChildren ? open : undefined}
            data-row-id={row.id}
            data-error={row.error || undefined}
            onClick={() => onSelect(row)}
            className={cn(
              "relative flex h-8 cursor-pointer items-center pr-4 transition-colors",
              selected ? "bg-muted font-medium" : "hover:bg-muted/50",
            )}
            style={{ paddingLeft }}
          >
            <IndentGuides depth={row.depth} />
            <span className="flex w-4 shrink-0 justify-center">
              {row.hasChildren && (
                <button
                  type="button"
                  aria-label={open ? "Collapse" : "Expand"}
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggle(row);
                  }}
                  className="flex size-4 items-center justify-center rounded-sm text-muted-foreground hover:text-foreground"
                >
                  <ChevronRight className={cn("size-3 transition-transform", open && "rotate-90")} />
                </button>
              )}
            </span>
            <Glyph glyph={row.glyph} />
            <span className="ml-2 flex min-w-0 flex-1 items-center gap-2">
              {row.error && (
                <span aria-label="failed" role="img" className="size-1.5 shrink-0 rounded-full bg-destructive" />
              )}
              <span className={cn("max-w-[80%] shrink-0 truncate", row.kind === "span" && row.span?.type === "tool" && "font-mono text-xs")}>
                {row.label}
              </span>
              {row.detail && (
                <span className="min-w-0 flex-1 truncate text-xs font-normal text-muted-foreground">{row.detail}</span>
              )}
            </span>
            <RowMeta row={row} />
            <Timeline row={row} total={traceDurationMs} selected={selected} />
          </li>
        );
      })}
    </ul>
  );
}
