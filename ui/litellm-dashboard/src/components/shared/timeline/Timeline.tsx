"use client";

import { X } from "lucide-react";
import moment from "moment";
import { useRef, useState, type RefObject } from "react";
import { useResizeObserver } from "usehooks-ts";

import type { DotBand, DotColumn } from "@/components/shared/dotField/dots";
import { cn } from "@/lib/cva.config";

import type { TimeWindow } from "../timeRange/timeRange";

export const TIMELINE_BUCKETS = 60;
const TICKS = 6;
const MINUTE_MS = 60 * 1000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;
const EDGE_FORMAT = "MMM DD, HH:mm";

export interface TimeBucket extends DotColumn {
  startMs: number;
  endMs: number;
}

export type ItemNoun = { readonly singular: string; readonly plural: string };

/** Compact window length, Logfire-style: "45m", "6h 12m", "7d", "152d 23h". */
export function formatSpan(ms: number): string {
  const totalMinutes = Math.max(0, Math.round(ms / MINUTE_MS));
  const days = Math.floor(totalMinutes / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const minutes = totalMinutes % 60;
  if (days > 0) return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
  if (hours > 0) return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  return `${minutes}m`;
}

const tickFormat = (range: TimeWindow): string => (range.endMs - range.startMs > 2 * DAY_MS ? EDGE_FORMAT : "HH:mm");

const pct = (value: number): string => `${value * 100}%`;

/** Keep the first / last tick label inside the strip; center the rest on their tick. */
const tickShift = (t: number): string => {
  if (t === 0) return "translateX(0)";
  if (t === 1) return "translateX(-100%)";
  return "translateX(-50%)";
};

interface TimelineProps {
  buckets: readonly TimeBucket[];
  selection: TimeWindow | null;
  onSelect: (selection: TimeWindow | null) => void;
  /** What one counted item is called in the hover tooltip. */
  noun: ItemNoun;
}

function BucketBar({ bucket, max, dimmed }: { bucket: TimeBucket; max: number; dimmed: boolean }) {
  const ok = bucket.total - bucket.failed;
  return (
    <div
      className={cn(
        "pointer-events-none flex h-full flex-1 flex-col justify-end px-[2px] transition-opacity",
        dimmed && "opacity-25",
      )}
      data-testid="timeline-bucket"
      data-total={bucket.total}
    >
      {bucket.failed > 0 && (
        <div className="rounded-t-[2px] bg-[#ef4444]" style={{ height: pct(bucket.failed / max) }} />
      )}
      {ok > 0 && (
        <div
          className={cn("bg-[#2b3fd6]", bucket.failed === 0 && "rounded-t-[2px]")}
          style={{ height: pct(ok / max) }}
        />
      )}
    </div>
  );
}

function BucketTooltip({
  bucket,
  index,
  bucketCount,
  noun,
}: {
  bucket: TimeBucket;
  index: number;
  bucketCount: number;
  noun: ItemNoun;
}) {
  return (
    <div
      className="pointer-events-none absolute top-full z-floating mt-1 rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground tabular-nums shadow-md"
      style={{ left: pct(Math.min(0.8, index / bucketCount)) }}
      role="tooltip"
    >
      <div>
        {moment(bucket.startMs).format(EDGE_FORMAT)} to {moment(bucket.endMs).format("HH:mm")}
      </div>
      <div>
        {bucket.total} {bucket.total === 1 ? noun.singular : noun.plural}
        {bucket.failed > 0 && `, ${bucket.failed} failed`}
      </div>
      <div className="text-muted-foreground">Drag to zoom</div>
    </div>
  );
}

const MIN_DURATION_LABEL_PX = 120;

export type Band = DotBand;

export type DragMode = "select" | "move" | "resize-lo" | "resize-hi";

export interface DragState {
  mode: DragMode;
  origin: number;
  band: Band;
}

/** The band a drag produces when the pointer is over bucket `at`; always within [0, buckets - 1]. */
export function dragUpdate(drag: DragState, at: number, buckets = TIMELINE_BUCKETS): Band {
  const last = buckets - 1;
  const clamp = (i: number) => Math.min(last, Math.max(0, i));
  const { band, origin } = drag;
  if (drag.mode === "select") return { lo: Math.min(origin, clamp(at)), hi: Math.max(origin, clamp(at)) };
  if (drag.mode === "resize-lo") return { lo: Math.min(clamp(at), band.hi), hi: band.hi };
  if (drag.mode === "resize-hi") return { lo: band.lo, hi: Math.max(clamp(at), band.lo) };
  const width = band.hi - band.lo;
  const lo = Math.min(last - width, Math.max(0, band.lo + (at - origin)));
  return { lo, hi: lo + width };
}

/** Bucket band covered by a selected window, or null when nothing is selected. */
export function bandForWindow(buckets: readonly TimeBucket[], selection: TimeWindow | null): Band | null {
  if (selection === null) return null;
  const inside = buckets.flatMap((b, i) => (b.startMs >= selection.startMs && b.endMs <= selection.endMs ? [i] : []));
  return inside.length > 0 ? { lo: inside[0], hi: inside[inside.length - 1] } : null;
}

const edgeFormat = (range: TimeWindow): string => (range.endMs - range.startMs < DAY_MS ? "HH:mm" : EDGE_FORMAT);

/** The selected window drawn as a flat bracket: resize handles on each edge, times outside, span centered. */
function SelectionBracket({
  band,
  window,
  format,
  stripWidth,
  bucketCount,
  onHandleDown,
}: {
  band: Band;
  window: TimeWindow;
  format: string;
  stripWidth: number;
  bucketCount: number;
  onHandleDown: (mode: DragMode) => (e: React.PointerEvent) => void;
}) {
  const leftFrac = band.lo / bucketCount;
  const rightFrac = (band.hi + 1) / bucketCount;
  const widthPx = (rightFrac - leftFrac) * stripWidth;
  const handle = "absolute inset-y-0 w-[2px] cursor-ew-resize bg-foreground";
  const tick =
    "before:absolute before:top-0 before:h-[2px] before:w-2 before:bg-inherit after:absolute after:bottom-0 after:h-[2px] after:w-2 after:bg-inherit";
  const edgeLabel = "pointer-events-none absolute -bottom-5 text-xs whitespace-nowrap text-foreground tabular-nums";
  return (
    <>
      <div
        className="absolute inset-y-0 cursor-grab bg-foreground/[0.04] active:cursor-grabbing"
        style={{ left: pct(leftFrac), width: pct(rightFrac - leftFrac) }}
        data-testid="timeline-selection"
        onPointerDown={onHandleDown("move")}
      >
        <span
          className={cn(handle, tick, "-left-px before:left-0 after:left-0")}
          data-testid="timeline-handle-lo"
          onPointerDown={onHandleDown("resize-lo")}
        />
        <span
          className={cn(handle, tick, "-right-px before:right-0 after:right-0")}
          data-testid="timeline-handle-hi"
          onPointerDown={onHandleDown("resize-hi")}
        />
        {widthPx >= MIN_DURATION_LABEL_PX && (
          <span className="pointer-events-none absolute inset-x-0 top-1 text-center text-xs font-medium text-foreground tabular-nums">
            {formatSpan(window.endMs - window.startMs)}
          </span>
        )}
      </div>
      <span className={cn(edgeLabel, leftFrac < 0.12 ? "" : "-translate-x-full pr-1")} style={{ left: pct(leftFrac) }}>
        {moment(window.startMs).format(format)}
      </span>
      <span className={cn(edgeLabel, rightFrac > 0.88 ? "-translate-x-full" : "pl-1")} style={{ left: pct(rightFrac) }}>
        {moment(window.endMs).format(format)}
      </span>
    </>
  );
}

export function timelineTicks(range: TimeWindow, width: number): number[] {
  const labelWidth = tickFormat(range) === EDGE_FORMAT ? 140 : 80;
  const count = Math.max(2, Math.min(TICKS, Math.floor(width / labelWidth)));
  return Array.from({ length: count }, (_, i) => i / (count - 1));
}

function TickAxis({ range, width }: { range: TimeWindow; width: number }) {
  const format = tickFormat(range);
  const ticks = timelineTicks(range, width);
  return (
    <div className="relative mt-0.5 h-5">
      {ticks.map((t) => (
        <div
          key={t}
          className={cn(
            "absolute top-0 flex flex-col",
            t === 0 && "items-start",
            t === 1 && "items-end",
            t > 0 && t < 1 && "items-center",
          )}
          style={{ left: pct(t), transform: tickShift(t) }}
        >
          <span className="h-1 w-px bg-border" />
          <span className="text-xs whitespace-nowrap text-muted-foreground tabular-nums">
            {moment(range.startMs + (range.endMs - range.startMs) * t).format(format)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Histogram over the window covered by `buckets`. Drag to select; drag the bracket or its edges to adjust; Esc clears. */
export function Timeline({ buckets, selection, onSelect, noun }: TimelineProps) {
  const bucketCount = buckets.length;
  const [hover, setHover] = useState<number | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [draft, setDraft] = useState<Band | null>(null);
  const areaRef = useRef<HTMLDivElement>(null);
  const { width: stripWidth = 0 } = useResizeObserver({
    ref: areaRef as RefObject<HTMLDivElement>,
    box: "border-box",
  });

  const committed = bandForWindow(buckets, selection);
  const band = drag ? draft : committed;
  const windowOf = (b: Band): TimeWindow => ({ startMs: buckets[b.lo].startMs, endMs: buckets[b.hi].endMs });

  const indexAt = (clientX: number): number => {
    const rect = areaRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return 0;
    return Math.min(bucketCount - 1, Math.max(0, Math.floor(((clientX - rect.left) / rect.width) * bucketCount)));
  };
  const begin = (mode: DragMode, e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    areaRef.current?.setPointerCapture?.(e.pointerId);
    const at = indexAt(e.clientX);
    const start = mode === "select" || committed === null ? { lo: at, hi: at } : committed;
    setDrag({ mode: committed === null ? "select" : mode, origin: at, band: start });
    setDraft(start);
  };
  const onHandleDown = (mode: DragMode) => (e: React.PointerEvent) => begin(mode, e);
  const onMove = (e: React.PointerEvent) => {
    const at = indexAt(e.clientX);
    setHover(at);
    if (drag) setDraft(dragUpdate(drag, at, bucketCount));
  };
  const onUp = (e: React.PointerEvent) => {
    areaRef.current?.releasePointerCapture?.(e.pointerId);
    if (!drag || !draft) return;
    const finished = draft;
    const wasSelect = drag.mode === "select";
    setDrag(null);
    setDraft(null);
    if (wasSelect && finished.lo === finished.hi) {
      onSelect(null);
      return;
    }
    onSelect(windowOf(finished));
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "Escape" || selection === null) return;
    e.preventDefault();
    onSelect(null);
  };

  if (bucketCount === 0) return null;

  const range: TimeWindow = { startMs: buckets[0].startMs, endMs: buckets[bucketCount - 1].endMs };
  const max = Math.max(1, ...buckets.map((b) => b.total));
  const labelFormat = edgeFormat(range);

  return (
    <div
      className="relative shrink-0 border-b border-border px-4 pt-3 pb-1 outline-none select-none"
      data-testid="timeline"
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      <div
        ref={areaRef}
        role="presentation"
        data-testid="timeline-area"
        className="relative flex h-12 cursor-crosshair touch-none items-end border-b border-border"
        onPointerDown={(e) => begin("select", e)}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => onSelect(null)}
      >
        {hover !== null && !drag && (
          <div
            className="pointer-events-none absolute inset-y-0 w-px bg-foreground/30"
            style={{ left: pct((hover + 0.5) / bucketCount) }}
            data-testid="timeline-cursor"
          />
        )}
        {buckets.map((b, i) => (
          <BucketBar key={b.startMs} bucket={b} max={max} dimmed={band !== null && (i < band.lo || i > band.hi)} />
        ))}
        {band && (
          <SelectionBracket
            band={band}
            window={windowOf(band)}
            format={labelFormat}
            stripWidth={stripWidth}
            bucketCount={bucketCount}
            onHandleDown={onHandleDown}
          />
        )}
      </div>
      <div className="mt-1">
        <TickAxis range={range} width={stripWidth} />
      </div>
      {hover !== null && !drag && (
        <BucketTooltip bucket={buckets[hover]} index={hover} bucketCount={bucketCount} noun={noun} />
      )}
      {selection && (
        <button
          type="button"
          onClick={() => onSelect(null)}
          aria-label="Clear time zoom"
          className="absolute top-1 right-3 inline-flex size-5 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <X className="size-3" />
        </button>
      )}
    </div>
  );
}
