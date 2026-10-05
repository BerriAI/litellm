"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentProps,
  type RefObject,
} from "react";
import { useMediaQuery, useResizeObserver } from "usehooks-ts";

import { cn } from "@/lib/cva.config";

import {
  DOT_COLS,
  DOT_PITCH,
  FAILED_COLOR,
  FIELD_HEIGHT,
  columnDots,
  type Dot,
  type DotBand,
  type DotColumn,
} from "./dots";

interface DotFieldState {
  columns: readonly DotColumn[];
  max: number;
  band: DotBand | null;
  hover: number | null;
}

const DotFieldContext = createContext<DotFieldState | null>(null);

function useDotField(): DotFieldState {
  const state = useContext(DotFieldContext);
  if (!state) throw new Error("DotField parts must be rendered inside DotFieldRoot");
  return state;
}

export type DotFieldRootProps = ComponentProps<"div"> & {
  columns: readonly DotColumn[];
  band?: DotBand | null;
  hover?: number | null;
};

/** Positioned box sized to the dot grid. Overlays (cursors, brackets, hit areas) compose as children. */
export function DotFieldRoot({ columns, band = null, hover = null, className, style, ...props }: DotFieldRootProps) {
  const state = useMemo<DotFieldState>(
    () => ({ columns, max: Math.max(1, ...columns.map((column) => column.total)), band, hover }),
    [columns, band, hover],
  );
  return (
    <DotFieldContext.Provider value={state}>
      <div
        data-slot="dot-field"
        className={cn("relative", className)}
        style={{ height: FIELD_HEIGHT, ...style }}
        {...props}
      />
    </DotFieldContext.Provider>
  );
}

function dotColor(lit: boolean, dot: Dot, grid: string): string {
  if (!lit) return grid;
  if (dot.kind === "failed") return FAILED_COLOR;
  return dot.kind === "series" ? dot.color : grid;
}

function dotRadius(lit: boolean, hovered: boolean): number {
  if (!lit) return 0.9;
  return hovered ? 2.4 : 1.9;
}

function drawDots(canvas: HTMLCanvasElement, { columns, max, band, hover }: DotFieldState, progress: number) {
  const context = canvas.getContext("2d");
  const width = canvas.clientWidth;
  if (!context || width === 0) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(FIELD_HEIGHT * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, FIELD_HEIGHT);
  const grid = document.documentElement.classList.contains("dark") ? "rgba(255,255,255,0.08)" : "rgba(15,23,42,0.08)";
  const columnWidth = width / columns.length;
  const pitch = columnWidth / DOT_COLS;
  columns.forEach((column, i) => {
    const dots = columnDots(column, max, i * 7919 + column.total);
    const visible = Math.ceil(dots.length * progress);
    const dimmed = band !== null && (i < band.lo || i > band.hi);
    dots.forEach((dot, index) => {
      const lit = dot.kind !== "grid" && index < visible;
      context.globalAlpha = lit ? (dimmed ? 0.15 : 1) * (column.opacity ?? 1) : 1;
      context.fillStyle = dotColor(lit, dot, grid);
      context.beginPath();
      context.arc(
        i * columnWidth + pitch * ((index % DOT_COLS) + 0.5),
        FIELD_HEIGHT - DOT_PITCH * (Math.floor(index / DOT_COLS) + 0.5),
        dotRadius(lit, hover === i),
        0,
        Math.PI * 2,
      );
      context.fill();
    });
  });
  context.globalAlpha = 1;
}

function useRiseIn(): number {
  const reduceMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    if (reduceMotion) {
      setProgress(1);
      return;
    }
    const start = performance.now();
    let frame = requestAnimationFrame(function rise(now) {
      const t = Math.min(1, (now - start) / 700);
      setProgress(1 - (1 - t) ** 3);
      if (t < 1) frame = requestAnimationFrame(rise);
    });
    return () => cancelAnimationFrame(frame);
  }, [reduceMotion]);
  return progress;
}

export type DotFieldCanvasProps = ComponentProps<"canvas">;

/** The dots themselves, painted on a canvas pinned to the bottom of the root. */
export function DotFieldCanvas({ className, style, ...props }: DotFieldCanvasProps) {
  const state = useDotField();
  const canvas = useRef<HTMLCanvasElement>(null);
  const progress = useRiseIn();
  const { width, height } = useResizeObserver({ ref: canvas as RefObject<HTMLCanvasElement> });
  useEffect(() => {
    if (canvas.current) drawDots(canvas.current, state, progress);
  }, [state, progress, width, height]);
  return (
    <canvas
      ref={canvas}
      data-slot="dot-field-canvas"
      aria-hidden="true"
      className={cn("pointer-events-none absolute inset-x-0 bottom-0 w-full", className)}
      style={{ height: FIELD_HEIGHT, ...style }}
      {...props}
    />
  );
}
