"use client";

import { ChevronDown, ChevronsRight, ChevronUp } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { cn } from "@/lib/cva.config";

import { RunView } from "./TraceDrawer";
import type { TraceSummary } from "./traceTypes";

const WIDTH_KEY = "litellm.agentTraces.drawerWidth";
const MIN_WIDTH = 700;
const MIN_LEFT_GAP = 100;
const DEFAULT_FRACTION = 0.56;
const KEY_STEP = 32;

const viewportWidth = (): number => (typeof window === "undefined" ? 1440 : window.innerWidth);

/** At least 700px when the screen allows, never wider than the screen minus a 100px strip of list (or the full screen). */
export const clampDrawerWidth = (width: number, viewport: number): number => {
  const max = Math.max(Math.min(MIN_WIDTH, viewport), viewport - MIN_LEFT_GAP);
  return Math.round(Math.min(Math.max(width, Math.min(MIN_WIDTH, max)), max));
};

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

const readStoredWidth = (): number | null => {
  try {
    const raw = window.localStorage.getItem(WIDTH_KEY);
    const parsed = raw === null ? Number.NaN : Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
  } catch {
    return null;
  }
};

const storeWidth = (width: number): void => {
  try {
    window.localStorage.setItem(WIDTH_KEY, String(width));
  } catch {
    return;
  }
};

const isTypingTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement &&
  target.matches("input, textarea, select, [contenteditable='true'], [role='combobox']");

function useDrawerWidth() {
  const [width, setWidth] = useState(() =>
    clampDrawerWidth(readStoredWidth() ?? viewportWidth() * DEFAULT_FRACTION, viewportWidth()),
  );
  const update = useCallback((next: number) => {
    const clamped = clampDrawerWidth(next, viewportWidth());
    setWidth(clamped);
    storeWidth(clamped);
  }, []);
  useEffect(() => {
    const onResize = () => setWidth((w) => clampDrawerWidth(w, viewportWidth()));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return [width, update] as const;
}

function ResizeHandle({ width, onResize }: { width: number; onResize: (width: number) => void }) {
  const [dragging, setDragging] = useState(false);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };
  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragging) onResize(viewportWidth() - event.clientX);
  };
  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft") onResize(width + KEY_STEP);
    else if (event.key === "ArrowRight") onResize(width - KEY_STEP);
    else return;
    event.preventDefault();
    event.stopPropagation();
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize trace panel"
      aria-valuenow={width}
      aria-valuemin={MIN_WIDTH}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
      onKeyDown={onKeyDown}
      className="group/handle absolute inset-y-0 -left-2.5 z-raised flex w-5 cursor-col-resize justify-center outline-none"
    >
      <span
        className={cn(
          "h-full w-[0.67px] bg-trace-line transition-[width,background-color] duration-150 group-hover/handle:w-0.5 group-focus-visible/handle:w-0.5 group-focus-visible/handle:bg-trace-brand motion-reduce:transition-none",
          dragging && "w-0.5 bg-trace-brand",
        )}
      />
    </div>
  );
}

function HeaderButton({
  label,
  disabled = false,
  onClick,
  children,
}: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="grid size-7 place-items-center rounded-[4px] text-trace-key transition-colors duration-150 hover:bg-trace-row-hover hover:text-trace-text disabled:pointer-events-none disabled:opacity-40 motion-reduce:transition-none"
    >
      {children}
    </button>
  );
}

interface RunDrawerProps {
  trace: TraceSummary | null;
  runs: readonly TraceSummary[];
  accessToken: string;
  onSelect: (trace: TraceSummary | null) => void;
}

const runKey = (run: TraceSummary): string => run.trace_ref || run.trace_id;

/** Right-side drawer over the runs list: resizable, keeps the list clickable, swaps runs in place. */
export function RunDrawer({ trace, runs, accessToken, onSelect }: RunDrawerProps) {
  const [width, setWidth] = useDrawerWidth();
  const [lastShown, setLastShown] = useState<TraceSummary | null>(trace);
  const [exitedKey, setExitedKey] = useState<string | null>(null);
  if (trace !== null && trace !== lastShown) setLastShown(trace);
  const shown = trace ?? lastShown;
  const closing = trace === null && shown !== null;
  if (trace !== null && exitedKey !== null) setExitedKey(null);
  if (closing && exitedKey !== runKey(shown) && prefersReducedMotion()) setExitedKey(runKey(shown));

  const index = trace === null ? -1 : runs.findIndex((run) => runKey(run) === runKey(trace));
  const step = useCallback(
    (delta: number) => {
      const next = runs[index + delta];
      if (next) onSelect(next);
    },
    [runs, index, onSelect],
  );

  useEffect(() => {
    if (trace === null) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const modified = event.metaKey || event.ctrlKey || event.altKey;
      if (modified || isTypingTarget(event.target)) return;
      if (event.key === "Escape") {
        event.preventDefault();
        onSelect(null);
      } else if (event.key === "j") {
        event.preventDefault();
        step(1);
      } else if (event.key === "k") {
        event.preventDefault();
        step(-1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [trace, step, onSelect]);

  if (shown === null || (closing && exitedKey === runKey(shown))) return null;
  return (
    <aside
      aria-label="Trace details"
      data-testid="run-drawer"
      style={{ width }}
      onAnimationEnd={(event) => {
        if (closing && event.target === event.currentTarget) setExitedKey(runKey(shown));
      }}
      className={cn(
        "fixed inset-y-0 right-0 z-overlay flex origin-right flex-col bg-trace-surface shadow-[0_10px_15px_-3px_rgba(16,24,40,0.1),0_4px_6px_-4px_rgba(16,24,40,0.1)] motion-reduce:animate-none",
        closing ? "animate-trace-drawer-out" : "animate-trace-drawer-in",
      )}
    >
      <ResizeHandle width={width} onResize={setWidth} />
      <div className="flex h-[37px] shrink-0 items-center gap-1 border-b border-trace-line px-2">
        <HeaderButton label="Close (Esc)" onClick={() => onSelect(null)}>
          <ChevronsRight className="size-4" />
        </HeaderButton>
        <span className="mx-1 h-4 w-px bg-trace-line" />
        <HeaderButton label="Next trace (J)" disabled={index < 0 || index >= runs.length - 1} onClick={() => step(1)}>
          <ChevronDown className="size-4" />
        </HeaderButton>
        <HeaderButton label="Previous trace (K)" disabled={index <= 0} onClick={() => step(-1)}>
          <ChevronUp className="size-4" />
        </HeaderButton>
        {index >= 0 && (
          <span className="ml-1 font-mono text-[11px] text-trace-key tabular-nums">
            {index + 1} / {runs.length}
          </span>
        )}
      </div>
      <div className="flex min-h-0 flex-1 flex-col">
        <RunView
          key={runKey(shown)}
          traceId={shown.trace_id}
          traceRef={shown.trace_ref}
          accessToken={accessToken}
          onBack={() => onSelect(null)}
          embedded
        />
      </div>
    </aside>
  );
}
