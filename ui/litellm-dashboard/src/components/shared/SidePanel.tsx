"use client";

import { ArrowLeft, ChevronDown, ChevronsRight, ChevronUp, Maximize2, Minimize2, X } from "lucide-react";
import { useCallback, useRef, useState, type ReactNode, type RefObject } from "react";
import { useEventListener, useLocalStorage, useMediaQuery, useOnClickOutside, useWindowSize } from "usehooks-ts";

import { cn } from "@/lib/cva.config";
import { ignoresLetterShortcut } from "@/components/view_logs/letterShortcut";

const WIDTH_KEY = "litellm.agentTraces.drawerWidth";
const MIN_WIDTH = 700;
const MIN_LEFT_GAP = 100;
const DEFAULT_FRACTION = 0.56;
const KEY_STEP = 32;

export const clampDrawerWidth = (width: number, viewport: number): number => {
  const max = Math.max(Math.min(MIN_WIDTH, viewport), viewport - MIN_LEFT_GAP);
  return Math.round(Math.min(Math.max(width, Math.min(MIN_WIDTH, max)), max));
};

export const PANEL_TRIGGER = { "data-side-panel-trigger": "" } as const;

const KEEPS_PANEL_OPEN =
  "[data-side-panel-trigger], [role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper]";

const OWNS_ITS_KEYS = "[role='dialog'], [role='menu'], [role='listbox']";

const insideKeyOwner = (event: KeyboardEvent): boolean =>
  event.target instanceof Element && event.target.closest(OWNS_ITS_KEYS) !== null;

function useDrawerWidth() {
  const { width: viewport } = useWindowSize();
  const [stored, setStored] = useLocalStorage<number | null>(WIDTH_KEY, null, {
    serializer: String,
    deserializer: (raw) => (Number.isFinite(Number(raw)) ? Number(raw) : null),
  });
  const width = clampDrawerWidth(stored ?? viewport * DEFAULT_FRACTION, viewport);
  const update = useCallback((next: number) => setStored(clampDrawerWidth(next, window.innerWidth)), [setStored]);
  return [width, update] as const;
}

function useExitPresence<T>(item: T | null, itemKey: (item: T) => string) {
  const reduceMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const [lastShown, setLastShown] = useState<T | null>(item);
  const [exitedKey, setExitedKey] = useState<string | null>(null);
  if (item !== null && item !== lastShown) setLastShown(item);
  if (item !== null && exitedKey !== null) setExitedKey(null);
  const shown = item ?? lastShown;
  const closing = item === null && shown !== null;
  const exited = closing && (reduceMotion || exitedKey === itemKey(shown));
  const onExited = () => {
    if (closing) setExitedKey(itemKey(shown));
  };
  return { shown: exited ? null : shown, closing, onExited };
}

function ResizeHandle({ noun, width, onResize }: { noun: string; width: number; onResize: (width: number) => void }) {
  const [dragging, setDragging] = useState(false);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };
  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragging) onResize(window.innerWidth - event.clientX);
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
      aria-label={`Resize ${noun} panel`}
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
          "h-full w-[0.67px] bg-border transition-[width,background-color] duration-150 group-hover/handle:w-0.5 group-focus-visible/handle:w-0.5 group-focus-visible/handle:bg-trace-brand motion-reduce:transition-none",
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
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="grid size-7 place-items-center rounded-[4px] text-muted-foreground transition-colors duration-150 hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-40 motion-reduce:transition-none"
    >
      {children}
    </button>
  );
}

export interface SidePanelProps<T> {
  item: T | null;
  itemKey: (item: T) => string;
  noun: string;
  label: string;
  testId?: string;
  index: number;
  total: number;
  onStep: (delta: number) => void;
  onClose: () => void;
  fullScreen: boolean;
  onFullScreenChange: (fullScreen: boolean) => void;
  children: (item: T) => ReactNode;
}

export function SidePanel<T>({
  item,
  itemKey,
  noun,
  label,
  testId,
  index,
  total,
  onStep,
  onClose,
  fullScreen,
  onFullScreenChange,
  children,
}: SidePanelProps<T>) {
  const [width, setWidth] = useDrawerWidth();
  const panelRef = useRef<HTMLElement>(null);
  const { shown, closing, onExited } = useExitPresence(item, itemKey);

  useEventListener("keydown", (event) => {
    if (item === null || ignoresLetterShortcut(event) || insideKeyOwner(event)) return;
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    } else if (event.key === "j") {
      event.preventDefault();
      onStep(1);
    } else if (event.key === "k") {
      event.preventDefault();
      onStep(-1);
    }
  });

  useOnClickOutside(panelRef as RefObject<HTMLElement>, (event) => {
    if (item === null || (event instanceof MouseEvent && event.button !== 0)) return;
    if (event.target instanceof Element && event.target.closest(KEEPS_PANEL_OPEN) !== null) return;
    onClose();
  });

  if (shown === null) return null;
  return (
    <aside
      ref={panelRef}
      aria-label={label}
      data-testid={testId}
      style={{ width: fullScreen ? "100%" : width }}
      onAnimationEnd={(event) => {
        if (event.target === event.currentTarget) onExited();
      }}
      className={cn(
        "fixed inset-y-0 right-0 z-overlay flex origin-right flex-col bg-background shadow-[0_10px_15px_-3px_rgba(16,24,40,0.1),0_4px_6px_-4px_rgba(16,24,40,0.1)] motion-reduce:animate-none",
        closing ? "animate-trace-drawer-out" : "animate-trace-drawer-in",
      )}
    >
      {!fullScreen && <ResizeHandle noun={noun} width={width} onResize={setWidth} />}
      <div className="flex h-[37px] shrink-0 items-center gap-1 border-b border-border px-2">
        <HeaderButton label="Close (Esc)" onClick={onClose}>
          <ChevronsRight className="size-4" />
        </HeaderButton>
        <span className="mx-1 h-4 w-px bg-border" />
        <HeaderButton label={`Next ${noun} (J)`} disabled={index < 0 || index >= total - 1} onClick={() => onStep(1)}>
          <ChevronDown className="size-4" />
        </HeaderButton>
        <HeaderButton label={`Previous ${noun} (K)`} disabled={index <= 0} onClick={() => onStep(-1)}>
          <ChevronUp className="size-4" />
        </HeaderButton>
        {index >= 0 && (
          <span className="ml-1 font-mono text-[11px] text-muted-foreground tabular-nums">
            {index + 1} / {total}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <HeaderButton
            label={fullScreen ? "Exit full screen" : "Enter full screen"}
            onClick={() => onFullScreenChange(!fullScreen)}
          >
            {fullScreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
          </HeaderButton>
          <HeaderButton label={`Close ${noun} (Esc)`} onClick={onClose}>
            <X className="size-4" />
          </HeaderButton>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col">{children(shown)}</div>
    </aside>
  );
}

export function PanelBackLink({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="-ml-1 inline-flex max-w-full items-center gap-1 rounded px-1 text-xs text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft aria-hidden="true" className="size-3 shrink-0" />
      <span className="truncate">{label}</span>
    </button>
  );
}
