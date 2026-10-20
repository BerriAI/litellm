"use client";

import { mergeProps } from "@base-ui/react/merge-props";
import { useRender } from "@base-ui/react/use-render";
import { ArrowLeft, ChevronDown, ChevronsRight, ChevronUp, Maximize2, Minimize2, X } from "lucide-react";
import {
  createContext,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { useLocalStorage, useMediaQuery, useOnClickOutside, useWindowSize } from "usehooks-ts";

import { ShortcutHints } from "@/components/shared/ShortcutHints";
import { useShortcut } from "@/components/shared/useShortcut";
import { cn } from "@/lib/cva.config";

const MIN_WIDTH = 700;
const MIN_LEFT_GAP = 100;
const DEFAULT_FRACTION = 0.56;
const KEY_STEP = 32;
const ROW_SLOT = "inspector-row";
const OVERLAYS = "[data-slot='sheet-overlay'], [data-slot='dialog-overlay'], [data-slot='alert-dialog-overlay']";
const KEEPS_PANEL_OPEN = `[data-slot='${ROW_SLOT}'], [role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper], ${OVERLAYS}`;

export const clampPanelWidth = (width: number, viewport: number): number => {
  const max = Math.max(Math.min(MIN_WIDTH, viewport), viewport - MIN_LEFT_GAP);
  return Math.round(Math.min(Math.max(width, Math.min(MIN_WIDTH, max)), max));
};

interface InspectorState<T> {
  readonly itemKey: (item: T) => string;
  readonly selected: T | null;
  readonly index: number;
  readonly total: number;
  readonly noun: string;
  readonly storageKey: string;
  readonly fullScreen: boolean;
  readonly toggle: (item: T) => void;
  readonly step: (delta: number) => void;
  readonly close: () => void;
  readonly setFullScreen: (fullScreen: boolean) => void;
  /** Called by an open Inspector nested inside this one; returns the release. */
  readonly claim: () => () => void;
}

const InspectorContext = createContext<unknown>(null);

export function useInspector<T>(): InspectorState<T> {
  const state = useContext(InspectorContext);
  if (state === null) throw new Error("Inspector parts must be rendered inside Inspector.Root");
  return state as InspectorState<T>;
}

export interface InspectorRootProps<T> {
  readonly items: readonly T[];
  readonly itemKey: (item: T) => string;
  readonly selected: T | null;
  readonly onSelectedChange: (item: T | null) => void;
  /** Names the item in button labels and the J/K hint, e.g. "trace". */
  readonly noun: string;
  /** localStorage key remembering the panel width for this surface. */
  readonly storageKey: string;
  readonly fullScreen?: boolean;
  readonly defaultFullScreen?: boolean;
  readonly onFullScreenChange?: (fullScreen: boolean) => void;
  readonly children: ReactNode;
}

/** Owns which item is open: rows toggle it, J/K step through `items`, Escape closes, the panel shows it. */
function Root<T>({
  items,
  itemKey,
  selected,
  onSelectedChange,
  noun,
  storageKey,
  fullScreen: controlledFullScreen,
  defaultFullScreen = false,
  onFullScreenChange,
  children,
}: InspectorRootProps<T>) {
  const parent = useContext(InspectorContext) as InspectorState<never> | null;
  const [claims, setClaims] = useState(0);
  const claim = useCallback(() => {
    setClaims((count) => count + 1);
    return () => setClaims((count) => count - 1);
  }, []);
  const [ownFullScreen, setOwnFullScreen] = useState(defaultFullScreen);
  const fullScreen = controlledFullScreen ?? ownFullScreen;
  const setFullScreen = (next: boolean) => {
    setOwnFullScreen(next);
    onFullScreenChange?.(next);
  };
  const selectedKey = selected === null ? null : itemKey(selected);
  const index = selectedKey === null ? -1 : items.findIndex((item) => itemKey(item) === selectedKey);
  const open = selected !== null;
  const parentClaim = parent?.claim;
  useEffect(() => (parentClaim && open ? parentClaim() : undefined), [parentClaim, open]);
  const keys = open && claims === 0;
  const step = (delta: number) => {
    const next = items[index + delta];
    if (next !== undefined) onSelectedChange(next);
  };
  const close = () => onSelectedChange(null);
  const toggle = (item: T) => onSelectedChange(itemKey(item) === selectedKey ? null : item);
  useShortcut("escape", close, { enabled: keys, description: "close" });
  useShortcut("j", () => step(1), { enabled: keys, description: noun });
  useShortcut("k", () => step(-1), { enabled: keys, description: noun });
  const state: InspectorState<T> = {
    itemKey,
    selected,
    index,
    total: items.length,
    noun,
    storageKey,
    fullScreen,
    toggle,
    step,
    close,
    setFullScreen,
    claim,
  };
  return <InspectorContext.Provider value={state}>{children}</InspectorContext.Provider>;
}

export type InspectorRowProps<T> = useRender.ComponentProps<"div"> & { readonly item: T };

const activates = (event: KeyboardEvent<HTMLElement>): boolean =>
  event.target === event.currentTarget && (event.key === "Enter" || event.key === " ");

/** A list entry that opens its item on click or Enter/Space and reads `data-state="selected"` while open. */
function Row<T>({ item, render, ...props }: InspectorRowProps<T>) {
  const { itemKey, selected, toggle } = useInspector<T>();
  const isSelected = selected !== null && itemKey(selected) === itemKey(item);
  const own = {
    "aria-selected": isSelected,
    onClick: () => toggle(item),
    onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
      if (!activates(event)) return;
      event.preventDefault();
      toggle(item);
    },
  };
  const rendered = {
    defaultTagName: "div" as const,
    render,
    state: { slot: ROW_SLOT, state: isSelected ? "selected" : "idle" },
    props: mergeProps<"div">(own, props),
  };
  return useRender(rendered);
}

function usePanelWidth(storageKey: string) {
  const { width: viewport } = useWindowSize();
  const [stored, setStored] = useLocalStorage<number | null>(storageKey, null, {
    serializer: String,
    deserializer: (raw) => (Number.isFinite(Number(raw)) ? Number(raw) : null),
  });
  const width = clampPanelWidth(stored ?? viewport * DEFAULT_FRACTION, viewport);
  const update = useCallback((next: number) => setStored(clampPanelWidth(next, window.innerWidth)), [setStored]);
  return [width, update] as const;
}

/** Keeps the last item on screen through the exit animation, then lets the panel unmount. */
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

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging) onResize(window.innerWidth - event.clientX);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
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

function Header() {
  const { index, total, noun, step, close, fullScreen, setFullScreen } = useInspector();
  return (
    <div data-slot="inspector-header" className="flex h-[37px] shrink-0 items-center gap-1 border-b border-border px-2">
      <HeaderButton label="Close (Esc)" onClick={close}>
        <ChevronsRight className="size-4" />
      </HeaderButton>
      <span className="mx-1 h-4 w-px bg-border" />
      <HeaderButton label={`Next ${noun} (J)`} disabled={index < 0 || index >= total - 1} onClick={() => step(1)}>
        <ChevronDown className="size-4" />
      </HeaderButton>
      <HeaderButton label={`Previous ${noun} (K)`} disabled={index <= 0} onClick={() => step(-1)}>
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
          onClick={() => setFullScreen(!fullScreen)}
        >
          {fullScreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
        </HeaderButton>
        <HeaderButton label={`Close ${noun} (Esc)`} onClick={close}>
          <X className="size-4" />
        </HeaderButton>
      </div>
    </div>
  );
}

export interface InspectorPanelProps<T> {
  readonly label: string;
  readonly testId?: string;
  readonly children: (item: T) => ReactNode;
}

/** Resizable right-hand panel showing the open item; the list stays clickable beside it. */
function Panel<T>({ label, testId, children }: InspectorPanelProps<T>) {
  const { selected, itemKey, noun, storageKey, fullScreen, close } = useInspector<T>();
  const [width, setWidth] = usePanelWidth(storageKey);
  const panelRef = useRef<HTMLElement>(null);
  const { shown, closing, onExited } = useExitPresence(selected, itemKey);

  useOnClickOutside(panelRef as RefObject<HTMLElement>, (event) => {
    if (selected === null || (event instanceof MouseEvent && event.button !== 0)) return;
    if (event.target instanceof Element && event.target.closest(KEEPS_PANEL_OPEN) !== null) return;
    close();
  });

  if (shown === null) return null;
  return (
    <aside
      ref={panelRef}
      aria-label={label}
      data-slot="inspector-panel"
      data-state={closing ? "closing" : "open"}
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
      <Header />
      <div className="flex min-h-0 flex-1 flex-col">{children(shown)}</div>
      <ShortcutHints className="min-h-10 shrink-0 border-t border-border px-3 py-1.5" />
    </aside>
  );
}

/** Returns from content stacked inside the panel, such as evidence opened over a finding. */
function BackLink({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      data-slot="inspector-back"
      className="-ml-1 inline-flex max-w-full items-center gap-1 rounded px-1 text-xs text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft aria-hidden="true" className="size-3 shrink-0" />
      <span className="truncate">{label}</span>
    </button>
  );
}

export const Inspector = { Root, Row, Panel, BackLink } as const;
