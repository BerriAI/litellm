"use client";

import * as React from "react";
import { Maximize2Icon, Minimize2Icon } from "lucide-react";

import { cn } from "@/lib/cva.config";
import { Button } from "@/components/ui/button";
import { SheetContent } from "@/components/ui/sheet";
import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";

type ResizableSheetContentProps = Omit<React.ComponentProps<typeof SheetContent>, "side" | "style"> & {
  storageKey: string;
  defaultWidthPercent?: number;
  minWidthPercent?: number;
};

function clampWidth(value: number, min: number) {
  return Math.min(100, Math.max(min, value));
}

function readStoredWidth(storageKey: string, defaultWidthPercent: number, minWidthPercent: number) {
  const stored = Number(getLocalStorageItem(storageKey));
  if (!Number.isFinite(stored)) return defaultWidthPercent;
  if (stored < minWidthPercent || stored > 100) return defaultWidthPercent;
  return stored;
}

function ResizableSheetContent({
  className,
  children,
  storageKey,
  defaultWidthPercent = 75,
  minWidthPercent = 40,
  ...props
}: ResizableSheetContentProps) {
  const [width, setWidth] = React.useState(() => readStoredWidth(storageKey, defaultWidthPercent, minWidthPercent));
  const widthRef = React.useRef(width);
  const lastNonFullWidthRef = React.useRef(defaultWidthPercent);
  const cleanupRef = React.useRef<(() => void) | null>(null);
  const isFull = width >= 100;

  React.useEffect(() => () => cleanupRef.current?.(), []);

  const applyWidth = (next: number, persist: boolean) => {
    const clamped = clampWidth(next, minWidthPercent);
    widthRef.current = clamped;
    setWidth(clamped);
    if (persist) setLocalStorageItem(storageKey, String(clamped));
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    cleanupRef.current?.();
    const onMove = (e: PointerEvent) => {
      applyWidth(((window.innerWidth - e.clientX) / window.innerWidth) * 100, false);
    };
    const onUp = () => {
      cleanupRef.current?.();
      cleanupRef.current = null;
      setLocalStorageItem(storageKey, String(widthRef.current));
    };
    cleanupRef.current = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const toggle = () => {
    if (isFull) {
      applyWidth(lastNonFullWidthRef.current, true);
      return;
    }
    lastNonFullWidthRef.current = width;
    applyWidth(100, true);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      applyWidth(widthRef.current + 5, true);
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      applyWidth(widthRef.current - 5, true);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      applyWidth(100, true);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      applyWidth(minWidthPercent, true);
    }
  };

  return (
    <SheetContent
      side="right"
      className={cn(
        "data-[side=right]:w-full data-[side=right]:sm:w-(--sheet-width) data-[side=right]:sm:min-w-[min(720px,100%)] data-[side=right]:sm:max-w-none",
        className,
      )}
      style={{ "--sheet-width": `${width}%` } as React.CSSProperties}
      {...props}
    >
      {children}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize drawer"
        data-slot="sheet-resize-handle"
        className="absolute inset-y-0 left-0 hidden w-1.5 cursor-col-resize touch-none select-none hover:bg-border focus-visible:bg-border focus-visible:outline-none sm:block"
        tabIndex={0}
        aria-valuenow={Math.round(width)}
        aria-valuemin={minWidthPercent}
        aria-valuemax={100}
        onPointerDown={onPointerDown}
        onKeyDown={onKeyDown}
      />
      <Button
        variant="ghost"
        size="icon-sm"
        className="absolute top-4 right-14 hidden sm:inline-flex"
        aria-label={isFull ? "Collapse drawer" : "Expand drawer"}
        onClick={toggle}
      >
        {isFull ? <Minimize2Icon /> : <Maximize2Icon />}
      </Button>
    </SheetContent>
  );
}

export { ResizableSheetContent };
